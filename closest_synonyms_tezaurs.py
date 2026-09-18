import argparse
import heapq
import time
import xml.etree.ElementTree as ET
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TildeAI/TildeOpen-30b"
DEFAULT_XML = "tezaurs_2026_1_wordforms_tei.xml"


def format_duration(seconds):
    """Format elapsed seconds as HH:MM:SS.mmm."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def direct_children(element, name):
    for child in element:
        if local_name(child.tag) == name:
            yield child


def direct_child(element, name):
    return next(direct_children(element, name), None)


def get_entry_lemma(entry):
    outer = direct_child(entry, "form")
    if outer is None:
        return None
    for orth in direct_children(outer, "orth"):
        if orth.get("type") == "lemma" and orth.text:
            return orth.text.strip()
    return None


def gram_values(form):
    values = {}
    group = direct_child(form, "gramGrp")
    if group is not None:
        for gram in direct_children(group, "gram"):
            if gram.get("type") and gram.text:
                values[gram.get("type")] = gram.text.strip()
    return values


def form_word(form):
    orth = direct_child(form, "orth")
    return orth.text.strip() if orth is not None and orth.text else None


def is_nominative_singular_noun(form):
    if form.get("type") != "inflection":
        return False
    g = gram_values(form)
    return (g.get("Locījums") == "Nominatīvs"
            and g.get("Skaitlis") == "Vienskaitlis"
            and g.get("Vārdšķira") == "Lietvārds")


def scan_tezaurs(xml_path, query_lemma):
    """Return unique nominative-singular nouns and all inflections of query lemma."""
    nouns, noun_seen = [], set()
    query_inflections = {query_lemma}
    query_entries = 0
    entries = 0

    scan_start = time.perf_counter()

    print(f'Scanning: {xml_path}')
    print(f'Query lemma: "{query_lemma}"')

    for _, entry in ET.iterparse(xml_path, events=("end",)):
        if local_name(entry.tag) != "entry":
            continue
        if entry.get("type") != "supplemental":
            entry.clear()
            continue

        entries += 1
        outer = direct_child(entry, "form")
        if outer is None:
            entry.clear()
            continue

        lemma = get_entry_lemma(entry)
        is_query = lemma == query_lemma
        if is_query:
            query_entries += 1

        for form in direct_children(outer, "form"):
            if form.get("type") != "inflection":
                continue
            word = form_word(form)
            if not word:
                continue

            # Exclude ALL singular/plural declensions of the query lemma.
            if is_query:
                query_inflections.add(word)

            # Search source: nouns that are nominative + singular only.
            if is_nominative_singular_noun(form) and word not in noun_seen:
                noun_seen.add(word)
                nouns.append(word)

        entry.clear()

    print(f"Supplemental entries scanned: {entries:,}")
    print(f"Unique nominative-singular nouns: {len(nouns):,}")
    print(f"Matching query entries: {query_entries}")
    print(f"Excluded query forms ({len(query_inflections)}): "
          + ", ".join(sorted(query_inflections)))
    scan_elapsed = time.perf_counter() - scan_start
    print(f"Scanning time: {format_duration(scan_elapsed)}")
    return nouns, query_inflections


def load_model():
    print("\nLoading tokenizer...")
    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer_elapsed = time.perf_counter() - tokenizer_start
    print(f"Tokenizer loaded in: {format_duration(tokenizer_elapsed)}")

    print("Loading model...")
    model_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    model_elapsed = time.perf_counter() - model_start
    print(f"Model loaded in: {format_duration(model_elapsed)}")
    print(f"Tokenizer + model load time: {format_duration(tokenizer_elapsed + model_elapsed)}")
    return tokenizer, model


def get_word_vector(word, tokenizer, model):
    """Mean of input-token embeddings, same word-level definition as before."""
    layer = model.get_input_embeddings()
    ids = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if ids.numel() == 0:
        raise ValueError(f'No tokens produced for "{word}"')
    with torch.no_grad():
        vectors = layer(ids.to(layer.weight.device))
        vector = vectors.detach().float().mean(dim=0).cpu()
    return vector, ids.tolist()


def closest_synonyms_from_list(word, candidate_words, excluded_words,
                               tokenizer, model, top_k=10):
    """
    Rank unique nominative-singular nouns by Euclidean distance.
    Query lemma and all its singular/plural inflections are excluded.

    'Synonym' here means embedding-nearest noun; it is not a lexical
    synonym guarantee.
    """
    processing_start = time.perf_counter()
    query_vector, query_ids = get_word_vector(word, tokenizer, model)
    excluded = set(excluded_words) | {word}
    seen = set()
    best = []
    compared = skipped = 0
    total = len(candidate_words)

    print(f"\nComparing against {total:,} unique nominative-singular nouns...")

    for i, candidate in enumerate(candidate_words, 1):
        if candidate in seen or candidate in excluded:
            skipped += 1
            continue
        seen.add(candidate)

        candidate_vector, candidate_ids = get_word_vector(candidate, tokenizer, model)
        distance = torch.dist(query_vector, candidate_vector, p=2).item()
        cosine = torch.nn.functional.cosine_similarity(
            query_vector.unsqueeze(0), candidate_vector.unsqueeze(0)
        ).item()
        compared += 1

        # Negative distance => root is worst retained candidate.
        item = (-distance, candidate, cosine, candidate_ids)
        if len(best) < top_k:
            heapq.heappush(best, item)
        elif distance < -best[0][0]:
            heapq.heapreplace(best, item)

        if i % 5000 == 0 or i == total:
            elapsed = time.perf_counter() - processing_start
            rate = compared / elapsed if elapsed > 0 else 0.0
            remaining = total - i
            eta = remaining / rate if rate > 0 else 0.0
            print(
                f"Processed {i:,}/{total:,} "
                f"(compared {compared:,}, skipped {skipped:,}) | "
                f"elapsed {format_duration(elapsed)} | "
                f"{rate:.1f} words/s | "
                f"ETA {format_duration(eta)}"
            )

    results = [(w, -neg_d, cos, ids) for neg_d, w, cos, ids in best]
    results.sort(key=lambda x: x[1])

    print("\n" + "=" * 105)
    print(f'CLOSEST NOMINATIVE-SINGULAR NOUNS TO: "{word}"')
    print(f"Query token IDs: {query_ids}")
    print("=" * 105)

    for rank, (candidate, distance, cosine, ids) in enumerate(results, 1):
        print(f"{rank:2}. {candidate:<30} Euclidean: {distance:10.6f}   "
              f"Cosine: {cosine:9.6f}   Token IDs: {ids}")

    processing_elapsed = time.perf_counter() - processing_start
    print(f"\nProcessing time: {format_duration(processing_elapsed)}")
    if compared:
        print(f"Average processing time per compared word: "
              f"{processing_elapsed / compared * 1000:.3f} ms")
        print(f"Average processing speed: {compared / processing_elapsed:.1f} words/s")
    return results


def main():
    total_start = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Search Tēzaurs nominative-singular nouns by TildeOpen word-level embedding."
    )
    parser.add_argument("word", help='Query lemma, e.g. "māja"')
    parser.add_argument("--xml", default=DEFAULT_XML,
                        help=f"Tēzaurs wordforms XML (default: {DEFAULT_XML})")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    candidates, excluded = scan_tezaurs(args.xml, args.word)
    if not candidates:
        print("No nominative-singular noun candidates found.")
        return

    tokenizer, model = load_model()
    closest_synonyms_from_list(
        args.word, candidates, excluded, tokenizer, model, args.top_k
    )

    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 105)
    print(f"TOTAL EXECUTION TIME: {format_duration(total_elapsed)}")
    print("=" * 105)


if __name__ == "__main__":
    main()
