import argparse
import xml.etree.ElementTree as ET

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "TildeAI/TildeOpen-30b"
DEFAULT_XML = "tezaurs_2026_1_wordforms_tei.xml"


def local_name(tag: str) -> str:
    """Return an XML tag name without a namespace prefix."""
    return tag.rsplit("}", 1)[-1]


def direct_children(element, tag_name):
    """Yield direct children with the requested local tag name."""
    for child in element:
        if local_name(child.tag) == tag_name:
            yield child


def direct_child(element, tag_name):
    """Return the first direct child with the requested local tag name."""
    return next(direct_children(element, tag_name), None)


def get_entry_lemma(entry):
    """
    Read the lemma from:
        <entry type="supplemental">
          <form>
            <orth type="lemma">...</orth>
    """
    outer_form = direct_child(entry, "form")
    if outer_form is None:
        return None

    for child in direct_children(outer_form, "orth"):
        if child.get("type") == "lemma" and child.text:
            return child.text.strip()

    return None


def get_inflection_forms(entry):
    """
    Return unique <orth> values from direct:
        <form type="inflection"> ... </form>
    children of the supplemental entry's outer <form>.

    Order from the XML is preserved.
    """
    outer_form = direct_child(entry, "form")
    if outer_form is None:
        return []

    result = []
    seen = set()

    for form in direct_children(outer_form, "form"):
        if form.get("type") != "inflection":
            continue

        orth = direct_child(form, "orth")
        if orth is None or not orth.text:
            continue

        word = orth.text.strip()

        if word and word not in seen:
            seen.add(word)
            result.append(word)

    return result


def load_forms_for_lemma(xml_path: str, lemma: str):
    """
    Stream the large Tēzaurs wordforms XML and collect all unique inflected
    forms from every <entry type="supplemental"> whose
    <orth type="lemma"> equals `lemma`.

    The full XML is not loaded into memory.
    """
    forms = []
    seen = set()
    matching_entries = 0

    print(f'Searching XML for lemma: "{lemma}"')
    print(f"XML: {xml_path}")

    # Use END events so each <entry> is fully populated before inspection.
    context = ET.iterparse(xml_path, events=("end",))

    for _, elem in context:
        if local_name(elem.tag) != "entry":
            continue

        if elem.get("type") == "supplemental":
            entry_lemma = get_entry_lemma(elem)

            if entry_lemma == lemma:
                matching_entries += 1

                for word in get_inflection_forms(elem):
                    if word not in seen:
                        seen.add(word)
                        forms.append(word)

        # Important for a multi-GB XML file: release this completed entry.
        elem.clear()

    print(f"Matching supplemental entries: {matching_entries}")
    print(f"Unique forms found: {len(forms)}")

    return forms


def load_model():
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model.eval()

    print("Model loaded.")

    return tokenizer, model


def get_word_vector(word: str, tokenizer, model):
    """
    Build the same word-level embedding used in the previous experiment:
    average the input embeddings of all tokenizer tokens composing the word.
    """
    embedding_layer = model.get_input_embeddings()

    encoded = tokenizer(
        word,
        add_special_tokens=False,
        return_tensors="pt"
    )

    token_ids = encoded["input_ids"][0]

    if token_ids.numel() == 0:
        raise ValueError(f'Tokenizer produced no tokens for "{word}".')

    token_ids_device = token_ids.to(embedding_layer.weight.device)

    with torch.no_grad():
        vectors = embedding_layer(token_ids_device)
        word_vector = vectors.detach().float().mean(dim=0).cpu()

    return word_vector, token_ids.tolist()


def closest_words_from_list(
    word: str,
    candidate_words,
    tokenizer,
    model,
    top_k: int = 10
):
    """
    Compare `word` with complete candidate words using word-level embeddings.
    Euclidean distance is calculated between averaged token embeddings.
    """
    query_vector, query_ids = get_word_vector(
        word,
        tokenizer,
        model
    )

    results = []

    for candidate in candidate_words:
        if candidate == word:
            continue

        candidate_vector, candidate_ids = get_word_vector(
            candidate,
            tokenizer,
            model
        )

        distance = torch.dist(
            query_vector,
            candidate_vector,
            p=2
        ).item()

        cosine = torch.nn.functional.cosine_similarity(
            query_vector.unsqueeze(0),
            candidate_vector.unsqueeze(0)
        ).item()

        results.append(
            (
                candidate,
                distance,
                cosine,
                candidate_ids
            )
        )

    results.sort(key=lambda item: item[1])
    results = results[:top_k]

    print("\n" + "=" * 100)
    print(f'CLOSEST INFLECTIONS TO: "{word}"')
    print(f"Query token IDs: {query_ids}")
    print("=" * 100)

    if not results:
        print("No other unique forms are available for comparison.")
        return []

    for rank, (candidate, distance, cosine, token_ids) in enumerate(
        results,
        start=1
    ):
        print(
            f"{rank:2}. "
            f"{candidate:<25} "
            f"Euclidean: {distance:10.6f}   "
            f"Cosine: {cosine:9.6f}   "
            f"Token IDs: {token_ids}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find the closest TildeOpen word-level embeddings among all "
            "unique Tēzaurs inflection forms belonging to a lemma."
        )
    )

    parser.add_argument(
        "word",
        help='Lemma/query word, for example: māja'
    )

    parser.add_argument(
        "--xml",
        default=DEFAULT_XML,
        help=f"Tēzaurs wordforms TEI XML path (default: {DEFAULT_XML})"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of closest forms to display (default: 10)"
    )

    args = parser.parse_args()

    # Parse the huge XML BEFORE loading the 30B model, so XML parsing does not
    # compete with model memory and a missing lemma fails quickly.
    candidate_words = load_forms_for_lemma(
        args.xml,
        args.word
    )

    if not candidate_words:
        print(f'\nNo supplemental inflection forms found for lemma "{args.word}".')
        return

    print("\nUnique candidate forms:")
    for candidate in candidate_words:
        print(f"  {candidate}")

    tokenizer, model = load_model()

    closest_words_from_list(
        word=args.word,
        candidate_words=candidate_words,
        tokenizer=tokenizer,
        model=model,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
