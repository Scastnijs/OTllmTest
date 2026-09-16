import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "TildeAI/TildeOpen-30b"


print("Loading tokenizer...")

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

print("Model loaded.")


def compare_words(word1: str, word2: str):
    """
    Print:
    - token(s)
    - token ID(s)
    - embedding vector(s)
    - averaged word embedding
    - Euclidean distance
    - cosine similarity
    """

    embedding_layer = model.get_input_embeddings()

    def get_word_embedding(word):
        # Do not add BOS/EOS tokens.
        encoded = tokenizer(
            word,
            add_special_tokens=False,
            return_tensors="pt"
        )

        token_ids = encoded["input_ids"][0]

        print(f"\nWord: {word}")
        print("-" * 80)

        vectors = []

        for token_id in token_ids:
            token_id_int = token_id.item()

            token_text = tokenizer.convert_ids_to_tokens(token_id_int)

            # Embedding lookup.
            # Shape becomes: [6144]
            vector = embedding_layer(
                token_id.to(embedding_layer.weight.device)
            )

            vector = vector.detach().float().cpu()

            vectors.append(vector)

            print(f"Token:    {token_text}")
            print(f"Token ID: {token_id_int}")
            print(f"Shape:    {tuple(vector.shape)}")

            print("Embedding vector:")
            print(vector)

            print()

        # If a word consists of multiple tokens,
        # average their embeddings into one vector.
        word_vector = torch.stack(vectors).mean(dim=0)

        print("Word-level embedding:")
        print(word_vector)

        print(f"Word vector shape: {tuple(word_vector.shape)}")

        return word_vector

    vector1 = get_word_embedding(word1)
    vector2 = get_word_embedding(word2)

    # Euclidean distance
    euclidean_distance = torch.dist(
        vector1,
        vector2,
        p=2
    ).item()

    # Cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(
        vector1.unsqueeze(0),
        vector2.unsqueeze(0)
    ).item()

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"Word 1: {word1}")
    print(f"Word 2: {word2}")

    print(f"\nEuclidean distance: {euclidean_distance:.6f}")
    print(f"Cosine similarity:  {cosine_similarity:.6f}")

def closest_words(word: str, top_k: int = 10):
    """
    Find the closest standalone vocabulary words to `word`
    using Euclidean distance between input embeddings.

    The query word may consist of multiple tokens. Its word-level
    embedding is the average of those token embeddings, matching
    the logic used by compare_words().

    Candidate words are standalone vocabulary tokens that decode
    to word-like strings.
    """

    embedding_layer = model.get_input_embeddings()
    device = embedding_layer.weight.device

    # ---------------------------------------------------------
    # 1. Create word-level embedding for the input word
    # ---------------------------------------------------------
    encoded = tokenizer(
        word,
        add_special_tokens=False,
        return_tensors="pt"
    )

    token_ids = encoded["input_ids"][0]

    vectors = []

    for token_id in token_ids:
        vector = embedding_layer(
            token_id.to(device)
        )

        vectors.append(
            vector.detach().float()
        )

    query_vector = torch.stack(vectors).mean(dim=0)

    print(f"\nWord: {word}")
    print(f"Token IDs: {token_ids.tolist()}")
    print(f"Number of tokens: {len(token_ids)}")

    # ---------------------------------------------------------
    # 2. Calculate distance to every token embedding
    # ---------------------------------------------------------
    weights = embedding_layer.weight.detach().float()

    distances = torch.norm(
        weights - query_vector.unsqueeze(0),
        p=2,
        dim=1
    )

    # Sort all vocabulary entries from nearest to farthest
    sorted_ids = torch.argsort(distances)

    # ---------------------------------------------------------
    # 3. Keep only entries that look like standalone words
    # ---------------------------------------------------------
    results = []
    seen = set()

    special_ids = set(tokenizer.all_special_ids)

    for token_id_tensor in sorted_ids:
        token_id = token_id_tensor.item()

        if token_id in special_ids:
            continue

        # Decode the token rather than displaying tokenizer
        # internals such as SentencePiece/BPE markers.
        candidate = tokenizer.decode(
            [token_id],
            skip_special_tokens=True
        ).strip()

        if not candidate:
            continue

        # Keep alphabetic standalone words.
        # str.isalpha() works with Latvian Unicode letters too.
        if not candidate.isalpha():
            continue

        candidate_normalized = candidate.lower()

        # Exclude the query itself
        if candidate_normalized == word.lower():
            continue

        # Avoid duplicate decoded words
        if candidate_normalized in seen:
            continue

        seen.add(candidate_normalized)

        results.append(
            (
                candidate,
                token_id,
                distances[token_id].item()
            )
        )

        if len(results) == top_k:
            break

    # ---------------------------------------------------------
    # 4. Print results
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{top_k} CLOSEST WORDS TO: {word}")
    print("=" * 80)

    for rank, (candidate, token_id, distance) in enumerate(
        results,
        start=1
    ):
        print(
            f"{rank:2}. "
            f"{candidate:<25} "
            f"Token ID: {token_id:<8} "
            f"Distance: {distance:.6f}"
        )

    return results

def get_word_vector(word: str):
    """
    Return a word-level embedding using the same method
    as compare_words():

    1. tokenize without BOS/EOS
    2. get input embedding for every token
    3. average token embeddings
    """

    embedding_layer = model.get_input_embeddings()

    encoded = tokenizer(
        word,
        add_special_tokens=False,
        return_tensors="pt"
    )

    token_ids = encoded["input_ids"][0]

    vectors = []

    for token_id in token_ids:
        vector = embedding_layer(
            token_id.to(embedding_layer.weight.device)
        )

        vector = vector.detach().float().cpu()

        vectors.append(vector)

    word_vector = torch.stack(vectors).mean(dim=0)

    return word_vector, token_ids


def test_inflections(words: list[str]):
    """
    Compare Latvian inflected forms.

    Prints:
    - tokenizer representation
    - distance from the first/base word
    - complete pairwise Euclidean-distance matrix
    - forms sorted by distance from the base word
    """

    if len(words) < 2:
        raise ValueError("Provide at least two word forms.")

    print("\n" + "=" * 90)
    print("LATVIAN INFLECTION EMBEDDING TEST")
    print("=" * 90)

    vectors = {}
    token_info = {}

    # ---------------------------------------------------------
    # 1. Calculate word-level embeddings
    # ---------------------------------------------------------

    for word in words:

        vector, token_ids = get_word_vector(word)

        vectors[word] = vector
        token_info[word] = token_ids.tolist()

        tokens = tokenizer.convert_ids_to_tokens(
            token_ids.tolist()
        )

        print(f"\nWord:      {word}")
        print(f"Tokens:    {tokens}")
        print(f"Token IDs: {token_ids.tolist()}")
        print(f"Token count: {len(token_ids)}")

    # ---------------------------------------------------------
    # 2. Use first word as base form
    # ---------------------------------------------------------

    base_word = words[0]
    base_vector = vectors[base_word]

    print("\n" + "=" * 90)
    print(f"DISTANCE FROM BASE WORD: {base_word}")
    print("=" * 90)

    distances_from_base = []

    for word in words:

        distance = torch.dist(
            base_vector,
            vectors[word],
            p=2
        ).item()

        cosine = torch.nn.functional.cosine_similarity(
            base_vector.unsqueeze(0),
            vectors[word].unsqueeze(0)
        ).item()

        distances_from_base.append(
            (word, distance, cosine)
        )

    distances_from_base.sort(
        key=lambda x: x[1]
    )

    for rank, (word, distance, cosine) in enumerate(
        distances_from_base,
        start=1
    ):
        print(
            f"{rank:2}. "
            f"{word:<15} "
            f"Euclidean: {distance:10.6f}   "
            f"Cosine: {cosine:.6f}"
        )

    # ---------------------------------------------------------
    # 3. Pairwise distance matrix
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("PAIRWISE EUCLIDEAN DISTANCES")
    print("=" * 90)

    column_width = 12

    print(
        "".ljust(column_width),
        end=""
    )

    for word in words:
        print(
            word[:column_width - 1].ljust(column_width),
            end=""
        )

    print()

    for word1 in words:

        print(
            word1[:column_width - 1].ljust(column_width),
            end=""
        )

        for word2 in words:

            distance = torch.dist(
                vectors[word1],
                vectors[word2],
                p=2
            ).item()

            print(
                f"{distance:<12.4f}",
                end=""
            )

        print()

    # ---------------------------------------------------------
    # 4. Find globally closest pair
    # ---------------------------------------------------------

    pairs = []

    for i in range(len(words)):

        for j in range(i + 1, len(words)):

            word1 = words[i]
            word2 = words[j]

            distance = torch.dist(
                vectors[word1],
                vectors[word2],
                p=2
            ).item()

            pairs.append(
                (word1, word2, distance)
            )

    pairs.sort(
        key=lambda x: x[2]
    )

    print("\n" + "=" * 90)
    print("CLOSEST INFLECTION PAIRS")
    print("=" * 90)

    for rank, (word1, word2, distance) in enumerate(
        pairs,
        start=1
    ):

        print(
            f"{rank:2}. "
            f"{word1:<12} ↔ "
            f"{word2:<12} "
            f"{distance:.6f}"
        )

# compare_words("māja", "māju")

# closest_words("māja")

maja_forms = [
    "māja",      # nominative singular
    "mājas",     # genitive singular
    "mājai",     # dative singular
    "māju",      # accusative singular
    "mājā",      # locative singular
    "mājas",     # nominative plural / genitive singular
    "mājām",     # dative plural
    "mājās"      # locative plural
]

test_inflections(maja_forms)
