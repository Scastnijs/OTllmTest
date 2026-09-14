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


compare_words("māja", "māju")