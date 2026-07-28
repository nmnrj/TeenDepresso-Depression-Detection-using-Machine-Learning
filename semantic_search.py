import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 70)
print("Semantic Search")
print("=" * 70)

# ----------------------------------------------------
# Load Embeddings
# ----------------------------------------------------

print("\nLoading Embeddings...")

embeddings = np.load("embeddings/sentence_embeddings.npy")

print("Embeddings Shape :", embeddings.shape)

# ----------------------------------------------------
# Load Sentences
# ----------------------------------------------------

df = pd.read_csv("embeddings/sentences.csv")

sentences = df["clean_text"].tolist()

print("Total Sentences :", len(sentences))

# ----------------------------------------------------
# Load Sentence Transformer
# ----------------------------------------------------

print("\nLoading Sentence Transformer...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded Successfully!")

# ----------------------------------------------------
# Search Loop
# ----------------------------------------------------

while True:

    query = input("\nEnter a sentence (or type 'exit'):\n\n")

    if query.lower() == "exit":
        break

    # Generate query embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # Get top 5 indices
    top_indices = similarities.argsort()[-5:][::-1]

    print("\nTop 5 Similar Sentences\n")

    for rank, idx in enumerate(top_indices, start=1):

        print("-" * 60)

        print(f"Rank       : {rank}")
        print(f"Similarity : {similarities[idx]:.4f}")
        print(f"Sentence   : {sentences[idx]}")

    print("-" * 60)