import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

print("=" * 70)
print("Generating Sentence Embeddings...")
print("=" * 70)

# Load dataset
df = pd.read_csv("data/processed/cleaned_dataset.csv")

# Remove empty rows
df.dropna(subset=["clean_text"], inplace=True)
df = df[df["clean_text"].str.strip() != ""]

sentences = df["clean_text"].tolist()

print(f"Total Sentences : {len(sentences)}")

# Load Sentence Transformer model
print("\nLoading Sentence Transformer Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded Successfully!")

# Generate embeddings
print("\nGenerating Embeddings...")

embeddings = model.encode(
    sentences,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nEmbedding Shape :", embeddings.shape)

# Create embeddings folder if it doesn't exist
os.makedirs("embeddings", exist_ok=True)

# Save embeddings
np.save("embeddings/sentence_embeddings.npy", embeddings)

# Save cleaned dataset to keep row alignment
df.to_csv("embeddings/sentences.csv", index=False)

print("\nEmbeddings Saved Successfully!")

print("\nSaved Files:")
print("embeddings/sentence_embeddings.npy")
print("embeddings/sentences.csv")

print("=" * 70)