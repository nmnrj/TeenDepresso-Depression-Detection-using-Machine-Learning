import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_text


# ==========================================================
# Load ML Model
# ==========================================================

print("\nLoading Depression Detection Model...")

model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("Model Loaded Successfully!")

# ==========================================================
# Load Semantic Search Data
# ==========================================================

print("\nLoading Semantic Embeddings...")

embeddings = np.load("embeddings/sentence_embeddings.npy")

df = pd.read_csv("embeddings/sentences.csv")
sentences = df["clean_text"].tolist()

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Semantic Search Ready!")

print("=" * 70)
print("TeenDepresso - Hybrid NLP System")
print("=" * 70)

while True:

    text = input("\nEnter your thoughts (type 'exit' to quit):\n\n")

    if text.lower() == "exit":
        break

    # ------------------------------------------------------
    # Classification
    # ------------------------------------------------------

    cleaned = preprocess_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    confidence = model.predict_proba(vector).max() * 100

    label = "Depression" if prediction == 1 else "Normal"

    critical_keywords = [
    "suicide",
    "kill myself",
    "end my life",
    "want to die",
    "don't want to live",
    "self harm",
    "harm myself"
]

    text_lower = text.lower()

    critical = any(keyword in text_lower for keyword in critical_keywords)

    # ------------------------------------------------------
    # Semantic Search
    # ------------------------------------------------------

    query_embedding = semantic_model.encode(
        [cleaned],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = similarities.argsort()[-3:][::-1]


    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    print("\n" + "=" * 70)

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2f}%")

    if critical:
        print("Risk Level : 🔴 CRITICAL")
    elif confidence >= 90:
        print("Status     : High Confidence")
    elif confidence >= 75:
        print("Status     : Medium Confidence")
    else:
        print("Status     : Low Confidence")

    print("\nTop Similar Sentences\n")

    for i, idx in enumerate(top_indices, start=1):

        print("-" * 60)

        print(f"{i}. Similarity : {similarities[idx]:.4f}")
        print(sentences[idx])

    print("-" * 60)

    # ------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------

    if prediction == 1:

        print("\nSuggestion:")
        print("• Consider reaching out to a trusted friend or family member.")
        print("• If these feelings persist, consult a mental health professional.")
        print("• Remember that seeking help is a sign of strength.")

    else:

        print("\nSuggestion:")
        print("• Continue maintaining healthy habits.")
        print("• Stay connected with family and friends.")
        print("• Take regular breaks and prioritize self-care.")

    print("\nDisclaimer:")
    print("This project is intended for educational and research purposes only.")
    print("It is NOT a medical diagnosis.")

    print("=" * 70)