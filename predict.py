import joblib

from preprocessing import preprocess_text

# Load Vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load Best Model
model = joblib.load("models/best_model.pkl")

print("=" * 70)
print("TeenDepresso - Depression Detection")
print("=" * 70)

while True:

    text = input("\nEnter your text (or type 'exit' to quit):\n\n")

    if text.lower() == "exit":
        print("\nThank you for using TeenDepresso!")
        break

    # Preprocess
    clean_text = preprocess_text(text)

    # TF-IDF
    vector = vectorizer.transform([clean_text])

    # Prediction
    prediction = model.predict(vector)[0]

    # Confidence
    probabilities = model.predict_proba(vector)[0]
    confidence = max(probabilities) * 100

    print("\n" + "-" * 60)

    if prediction == 1:
        print("Prediction : Depression")
        print("Suggestion : Consider talking to a trusted friend or a mental health professional if these feelings persist.")
    else:
        print("Prediction : Normal")
        print("Suggestion : No signs of depression detected from the provided text.")

    print(f"Confidence : {confidence:.2f}%")

    print("-" * 60)