import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 70)
print("Loading Dataset...")
print("=" * 70)

df = pd.read_csv("data/processed/cleaned_dataset.csv")
df.dropna(subset=["clean_text"], inplace=True)

df = df[df["clean_text"].str.strip() != ""]
print("Dataset Shape :", df.shape)

X = df["clean_text"]
y = df["label"]

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print("\nCreating TF-IDF Vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Vectorizer Saved Successfully!")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

results = []

best_model = None
best_f1 = 0
best_model_name = ""
print("\n" + "=" * 70)
print("Training Models...")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(f"\n{name}")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report\n")

    print(classification_report(y_test, predictions))

    print("Confusion Matrix\n")

    print(confusion_matrix(y_test, predictions))

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(model, f"models/{filename}")

if f1 > best_f1:
    best_f1 = f1
    best_model = model
    best_model_name = name

       

joblib.dump(best_model, "models/best_model.pkl")

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

print(results_df)

print("\nBest Model :", best_model_name)
print(f"F1 Score   : {best_f1:.4f}")

print("\nBest Model Saved Successfully!")