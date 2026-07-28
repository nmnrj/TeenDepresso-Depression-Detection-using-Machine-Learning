import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
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
print("Loading Clean Dataset...")
print("=" * 70)

df = pd.read_csv("data/processed/cleaned_dataset.csv")

# Safety check
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

print("TF-IDF Vectorizer Created Successfully!")

# ============================================================
# Parameter Grids
# ============================================================

lr_params = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}

svm_params = {
    "C": [0.01, 0.1, 1, 10, 100]
}

results = []

best_model = None
best_model_name = ""
best_f1 = 0

# ============================================================
# Logistic Regression Grid Search
# ============================================================

print("\n" + "=" * 70)
print("Tuning Logistic Regression...")
print("=" * 70)

lr_grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid=lr_params,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)

print("\nBest Parameters :", lr_grid.best_params_)
print("Best CV F1 Score :", round(lr_grid.best_score_, 4))

lr_model = lr_grid.best_estimator_

predictions = lr_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

results.append([
    "Logistic Regression",
    accuracy,
    precision,
    recall,
    f1
])

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, predictions))

if f1 > best_f1:
    best_f1 = f1
    best_model = lr_model
    best_model_name = "Logistic Regression"

# ============================================================
# Linear SVM Grid Search
# ============================================================

print("\n" + "=" * 70)
print("Tuning Linear SVM...")
print("=" * 70)

svm_grid = GridSearchCV(
    estimator=LinearSVC(),
    param_grid=svm_params,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

print("\nBest Parameters :", svm_grid.best_params_)
print("Best CV F1 Score :", round(svm_grid.best_score_, 4))

svm_model = svm_grid.best_estimator_

predictions = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

results.append([
    "Linear SVM",
    accuracy,
    precision,
    recall,
    f1
])

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, predictions))

if f1 > best_f1:
    best_f1 = f1
    best_model = svm_model
    best_model_name = "Linear SVM"

# ============================================================
# Save Models
# ============================================================

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(best_model, "models/best_model.pkl")

# Save tuned models separately
joblib.dump(lr_model, "models/tuned_logistic_regression.pkl")
joblib.dump(svm_model, "models/tuned_linear_svm.pkl")

# ============================================================
# Results
# ============================================================

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

print("\n" + "=" * 70)
print("TUNED MODEL COMPARISON")
print("=" * 70)

print(results_df)

print("\nBest Tuned Model :", best_model_name)
print(f"Best F1 Score    : {best_f1:.4f}")

print("\nModels Saved Successfully!")