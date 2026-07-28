# 🧠 TeenDepresso-Hybrid-NLP-Depression-Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![NLP](https://img.shields.io/badge/NLP-Sentence%20Transformers-green)
![License](https://img.shields.io/badge/License-MIT-red)

## 📌 Overview

TeenDepresso is a **Hybrid NLP-based Depression Detection System** developed to identify depressive text from user input using both **traditional machine learning** and **modern transformer-based semantic search**.

The project combines **TF-IDF + Logistic Regression** for text classification with **Sentence Transformers** for semantic similarity search, making the predictions more explainable by retrieving similar examples from the training dataset.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes only. It is **not** a medical diagnosis tool.

---

# 🚀 Features

- ✅ Text Cleaning & Preprocessing
- ✅ TF-IDF Feature Extraction
- ✅ Logistic Regression Classifier
- ✅ Hyperparameter Tuning using GridSearchCV
- ✅ Sentence Transformer Embeddings
- ✅ Semantic Search using Cosine Similarity
- ✅ Hybrid Prediction Pipeline
- ✅ Explainable AI using Similar Training Examples
- ✅ Confidence Score
- ✅ Risk-Level Detection
- ✅ User-Friendly Suggestions

---

# 🏗️ Project Architecture

```text
                    User Input
                         │
                         ▼
                 Text Preprocessing
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
TF-IDF Vectorizer             Sentence Transformer
        │                                 │
        ▼                                 ▼
Logistic Regression           Query Embedding
        │                                 │
        ▼                                 ▼
Depression Prediction      Cosine Similarity Search
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
            Explainable Hybrid Output
```

---

# 📂 Project Structure

```text
TeenDepresso/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── embeddings/
│   ├── sentence_embeddings.npy
│   └── sentences.csv
│
├── models/
│   ├── best_model.pkl
│   ├── logistic_regression.pkl
│   ├── linear_svm.pkl
│   ├── naive_bayes.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── prepare_clean_dataset.py
│   ├── train.py
│   ├── tune_models.py
│   ├── predict.py
│   ├── semantic_embeddings.py
│   ├── semantic_search.py
│   └── hybrid_predict.py
│
├── requirements.txt
├── README.md
└── main.py
```

---

# ⚙️ Technologies Used

- Python
- Scikit-Learn
- Pandas
- NumPy
- NLTK
- Sentence Transformers
- HuggingFace Transformers
- PyTorch
- Joblib

---

# 🧹 Text Preprocessing

The following preprocessing techniques were applied:

- Lowercasing
- URL Removal
- HTML Removal
- Emoji Removal
- Punctuation Removal
- Number Removal
- Tokenization
- Stopword Removal
- Lemmatization
- Whitespace Normalization

---

# 🤖 Machine Learning Models

The following classifiers were trained and evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

The final model was selected based on the **F1-score** after hyperparameter tuning.

---

# 🔍 Semantic Search

The project uses:

- **Sentence Transformer:** `all-MiniLM-L6-v2`
- **Embedding Dimension:** 384
- **Similarity Metric:** Cosine Similarity

Instead of relying only on classification, the system retrieves the **Top-3 semantically similar training examples** to make predictions more interpretable.

---

# 🖥️ Sample Output

<img width="1646" height="856" alt="Screenshot 2026-07-28 074400" src="https://github.com/user-attachments/assets/49946b57-47fb-404c-847c-ccfd3f0863c8" />


# 📊 Future Improvements

- Web Interface using Streamlit
- Explainable AI Dashboard
- Multilingual Support
- Emotion Detection
- Depression Severity Prediction
- RAG-based Mental Health Knowledge Retrieval
- LLM-powered Response Generation

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/nmnrj/TeenDepresso-Depression-Detection-using-Machine-Learning.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python src/hybrid_predict.py
```

---

# 📈 Results

- Hybrid NLP Pipeline
- Explainable Predictions
- Semantic Similarity Search
- Confidence Estimation
- Risk-Level Detection

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Naman Raj**

M.Tech (Data Science)

National Institute of Technology Patna

GitHub: https://github.com/nmnrj
