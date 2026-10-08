"""Train a spam classifier (TF-IDF + Naive Bayes) and save it to model.pkl."""
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. Load data (columns: label, message)
df = pd.read_csv("spam.csv")
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 2. Split into train / test
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. Build model: text -> TF-IDF numbers -> Naive Bayes
model = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2)),
    MultinomialNB(alpha=0.1),
)
model.fit(X_train, y_train)

# 4. Evaluate
pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.2%}\n")
print(classification_report(y_test, pred, target_names=["ham", "spam"]))

# 5. Save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to model.pkl")
