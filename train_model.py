import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# -----------------------------
# 1. Load dataset (from Kaggle download in /data/)
# -----------------------------
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels: 0 = fake, 1 = true
fake["label"] = 0
true["label"] = 1

# Merge datasets
df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

X = df["text"]
y = df["label"]

# -----------------------------
# 2. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Build pipeline (TF-IDF + Logistic Regression)
# -----------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000)),
])

# -----------------------------
# 4. Train model
# -----------------------------
print("Training model...")
pipeline.fit(X_train, y_train)
print("✅ Training complete")

# -----------------------------
# 5. Save model + vectorizer
# -----------------------------
os.makedirs("backend", exist_ok=True)
joblib.dump(pipeline.named_steps["clf"], "backend/model.joblib")
joblib.dump(pipeline.named_steps["tfidf"], "backend/vectorizer.joblib")

print("✅ Model and vectorizer saved in backend/")
