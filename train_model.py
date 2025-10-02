import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Paths
DATA_DIR = "data"
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
REAL_PATH = os.path.join(DATA_DIR, "True.csv")

print("🔹 Loading dataset...")
fake = pd.read_csv(FAKE_PATH)
real = pd.read_csv(REAL_PATH)

# Add labels
fake["label"] = 0   # Fake = 0
real["label"] = 1   # Real = 1

# Combine datasets
df = pd.concat([fake, real], axis=0).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

print(f"✅ Dataset loaded: {df.shape[0]} samples")

# Use only title + text
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

X = df["content"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("🔹 Training Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {acc*100:.2f}%")

# Save model + vectorizer
os.makedirs("backend", exist_ok=True)
joblib.dump(model, "backend/model.joblib")
joblib.dump(vectorizer, "backend/vectorizer.joblib")

print("💾 Model and vectorizer saved to backend/")
