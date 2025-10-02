import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib, os

HERE = os.path.dirname(__file__)
MODEL_OUT = os.path.join(HERE, "model.joblib")
VEC_OUT = os.path.join(HERE, "vectorizer.joblib")

# 🔹 Load dataset
DATA_CSV = os.path.join(HERE, "news.csv")

if not os.path.exists(DATA_CSV):
    raise FileNotFoundError("⚠️ news.csv dataset not found! Place it in backend/ folder.")

df = pd.read_csv(DATA_CSV)

# Assume dataset has columns 'text' and 'label' (adjust if different)
X = df["text"].astype(str)
y = df["label"].astype(str)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.8, min_df=5, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

# Logistic Regression Classifier
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# Save trained model + vectorizer
joblib.dump(model, MODEL_OUT)
joblib.dump(vectorizer, VEC_OUT)

print("✅ Model trained and saved!")
print("📊 Training accuracy:", model.score(X_train_vec, y_train))

# Evaluate on test set
X_test_vec = vectorizer.transform(X_test)
print("📊 Test accuracy:", model.score(X_test_vec, y_test))
