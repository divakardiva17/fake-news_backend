import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Tiny sample dataset ---
texts = [
    "Breaking news: The president will resign tomorrow",   # real
    "Shocking! Aliens land in New York City",              # fake
    "Scientists discover cure for common cold",            # real
    "You won’t believe this miracle weight loss trick",    # fake
]
labels = [1, 0, 1, 0]  # 1 = REAL, 0 = FAKE

# --- Train vectorizer + model ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

# --- Save artifacts ---
joblib.dump(model, "backend/model.joblib")
joblib.dump(vectorizer, "backend/vectorizer.joblib")

print("✅ model.joblib and vectorizer.joblib created in backend/")
