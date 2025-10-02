import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib, os

HERE = os.path.dirname(__file__)
MODEL_OUT = os.path.join(HERE, "model.joblib")
VEC_OUT = os.path.join(HERE, "vectorizer.joblib")

DATA_CSV = os.path.join(HERE, "fake_news_sample.csv")

if not os.path.exists(DATA_CSV):
    df = pd.DataFrame({
        "text": [
            "Government confirms new benefits for citizens",
            "Breaking: celebrity endorses miracle cure",
            "Study shows benefits of walking for health",
            "Shocking: politician involved in secret cult"
        ],
        "label": ["real", "fake", "real", "fake"]
    })
else:
    df = pd.read_csv(DATA_CSV)

X = df["text"].astype(str)
y = df["label"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.8, min_df=1)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

joblib.dump(model, MODEL_OUT)
joblib.dump(vectorizer, VEC_OUT)

print("✅ Model saved:", MODEL_OUT)
print("✅ Vectorizer saved:", VEC_OUT)
