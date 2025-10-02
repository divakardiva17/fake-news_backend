import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
real = pd.read_csv("data/True.csv")
fake = pd.read_csv("data/Fake.csv")

real["label"] = 1   # REAL
fake["label"] = 0   # FAKE

df = pd.concat([real, fake]).sample(frac=1).reset_index(drop=True)

X = df["text"]
y = df["label"]

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Save artifacts
joblib.dump(model, "backend/model.joblib")
joblib.dump(vectorizer, "backend/vectorizer.joblib")

print("✅ Saved model.joblib & vectorizer.joblib")

