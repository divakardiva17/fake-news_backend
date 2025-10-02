from kaggle.api.kaggle_api_extended import KaggleApi
import os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

HERE = os.path.dirname(__file__)
csv_path = os.path.join(HERE, "news.csv")

# Download dataset from Kaggle if not exists
if not os.path.exists(csv_path):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("clmentbisaillon/fake-and-real-news-dataset", path=HERE, unzip=True)
    df_true = pd.read_csv(os.path.join(HERE, "True.csv"))
    df_fake = pd.read_csv(os.path.join(HERE, "Fake.csv"))
    df_true["label"] = "real"
    df_fake["label"] = "fake"
    df = pd.concat([df_true[["text","label"]], df_fake[["text","label"]]])
    df.to_csv(csv_path, index=False)

# Train model
df = pd.read_csv(csv_path)
X = df["text"].astype(str)
y = df["label"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.8, min_df=5, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

joblib.dump(model, os.path.join(HERE, "model.joblib"))
joblib.dump(vectorizer, os.path.join(HERE, "vectorizer.joblib"))
