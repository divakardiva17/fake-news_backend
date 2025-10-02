from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, os

# ML imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
VEC_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.joblib")

model, vectorizer = None, None

def train_dummy_model():
    """Train a tiny fallback model if no model files exist."""
    global model, vectorizer

    print("⚠️ No model found. Training a dummy model...")
    df = pd.DataFrame({
        "text": [
            "Government confirms new benefits for citizens",
            "Breaking: celebrity endorses miracle cure",
            "Study shows benefits of walking for health",
            "Shocking: politician involved in secret cult"
        ],
        "label": ["real", "fake", "real", "fake"]
    })

    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.8, min_df=1)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)

    print("✅ Dummy model trained & saved.")

def load_artifacts():
    """Load existing model or train fallback if missing."""
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
        print("✅ Model & vectorizer loaded.")
    else:
        train_dummy_model()

@app.route("/", methods=["GET"])
def home():
    return {"message": "✅ Fake News Detector API is running!"}

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        if model is None or vectorizer is None:
            load_artifacts()

        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0].max())

        return jsonify({"label": str(pred), "probability": proba})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=True)
