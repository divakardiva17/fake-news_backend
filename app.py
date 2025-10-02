from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "backend/model.joblib"
VEC_PATH = "backend/vectorizer.joblib"

# Load pre-trained model + vectorizer
model, vectorizer = None, None
if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    print("✅ Model and vectorizer loaded")
else:
    print("⚠️ Model or vectorizer not found. Please add them in backend/")

@app.route("/")
def home():
    return {"message": "Fake News Detector API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model not available"}), 500
    
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].max()

    return jsonify({
        "prediction": "REAL" if pred == 1 else "FAKE",
        "confidence": round(float(proba) * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
