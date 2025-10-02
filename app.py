from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load pre-trained model + vectorizer
model = joblib.load("backend/model.joblib")
vectorizer = joblib.load("backend/vectorizer.joblib")

@app.route("/")
def home():
    return {"message": "Fake News Detector API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
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
