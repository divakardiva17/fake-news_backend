from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load model + vectorizer (must exist in backend/ after training)
MODEL_PATH = os.path.join("backend", "model.joblib")
VECTORIZER_PATH = os.path.join("backend", "vectorizer.joblib")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and vectorizer loaded successfully")
except Exception as e:
    print("❌ Error loading model/vectorizer:", e)
    model, vectorizer = None, None


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fake News Detector API is running!"})


@app.route("/api/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded. Please train first."}), 500

    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Transform and predict
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][prediction]

        result = "REAL" if prediction == 1 else "FAKE"
        return jsonify({
            "prediction": result,
            "confidence": round(float(proba) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

