from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
VEC_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.joblib")

model, vectorizer = None, None

def load_artifacts():
    global model, vectorizer
    if model is None or vectorizer is None:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    load_artifacts()
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0].max())

    return jsonify({"label": str(pred), "probability": proba})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


