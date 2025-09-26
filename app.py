from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# If using NLTK resources, ensure you have downloaded them
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

app = Flask(__name__)
CORS(app)

# Load your trained model
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Also load any vectorizer you used (TF-IDF, CountVectorizer) 
# Suppose you saved it similarly as "vectorizer.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # basic cleaning & preprocessing
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove non-letters
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    cleaned = " ".join(words)
    return cleaned

@app.route("/")
def home():
    return "Fake News Detection API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Expect JSON: {"text": "some news article text here"}
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    raw_text = data["text"]
    processed = preprocess_text(raw_text)
    vect = vectorizer.transform([processed])  # transform into feature vector
    prediction = model.predict(vect)[0]
    # If your labels are numeric (0/1), you may want to map to "FAKE"/"REAL"
    # Example:
    label_map = {0: "FAKE", 1: "REAL"}
    label = label_map.get(prediction, str(prediction))
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
