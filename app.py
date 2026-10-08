"""Simple Flask app: paste a message, find out if it is spam."""
import os
import pickle
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Train automatically if the model file is missing (e.g. fresh deploy)
if not os.path.exists("model.pkl"):
    import train  # noqa: F401  (running train.py creates model.pkl)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def check(text: str) -> dict:
    spam_prob = float(model.predict_proba([text])[0][1])
    return {
        "label": "spam" if spam_prob >= 0.5 else "ham",
        "spam_probability": round(spam_prob * 100, 1),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not text:
        return jsonify(error="Please enter a message"), 400
    return jsonify(check(text))


if __name__ == "__main__":
    app.run(debug=True)
