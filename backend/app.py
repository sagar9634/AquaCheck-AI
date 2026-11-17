"""
app.py
------
Flask web server for AquaCheckAI. It
  * serves the website (../frontend)
  * exposes the ML model as an API  -> POST /api/predict
  * exposes the AI assistant        -> POST /api/chat

Run:  python app.py      then open  http://127.0.0.1:5000
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

import chatbot

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(HERE, "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND, static_url_path="")
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://aquacheck-ai-frontend.vercel.app"]
    }
})

model = joblib.load(os.path.join(HERE, "model", "model.joblib"))
with open(os.path.join(HERE, "model", "meta.json")) as fh:
    META = json.load(fh)
FEATURES = [f["key"] for f in META["features"]]
FEATURE_INFO = {f["key"]: f for f in META["features"]}


def probability(values):
    """Probability (0-1) that a sample is potable."""
    row = pd.DataFrame([values], columns=FEATURES)
    return float(model.predict_proba(row)[0][1])


@app.route("/")
def home():
    return send_from_directory(FRONTEND, "index.html")


@app.route("/api/meta")
def meta():
    return jsonify(META)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    values = data.get("values", {})
    try:
        values = {f: float(values[f]) for f in FEATURES}
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": f"Please send numeric values for: {', '.join(FEATURES)}"}), 400

    p = probability(values)

    # Simple explanation: replace each feature with the dataset median and see
    # how much the probability changes. Big positive impact = that value helps.
    contributions = []
    for f in FEATURES:
        tweaked = dict(values, **{f: FEATURE_INFO[f]["median"]})
        impact = (p - probability(tweaked)) * 100
        contributions.append({"feature": f, "label": FEATURE_INFO[f]["label"],
                              "unit": FEATURE_INFO[f]["unit"], "value": values[f],
                              "impact": round(impact, 2)})
    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)

    status = []
    for f in FEATURES:
        info, v = FEATURE_INFO[f], values[f]
        state = "ok" if info["safe_low"] <= v <= info["safe_high"] else ("low" if v < info["safe_low"] else "high")
        status.append({"feature": f, "label": info["label"], "state": state})

    return jsonify({
        "potable": p >= 0.5,
        "probability": round(p * 100, 1),
        "contributions": contributions,
        "status": status,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    return jsonify(chatbot.reply(data.get("message", ""), data.get("history", []), data.get("context")))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
