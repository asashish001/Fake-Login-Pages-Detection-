import os
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARTIFACT_PATH = os.path.join(MODELS_DIR, "train02_results.json")

if not os.path.isfile(ARTIFACT_PATH):
    raise FileNotFoundError(
        f"Missing: {ARTIFACT_PATH}\n"
        "Run training first: python pd2/train02.py"
    )

with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
    artifact = json.load(f)

MODEL_PATHS = artifact.get("models", [])
if not MODEL_PATHS or not all(os.path.isfile(p) for p in MODEL_PATHS):
    raise FileNotFoundError("One or more ensemble model files are missing in train02 artifact.")

IMG_SIZE = tuple(artifact.get("img_size", [300, 300]))
PRED_THRESHOLD = float(artifact.get("results", {}).get("threshold", 0.40))
ENSEMBLE_MODELS = [tf.keras.models.load_model(p, compile=False) for p in MODEL_PATHS]

template_dir = os.path.join(PROJECT_ROOT, "templates")
static_dir = os.path.join(PROJECT_ROOT, "static")
if not os.path.isdir(template_dir):
    template_dir = PROJECT_ROOT
if not os.path.isdir(static_dir):
    static_dir = PROJECT_ROOT

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)


def prepare_image(file_storage):
    img = Image.open(file_storage.stream).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/")
def home():
    return render_template("index.html", threshold=PRED_THRESHOLD)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not file.filename.lower().endswith(".png"):
        return jsonify({"error": "Only .png files are allowed"}), 400

    x = prepare_image(file)
    probs = [float(m.predict(x, verbose=0)[0][0]) for m in ENSEMBLE_MODELS]
    phishing_prob = float(np.mean(probs))

    label = "Phishing" if phishing_prob >= PRED_THRESHOLD else "Real"
    confidence = phishing_prob if label == "Phishing" else (1.0 - phishing_prob)
    return jsonify(
        {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "phishing_probability": round(phishing_prob * 100, 2),
            "threshold": PRED_THRESHOLD,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
