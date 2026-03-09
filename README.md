# Fake Login Page Detection (Computer Vision + Deep Learning)

This project detects whether a login-page screenshot is **Real** or **Phishing** using a transfer-learning CNN (MobileNetV2) and serves predictions through a Flask web app.

## Features
- TensorFlow/Keras image pipeline (`image_dataset_from_directory`)
- Transfer learning with MobileNetV2
- Class-weighted training (phishing class emphasized to reduce false negatives)
- Evaluation with Confusion Matrix, ROC-AUC, and F1-score
- Real-time Flask inference API + modern frontend UI (upload, preview, confidence bar)

## Project Structure
```text
LPD/
├─ Dataset/
│  ├─ real/
│  └─ phishing/
├─ models/
│  ├─ phish_detector.keras
│  ├─ train_test_metrics.json
│  ├─ eval_results.json
│  └─ roc_curve.png
├─ train.py
├─ evaluate.py
├─ app.py
├─ index.html
└─ app.js
```

## Requirements
- Python 3.10+ (works with your Python 3.13 setup)
- pip packages:
  - `tensorflow`
  - `scikit-learn`
  - `flask`
  - `pillow`
  - `matplotlib`
  - `numpy`

Install:
```bash
pip install tensorflow scikit-learn flask pillow matplotlib numpy
```

## Dataset Format
Keep screenshots as `.png` files in:
```text
Dataset/
  real/
  phishing/
```

`train.py` and `evaluate.py` auto-detect dataset folders named: `data`, `Data`, `dataset`, or `Dataset`.

## Train
```bash
python train.py
```

Output:
- Trained model: `models/phish_detector.keras`
- Training metrics: `models/train_test_metrics.json`

## Evaluate
```bash
python evaluate.py
```

Outputs:
- `models/eval_results.json`
- `models/roc_curve.png`

Latest run (threshold = `0.40`):
- Confusion Matrix: `TN=69, FP=93, FN=14, TP=160`
- ROC-AUC: `0.7838`
- F1 (phishing): `0.7494`

## Run Web App
```bash
python app.py
```

Then open:
- `http://127.0.0.1:5000`

### Web App Behavior
- Upload `.png` screenshot
- Live preview
- Predicts `Real` or `Phishing`
- Shows phishing probability + confidence bar

## API Endpoint
`POST /predict`

Form-data:
- `file`: PNG image

Response example:
```json
{
  "label": "Phishing",
  "confidence": 87.15,
  "phishing_probability": 87.15,
  "threshold": 0.4
}
```

## Notes
- `FN_phishing_as_real` is the key safety metric for this project.
- Lowering prediction threshold (currently `0.40`) usually reduces FN but can increase FP.
- TensorFlow oneDNN log messages are informational, not errors.
