import os
import json
import numpy as np
import tensorflow as tf

from train02 import (
    DATA_DIR,
    MODELS_DIR,
    TTA_PASSES,
    ensure_dataset_layout,
    collect_samples,
    clean_samples,
    stratified_group_split,
    build_dataset,
    labels_array,
    predict_probs,
    evaluate_probs,
)


ARTIFACT_PATH = os.path.join(MODELS_DIR, "train02_results.json")


def main():
    if not os.path.isfile(ARTIFACT_PATH):
        raise FileNotFoundError(
            f"Missing artifact: {ARTIFACT_PATH}\n"
            "Run training first: python pd2/train02.py"
        )

    ensure_dataset_layout(DATA_DIR)
    with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
        artifact = json.load(f)

    model_paths = artifact.get("models", [])
    if not model_paths:
        raise ValueError("No model paths found in train02_results.json")
    missing = [p for p in model_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Missing ensemble model file(s):\n" + "\n".join(missing))

    threshold = float(artifact["results"]["threshold"])

    raw_samples = collect_samples(DATA_DIR)
    cleaned_samples, clean_report_now = clean_samples(raw_samples)
    _, _, test_samples = stratified_group_split(cleaned_samples)
    test_ds = build_dataset(test_samples, training=False)
    y_test = labels_array(test_samples)

    probs = []
    for path in model_paths:
        model = tf.keras.models.load_model(path, compile=False)
        p = predict_probs(model, test_ds, tta=True, tta_passes=TTA_PASSES)
        probs.append(p)
    y_prob = np.mean(np.stack(probs, axis=0), axis=0)

    results = evaluate_probs(y_test, y_prob, threshold)
    out = {
        "artifact_path": ARTIFACT_PATH,
        "dataset_dir": DATA_DIR,
        "model_paths": model_paths,
        "tta_passes": TTA_PASSES,
        "threshold": threshold,
        "clean_report_current_run": clean_report_now,
        "test_size": int(len(y_test)),
        "results": results,
    }

    out_path = os.path.join(MODELS_DIR, "eval02_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
