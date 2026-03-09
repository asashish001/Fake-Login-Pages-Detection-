import os
import re
import json
import math
import hashlib
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight


# ----------------------------
# Config
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "Dataset"),
    os.path.join(PROJECT_ROOT, "dataset"),
    os.path.join(PROJECT_ROOT, "Data"),
    os.path.join(PROJECT_ROOT, "data"),
    os.path.join(SCRIPT_DIR, "Dataset"),
    os.path.join(SCRIPT_DIR, "dataset"),
    os.path.join(SCRIPT_DIR, "Data"),
    os.path.join(SCRIPT_DIR, "data"),
]
DATA_DIR = next((d for d in DATASET_CANDIDATES if os.path.isdir(d)), DATASET_CANDIDATES[0])
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = (300, 300)  # upgraded input size
BATCH_SIZE = 24
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 12
PHISHING_WEIGHT_BOOST = 1.8
MIN_PHISHING_RECALL = 0.90
TTA_PASSES = 5

# Heuristic: keep pages likely to be login/auth flows by filename
LOGIN_HINTS = {
    "login", "signin", "sign-in", "auth", "account", "verify", "verification",
    "password", "secure", "bank", "paypal", "outlook", "mail", "credential",
    "session", "unlock", "update", "confirm", "webscr",
}


@dataclass
class Sample:
    path: str
    label: int  # 0 real, 1 phishing
    domain: str
    source_folder: str


def ensure_dataset_layout(dataset_dir: str) -> None:
    real_dir = os.path.join(dataset_dir, "real")
    phishing_dir = os.path.join(dataset_dir, "phishing")
    if not (os.path.isdir(real_dir) and os.path.isdir(phishing_dir)):
        raise FileNotFoundError(
            "Dataset folder is invalid.\n"
            "Expected: <dataset_dir>/real and <dataset_dir>/phishing.\n"
            f"Detected dataset path: {dataset_dir}"
        )


def sha1_file(path: str, block_size: int = 65536) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def average_hash(path: str, size: int = 8) -> str:
    # Lightweight near-duplicate hash (no extra dependency)
    try:
        img = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        mean = arr.mean()
        bits = (arr > mean).astype(np.uint8).flatten()
        return "".join(bits.astype(str))
    except Exception:
        return ""


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def extract_domain_from_filename(filename: str) -> str:
    # Handles many noisy filename formats; keeps split consistency by origin
    base = os.path.splitext(filename)[0].lower()
    base = re.sub(r"^(genuine_|phishing_)", "", base)
    base = re.sub(r"_aug\d+$", "", base)
    m = re.search(r"([a-z0-9-]+\.[a-z]{2,})(?:[._/]|$)", base)
    if m:
        return m.group(1)
    return base[:80]


def likely_login_page(filename: str) -> bool:
    low = filename.lower()
    return any(k in low for k in LOGIN_HINTS)


def collect_samples(dataset_dir: str) -> List[Sample]:
    samples: List[Sample] = []
    for folder_name, label in [("real", 0), ("phishing", 1)]:
        folder = os.path.join(dataset_dir, folder_name)
        for name in os.listdir(folder):
            if not name.lower().endswith(".png"):
                continue
            path = os.path.join(folder, name)
            domain = extract_domain_from_filename(name)
            samples.append(Sample(path=path, label=label, domain=domain, source_folder=folder_name))
    return samples


def clean_samples(raw_samples: List[Sample]) -> Tuple[List[Sample], Dict]:
    # 1) exact dedup by SHA1
    by_sha: Dict[str, Sample] = {}
    removed_exact = 0
    for s in raw_samples:
        key = sha1_file(s.path)
        if key in by_sha:
            removed_exact += 1
            continue
        by_sha[key] = s
    samples = list(by_sha.values())

    # 2) near-dup by aHash (same class+domain, tiny Hamming distance)
    buckets: Dict[Tuple[int, str], List[Tuple[str, Sample]]] = {}
    for s in samples:
        ah = average_hash(s.path)
        buckets.setdefault((s.label, s.domain), []).append((ah, s))

    filtered: List[Sample] = []
    removed_near = 0
    for _, hashed_samples in buckets.items():
        kept_hashes: List[str] = []
        for ah, s in hashed_samples:
            if not ah:
                filtered.append(s)
                continue
            if any(hamming(ah, kh) <= 5 for kh in kept_hashes):
                removed_near += 1
                continue
            kept_hashes.append(ah)
            filtered.append(s)

    # 3) heuristic non-login filter only on real pages (keeps phishing diversity)
    final_samples: List[Sample] = []
    removed_non_login_real = 0
    for s in filtered:
        name = os.path.basename(s.path)
        if s.label == 0 and not likely_login_page(name):
            removed_non_login_real += 1
            continue
        final_samples.append(s)

    report = {
        "raw_count": len(raw_samples),
        "after_exact_dedup": len(samples),
        "removed_exact_duplicates": removed_exact,
        "after_near_dedup": len(filtered),
        "removed_near_duplicates": removed_near,
        "final_count": len(final_samples),
        "removed_non_login_real_heuristic": removed_non_login_real,
    }
    return final_samples, report


def stratified_group_split(
    samples: List[Sample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not math.isclose(val_ratio + test_ratio, 0.30, abs_tol=1e-6):
        raise ValueError("This splitter expects val_ratio + test_ratio == 0.30")

    y = np.array([s.label for s in samples], dtype=np.int32)
    g = np.array([s.domain for s in samples])
    idx = np.arange(len(samples))

    # First split: train vs temp (70/30)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, temp_idx = next(sgkf.split(idx, y, groups=g))

    temp_samples = [samples[i] for i in temp_idx]
    y_temp = np.array([s.label for s in temp_samples], dtype=np.int32)
    g_temp = np.array([s.domain for s in temp_samples])
    idx_temp = np.arange(len(temp_samples))

    # Second split temp into val/test (50/50 => 15/15 of total)
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED + 1)
    val_rel_idx, test_rel_idx = next(sgkf2.split(idx_temp, y_temp, groups=g_temp))

    train_samples = [samples[i] for i in train_idx]
    val_samples = [temp_samples[i] for i in val_rel_idx]
    test_samples = [temp_samples[i] for i in test_rel_idx]
    return train_samples, val_samples, test_samples


def decode_png(path: tf.Tensor, label: tf.Tensor):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    return img, label


def build_dataset(samples: List[Sample], training: bool) -> tf.data.Dataset:
    paths = [s.path for s in samples]
    labels = [s.label for s in samples]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_png, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_augmentation() -> tf.keras.Sequential:
    # Heavier augmentation to improve robustness to screenshot artifacts
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.20),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.20),
            tf.keras.layers.GaussianNoise(0.03),
        ],
        name="augmentation",
    )


def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred_f = tf.clip_by_value(tf.cast(y_pred, tf.float32), eps, 1.0 - eps)

        p_t = y_true_f * y_pred_f + (1.0 - y_true_f) * (1.0 - y_pred_f)
        alpha_t = y_true_f * alpha + (1.0 - y_true_f) * (1.0 - alpha)
        focal = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal)

    return loss


def build_model(model_name: str):
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    aug = build_augmentation()
    x = aug(inputs)

    if model_name == "efficientnetb0":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
    elif model_name == "resnet50v2":
        preprocess = tf.keras.applications.resnet_v2.preprocess_input
        base = tf.keras.applications.ResNet50V2(
            include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    base.trainable = False
    x = preprocess(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="phishing_prob")(x)
    model = tf.keras.Model(inputs, outputs, name=f"{model_name}_clf")
    return model, base


def compile_model(model: tf.keras.Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=binary_focal_loss(gamma=2.0, alpha=0.35),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )


def get_class_weights(train_samples: List[Sample], boost: float) -> Dict[int, float]:
    y = np.array([s.label for s in train_samples], dtype=np.int32)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(weights[0]), 1: float(weights[1] * boost)}


def predict_probs(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
    tta: bool = False,
    tta_passes: int = 5,
) -> np.ndarray:
    if not tta:
        p = model.predict(ds, verbose=0).reshape(-1)
        return p

    probs = []
    for _ in range(tta_passes):
        # Rebuild a stochastic-augmentation view by remapping through dataset pipeline
        # We invoke model in training=True for active dropout + augmentation.
        batch_probs = []
        for xb, _ in ds:
            pr = model(xb, training=True).numpy().reshape(-1)
            batch_probs.extend(pr.tolist())
        probs.append(np.array(batch_probs, dtype=np.float32))
    return np.mean(np.stack(probs, axis=0), axis=0)


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.90) -> float:
    # Pick threshold maximizing F1 under recall constraint for phishing class
    best_t, best_f1 = 0.5, -1.0
    thresholds = np.linspace(0.05, 0.95, 181)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        recall = tp / (tp + fn + 1e-9)
        if recall < min_recall:
            continue
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    # fallback: threshold by Youden from PR data if recall constraint impossible
    if best_f1 < 0:
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
        f1s = (2 * precision * recall) / (precision + recall + 1e-9)
        idx = int(np.nanargmax(f1s))
        if idx >= len(thresholds_pr):
            return 0.5
        return float(thresholds_pr[idx])
    return best_t


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict:
    y_pred = (y_prob >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    acc = (tn + tp) / max(1, len(y_true))
    phishing_recall = tp / max(1, tp + fn)
    phishing_precision = tp / max(1, tp + fp)
    return {
        "threshold": threshold,
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "f1_phishing": float(f1),
        "phishing_recall": float(phishing_recall),
        "phishing_precision": float(phishing_precision),
        "confusion_matrix": {
            "TN_real": int(tn),
            "FP_real_as_phishing": int(fp),
            "FN_phishing_as_real": int(fn),
            "TP_phishing": int(tp),
        },
    }


def model_callbacks(prefix: str) -> List[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{prefix}.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_one_model(
    model_name: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Dict[int, float],
) -> str:
    model, base = build_model(model_name)
    compile_model(model, lr=1e-3)
    cbs = model_callbacks(model_name)

    print(f"\n--- {model_name}: stage 1 (frozen backbone) ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1,
    )

    print(f"\n--- {model_name}: stage 2 (fine-tune top layers) ---")
    base.trainable = True
    freeze_upto = int(0.75 * len(base.layers))
    for layer in base.layers[:freeze_upto]:
        layer.trainable = False
    compile_model(model, lr=1e-5)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1,
    )
    return os.path.join(MODELS_DIR, f"{model_name}.keras")


def labels_array(samples: List[Sample]) -> np.ndarray:
    return np.array([s.label for s in samples], dtype=np.int32)


def main():
    ensure_dataset_layout(DATA_DIR)
    print(f"Using dataset: {DATA_DIR}")

    raw_samples = collect_samples(DATA_DIR)
    cleaned_samples, clean_report = clean_samples(raw_samples)
    print("Cleaning report:", json.dumps(clean_report, indent=2))

    train_samples, val_samples, test_samples = stratified_group_split(cleaned_samples)
    split_report = {
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "train_pos": int(sum(s.label for s in train_samples)),
        "val_pos": int(sum(s.label for s in val_samples)),
        "test_pos": int(sum(s.label for s in test_samples)),
    }
    print("Split report:", json.dumps(split_report, indent=2))

    train_ds = build_dataset(train_samples, training=True)
    val_ds = build_dataset(val_samples, training=False)
    test_ds = build_dataset(test_samples, training=False)

    class_weights = get_class_weights(train_samples, boost=PHISHING_WEIGHT_BOOST)
    print("Class weights:", class_weights)

    # Ensemble models
    ensemble_names = ["efficientnetb0", "resnet50v2"]
    model_paths = []
    for name in ensemble_names:
        path = train_one_model(name, train_ds, val_ds, class_weights)
        model_paths.append(path)

    # Validation predictions for threshold tuning
    y_val = labels_array(val_samples)
    val_probs_ensemble = []
    for path in model_paths:
        m = tf.keras.models.load_model(path, compile=False)
        p = predict_probs(m, val_ds, tta=True, tta_passes=TTA_PASSES)
        val_probs_ensemble.append(p)
    val_prob = np.mean(np.stack(val_probs_ensemble, axis=0), axis=0)
    best_threshold = tune_threshold(y_val, val_prob, min_recall=MIN_PHISHING_RECALL)
    print(f"Best threshold (val): {best_threshold:.4f}")

    # Test set evaluation with TTA + ensemble
    y_test = labels_array(test_samples)
    test_probs_ensemble = []
    for path in model_paths:
        m = tf.keras.models.load_model(path, compile=False)
        p = predict_probs(m, test_ds, tta=True, tta_passes=TTA_PASSES)
        test_probs_ensemble.append(p)
    test_prob = np.mean(np.stack(test_probs_ensemble, axis=0), axis=0)

    results = evaluate_probs(y_test, test_prob, best_threshold)
    print("Final test metrics:", json.dumps(results, indent=2))

    artifact = {
        "dataset_dir": DATA_DIR,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "models": model_paths,
        "class_weights": class_weights,
        "clean_report": clean_report,
        "split_report": split_report,
        "results": results,
    }
    out_path = os.path.join(MODELS_DIR, "train02_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"Saved artifact: {out_path}")


if __name__ == "__main__":
    main()
