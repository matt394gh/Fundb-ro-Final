import json
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Pfade definieren
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"
CONFIG_PATH = MODEL_DIR / "model_config.json"


def load_config() -> dict:
    """Lädt die Modell-Konfiguration aus json."""
    default_config = {
        "image_width": 224,
        "image_height": 224,
        "normalization": "0_1",
        "confidence_threshold": 0.55,
        "model_type": "keras_h5"
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                default_config.update(config)
        except Exception:
            pass
    return default_config


def load_labels() -> list:
    """Lädt und bereinigt die Label-Datei."""
    if not LABELS_PATH.exists():
        return ["Sonstiges"]

    labels = []
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Nummern entfernen (z.B. "0 T-Shirt" -> "T-Shirt")
            parts = line.split(" ", 1)
            if len(parts) > 1 and parts[0].replace(":", "").replace("-", "").isdigit():
                clean_label = parts[1].strip()
            else:
                clean_label = line
            labels.append(clean_label)

    return labels if labels else ["Sonstiges"]


@st.cache_resource
def load_keras_model():
    """Lädt das Keras-Modell mit Caching."""
    if not MODEL_PATH.exists():
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des KI-Modells: {e}")
        return None


def predict_clothing(image_file) -> dict:
    """Führt die Bildklassifikation durch."""
    config = load_config()
    labels = load_labels()
    model = load_keras_model()

    if model is None:
        return {
            "label": "Sonstiges (Modell nicht geladen)",
            "confidence": 0.0,
            "probabilities": {}
        }

    # 1. Bild öffnen & RGB
    img = Image.open(image_file).convert("RGB")

    # 2. Resizing
    target_size = (config.get("image_width", 224), config.get("image_height", 224))
    img = img.resize(target_size)

    # 3. Array & Normalisierung
    img_array = np.asarray(img, dtype=np.float32)

    norm_type = config.get("normalization", "0_1")
    if norm_type == "0_1":
        img_array = img_array / 255.0
    elif norm_type == "minus1_1":
        img_array = (img_array / 127.5) - 1.0

    # 4. Batch Dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # 5. Prediction
    preds = model.predict(img_array)[0]

    # Matching mit Labels
    probs = {}
    for idx, prob in enumerate(preds):
        lbl = labels[idx] if idx < len(labels) else f"Klasse_{idx}"
        probs[lbl] = float(prob)

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_label, top_conf = sorted_probs[0]

    threshold = config.get("confidence_threshold", 0.55)
    final_label = top_label if top_conf >= threshold else "Unbekannt / Nicht eindeutig"

    return {
        "label": final_label,
        "confidence": top_conf,
        "probabilities": dict(sorted_probs[:3])
    }


def get_model_info() -> dict:
    """Gibt Statusinformationen zum Modell zurück."""
    model = load_keras_model()
    labels = load_labels()
    config = load_config()

    return {
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "labels_count": len(labels),
        "labels": labels,
        "config": config
    }
