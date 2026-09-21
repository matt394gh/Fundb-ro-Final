import os
import json
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import streamlit as st

# Pfade definieren (Hauptverzeichnis des Projekts)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
CONFIG_PATH = BASE_DIR / "config.json"
LABELS_PATH = BASE_DIR / "labels.txt"


def load_config() -> dict:
    """Lädt die Konfigurationsdatei."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Fehler beim Laden von config.json: {e}")
    return {}


def load_labels() -> list:
    """Lädt die Labels und sortiert sie strikt nach ihrer Index-Nummer."""
    if not LABELS_PATH.exists():
        st.warning(f"⚠️ Label-Datei nicht gefunden unter: `{LABELS_PATH}`")
        return ["Pullover", "Hose", "Schuhe", "Andere", "T-Shirt"]

    labels_dict = {}
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    idx = int(parts[0])
                    label_name = parts[1].strip()
                    labels_dict[idx] = label_name

        sorted_labels = [labels_dict[i] for i in sorted(labels_dict.keys())]
        return sorted_labels if sorted_labels else ["Pullover", "Hose", "Schuhe", "Andere", "T-Shirt"]

    except Exception as e:
        st.error(f"Fehler beim Lesen von labels.txt: {e}")
        return ["Pullover", "Hose", "Schuhe", "Andere", "T-Shirt"]


@st.cache_resource
def load_keras_model():
    """Lädt das Keras-Modell und speichert es im Cache."""
    if not MODEL_PATH.exists():
        st.error(f"❌ Modell-Datei nicht gefunden unter: `{MODEL_PATH}`")
        return None

    if MODEL_PATH.stat().st_size < 1000000:
        st.error(
            f"❌ Die Datei `{MODEL_PATH.name}` ist zu klein ({MODEL_PATH.stat().st_size} Bytes). "
            "Vermutlich wurde nur ein Git-Pointer hochgeladen!"
        )
        return None

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Fehler beim Laden des KI-Modells (`{MODEL_PATH.name}`): {e}")
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

    try:
        # 1. Bild öffnen & in RGB umwandeln
        img = Image.open(image_file).convert("RGB")
        
        # 2. Genau wie Teachable Machine: Quadratisch zuschneiden und auf 224x224 skalieren
        target_size = (224, 224)
        img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)

        # 3. In NumPy-Array umwandeln
        img_array = np.asarray(img, dtype=np.float32)

        # 4. Teachable Machine Formel: Werte von [0, 255] auf [-1, 1] umrechnen
        normalized_img = (img_array / 127.5) - 1.0

        # 5. Batch-Dimension hinzufügen
        img_array = np.expand_dims(normalized_img, axis=0)

        # 6. Vorhersage ausführen
        preds = model.predict(img_array)[0]

        # DEBUG: Zeigt im Streamlit Cloud Log die echten Prozentwerte an
        print("ROH-VORHERSAGE VOM MODELL:", preds)

        # 7. Ergebnisse den Labels zuordnen
        probs = {}
        for idx, prob in enumerate(preds):
            lbl = labels[idx] if idx < len(labels) else f"Klasse_{idx}"
            probs[lbl] = float(prob)

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_label, top_conf = sorted_probs[0]

        threshold = config.get("confidence_threshold", 0.30)
        final_label = top_label if top_conf >= threshold else "Unbekannt / Nicht eindeutig"

        return {
            "label": final_label,
            "confidence": top_conf,
            "probabilities": dict(sorted_probs[:3])
        }

    except Exception as e:
        st.error(f"Fehler bei der Bildanalyse: {e}")
        return {
            "label": "Fehler bei Analyse",
            "confidence": 0.0,
            "probabilities": {}
        }

def get_model_info() -> dict:
    """Gibt Statusinformationen zum Modell zurück."""
    model = load_keras_model()
    labels = load_labels()
    config = load_config()

    file_exists = MODEL_PATH.exists()
    file_size_mb = round(MODEL_PATH.stat().st_size / (1024 * 1024), 2) if file_exists else 0

    return {
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "file_exists": file_exists,
        "file_size_mb": f"{file_size_mb} MB",
        "labels_count": len(labels),
        "labels": labels,
        "config": config
    }
