from pathlib import Path
from PIL import Image
import uuid

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"


def ensure_upload_dir():
    """Stellt sicher, dass das Upload-Verzeichnis existiert."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_image(uploaded_file) -> str:
    """Speichert ein hochgeladenes Bild sicher als JPG ab."""
    ensure_upload_dir()

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = UPLOAD_DIR / filename

    img = Image.open(uploaded_file)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img.save(filepath, "JPEG", quality=85)
    return str(filepath)
