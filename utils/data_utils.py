import json
from pathlib import Path
import uuid
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JSON_PATH = DATA_DIR / "lost_items.json"


def ensure_data_dir():
    """Stellt sicher, dass das Datenverzeichnis existiert."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not JSON_PATH.exists():
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_items() -> list:
    """Lädt alle Fundstücke sicher aus der JSON-Datei."""
    ensure_data_dir()
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_all_items(items: list):
    """Speichert die Gesamtliste der Fundstücke."""
    ensure_data_dir()
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def generate_item_id() -> str:
    """Generiert eine verständliche Fundstück-ID."""
    year = datetime.now().year
    items = load_items()
    count = len(items) + 1
    return f"FB-{year}-{count:04d}"


def save_item(item_data: dict) -> str:
    """Fügt ein neues Fundstück hinzu."""
    items = load_items()
    item_id = generate_item_id()
    item_data["id"] = item_id
    items.append(item_data)
    save_all_items(items)
    return item_id


def update_item(item_id: str, changes: dict) -> bool:
    """Aktualisiert ein bestehendes Fundstück."""
    items = load_items()
    updated = False
    for item in items:
        if item.get("id") == item_id:
            item.update(changes)
            updated = True
            break
    if updated:
        save_all_items(items)
    return updated


def delete_item(item_id: str) -> bool:
    """Löscht ein Fundstück."""
    items = load_items()
    new_items = [i for i in items if i.get("id") != item_id]
    if len(new_items) < len(items):
        save_all_items(new_items)
        return True
    return False


def get_item_by_id(item_id: str) -> dict:
    """Sucht ein Fundstück nach ID."""
    items = load_items()
    for item in items:
        if item.get("id") == item_id:
            return item
    return None


def filter_items(items: list, query: str = None, category: str = None, color: str = None, location: str = None, status: str = None) -> list:
    """Filtert Fundstücke nach verschiedenen Kriterien."""
    result = items

    if query:
        q = query.lower()
        result = [
            i for i in result
            if q in i.get("description", "").lower()
            or q in i.get("brand", "").lower()
            or q in i.get("id", "").lower()
            or q in i.get("special_features", "").lower()
        ]

    if category and category != "Alle":
        result = [i for i in result if i.get("category") == category]

    if color and color != "Alle":
        result = [i for i in result if i.get("color") == color]

    if location and location != "Alle":
        result = [i for i in result if i.get("location_found") == location]

    if status and status != "Alle":
        result = [i for i in result if i.get("status") == status]

    return result


def get_unique_categories(items: list) -> list:
    return sorted(list(set(i.get("category") for i in items if i.get("category"))))


def get_unique_colors(items: list) -> list:
    return sorted(list(set(i.get("color") for i in items if i.get("color"))))


def get_unique_locations(items: list) -> list:
    return sorted(list(set(i.get("location_found") for i in items if i.get("location_found"))))


def export_data_as_json() -> str:
    """Exportiert die Fundstückdaten als JSON-String."""
    items = load_items()
    return json.dumps(items, ensure_ascii=False, indent=2)
