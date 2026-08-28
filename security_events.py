"""
Lightweight security-event store for spoof / unknown images.
Additive only – does not touch any authentication or liveness logic.
"""

import json
import threading
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_lock = threading.Lock()
_ROOT = Path(__file__).parent.absolute()
_EVENTS_FILE = _ROOT / "data" / "security_events.json"
_SPOOF_DIR = (_ROOT / "spoofs_detected").resolve()
_UNKNOWN_DIR = (_ROOT / "unknown_detected").resolve()
_ALLOWED_DIRS = {_SPOOF_DIR, _UNKNOWN_DIR}
_ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


def _ensure_store() -> None:
    _EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _EVENTS_FILE.exists():
        with open(_EVENTS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)


def _load() -> List[Dict[str, Any]]:
    _ensure_store()
    try:
        with open(_EVENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(events: List[Dict[str, Any]]) -> None:
    _ensure_store()
    with open(_EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)


def create_event(
    event_type: str,
    filename: str,
    image_path: Path,
    score: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if event_type not in ("spoof", "unknown"):
        return None

    try:
        resolved = image_path.resolve()
        if resolved.parent not in _ALLOWED_DIRS:
            return None
        if resolved.suffix.lower() not in _ALLOWED_EXT:
            return None
        if not resolved.is_file():
            return None
    except Exception:
        return None

    with _lock:
        events = _load()

        if any(e.get("filename") == filename for e in events):
            print(f"[EVENT] Duplicate skipped: {filename}")
            return None

        event = {
            "event_id": str(uuid.uuid4()),
            "type": event_type,
            "filename": filename,
            "image_path": str(resolved),
            "timestamp": datetime.now().isoformat(),
            "score": float(score) if score is not None else None,
        }

        events.append(event)

        if len(events) > 200:
            events = events[-200:]

        _save(events)
        print(f"[EVENT] Security event created: {event['event_id']} ({event_type}) {filename}")
        return event


def list_events(limit: int = 30) -> List[Dict[str, Any]]:
    with _lock:
        events = _load()
    return list(reversed(events[-limit:]))


def get_event(event_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        events = _load()
    for e in events:
        if e.get("event_id") == event_id:
            return e
    return None


def safe_image_path(event_id: str) -> Optional[Path]:
    event = get_event(event_id)
    if not event:
        return None
    try:
        p = Path(event["image_path"]).resolve()
        if p.parent not in _ALLOWED_DIRS:
            return None
        if p.suffix.lower() not in _ALLOWED_EXT:
            return None
        if not p.is_file():
            return None
        return p
    except Exception:
        return None


def delete_event(event_id: str) -> bool:
    """Delete one event and its image file. Returns True if deleted."""
    with _lock:
        events = _load()
        target = None
        for e in events:
            if e.get("event_id") == event_id:
                target = e
                break

        if target is None:
            return False

        # Delete image file safely
        try:
            p = Path(target["image_path"]).resolve()
            if p.parent in _ALLOWED_DIRS and p.is_file():
                os.remove(p)
                print(f"[EVENT] Image deleted: {p.name}")
        except Exception as ex:
            print(f"[EVENT] Failed to delete image: {ex}")

        # Remove from store
        events = [e for e in events if e.get("event_id") != event_id]
        _save(events)
        print(f"[EVENT] Event deleted: {event_id}")
        return True


def delete_all_events() -> int:
    """Delete all events and their image files. Returns number deleted."""
    with _lock:
        events = _load()
        count = 0
        for e in events:
            try:
                p = Path(e["image_path"]).resolve()
                if p.parent in _ALLOWED_DIRS and p.is_file():
                    os.remove(p)
                    count += 1
            except Exception:
                pass
        _save([])
        print(f"[EVENT] All events deleted ({count} images removed)")
        return count