from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image

from .paths import WORLDFM_END_ROOT
from .precache_frames import list_precache_frame_paths, resolve_precache_dir


SCENES_DIR = WORLDFM_END_ROOT / "scenes"
MANIFEST_PATH = SCENES_DIR / "manifest.json"


@dataclass
class SceneDef:
    name: str
    init_image: Path
    thumbnail_url: str
    precache_dir: Optional[Path] = None


def _ensure_layout() -> None:
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    (WORLDFM_END_ROOT / "static" / "scenes").mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text(json.dumps({"scenes": []}, indent=2), encoding="utf-8")


def _sync_thumbnails_from_manifest(raw: dict[str, Any]) -> None:
    static_root = WORLDFM_END_ROOT / "static"
    for item in raw.get("scenes", []):
        rel = str(item.get("init_image") or "").strip()
        if not rel:
            continue
        spath = (SCENES_DIR / rel).resolve()
        if not spath.is_file():
            continue
        thumb_url = str(item.get("thumbnail") or "").strip()
        if not thumb_url:
            thumb_url = f"/static/scenes/{spath.stem}_thumb.jpg"
        if not thumb_url.startswith("/static/"):
            continue
        rel_under_static = thumb_url[len("/static/") :].lstrip("/")
        tpath = static_root / rel_under_static
        if tpath.is_file():
            continue
        tpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(spath).convert("RGB")
            img.thumbnail((320, 240), Image.Resampling.LANCZOS)
            img.save(tpath, format="JPEG", quality=85)
        except OSError:
            pass


def load_scenes() -> List[SceneDef]:
    _ensure_layout()
    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, dict):
        return []
    _sync_thumbnails_from_manifest(raw)
    out: List[SceneDef] = []
    for item in raw.get("scenes", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        rel = str(item.get("init_image") or "").strip()
        if not name or not rel:
            continue
        init_path = (SCENES_DIR / rel).resolve()
        if not init_path.is_file():
            continue
        thumb = str(item.get("thumbnail") or "")
        if not thumb:
            thumb = f"/static/scenes/{init_path.stem}_thumb.jpg"
        mp = item.get("precache_dir")
        mp_str = str(mp).strip() if mp is not None and str(mp).strip() else None
        pdir = resolve_precache_dir(name, mp_str)
        if pdir is not None and not list_precache_frame_paths(pdir):
            pdir = None
        out.append(SceneDef(name=name, init_image=init_path, thumbnail_url=thumb, precache_dir=pdir))
    return out


def scene_by_name(name: str) -> SceneDef | None:
    for s in load_scenes():
        if s.name == name:
            return s
    return None


def list_scenes_api() -> dict[str, Any]:
    scenes = []
    for s in load_scenes():
        scenes.append({"name": s.name, "thumbnail": s.thumbnail_url})
    return {"scenes": scenes}
