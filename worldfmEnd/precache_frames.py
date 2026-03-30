"""Pre-baked frames under models/worldfm/outputs/<scene>/ (e.g. output_0000.png) for fast WS demo."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional

from .paths import WMFACTORY_ROOT

OUTPUTS_ROOT = WMFACTORY_ROOT / "models" / "worldfm" / "outputs"


def list_precache_frame_paths(pdir: Path) -> List[Path]:
    """Prefer output_####.png sorted by index; else any *.png sorted by name."""
    if not pdir.is_dir():
        return []
    numbered = list(pdir.glob("output_*.png"))
    if numbered:

        def sort_key(p: Path) -> tuple[int, str]:
            m = re.search(r"output_(\d+)", p.name, re.I)
            return (int(m.group(1)), p.name) if m else (0, p.name)

        return sorted(numbered, key=sort_key)
    return sorted(p for p in pdir.glob("*.png") if p.is_file())


def resolve_precache_dir(scene_name: str, manifest_precache: Optional[str]) -> Optional[Path]:
    """
    Directory containing pre-rendered PNGs. Optional manifest key precache_dir:
    relative name under outputs/ (e.g. mario) or absolute path.
    If omitted, tries outputs/<scene_name>/ when it exists.
    """
    if os.getenv("WORLDFMEND_FORCE_FULL_PIPELINE", "").strip() in ("1", "true", "yes"):
        return None

    candidates: List[Path] = []
    if manifest_precache:
        mp = Path(manifest_precache.strip())
        if mp.is_absolute():
            candidates.append(mp)
        else:
            candidates.append(OUTPUTS_ROOT / mp)
    candidates.append(OUTPUTS_ROOT / scene_name)

    extra = os.getenv("WORLDFMEND_PRECACHE_DIR", "").strip()
    if extra:
        ep = Path(extra)
        candidates.insert(0, ep if ep.is_absolute() else OUTPUTS_ROOT / ep)

    seen: set[Path] = set()
    for p in candidates:
        try:
            r = p.resolve()
        except OSError:
            continue
        if r in seen:
            continue
        seen.add(r)
        if r.is_dir() and list_precache_frame_paths(r):
            return r
    return None


def load_keyframes_manifest(pdir: Path) -> Optional[list[dict[str, Any]]]:
    """keyframes.json from WM_WORLDFM_EXPORT_KEYFRAMES (yaw_deg/pitch_deg per output_*.png)."""
    fp = pdir / "keyframes.json"
    if not fp.is_file():
        return None
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass
    return None


def yaw_delta_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def nearest_keyframe_index(
    keyframes: list[dict[str, Any]],
    yaw_deg: float,
    pitch_deg: float,
    pitch_weight: float = 0.5,
) -> int:
    best_i = 0
    best = float("inf")
    for i, k in enumerate(keyframes):
        y = float(k.get("yaw_deg", 0.0))
        p = float(k.get("pitch_deg", 0.0))
        dy = yaw_delta_deg(yaw_deg, y)
        dp = abs(pitch_deg - p)
        d = dy + pitch_weight * dp
        if d < best:
            best = d
            best_i = i
    return best_i


def path_index_for_output_index(paths: list[Path], output_index: int) -> int:
    for pi, pth in enumerate(paths):
        m = re.search(r"output_(\d+)", pth.name, re.I)
        if m and int(m.group(1)) == output_index:
            return pi
    return min(output_index, len(paths) - 1) if paths else 0
