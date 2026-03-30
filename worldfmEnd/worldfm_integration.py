from __future__ import annotations

from typing import Any

from .paths import ensure_worldfm_on_path

_svc: Any = None


def get_svc():
    """Lazy import of services/worldfm/app.py singleton."""
    global _svc
    if _svc is None:
        ensure_worldfm_on_path()
        import app as worldfm_app  # type: ignore  # noqa: WPS433 — module in services/worldfm

        _svc = worldfm_app.svc
    return _svc
