from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

SERVICE_PYTHON = "/home/mengfei/miniconda3/envs/WorldFM/bin/python"

@dataclass
class StepResult:
    frame_base64: str
    reward: float
    ended: bool
    truncated: bool
    extra: Dict[str, Any]


class WorldModelAdapter(ABC):
    """Unified runtime adapter for a world model backend."""

    model_id: str

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def start_session(self, init_image_bytes: Optional[bytes]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def reset_session(self, init_image_bytes: Optional[bytes]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Dict[str, Any]) -> StepResult:
        raise NotImplementedError
