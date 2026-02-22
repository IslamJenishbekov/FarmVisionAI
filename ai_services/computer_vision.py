"""Computer vision helpers for cow analysis."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Mapping, TYPE_CHECKING

from PIL import Image

from schemas import AnalyzeResponse

if TYPE_CHECKING:
    from ultralytics import YOLO


_MODEL_PATH = Path("models/prod1.pt")
_MODEL: "YOLO | None" = None
_MODEL_LOCK = Lock()


def _load_model() -> "YOLO":
    """Load and cache the YOLO model used for inference."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {_MODEL_PATH}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - runtime check
            raise RuntimeError(
                "Missing dependency: ultralytics. Install it to run inference."
            ) from exc

        _MODEL = YOLO(str(_MODEL_PATH))
        return _MODEL


def _to_box_list(value: list[float]) -> list[int]:
    """Convert a YOLO xyxy box to the API's [x1, x2, y1, y2] format."""
    x1, y1, x2, y2 = value
    return [
        int(round(x1)),
        int(round(x2)),
        int(round(y1)),
        int(round(y2)),
    ]


def _extract_boxes(result: Any, names: Mapping[int, str]) -> dict[str, list[list[int]]]:
    """Extract selected label boxes from a YOLO result."""
    buckets: dict[str, list[list[int]]] = {
        "cow": [],
        "wolf": [],
        "person": [],
    }

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return buckets

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    for coords, cls_id in zip(xyxy, cls):
        label = names.get(int(cls_id), str(cls_id))
        if label in buckets:
            buckets[label].append(_to_box_list(coords.tolist()))

    return buckets


def _normalize_names(names: Any) -> dict[int, str]:
    """Normalize YOLO names to an int-to-label mapping."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def analyze_cows(
    image: Image.Image,
    add_info: list[Mapping[str, Any]],
) -> AnalyzeResponse:
    """Analyze a cow image and return detection results.

    Uses models/prod1.pt to detect objects and maps selected labels:
    - cow -> cows_num
    - wolf -> hunter boxes
    - person -> thief boxes
    """

    model = _load_model()
    results = model.predict(image, verbose=False)
    if not results:
        boxes_by_label: dict[str, list[list[int]]] = {
            "cow": [],
            "wolf": [],
            "person": [],
        }
    else:
        result = results[0]
        names = _normalize_names(getattr(model, "names", {}))
        boxes_by_label = _extract_boxes(result, names)

    return AnalyzeResponse(
        cows_num=len(boxes_by_label["cow"]),
        ill_cow=[],
        hunter=boxes_by_label["wolf"],
        thief=boxes_by_label["person"],
        pregnant=[],
        info={},
    )
