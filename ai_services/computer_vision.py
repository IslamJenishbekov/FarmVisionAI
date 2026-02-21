"""Computer vision helpers for cow analysis."""

from __future__ import annotations

from typing import Any, Mapping

from PIL import Image

from schemas import AnalyzeResponse


def analyze_cows(
    image: Image.Image,
    add_info: Mapping[str, Any],
) -> AnalyzeResponse:
    """Analyze a cow image and return detection results.

    This is a placeholder implementation to keep the contract stable.
    """

    _ = image
    _ = add_info
    return AnalyzeResponse(
        cows_num=0,
        ill_cow=[],
        hunter=[],
        thief=[],
        pregnant=[],
        info={},
    )
