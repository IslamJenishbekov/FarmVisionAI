"""Pydantic models for request/response validation."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AskQuestionParams(BaseModel):
    """Query parameters for asking a farm-related question."""

    user_text: str = Field(..., min_length=1)
    history: list[dict[str, str]] | None = None

    @field_validator("history", mode="before")
    @classmethod
    def parse_history(cls, value: Any) -> Any:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("history must be valid JSON") from exc
        return value

    @field_validator("history")
    @classmethod
    def validate_history(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("history must be a list of dicts")
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("history items must be dicts")
            for key, val in item.items():
                if not isinstance(key, str) or not isinstance(val, str):
                    raise ValueError("history dict keys/values must be strings")
        return value


def parse_add_info(value: Any) -> list[dict[str, Any]]:
    """Parse add_info JSON into a list of dictionaries."""

    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("add_info must be valid JSON") from exc
    else:
        parsed = value

    # Compatibility path: accept a single object and normalize to a one-item list.
    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        raise ValueError("add_info must be a JSON array of objects")
    if not all(isinstance(item, dict) for item in parsed):
        raise ValueError("add_info items must be objects")
    return parsed


class AnalyzeResponse(BaseModel):
    """Response model for cow image analysis."""

    cows_num: int = 0
    ill_cow: list[list[int]] = Field(default_factory=list)
    hunter: list[list[int]] = Field(default_factory=list)
    thief: list[list[int]] = Field(default_factory=list)
    pregnant: list[list[int]] = Field(default_factory=list)
    info: dict[str, Any] = Field(default_factory=dict)
