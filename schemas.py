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
