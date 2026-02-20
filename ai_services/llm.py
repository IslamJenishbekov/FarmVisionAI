"""Gemini LLM client logic for the farm assistant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import requests

def _load_dotenv(path: str = ".env") -> dict[str, str]:
    """Load .env file key-value pairs without external dependencies."""

    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return {}

    data: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            data[key] = value
    return data


SYSTEM_PROMPT = (
    "Ты — помощник фермера. Отвечай на вопросы по ведению фермы, "
    "животным, технике, кормлению, ветеринарии и учёту. "
    "Если вопрос выходит за рамки фермы, дай краткий полезный ответ "
    "и при необходимости задай уточняющий вопрос. "
    "Будь точным, лаконичным и безопасным."
)


@dataclass(frozen=True)
class GeminiConfig:
    """Configuration for calling the Gemini API."""

    api_key: str
    model: str
    base_url: str
    timeout_s: int


class GeminiClientError(RuntimeError):
    """Raised when Gemini API returns an error or unexpected payload."""


def _build_config_from_env() -> GeminiConfig:
    dotenv_values = _load_dotenv()
    api_key = dotenv_values.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env")

    model = dotenv_values.get("GEMINI_MODEL", "gemini-3-pro-preview").strip()
    base_url = dotenv_values.get(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
    ).rstrip("/")
    timeout_raw = dotenv_values.get("GEMINI_TIMEOUT_S", "30").strip()
    try:
        timeout_s = int(timeout_raw)
    except ValueError as exc:
        raise ValueError("GEMINI_TIMEOUT_S must be an integer") from exc

    return GeminiConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_s=timeout_s,
    )


def _history_to_contents(
    history: Sequence[Mapping[str, str]], user_text: str
) -> list[dict]:
    contents: list[dict] = []

    for item in history:
        for question, answer in item.items():
            if question:
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": str(question)}],
                    }
                )
            if answer:
                contents.append(
                    {
                        "role": "model",
                        "parts": [{"text": str(answer)}],
                    }
                )

    contents.append(
        {
            "role": "user",
            "parts": [{"text": user_text}],
        }
    )
    return contents


def _extract_answer(payload: Mapping[str, object]) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        raise GeminiClientError("No candidates in Gemini response")

    first = candidates[0]
    if not isinstance(first, Mapping):
        raise GeminiClientError("Unexpected candidates format")

    content = first.get("content", {})
    if not isinstance(content, Mapping):
        raise GeminiClientError("Unexpected content format")

    parts = content.get("parts", [])
    if not isinstance(parts, list):
        raise GeminiClientError("Unexpected parts format")

    for part in parts:
        if isinstance(part, Mapping) and "text" in part:
            text = part.get("text", "")
            if isinstance(text, str) and text.strip():
                return text.strip()

    raise GeminiClientError("No text part in Gemini response")


def generate_llm_answer(
    history: Sequence[Mapping[str, str]],
    user_text: str,
    config: GeminiConfig | None = None,
) -> str:
    """Generate an answer using Gemini based on history and user text."""

    if not user_text:
        raise ValueError("user_text must be non-empty")

    if config is None:
        config = _build_config_from_env()

    url = f"{config.base_url}/models/{config.model}:generateContent?key={config.api_key}"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": _history_to_contents(history, user_text),
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 512,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=config.timeout_s)
    except requests.RequestException as exc:
        raise GeminiClientError("Failed to call Gemini API") from exc

    if response.status_code != 200:
        raise GeminiClientError(
            f"Gemini API error {response.status_code}: {response.text[:500]}"
        )

    data = response.json()
    return _extract_answer(data)
