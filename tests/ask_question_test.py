"""Manual client for /ask-question endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


def _parse_history(raw: str) -> list[dict[str, str]]:
    """Parse JSON list of dicts from CLI."""

    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("history must be valid JSON") from exc

    if not isinstance(value, list):
        raise ValueError("history must be a JSON list of dicts")

    for item in value:
        if not isinstance(item, dict):
            raise ValueError("history items must be dicts")
        for key, val in item.items():
            if not isinstance(key, str) or not isinstance(val, str):
                raise ValueError("history dict keys/values must be strings")

    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a request to /ask-question and print the response.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the API (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--user-text",
        default="Кто ты?",
        help="Question text to send to the endpoint.",
    )
    parser.add_argument(
        "--history-json",
        default="[]",
        help="History as JSON list of dicts.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        history = _parse_history(args.history_json)
    except ValueError as exc:
        print(f"Invalid history: {exc}", file=sys.stderr)
        return 2

    params = {
        "user_text": args.user_text,
        "history": json.dumps(history, ensure_ascii=False),
    }

    try:
        response = requests.get(
            f"{args.base_url.rstrip('/')}/ask-question",
            params=params,
            timeout=30,
        )
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(f"Status: {response.status_code}")
    print(response.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
