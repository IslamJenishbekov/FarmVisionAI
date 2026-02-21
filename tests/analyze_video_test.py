"""Manual client for /analyze endpoint using video frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import requests


BOX_COLORS = {
    "ill_cow": (0, 0, 255),
    "hunter": (0, 165, 255),
    "thief": (255, 0, 0),
    "pregnant": (255, 0, 255),
}


def _parse_add_info(raw: str) -> dict[str, Any]:
    """Parse add_info JSON from CLI."""

    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("add_info must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("add_info must be a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send video frames to /analyze and save results.",
    )
    parser.add_argument(
        "--video-path",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the API (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=25,
        help="Process every Nth frame (default: 25).",
    )
    parser.add_argument(
        "--add-info-json",
        default="{}",
        help="add_info as JSON object (default: {}).",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to save frames and JSON responses (default: data).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30).",
    )
    return parser


def _draw_boxes(frame, boxes: list[list[Any]], color: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, x2, y1, y2 = (int(round(float(v))) for v in box)
        except (TypeError, ValueError):
            continue

        left = max(0, min(x1, x2))
        right = min(width - 1, max(x1, x2))
        top = max(0, min(y1, y2))
        bottom = min(height - 1, max(y1, y2))
        if right <= left or bottom <= top:
            continue
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _request_frame(
    url: str,
    frame,
    add_info: dict[str, Any],
    timeout: int,
) -> requests.Response:
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")

    files = {"image": ("frame.jpg", encoded.tobytes(), "image/jpeg")}
    data = {"add_info": json.dumps(add_info, ensure_ascii=False)}
    return requests.request("GET", url, files=files, data=data, timeout=timeout)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        add_info = _parse_add_info(args.add_info_json)
    except ValueError as exc:
        print(f"Invalid add_info: {exc}", file=sys.stderr)
        return 2

    if args.step <= 0:
        print("step must be a positive integer", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {args.video_path}", file=sys.stderr)
        return 1

    frame_index = 0
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % args.step != 0:
                frame_index += 1
                continue

            url = f"{args.base_url.rstrip('/')}/analyze"
            response_payload: Any
            status_code = 0
            try:
                response = _request_frame(url, frame, add_info, args.timeout)
                status_code = response.status_code
                try:
                    response_payload = response.json()
                except ValueError:
                    response_payload = {"status": status_code, "body": response.text}
            except requests.RequestException as exc:
                response_payload = {"error": str(exc)}
            except RuntimeError as exc:
                response_payload = {"error": str(exc)}

            json_path = output_dir / f"frame_{frame_index:06d}.json"
            _save_json(json_path, response_payload)

            if status_code == 200 and isinstance(response_payload, dict):
                for key, color in BOX_COLORS.items():
                    boxes = response_payload.get(key, [])
                    if isinstance(boxes, list):
                        _draw_boxes(frame, boxes, color)

            image_path = output_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(image_path), frame)

            processed += 1
            frame_index += 1
    finally:
        cap.release()

    print(f"Processed frames: {processed}")
    print(f"Saved results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
