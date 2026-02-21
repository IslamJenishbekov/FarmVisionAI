"""Cut a segment from an MP4 video using mm:ss timestamps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def _parse_mmss(value: str) -> int:
    """Parse mm:ss into total seconds."""

    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError("time must be in mm:ss format")
    minutes_str, seconds_str = parts
    if not minutes_str.isdigit() or not seconds_str.isdigit():
        raise ValueError("time must contain only digits")
    minutes = int(minutes_str)
    seconds = int(seconds_str)
    if seconds >= 60:
        raise ValueError("seconds must be between 0 and 59")
    return minutes * 60 + seconds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cut a segment from an MP4 video. Audio is not preserved.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input MP4 file.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start time in mm:ss.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End time in mm:ss.",
    )
    parser.add_argument(
        "--output",
        default="output.mp4",
        help="Output path (default: output.mp4).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        start_s = _parse_mmss(args.start)
        end_s = _parse_mmss(args.end)
    except ValueError as exc:
        print(f"Invalid time: {exc}", file=sys.stderr)
        return 2

    if end_s <= start_s:
        print("end time must be greater than start time", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Failed to read FPS from video", file=sys.stderr)
        cap.release()
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        print("Failed to read frame size from video", file=sys.stderr)
        cap.release()
        return 1

    start_frame = int(round(start_s * fps))
    end_frame = int(round(end_s * fps))
    if end_frame <= start_frame:
        print("Computed frame range is empty", file=sys.stderr)
        cap.release()
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0 and start_frame >= total_frames:
        print("Start time exceeds video duration", file=sys.stderr)
        cap.release()
        return 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Failed to open output file: {args.output}", file=sys.stderr)
        cap.release()
        return 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    written = 0
    try:
        while current_frame < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
            current_frame += 1
    finally:
        writer.release()
        cap.release()

    print(f"Written frames: {written}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
