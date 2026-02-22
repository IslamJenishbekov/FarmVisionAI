"""Cut a segment from a video using mm:ss timestamps."""

from __future__ import annotations

import argparse
import subprocess
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


def _format_hhmmss(total_seconds: int) -> str:
    """Format seconds into HH:MM:SS for ffmpeg."""

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cut a segment from a video. OpenCV path drops audio; ffmpeg fallback keeps it."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video file.",
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
        default="2.mp4",
        help="Output path (default: 2.mp4).",
    )
    return parser


def _cut_with_opencv(
    input_path: Path,
    start_s: int,
    end_s: int,
    output_path: str,
) -> tuple[bool, int, str]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False, 0, f"Failed to open video: {input_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return False, 0, "Failed to read FPS from video"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return False, 0, "Failed to read frame size from video"

    start_frame = int(round(start_s * fps))
    end_frame = int(round(end_s * fps))
    if end_frame <= start_frame:
        cap.release()
        return False, 0, "Computed frame range is empty"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0 and start_frame >= total_frames:
        cap.release()
        return False, 0, "Start time exceeds video duration"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False, 0, f"Failed to open output file: {output_path}"

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

    if written == 0:
        return False, 0, "No frames were written (decoder failure likely)"

    return True, written, ""


def _cut_with_ffmpeg(
    input_path: Path,
    start_s: int,
    end_s: int,
    output_path: str,
) -> int:
    start_ts = _format_hhmmss(start_s)
    duration_ts = _format_hhmmss(end_s - start_s)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        start_ts,
        "-i",
        str(input_path),
        "-t",
        duration_ts,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("ffmpeg not found in PATH", file=sys.stderr)
        return 1

    if result.returncode != 0:
        print(f"ffmpeg failed with code {result.returncode}", file=sys.stderr)
        return result.returncode

    print(f"Output: {output_path}")
    return 0


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

    ok, written, error = _cut_with_opencv(
        input_path=input_path,
        start_s=start_s,
        end_s=end_s,
        output_path=args.output,
    )
    if ok:
        print(f"Written frames: {written}")
        print(f"Output: {args.output}")
        return 0

    print(f"OpenCV cut failed: {error}", file=sys.stderr)
    print("Falling back to ffmpeg...", file=sys.stderr)
    return _cut_with_ffmpeg(
        input_path=input_path,
        start_s=start_s,
        end_s=end_s,
        output_path=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
