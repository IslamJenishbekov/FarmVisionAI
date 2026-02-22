"""Extract evenly spaced frames from selected videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def _get_total_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        cap.release()
        return total_frames
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def _build_indices(total_frames: int, frames_per_video: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Total frames must be positive")
    if frames_per_video <= 0:
        raise ValueError("frames_per_video must be positive")
    if frames_per_video == 1:
        return [0]
    step = (total_frames - 1) / (frames_per_video - 1)
    indices = [int(round(i * step)) for i in range(frames_per_video)]
    return indices


def _extract_frames(
    video_path: Path,
    output_dir: Path,
    frames_per_video: int,
    image_ext: str,
) -> int:
    total_frames = _get_total_frames(video_path)
    indices = _build_indices(total_frames, frames_per_video)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    saved = 0
    for out_index, frame_index in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_name = (
            f"video_{video_path.stem}_frame_{out_index:03d}.{image_ext}"
        )
        output_path = output_dir / frame_name
        if cv2.imwrite(str(output_path), frame):
            saved += 1
    cap.release()
    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract evenly spaced frames from videos in a numeric range."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data/selected_videos",
        help="Directory with input videos (default: data/selected_videos).",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/clear_frames",
        help="Directory to save extracted frames (default: tests/clear_frames).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=5,
        help="First video number to process (default: 5).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=12,
        help="Last video number to process (default: 12).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Number of frames to extract per video (default: 20).",
    )
    parser.add_argument(
        "--ext",
        default="png",
        help="Image extension for output frames (default: png).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.start > args.end:
        raise ValueError("--start cannot be greater than --end")

    total_saved = 0
    for index in range(args.start, args.end + 1):
        video_path = input_dir / f"{index}.mp4"
        if not video_path.exists():
            print(f"Missing video: {video_path}")
            continue
        try:
            saved = _extract_frames(
                video_path=video_path,
                output_dir=output_dir,
                frames_per_video=args.frames,
                image_ext=args.ext,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Failed on {video_path}: {exc}")
            continue
        print(f"{video_path.name}: saved {saved} frames")
        total_saved += saved

    print(f"Total frames saved: {total_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
