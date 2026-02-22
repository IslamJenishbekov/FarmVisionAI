"""Run YOLOv26s on a video and save annotated frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime check
    raise SystemExit(
        "Missing dependency: ultralytics. Install it to run this script."
    ) from exc
try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime check
    raise SystemExit(
        "Missing dependency: torch. Install it to run this script."
    ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLOv26s on every Nth frame of a video.",
    )
    parser.add_argument(
        "--model-path",
        default="models/yolo26s.pt",
        help="Path to YOLOv26s weights (default: models/yolo26s.pt).",
    )
    parser.add_argument(
        "--video-path",
        default="data/selected_videos/1.mp4",
        help="Path to input video (default: data/selected_videos/1.mp4).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=25,
        help="Process every Nth frame (default: 25).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/yolo_test",
        help="Directory for annotated frames (default: data/yolo_test).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device for inference (default: 0, i.e. cuda:0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    return parser


def _draw_boxes(frame, boxes, names) -> dict[str, int]:
    counts: dict[str, int] = {}
    if boxes is None or len(boxes) == 0:
        return counts

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), cls_id, score in zip(xyxy, cls, conf):
        label = names.get(cls_id, str(cls_id))
        counts[label] = counts.get(label, 0) + 1

        left, top, right, bottom = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(
            frame,
            text,
            (left, max(0, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return counts


def _ensure_gpu_available(device: str) -> None:
    device_str = str(device).lower()
    if device_str == "cpu":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available")

    device_count = torch.cuda.device_count()
    if device_str.isdigit():
        if int(device_str) >= device_count:
            raise RuntimeError(f"CUDA device {device_str} is not available")
    elif device_str.startswith("cuda:"):
        index = device_str.split("cuda:", 1)[1]
        if index.isdigit() and int(index) >= device_count:
            raise RuntimeError(f"CUDA device {device_str} is not available")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.step <= 0:
        print("step must be a positive integer", file=sys.stderr)
        return 2

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _ensure_gpu_available(args.device)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
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

            results = model.predict(
                frame,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            counts = _draw_boxes(frame, result.boxes, model.names)

            frame_name = f"frame_{frame_index:06d}.jpg"
            output_path = output_dir / frame_name
            cv2.imwrite(str(output_path), frame)

            if counts:
                summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            else:
                summary = "no detections"
            print(f"frame {frame_index:06d}: {summary}")

            processed += 1
            frame_index += 1
    finally:
        cap.release()

    print(f"Processed frames: {processed}")
    print(f"Saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
