"""Fine-tune YOLOv26m on the 8-Calves pmfeed frames."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime check
    raise SystemExit(
        "Missing dependency: ultralytics. Install it to run this script."
    ) from exc

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

@dataclass(frozen=True)
class Pair:
    image: Path
    label: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv26m on frames (YOLO labels).",
    )
    parser.add_argument(
        "--model-path",
        default="models/yolo26m.pt",
        help="Path to base YOLOv26m weights (default: models/yolo26m.pt).",
    )
    parser.add_argument(
        "--data-root",
        default="data/8-calves/data/pmfeed",
        help=(
            "Directory with .png/.jpg/.jpeg + .txt pairs "
            "(default: data/8-calves/data/pmfeed)."
        ),
    )
    parser.add_argument(
        "--images-dir",
        default="",
        help="Directory with input images (overrides --data-root).",
    )
    parser.add_argument(
        "--labels-dir",
        default="",
        help="Directory with label .txt files (overrides --data-root).",
    )
    parser.add_argument(
        "--output-dir",
        default="train/datasets/pmfeed_yolo",
        help="Output YOLO dataset directory (default: train/datasets/pmfeed_yolo).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio if --val-count is 0 (default: 0.1).",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=0,
        help="Exact number of validation frames (overrides --val-ratio).",
    )
    parser.add_argument(
        "--split",
        choices=["tail", "random"],
        default="tail",
        help="How to split data: 'tail' uses last frames for val (default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split=random (default: 42).",
    )
    parser.add_argument(
        "--link-method",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to materialize files into the dataset (default: hardlink).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it exists (destructive).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Rename existing output-dir to a timestamped backup before use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Training device (default: 0, i.e. cuda:0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader workers (default: 8).",
    )
    parser.add_argument(
        "--project",
        default="train/runs",
        help="Ultralytics project output directory (default: train/runs).",
    )
    parser.add_argument(
        "--name",
        default="yolo26m_pmfeed",
        help="Ultralytics run name (default: yolo26m_pmfeed).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20).",
    )
    parser.add_argument(
        "--cache",
        choices=["false", "true", "ram"],
        default="false",
        help="Ultralytics cache setting (default: false).",
    )
    parser.add_argument(
        "--class-name",
        default="cow",
        help="Class name for dataset.yaml (default: cow).",
    )
    parser.add_argument(
        "--classes-path",
        default="data/annotated_labels/classes.txt",
        help=(
            "Path to classes.txt with one class per line "
            "(default: data/annotated_labels/classes.txt)."
        ),
    )
    parser.add_argument(
        "--val-on-train",
        action="store_true",
        help="Use the full dataset for both train and val (no split).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare dataset but skip training.",
    )
    return parser


def _parse_frame_id(stem: str) -> int | None:
    parts = stem.split("_")
    if not parts:
        return None
    last = parts[-1]
    if last.isdigit():
        return int(last)
    return None


def _sort_key(path: Path) -> tuple[int, str]:
    frame_id = _parse_frame_id(path.stem)
    if frame_id is None:
        return (sys.maxsize, path.stem)
    return (frame_id, path.stem)


def _collect_images(data_root: Path) -> list[Path]:
    """Collect image files with supported extensions from a directory."""
    images: list[Path] = []
    for extension in _IMAGE_EXTENSIONS:
        images.extend(data_root.glob(f"*{extension}"))
    return sorted(images, key=_sort_key)


def _collect_pairs(data_root: Path) -> list[Pair]:
    images = _collect_images(data_root)
    pairs: list[Pair] = []
    missing_labels = 0
    for image_path in images:
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            missing_labels += 1
            continue
        pairs.append(Pair(image=image_path, label=label_path))

    if missing_labels:
        print(
            f"Warning: {missing_labels} images missing labels.",
            file=sys.stderr,
        )
    return pairs


def _collect_pairs_split(images_dir: Path, labels_dir: Path) -> list[Pair]:
    images = _collect_images(images_dir)
    pairs: list[Pair] = []
    missing_labels = 0
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        pairs.append(Pair(image=image_path, label=label_path))

    if missing_labels:
        print(
            f"Warning: {missing_labels} images missing labels.",
            file=sys.stderr,
        )
    return pairs


def _split_pairs(
    pairs: Sequence[Pair],
    val_ratio: float,
    val_count: int,
    split: str,
    seed: int,
) -> tuple[list[Pair], list[Pair]]:
    total = len(pairs)
    if total < 2:
        raise ValueError("Need at least 2 samples to create train/val splits")

    if val_count <= 0:
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val-ratio must be between 0 and 1")
        val_count = max(1, int(total * val_ratio))

    if val_count >= total:
        raise ValueError("val-count must be меньше общего числа кадров")

    pairs_list = list(pairs)
    if split == "random":
        rng = random.Random(seed)
        rng.shuffle(pairs_list)
        val_pairs = pairs_list[:val_count]
        train_pairs = pairs_list[val_count:]
    else:
        val_pairs = pairs_list[-val_count:]
        train_pairs = pairs_list[:-val_count]

    return train_pairs, val_pairs


def _prepare_output_dir(output_dir: Path, overwrite: bool, backup: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = output_dir.with_name(f"{output_dir.name}_bak_{timestamp}")
            output_dir.rename(backup_dir)
        elif overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(
                f"Output dir exists and is not empty: {output_dir}. "
                "Use --overwrite or --backup."
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def _ensure_dirs(root: Path) -> dict[str, Path]:
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    val_images = root / "val" / "images"
    val_labels = root / "val" / "labels"
    for path in (train_images, train_labels, val_images, val_labels):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
    }


def _link_or_copy(src: Path, dst: Path, method: str) -> None:
    if dst.exists():
        dst.unlink()

    if method == "copy":
        shutil.copy2(src, dst)
        return

    if method == "symlink":
        os.symlink(src, dst)
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _materialize(
    pairs: Iterable[Pair],
    images_dir: Path,
    labels_dir: Path,
    link_method: str,
) -> None:
    for pair in pairs:
        _link_or_copy(pair.image, images_dir / pair.image.name, link_method)
        _link_or_copy(pair.label, labels_dir / pair.label.name, link_method)

def _load_classes(classes_path: Path, fallback: str) -> list[str]:
    if classes_path.exists():
        lines = [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
        ]
        classes = [line for line in lines if line]
        if not classes:
            raise ValueError("classes.txt is empty")
        return classes
    if fallback:
        return [fallback]
    raise FileNotFoundError(
        f"Classes file not found: {classes_path}. "
        "Provide --classes-path or --class-name."
    )


def _write_data_yaml(output_dir: Path, class_names: Sequence[str]) -> Path:
    safe_names = [name.replace('"', "") for name in class_names]
    content = "\n".join(
        [
            f"path: {output_dir}",
            "train: train/images",
            "val: val/images",
            f"nc: {len(safe_names)}",
            f'names: {safe_names}',
            "",
        ]
    )
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(content, encoding="utf-8")
    return data_yaml


def _cache_value(cache: str) -> bool | str:
    if cache == "false":
        return False
    if cache == "true":
        return True
    return "ram"


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    if args.images_dir and args.labels_dir:
        images_dir = Path(args.images_dir)
        labels_dir = Path(args.labels_dir)
        if not images_dir.exists():
            print(f"Images dir not found: {images_dir}", file=sys.stderr)
            return 1
        if not labels_dir.exists():
            print(f"Labels dir not found: {labels_dir}", file=sys.stderr)
            return 1
        pairs = _collect_pairs_split(images_dir, labels_dir)
    else:
        data_root = Path(args.data_root)
        if not data_root.exists():
            print(f"Data root not found: {data_root}", file=sys.stderr)
            return 1
        pairs = _collect_pairs(data_root)
    if not pairs:
        print("No image/label pairs found.", file=sys.stderr)
        return 1

    try:
        class_names = _load_classes(Path(args.classes_path), args.class_name)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.val_on_train:
        train_pairs = list(pairs)
        val_pairs = list(pairs)
    else:
        try:
            train_pairs, val_pairs = _split_pairs(
                pairs,
                val_ratio=args.val_ratio,
                val_count=args.val_count,
                split=args.split,
                seed=args.seed,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    output_dir = Path(args.output_dir)
    try:
        _prepare_output_dir(output_dir, args.overwrite, args.backup)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    dirs = _ensure_dirs(output_dir)
    _materialize(train_pairs, dirs["train_images"], dirs["train_labels"], args.link_method)
    _materialize(val_pairs, dirs["val_images"], dirs["val_labels"], args.link_method)
    data_yaml = _write_data_yaml(output_dir, class_names)

    print(f"Prepared dataset at: {output_dir}")
    print(f"Train samples: {len(train_pairs)}")
    print(f"Val samples: {len(val_pairs)}")
    print(f"Data YAML: {data_yaml}")

    if args.dry_run:
        print("Dry run requested; skipping training.")
        return 0

    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        cache=_cache_value(args.cache),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
