"""Split data/full_dataset into train/val/test (images only).

Default output:
    data/split_dataset/
        train/<doc_id>/...image files...
        val/<doc_id>/...image files...
        test/<doc_id>/...image files...
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SPLIT_MAP = {
    "train": [
        "100241706",
        "100249376",
        "100249476",
        "100249537",
        "200003076",
        "200003967",
        "200004148",
        "200006663",
        "200006665",
        "200008316",
        "200014685",
        "200015779",
        "200020019",
        "200021086",
        "200021644",
        "200021712",
        "200021763",
        "200021802",
        "200021853",
        "200021925",
        "200025191",
        "brsk00000",
        "hnsd00000",
        "umgy00000",
    ],
    "val": [
        "100249371",
        "100249416",
        "200005598",
        "200014740",
        "200021637",
        "200021660",
        "200021851",
        "200021869",
        "200022050",
    ],
    "test": [
        "200003803",
        "200010454",
        "200015843",
        "200017458",
        "200018243",
        "200019865",
        "200021063",
        "200021071",
        "200004107",
        "200005798",
        "200008003",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split data/full_dataset into train/val/test.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/full_dataset"),
        help="Source dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/split_dataset"),
        help="Output root for the split dataset.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="copy",
        help="How to materialize the split tree.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing split entries if they already exist.",
    )
    return parser.parse_args()


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def make_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def materialize(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing source path: {source}")

    if destination.exists() or destination.is_symlink():
        if overwrite:
            remove_path(destination)
        else:
            return

    make_parent(destination)
    if mode == "symlink":
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        return

    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def build_split(source_root: Path, output_root: Path, mode: str, overwrite: bool) -> None:
    for split_name, document_ids in SPLIT_MAP.items():
        split_root = output_root / split_name
        if overwrite and split_root.exists():
            remove_path(split_root)
        split_root.mkdir(parents=True, exist_ok=True)

        for document_id in document_ids:
            source_document = source_root / document_id
            if not source_document.exists():
                raise FileNotFoundError(f"Missing document folder: {source_document}")

            materialize(source_document / "images", split_root / document_id, mode, overwrite)


def main() -> None:
    args = parse_args()
    build_split(args.source_root, args.output_root, args.mode, args.overwrite)
    print(f"Split dataset created at: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()