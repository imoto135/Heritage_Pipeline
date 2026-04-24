#!/usr/bin/env python3
"""Run the recommended KenLM -> BERT evaluation pipeline.

This is a thin launcher around eval_kenlm_then_bert.py with practical defaults
based on the paper setting used in this repository.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LM_DIR = BASE_DIR.parent
DEFAULT_ARPA = LM_DIR / "handchar_kenlmo3.arpa"
DEFAULT_OUT = LM_DIR / "outputs" / "kenlm_then_bert" / "best"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run best KenLM -> BERT pipeline")

    # Required
    ap.add_argument("--coord-csv", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--bert-model", required=True)

    # Tuned defaults from paper-side experiment script comments
    ap.add_argument("--arpa", default=str(DEFAULT_ARPA))
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--kenlm-lambda", type=float, default=1.6)
    ap.add_argument("--bert-lambda", type=float, default=0.8)
    ap.add_argument("--bert-passes", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))

    # Optional flags
    ap.add_argument("--freeze-if-not-in-topk", action="store_true")
    ap.add_argument("--annotate-all", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--draw-ok", action="store_true")

    # Optional visualization/stats args
    ap.add_argument("--font-path", default="")
    ap.add_argument("--font-size", type=int, default=18)
    ap.add_argument("--box-width", type=int, default=4)
    ap.add_argument("--max-labels", type=int, default=2000)
    ap.add_argument("--oracle-k", type=int, default=5)
    ap.add_argument("--confusion-topn", type=int, default=200)
    ap.add_argument("--confusion-min-count", type=int, default=5)
    ap.add_argument("--min-char-count", type=int, default=5)
    ap.add_argument("--pages", default="all")
    ap.add_argument("--demo-page", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    script = BASE_DIR / "eval_kenlm_then_bert.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--coord-csv", args.coord_csv,
        "--images-dir", args.images_dir,
        "--gt-dir", args.gt_dir,
        "--arpa", args.arpa,
        "--ckpt", args.ckpt,
        "--classes", args.classes,
        "--model", args.model,
        "--img-size", str(args.img_size),
        "--topk", str(args.topk),
        "--beam-size", str(args.beam_size),
        "--kenlm-lambda", str(args.kenlm_lambda),
        "--bert-model", args.bert_model,
        "--bert-lambda", str(args.bert_lambda),
        "--bert-passes", str(args.bert_passes),
        "--device", args.device,
        "--out-dir", args.out_dir,
        "--font-path", args.font_path,
        "--font-size", str(args.font_size),
        "--box-width", str(args.box_width),
        "--max-labels", str(args.max_labels),
        "--oracle-k", str(args.oracle_k),
        "--confusion-topn", str(args.confusion_topn),
        "--confusion-min-count", str(args.confusion_min_count),
        "--min-char-count", str(args.min_char_count),
        "--pages", args.pages,
        "--demo-page", args.demo_page,
    ]

    if args.freeze_if_not_in_topk:
        cmd.append("--freeze-if-not-in-topk")
    if args.annotate_all:
        cmd.append("--annotate-all")
    if args.stats:
        cmd.append("--stats")
    if args.draw_ok:
        cmd.append("--draw-ok")

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
