"""Compare DDRM outputs while changing only the black-text mask (six cases)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / "ddrm_pilot"
PYTHON = Path("/home/imoto/miniconda3/envs/diffusion/bin/python")
PIPELINE = ROOT / "tmp_work/npj_dirt_control_20261002/source_snapshot/ddrm_npj_v2/scripts/pipeline_ddrm.py"
CHECKPOINT = ROOT / "modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth"
BACKGROUND = ROOT / "modules/ddrm_npj_v2/background/isemonogatari.png"
ROIS = {
    "01": (570, 600, 750, 850),
    "02": (700, 1200, 850, 850),
    "03": (750, 700, 850, 850),
    "04": (550, 160, 800, 820),
    "06": (80, 95, 620, 530),
    "07": (800, 750, 850, 900),
}
METHODS = ("csc", "sauvola", "howe")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def stage_case(case):
    page = case["name"]
    source = ROOT / case["source"]
    assert digest(source) == case["sha256"]
    image = Image.open(source)
    input_dir = OUT / "inputs" / page
    input_dir.mkdir(parents=True, exist_ok=True)
    input_link = input_dir / source.name
    if not input_link.exists():
        input_link.symlink_to(source)
    csc_dir = ROOT / "data/split_dataset_csc_mask_v2/test" / source.parent.name / page
    red_path = csc_dir / "03_text_red.png"
    assert np.asarray(Image.open(red_path).convert("L")).shape == (image.height, image.width)
    staged = {"id": case["id"], "page": page, "source": relative(source), "source_sha256": digest(source),
              "input_size": image.size, "red": relative(red_path), "red_sha256": digest(red_path), "masks": {}}
    for method in METHODS:
        black_path = (csc_dir / "02_text_black.png") if method == "csc" else (
            BASE / "three_method_comparison/masks" / page / (method + ".png"))
        black = np.asarray(Image.open(black_path).convert("L"))
        red = np.asarray(Image.open(red_path).convert("L"))
        assert black.shape == red.shape == (image.height, image.width)
        assert np.all((black == 0) | (black == 255))
        assert not np.any((black == 0) & (red == 0)), "black/red overlap needs a stated priority"
        dest = OUT / "masks" / method / page
        dest.mkdir(parents=True, exist_ok=True)
        for filename, src in (("02_text_black.png", black_path), ("03_text_red.png", red_path)):
            dst = dest / filename
            if not dst.exists():
                dst.symlink_to(src)
            assert digest(dst) == digest(src)
        staged["masks"][method] = {"black": relative(black_path), "black_sha256": digest(black_path),
                                     "black_pixels": int((black == 0).sum())}
    return staged


def run(case, method, gpu):
    page = case["page"]
    target = OUT / "outputs" / method / page / "result" / (page + ".png")
    if target.exists():
        assert Image.open(target).size == tuple(case["input_size"])
        return {"id": case["id"], "method": method, "seconds": None, "reused": True,
                "output": relative(target), "sha256": digest(target)}
    command = [str(PYTHON), "-B", str(PIPELINE), "--input_dir", str(OUT / "inputs" / page),
               "--output_dir", str(OUT / "outputs" / method), "--tmp_dir", str(OUT / "infer_tmp" / method / page),
               "--mask_dir", str(OUT / "masks" / method), "--ckpt", str(CHECKPOINT),
               "--back_ground", str(BACKGROUND), "--device", f"cuda:{gpu}",
               "--patch_size", "128", "--overlap", "32", "--shared_noise", "--no_wandb"]
    cwd = OUT / "work" / case["id"] / method
    cwd.mkdir(parents=True, exist_ok=True)
    log = OUT / "logs" / f"{case['id']}_{method}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {case['id']} {method} GPU {gpu}", flush=True)
    start = time.monotonic()
    with log.open("w") as f:
        proc = subprocess.run(command, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
                                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"))
    assert proc.returncode == 0, f"{log}: exit {proc.returncode}"
    assert target.exists() and Image.open(target).size == tuple(case["input_size"]), target
    result = {"id": case["id"], "method": method, "seconds": round(time.monotonic() - start, 2),
              "reused": False, "command": command, "output": relative(target), "sha256": digest(target)}
    save_json(OUT / "logs" / f"{case['id']}_{method}.json", result)
    print(f"DONE {case['id']} {method} {result['seconds']} sec", flush=True)
    return result


def run_case(case, gpu):
    return [run(case, method, gpu) for method in METHODS]


def render(cases):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    panels = []
    for case in cases:
        x, y, w, h = ROIS[case["id"]]
        paths = [ROOT / case["source"]] + [OUT / "outputs" / method / case["page"] / "result" /
                                              (case["page"] + ".png") for method in METHODS]
        row = []
        for title, path in zip(("Original", "CSC black", "Sauvola black", "Howe black"), paths):
            crop = Image.open(path).convert("RGB").crop((x, y, x+w, y+h))
            panel = Image.new("RGB", (w, h+48), "white")
            panel.paste(crop, (0, 48))
            ImageDraw.Draw(panel).text((12, 7), f"{case['id']}  {title}", font=font, fill="black")
            row.append(panel)
        single = Image.new("RGB", (w*4, h+48), "#dddddd")
        for i, panel in enumerate(row):
            single.paste(panel, (i*w, 0))
        single.save(OUT / f"{case['id']}_DDRM_3文字マスク比較.png")
        panels.extend(row)
    width = max(p.width for p in panels)
    row_heights = [max(p.height for p in panels[i*4:(i+1)*4]) for i in range(len(cases))]
    offsets = [0]
    for row_height in row_heights:
        offsets.append(offsets[-1] + row_height)
    board = Image.new("RGB", (width*4, offsets[-1]), "#dddddd")
    for i, panel in enumerate(panels):
        board.paste(panel, ((i % 4)*width, offsets[i // 4]))
    path = OUT / "6課題_DDRM_3文字マスク比較.png"
    board.save(path)
    return relative(path)


def main():
    candidates = json.loads((BASE / "test_candidates.json").read_text())["cases"]
    cases = [stage_case(c) for c in candidates if c["id"] in ROIS]
    assert [c["id"] for c in cases] == ["01", "02", "03", "04", "06", "07"]
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
                "factor": "black text mask only", "fixed_red": "saved CSC v2 red mask (zero red pixels in all six pages)",
                "ddrm": {"code": relative(PIPELINE), "code_sha256": digest(PIPELINE),
                         "checkpoint": relative(CHECKPOINT), "checkpoint_sha256": digest(CHECKPOINT),
                         "background": relative(BACKGROUND), "background_sha256": digest(BACKGROUND),
                         "seed": 999, "timesteps": 20, "patch_size": 128, "overlap": 32,
                         "shared_noise": True, "native_full_page": True},
                "rois_xywh": ROIS,
                "interpretation": "development examples selected and tuned on these pages, not independent test accuracy"}
    save_json(OUT / "protocol.json", protocol)
    def run_group(group, gpu):
        return [result for case in group for result in run_case(case, gpu)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(run_group, [cases[0], cases[1], cases[3]], 0)
        b = pool.submit(run_group, [cases[2], cases[4], cases[5]], 1)
        results = a.result() + b.result()
    figure = render(cases)
    save_json(OUT / "complete.json", {"results": results, "figure": figure})
    print(f"FIGURE {figure}", flush=True)


if __name__ == "__main__":
    main()
