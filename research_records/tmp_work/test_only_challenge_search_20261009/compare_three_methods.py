"""Compare fixed CSC v2, Sauvola and local Howe on six test-only cases."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "three_method_comparison"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
ROIS = {
    "01": [570, 600, 750, 850],
    "02": [700, 1200, 850, 850],
    "03": [750, 700, 850, 850],
    "04": [550, 160, 800, 820],
    "06": [80, 95, 620, 530],
    "07": [800, 750, 850, 900],
}
LABELS = [("original", "原画像"), ("csc", "CSC v2 黒文字"),
          ("sauvola", "Sauvola"), ("howe", "Howe 移植")]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    assert path.is_file(), path
    with Image.open(path) as image:
        assert image.size == size, (path, image.size, size)
        im = image.convert("L")
    assert im.getextrema() == (0, 255), path
    assert np.isin(np.asarray(im), (0, 255)).all(), path
    return im


def compute_classical(case: dict, rgb: np.ndarray, deep) -> tuple[dict[str, Path], dict]:
    name = case["name"]
    paths = {method: OUT / "masks" / name / f"{method}.png"
             for method in ("sauvola", "howe")}
    if all(path.is_file() for path in paths.values()):
        for path in paths.values():
            load_mask(path, (rgb.shape[1], rgb.shape[0]))
        return paths, {"status": "reused", "settings": "previous fixed run"}
    assert not any(path.exists() for path in paths.values()), "Partial masks require inspection"
    deep.SCRATCH = OUT / "scratch" / name
    deep.SCRATCH.mkdir(parents=True, exist_ok=True)
    gray = deep.base.gray(rgb)
    sauvola = deep.base.sauvola(gray, 61, .2)
    deep.save_mask(paths["sauvola"], sauvola)
    print(f"{case['id']} Sauvola done", flush=True)
    terms, terms_meta = deep.howe_terms(gray)
    howe, howe_meta = deep.howe(terms, 160)
    deep.save_mask(paths["howe"], howe)
    print(f"{case['id']} Howe done", flush=True)
    for key, expected in (("sauvola", sauvola), ("howe", howe)):
        assert np.array_equal(deep.read_mask(paths[key]), expected)
    return paths, {"status": "computed", "sauvola": {"window": 61, "k": .2, "R": .5},
                   "howe": {"c": 160, "terms": terms_meta, "cut": howe_meta}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = json.loads((HERE / "test_candidates.json").read_text())["cases"]
    assert [case["id"] for case in cases] == list(ROIS)
    spec = importlib.util.spec_from_file_location(
        "classical_deep", ROOT / "tmp_work/classical_deep_review_20261008/run.py")
    deep = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(deep)
    cv2.setNumThreads(2)

    max_w = max(roi[2] for roi in ROIS.values())
    gap, left, header, row_header, footer = 20, 25, 155, 68, 85
    width = 2 * left + 4 * max_w + 3 * gap
    height = header + sum(row_header + ROIS[c["id"]][3] + gap for c in cases) + footer
    sheet = Image.new("RGB", (width, height), "#f5f3ed")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype(FONT, 31)
    label_font = ImageFont.truetype(FONT, 22)
    small_font = ImageFont.truetype(FONT, 17)
    draw.text((left, 12), "test由来6課題：原画像・CSC・Sauvola・Howe", font=title_font, fill="#20323c")
    draw.text((left, 56), "全ページを原寸処理してから、各行で同一座標を切り出し。黒＝文字候補。", font=small_font, fill="#405462")
    draw.text((left, 82), "CSC v2は保存済みマスク。Sauvola 窓61・k=0.2、Howe移植 c=160。自動調整なし。", font=small_font, fill="#405462")
    for col, (_, label) in enumerate(LABELS):
        draw.text((left + col * (max_w + gap), 119), label, font=label_font, fill="#1d4357")

    records = []
    y = header
    for case in cases:
        source = ROOT / case["source"]
        assert sha(source) == case["sha256"]
        with Image.open(source) as src:
            original = src.convert("RGB")
        rgb = np.asarray(original)
        masks, provenance = compute_classical(case, rgb, deep)
        csc = ROOT / "data/split_dataset_csc_mask_v2/test" / case["name"].split("_")[0] / case["name"] / "02_text_black.png"
        paths = {"original": source, "csc": csc, **masks}
        full = {"original": original}
        for method in ("csc", "sauvola", "howe"):
            full[method] = load_mask(paths[method], original.size)
        x, yy, w, h = ROIS[case["id"]]
        assert 0 <= x < original.width and 0 <= yy < original.height
        assert x+w <= original.width and yy+h <= original.height
        box = (x, yy, x+w, yy+h)
        draw.rectangle((left-8, y, width-left+8, y+row_header+h+5), fill="white", outline="#cad2d5", width=2)
        draw.text((left, y+7), f"{case['id']} {case['label']}  [{case['assessment']}]", font=label_font, fill="#20323c")
        draw.text((left, y+39), f"{case['name']}  /  xywh={x},{yy},{w},{h}", font=small_font, fill="#536a77")
        panels = []
        for col, (method, _) in enumerate(LABELS):
            crop = full[method].crop(box).convert("RGB")
            px, py = left + col*(max_w+gap), y+row_header
            sheet.paste(crop, (px, py))
            draw.rectangle((px-1, py-1, px+w, py+h), outline="#8398a2", width=1)
            panels.append({"method": method, "path": str(paths[method].relative_to(ROOT)),
                           "sha256": sha(paths[method]), "canvas_xywh": [px, py, w, h]})
            assert np.array_equal(np.asarray(sheet.crop((px, py, px+w, py+h))), np.asarray(crop))
        records.append({"id": case["id"], "name": case["name"], "roi_xywh": ROIS[case["id"]],
                        "source_sha256": case["sha256"], "panels": panels,
                        "classical_provenance": provenance})
        y += row_header+h+gap
        print(f"{case['id']} row done", flush=True)
    draw.text((left, height-footer+12), "探索的な目視比較。05は除外。02・04は弱い代替、06は紙面損傷で本文二値化の失敗例ではない。", font=small_font, fill="#405462")
    draw.text((left, height-footer+42), "画素正解や独立テスト精度、手法の総合順位を示す図ではない。Howeは局所移植でCanny近似を使用。", font=small_font, fill="#405462")
    output = OUT / "6課題_CSC_Sauvola_Howe_比較.png"
    sheet.save(output)
    verification = {"output": str(output.relative_to(ROOT)), "output_sha256": sha(output),
                    "selection_sha256": sha(HERE / "test_candidates.json"), "panels_checked": 4*len(cases),
                    "conditions": {"CSC": "saved ACE+CSC v2 black-text mask", "Sauvola": "raw native RGB to gray; window=61 k=0.2 R=0.5",
                                   "Howe": "raw native RGB to gray; local base-energy port c=160; approximate Canny; no auto-tuning"},
                    "cases": records}
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2)+"\n")
    print(f"{output} / 24 exact panels", flush=True)


if __name__ == "__main__":
    main()
