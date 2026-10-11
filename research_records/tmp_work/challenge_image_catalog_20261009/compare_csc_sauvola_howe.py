"""Render the seven locked ROIs as original / CSC v2 / Sauvola / Howe.

01-05 reuse saved full-page classical masks. 06-07 are computed on the full
native page with the same fixed local conditions, then cropped for display.
"""
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
PRIOR = ROOT / "tmp_work/classical_limits_by_category_20261008/masks"
OUT = HERE / "csc_sauvola_howe"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
LABELS = (("original", "原画像"), ("csc", "CSC v2 黒文字"),
          ("sauvola", "Sauvola"), ("howe", "Howe 移植"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    assert path.exists(), path
    with Image.open(path) as image:
        assert image.size == size, (path, image.size, size)
        mask = image.convert("L")
        values = set(mask.getextrema())
        assert values.issubset({0, 255}), (path, values)
        return mask.copy()


def csc_path(case: dict) -> Path:
    parts = Path(case["original"]).parts
    if parts[1] == "split_dataset":
        split, book = parts[2:4]
    else:
        assert parts[1] in ("train", "val", "test")
        split, book = parts[1:3]
    return ROOT / "data/split_dataset_csc_mask_v2" / split / book / case["page"] / "02_text_black.png"


def classical_paths(case: dict) -> dict[str, Path]:
    if case["id"] <= "05":
        return {method: PRIOR / case["page"] / f"{method}.png"
                for method in ("sauvola", "howe")}
    return {method: OUT / "new_full_page_masks" / case["page"] / f"{method}.png"
            for method in ("sauvola", "howe")}


def compute_missing(case: dict, source: Path, paths: dict[str, Path]) -> dict:
    if all(path.exists() for path in paths.values()):
        return {"status": "reused", "method_settings": "Saved full-page masks"}
    assert case["id"] in ("06", "07")
    assert not any(path.exists() for path in paths.values()), "Partial result: inspect before recomputing"
    spec = importlib.util.spec_from_file_location(
        "classical_deep", ROOT / "tmp_work/classical_deep_review_20261008/run.py")
    deep = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(deep)
    deep.SCRATCH = OUT / "scratch" / case["page"]
    deep.SCRATCH.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    gray = deep.base.gray(rgb)
    sauvola = deep.base.sauvola(gray, 61, .2)
    deep.save_mask(paths["sauvola"], sauvola)
    print(f"{case['id']} Sauvola full page complete", flush=True)
    terms, terms_meta = deep.howe_terms(gray)
    howe, howe_meta = deep.howe(terms, 160)
    deep.save_mask(paths["howe"], howe)
    print(f"{case['id']} Howe full page complete", flush=True)
    for method, expected in (("sauvola", sauvola), ("howe", howe)):
        assert np.array_equal(deep.read_mask(paths[method]), expected)
    return {"status": "computed", "method_settings": {
        "sauvola": {"window": 61, "k": .2, "R": .5,
                     "input": "raw RGB converted by OpenCV to gray / 255"},
        "howe": {"c": 160, "terms": terms_meta, "cut": howe_meta,
                 "limit": "Local base-energy port; skimage Canny, no MATLAB canny_old or auto-tuning"}}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    locked = json.loads((HERE / "selection_lock.json").read_text())
    cases = locked["cases"]
    assert [case["id"] for case in cases] == [f"{i:02d}" for i in range(1, 8)]
    sources: list[dict] = []
    images: list[dict[str, Image.Image]] = []
    max_w = max(case["roi_xywh"][2] for case in cases)
    gap, left, head, row_header, foot = 22, 28, 132, 70, 92
    width = left * 2 + 4 * max_w + 3 * gap
    height = head + sum(row_header + case["roi_xywh"][3] + gap for case in cases) + foot
    canvas = Image.new("RGB", (width, height), "#f5f3ed")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(FONT, 31)
    label_font = ImageFont.truetype(FONT, 23)
    small_font = ImageFont.truetype(FONT, 18)
    draw.text((left, 12), "7課題：原画像・CSC・Sauvola・Howe", font=title_font, fill="#20323c")
    draw.text((left, 57), "同じ原画像座標を表示。黒＝黒文字候補。各手法は全ページで処理後に切り出し。", font=small_font, fill="#405462")
    draw.text((left, 83), "CSC v2は保存マスク。Sauvola窓61/k=.2、Howe移植c=160（Canny近似・自動調整なし）。", font=small_font, fill="#405462")
    for col, (_, label) in enumerate(LABELS):
        draw.text((left + col * (max_w + gap), 110), label, font=small_font, fill="#1d4357")
    y = head
    panel_records = []
    for case in cases:
        source = ROOT / case["original"]
        assert sha(source) == case["source_sha256"], source
        with Image.open(source) as image:
            raw = image.convert("RGB")
        masks = classical_paths(case)
        provenance = compute_missing(case, source, masks)
        paths = {"original": source, "csc": csc_path(case), **masks}
        full = {"original": raw}
        for method in ("csc", "sauvola", "howe"):
            full[method] = load_mask(paths[method], raw.size)
        x, yy, w, h = case["roi_xywh"]
        box = (x, yy, x + w, yy + h)
        draw.rectangle((left - 10, y + 2, width - left + 10, y + row_header + h + 4),
                       fill="white", outline="#cad2d5", width=2)
        draw.text((left, y + 9), f"{case['id']} {case['challenge']}  [{case['role']}]",
                  font=label_font, fill="#20323c")
        draw.text((left, y + 41), f"{case['page']}  /  xywh={x},{yy},{w},{h}",
                  font=small_font, fill="#536a77")
        panels = []
        for col, (method, _) in enumerate(LABELS):
            crop = full[method].crop(box).convert("RGB")
            px = left + col * (max_w + gap)
            py = y + row_header
            canvas.paste(crop, (px, py))
            draw.rectangle((px - 1, py - 1, px + w, py + h), outline="#8398a2", width=1)
            panels.append({"method": method, "source": str(paths[method].relative_to(ROOT)),
                           "source_sha256": sha(paths[method]), "canvas_xywh": [px, py, w, h]})
        panel_records.append({"id": case["id"], "page": case["page"],
                              "roi_xywh": case["roi_xywh"], "panels": panels,
                              "classical_mask_provenance": provenance})
        images.append({method: full[method].crop(box).convert("RGB") for method, _ in LABELS})
        sources.append({"id": case["id"], "source_sha256": sha(source)})
        y += row_header + h + gap
        print(f"{case['id']} row rendered", flush=True)
    draw.text((left, height - foot + 15), "01–03 主対象、04 成功対照。05は材質未確定、06は紙面損傷、07は裏移り。", font=small_font, fill="#405462")
    draw.text((left, height - foot + 46), "探索的な目視比較。正解マスクや独立テスト精度、手法の総合順位ではない。", font=small_font, fill="#405462")
    image_path = OUT / "7課題_CSC_Sauvola_Howe_比較.png"
    canvas.save(image_path)
    with Image.open(image_path) as saved_image:
        saved = np.asarray(saved_image.convert("RGB"))
    checked = 0
    for row, crops in zip(panel_records, images):
        for panel in row["panels"]:
            px, py, w, h = panel["canvas_xywh"]
            assert np.array_equal(saved[py:py+h, px:px+w],
                                  np.asarray(crops[panel["method"]]))
            checked += 1
    record = {"selection_lock_sha256": sha(HERE / "selection_lock.json"),
              "output": str(image_path.relative_to(ROOT)), "output_sha256": sha(image_path),
              "pixel_exact_panels": checked, "cases": panel_records,
              "notes": ["CSC v2 is the stored ACE+CSC black-text mask, not raw-image-only input.",
                        "Sauvola and Howe use the native raw RGB converted to grayscale.",
                        "Howe is a local base-energy port with approximate Canny, not an exact MATLAB run.",
                        "06 and 07 classical masks are newly computed with the saved 01-05 settings."]}
    (OUT / "verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"{image_path} / {checked} exact panels", flush=True)


if __name__ == "__main__":
    main()
