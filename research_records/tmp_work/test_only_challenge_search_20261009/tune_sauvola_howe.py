"""Exploratory fixed grid on the six selected test-origin development pages."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.feature import canny

from compare_three_methods import HERE, ROOT, ROIS


OUT = HERE / "parameter_tuning"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
WS = (31, 61, 101)
KS = (.10, .20, .30)
CS = (80, 160, 320)
HIGHS = (.25, .40, .50)


def howe_terms(gray: np.ndarray, high: float):
    I = gray.astype("float64") * 255
    dx = I[:-1, 1:] - I[:-1, :-1]
    dy = I[1:, :-1] - I[:-1, :-1]
    lap = np.gradient(dx, axis=1) + np.gradient(dy, axis=0)
    residual = I - ndi.gaussian_filter(I, 20, mode="reflect", radius=60)
    rms = np.sqrt(ndi.gaussian_filter(residual**2, 20, mode="constant", cval=0, radius=50))
    confident = residual / (rms + np.finfo(float).eps) > 2
    lap[confident[:-1, :-1]] = -500
    sm = ndi.gaussian_filter(gray, .6, mode="reflect", truncate=4)
    mag = np.hypot(ndi.sobel(sm, axis=0, mode="reflect"), ndi.sobel(sm, axis=1, mode="reflect"))
    maxmag = float(mag.max())
    edges = canny(gray, sigma=.6, low_threshold=.1*maxmag,
                  high_threshold=high*maxmag, mode="reflect")
    down = ~((edges[:-1, :-1] & (dy > 0)) | (edges[1:, :-1] & (dy <= 0)))
    right = ~((edges[:-1, :-1] & (dx > 0)) | (edges[:-1, 1:] & (dx <= 0)))
    return (1500-lap, 1500+lap, down[:-1, :].astype("float64"),
            right[:, :-1].astype("float64")), int(edges.sum())


def save_crop(mask: np.ndarray, roi: list[int], path: Path) -> dict:
    x, y, w, h = roi
    cut = mask[y:y+h, x:x+w]
    assert cut.shape == (h, w)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(cut, 0, 255).astype("uint8")).save(path)
    return {"path": str(path.relative_to(ROOT)), "black_pixels": int(cut.sum()),
            "black_fraction": round(float(cut.mean()), 6)}


def grid(case: dict, method: str, keys: list[str]) -> None:
    x, y, w, h = ROIS[case["id"]]
    gap, header, label_h = 18, 78, 38
    sheet = Image.new("RGB", (3*w+4*gap, header+3*(h+label_h+gap)+35), "#f6f4ef")
    d = ImageDraw.Draw(sheet)
    f1 = ImageFont.truetype(FONT, 27)
    f2 = ImageFont.truetype(FONT, 20)
    d.text((gap, 10), f"{case['id']} {case['label']}  {method} 固定候補", font=f1, fill="#20323c")
    d.text((gap, 46), f"{case['name']}  xywh={x},{y},{w},{h}  黒＝前景", font=f2, fill="#405462")
    for i, key in enumerate(keys):
        p = OUT / "crops" / method / case["id"] / f"{key}.png"
        im = Image.open(p).convert("RGB")
        px = gap + (i%3)*(w+gap)
        py = header + (i//3)*(h+label_h+gap)
        d.text((px, py), key, font=f2, fill="#1d4357")
        sheet.paste(im, (px, py+label_h))
        d.rectangle((px-1, py+label_h-1, px+w, py+label_h+h), outline="#8d9da3", width=1)
    dest = OUT / "grids" / f"{case['id']}_{method}_9条件.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)


def main() -> None:
    cases = json.loads((HERE / "test_candidates.json").read_text())["cases"]
    spec = importlib.util.spec_from_file_location(
        "classical_deep", ROOT / "tmp_work/classical_deep_review_20261008/run.py")
    deep = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(deep)
    cv2.setNumThreads(2)
    records = []
    for case in cases:
        name, cid = case["name"], case["id"]
        with Image.open(ROOT / case["source"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        gray = deep.base.gray(rgb)
        roi = ROIS[cid]
        entries = {"id": cid, "name": name, "roi_xywh": roi, "sauvola": {}, "howe": {}}
        sauvola_keys = []
        for w in WS:
            m, s, _ = deep.base.stats(gray, w)
            for k in KS:
                key = f"w{w}_k{k:.2f}"
                mask = gray < m*(1+k*(s/.5-1))
                path = OUT / "crops" / "sauvola" / cid / f"{key}.png"
                entries["sauvola"][key] = save_crop(mask, roi, path)
                sauvola_keys.append(key)
            print(cid, "Sauvola", w, "done", flush=True)
        grid(case, "sauvola", sauvola_keys)
        deep.SCRATCH = OUT / "scratch" / name
        deep.SCRATCH.mkdir(parents=True, exist_ok=True)
        howe_keys = []
        for high in HIGHS:
            terms, n_edges = howe_terms(gray, high)
            for c in CS:
                key = f"edge{high:.2f}_c{c}"
                mask, meta = deep.howe(terms, c)
                path = OUT / "crops" / "howe" / cid / f"{key}.png"
                entry = save_crop(mask, roi, path)
                entry.update(edge_pixels=n_edges, energy=meta["mincut_energy"])
                entries["howe"][key] = entry
                howe_keys.append(key)
            print(cid, "Howe edge", high, "done", flush=True)
        grid(case, "howe", howe_keys)
        records.append(entries)
        (OUT / "grid_results.json").write_text(json.dumps({
            "status": "partial" if len(records)<len(cases) else "complete",
            "role": "development tuning; not independent test accuracy",
            "sauvola": {"windows": WS, "k": KS, "R": .5},
            "howe": {"c": CS, "canny_high_relative": HIGHS,
                      "canny_low_relative": .1, "canny_sigma": .6,
                      "implementation": "local base-energy port, approximate Canny, no auto-tuning"},
            "cases": records}, ensure_ascii=False, indent=2)+"\n")
        print(cid, "all 18 done", flush=True)


if __name__ == "__main__":
    main()
