"""Compare case 07's gray image with neighboring mirrored pages.

This checks source attribution, not pixel-level text/ghost ground truth.
"""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "data/split_dataset/val/200021660"
MASK = ROOT / "data/split_dataset_csc_mask_v2/val/200021660/200021660_00028_1/02_text_black.png"
BOX = (1093, 1723, 768, 768)
NAMES = ["200021660_00027_1", "200021660_00027_2", "200021660_00028_2",
         "200021660_00029_1", "200021660_00029_2", "200021660_00030_1"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residual(gray):
    gray = gray.astype("float32")
    return cv2.GaussianBlur(gray, (0, 0), 35) - cv2.GaussianBlur(gray, (0, 0), 4)


def main():
    x, y, w, h = BOX
    source = BASE / "200021660_00028_1.jpg"
    front = np.asarray(Image.open(source).convert("RGB"))
    text_mask = (np.asarray(Image.open(MASK).convert("L"))[y:y+h, x:x+w] == 0).astype("uint8") * 255
    text_mask = cv2.dilate(text_mask, np.ones((3, 3), np.uint8))
    original = front[y:y+h, x:x+w]
    # Removing the darker front strokes exposes the pale gray image for comparison.
    removed = cv2.inpaint(original, text_mask, 9, cv2.INPAINT_TELEA)
    target = residual(cv2.cvtColor(removed, cv2.COLOR_RGB2GRAY))
    target = cv2.resize(target, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    results = []
    best_image = None
    for name in NAMES:
        path = BASE / (name + ".jpg")
        reverse = ImageOps.mirror(Image.open(path).convert("RGB"))
        reverse = reverse.resize((front.shape[1], front.shape[0]), Image.Resampling.LANCZOS)
        gray = cv2.cvtColor(np.asarray(reverse), cv2.COLOR_RGB2GRAY)
        patch = residual(gray)[y-180:y+h+180, x-180:x+w+180]
        patch = cv2.resize(patch, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        corr = cv2.matchTemplate(patch, target, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(corr)
        dx, dy = location[0] * 2 - 180, location[1] * 2 - 180
        results.append({"page": name, "source_sha256": sha(path),
                        "correlation": round(float(score), 6), "shift_xy": [dx, dy]})
        if name == "200021660_00027_2":
            best_image = reverse.crop((x+dx, y+dy, x+dx+w, y+dy+h))
    assert best_image is not None
    assert max(results, key=lambda r:r["correlation"])["page"] == "200021660_00027_2"
    assert results[1]["correlation"] > 2 * max(r["correlation"] for r in results if r is not results[1])
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 21)
    small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 17)
    gap, top = 24, 102
    canvas = Image.new("RGB", (w*3 + gap*4, h + top + 75), "#f8f7f3")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 11), "07 裏移りの発生源：直前ページとの照合", font=font, fill="#1b2c38")
    labels = ("表画像 00028_1", "表文字を除いた灰色像", "00027_2 を反転・位置合わせ")
    for i, (label, image) in enumerate(zip(labels, (Image.fromarray(original), Image.fromarray(removed), best_image))):
        px = gap + i * (w+gap)
        draw.text((px, 66), label, font=small, fill="#405364")
        canvas.paste(image, (px, top))
    draw.text((24, canvas.height-50),
              f"局所相関：00027_2 = {results[1]['correlation']:.3f}、他候補の最大 = {max(r['correlation'] for r in results if r is not results[1]):.3f}。画素正解ではない。",
              font=small, fill="#405364")
    canvas.save(HERE / "07_裏移り_前後ページ照合.png")
    record = {"target": "200021660_00028_1", "target_source_sha256": sha(source),
              "target_mask_sha256": sha(MASK), "roi_xywh": list(BOX),
              "method": "CSC v2 front-text mask dilated 3x3; OpenCV Telea inpaint radius 9; Gaussian residual sigma 35 minus sigma 4; mirrored candidate resized to target page; half-size TM_CCOEFF_NORMED over +/-180 native pixels",
              "results": results,
              "interpretation": "The immediately preceding 00027_2 image is a much stronger source match than other tested neighboring pages. This supports show-through from the reverse/adjacent leaf; it does not establish pixelwise text/ghost ground truth."}
    (HERE / "07_裏移り_前後ページ照合.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
