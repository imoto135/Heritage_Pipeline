"""Register a mirrored neighboring scan and audit conservative bleed removal.

This is a development diagnostic, not a pixel ground-truth generator.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import run_ddrm_pilot as driver

BASE = Path(__file__).resolve().parent
ROOT = driver.ROOT
OUT = BASE / 'bleed_pair_registration'
PAGE = '200021071_00033_2'
SOURCE_PAGE = '200021071_00033_1'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def read_mask(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) < 127


def landmarks(mask):
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    large = stats[:, 4] > 800
    large[0] = False
    distances = cv2.distanceTransform((~large[labels]).astype(np.uint8), cv2.DIST_L2, 5)
    nearest = np.full(n, np.inf)
    np.minimum.at(nearest, labels.ravel(), distances.ravel())
    x, y, w, h, area = stats.T
    H, W = mask.shape
    accept = ((area >= 10) & (area <= 800) & (w >= 4) & (w <= 35) &
              (h >= 4) & (h <= 35) & (w / np.maximum(h, 1) >= .4) &
              (w / np.maximum(h, 1) <= 2.5) & (area / np.maximum(w*h, 1) >= .45) &
              (nearest >= 25) & (centers[:, 0] > 100) & (centers[:, 0] < W-100) &
              (centers[:, 1] > 80) & (centers[:, 1] < H-80))
    accept[0] = False
    return centers[accept].astype(np.float32)


def board(path, arrays, titles, roi, width=None):
    x, y, w, h = roi
    font = ImageFont.truetype(FONT, 24)
    sheet = Image.new('RGB', (len(arrays)*(w+12)+12, h+70), '#f4f2ec')
    draw = ImageDraw.Draw(sheet)
    for i, (array, title) in enumerate(zip(arrays, titles)):
        px = 12+i*(w+12)
        draw.text((px, 12), title, font=font, fill='black')
        crop = Image.fromarray(array[y:y+h, x:x+w])
        sheet.paste(crop, (px, 58))
    if width:
        sheet = sheet.resize((width, round(sheet.height*width/sheet.width)), Image.Resampling.LANCZOS)
    sheet.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target_path = ROOT / 'data/test/200021071' / (PAGE+'.jpg')
    source_path = ROOT / 'data/test/200021071' / (SOURCE_PAGE+'.jpg')
    masks_root = ROOT / 'data/split_dataset_csc_mask_v2/test/200021071'
    source_mask = masks_root / SOURCE_PAGE / '02_text_black.png'
    target_mask = masks_root / PAGE / '02_text_black.png'
    howe_path = BASE / 'ddrm_pilot/masks/howe' / PAGE / '02_text_black.png'
    red_path = BASE / 'ddrm_pilot/masks/howe' / PAGE / '03_text_red.png'
    output_paths = [BASE / 'ddrm_pilot/outputs' / m / PAGE / 'result' / (PAGE+'.png')
                    for m in ('csc', 'howe')]
    dependencies = {driver.relative(p): driver.digest(p) for p in
                    [target_path, source_path, source_mask, target_mask, howe_path, red_path]+output_paths}
    raw = cv2.imread(str(target_path))
    H, W = raw.shape[:2]
    source = cv2.resize(cv2.imread(str(source_path))[:, ::-1], (W, H))
    mirrored_mask = cv2.resize(read_mask(source_mask)[:, ::-1].astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST) > 0
    sp, tp = landmarks(mirrored_mask), landmarks(read_mask(target_mask))
    d = np.linalg.norm(sp[:, None]-tp[None, :], axis=2)
    pairs = [(i, j) for i, j in enumerate(d.argmin(axis=1))
             if d[i, j] < 20 and d[:, j].argmin() == i]
    sp = np.array([sp[i] for i, j in pairs], dtype=np.float32)
    tp = np.array([tp[j] for i, j in pairs], dtype=np.float32)
    affine, accepted = cv2.estimateAffine2D(sp, tp, method=cv2.RANSAC,
                                           ransacReprojThreshold=5, maxIters=10000)
    assert affine is not None
    accepted = accepted.ravel().astype(bool)
    assert accepted.sum() >= 6
    assert .95 < np.linalg.det(affine[:, :2]) < 1.05
    errors = np.linalg.norm(np.c_[sp, np.ones(len(sp))] @ affine.T-tp, axis=1)
    ids = np.flatnonzero(accepted)
    loo = []
    for i in ids:
        train = ids[ids != i]
        fit = np.linalg.lstsq(np.c_[sp[train], np.ones(len(train))], tp[train], rcond=None)[0]
        loo.append(float(np.linalg.norm(np.r_[sp[i], 1.] @ fit-tp[i])))
    assert max(loo) < 5
    registered = cv2.warpAffine(source, affine, (W, H), borderValue=(255, 255, 255))
    registered_mask = cv2.warpAffine(mirrored_mask.astype(np.uint8), affine, (W, H),
                                    flags=cv2.INTER_NEAREST) > 0
    cv2.imwrite(str(OUT/'source_affine_registered.png'), registered)
    Image.fromarray(np.where(registered_mask, 0, 255).astype(np.uint8)).save(OUT/'source_registered_black.png')
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    source_gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
    howe, csc, red = read_mask(howe_path), read_mask(target_mask), read_mask(red_path)
    support = cv2.dilate((source_gray < 130).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(howe.astype(np.uint8), 8)
    remove = np.zeros_like(howe)
    rows = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        sl = np.s_[y:y+h, x:x+w]
        component = labels[sl] == i
        values = gray[sl][component]
        sfrac = float(support[sl][component].mean())
        dfrac = float((values < 130).mean())
        cfrac = float(csc[sl][component].mean())
        chosen = area >= 16 and sfrac >= .7 and dfrac == 0 and cfrac <= .05 and not red[sl][component].any()
        if chosen:
            remove[sl] |= component
        if (area >= 8 and dfrac < .15) or chosen:
            rows.append(dict(label=i, xywh=[x, y, w, h], area=area, source_support_fraction=sfrac,
                             target_dark_fraction=dfrac, csc_fraction=cfrac,
                             median_gray=float(np.median(values)), selected=chosen))
    candidate = howe & ~remove
    destination = OUT/'masks/paired'/PAGE
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(candidate, 0, 255).astype(np.uint8)).save(destination/'02_text_black.png')
    Image.open(red_path).save(destination/'03_text_red.png')
    identical = np.array_equal(candidate, howe)
    assert identical, 'Review newly selected components before any new inference'
    reused = OUT/'outputs/paired'/PAGE/'result'/(PAGE+'.png')
    reused.parent.mkdir(parents=True, exist_ok=True)
    if not reused.exists():
        reused.symlink_to(output_paths[1])
    assert driver.digest(reused) == driver.digest(output_paths[1])
    original_rgb, registered_rgb = raw[:, :, ::-1], registered[:, :, ::-1]
    outputs = [np.asarray(Image.open(p).convert('RGB')) for p in output_paths]
    overlay = original_rgb.copy()
    overlay[howe & ~csc] = [255, 60, 60]
    roi = driver.ROIS['07']
    board(OUT/'07_位置合わせと既存DDRM比較.png',
          [original_rgb, registered_rgb]+outputs,
          ['07 原画像', '直前ページの鏡像・位置合わせ', 'CSC → DDRM', 'Howe → DDRM'], roi)
    board(OUT/'07_位置合わせと既存DDRM比較_preview.png',
          [original_rgb, registered_rgb]+outputs,
          ['07 原画像', '直前ページの鏡像・位置合わせ', 'CSC → DDRM', 'Howe → DDRM'], roi, 1800)
    board(OUT/'07_マスク候補比較.png', [original_rgb, overlay,
          np.repeat(np.where(howe, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2),
          np.repeat(np.where(candidate, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)],
          ['原画像', 'Howe追加候補（赤）', '元のHoweマスク', '対応条件後（変更0画素）'], roi)
    # Check the actual matched landmarks in context, including the RANSAC outlier.
    gallery = Image.new('RGB', (640, len(sp)*185), 'white')
    draw = ImageDraw.Draw(gallery)
    font = ImageFont.truetype(FONT, 19)
    for i, (p, q) in enumerate(zip(sp, tp)):
        cx, cy = np.round(q).astype(int)
        for k, arr in enumerate([original_rgb, registered_rgb]):
            cr = Image.fromarray(arr).crop((cx-70, cy-65, cx+70, cy+65)).resize((280, 130))
            gallery.paste(cr, (k*320, i*185+45))
        draw.text((5, i*185+8), f'{i+1} 原画像／位置合わせ参照 '+
                  ('採用' if accepted[i] else '除外')+f' 残差 {errors[i]:.2f}px', font=font, fill='black')
    gallery.save(OUT/'07_対応点検証.png')
    contexts = Image.new('RGB', (1120, len(rows)*210), 'white')
    draw = ImageDraw.Draw(contexts)
    for i, row in enumerate(rows):
        x, y, w, h = row['xywh']
        cx, cy = x+w//2, y+h//2
        draw.text((5, i*210+3), f"成分 {row['label']} xywh={row['xywh']} 参照支持率 "
                  f"{row['source_support_fraction']:.2f}（未確定）", font=font, fill='black')
        for k, (arr, title) in enumerate(zip([original_rgb, registered_rgb]+outputs,
                ['原画像', '位置合わせ参照', 'CSC → DDRM', 'Howe → DDRM'])):
            draw.text((k*280+4, i*210+28), title, font=font, fill='black')
            cr = Image.fromarray(arr).crop((cx-90, cy-70, cx+90, cy+70)).resize((270, 150))
            contexts.paste(cr, (k*280, i*210+57))
            draw.rectangle((k*280+135-w*.75, i*210+132-h*150/280,
                            k*280+135+w*.75, i*210+132+h*150/280), outline='red', width=2)
    contexts.save(OUT/'07_残存候補の文脈比較.png')
    x, y, w, h = roi
    sl = np.s_[y:y+h, x:x+w]
    protocol = dict(target=driver.relative(target_path), source=driver.relative(source_path),
        transformation='horizontal mirror, resize to target, then affine source-to-target',
        source_points=sp.tolist(), target_points=tp.tolist(), affine=affine.tolist(),
        inliers=accepted.tolist(), residuals_px=errors.tolist(), leave_one_out_errors_px=loo,
        accepted_landmark_bounds_xy=[tp[accepted].min(axis=0).tolist(), tp[accepted].max(axis=0).tolist()],
        registration_limit='six spot correspondences concentrated on right side; left-side accuracy and local paper warp unverified',
        pixel_ground_truth=False, selected_removal_pixels=int(remove.sum()),
        selected_removal_pixels_roi=int(remove[sl].sum()), candidate_mask_identical=identical,
        red_mask_identical=np.array_equal(read_mask(destination/'03_text_red.png'), red),
        mask_rule=dict(source_gray_threshold=130, source_support_dilation_radius_px=5,
                       min_component_area=16, min_source_support=.7,
                       target_gray_threshold=130, max_target_dark_fraction=0,
                       max_csc_black_fraction=.05, protect_red_overlap=True),
        pale_component_diagnostics=rows, dependencies=dependencies,
        output_reused=True, reused_output_sha256=driver.digest(reused),
        development_examples=True, shared_stepwise_noise_guaranteed=False,
        script_sha256=driver.digest(Path(__file__)))
    driver.save_json(OUT/'registration_and_candidate.json', protocol)
    for rel, expected in dependencies.items():
        assert driver.digest(ROOT/rel) == expected
    print(json.dumps(dict(inliers=int(accepted.sum()), max_leave_one_out_px=max(loo),
                          selected_pixels=int(remove.sum()), identical=identical), ensure_ascii=False))


if __name__ == '__main__':
    main()
