"""Stage 0/1: page-specific pigment and counterfactual cross-section fits.

Known development rectangles only. No neural training, pixel GT, restoration,
or dense editing mask. Finite model sensitivity is not a probability or bound.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tmp_work/ink_background_profile_pilot_v0_20260930"
V6 = ROOT / "tmp_work/warm_ink_hold_v6_20260930"
os.environ["MPLCONFIGDIR"] = str(OUT / "plot_cache")
import cv2
import numpy as np
cv2.setNumThreads(1)

ROIS = [
    ("200021763_00002_2", "long_tail", [200, 2020, 490, 450]),
    ("200021763_00002_2", "upper_writing", [190, 1080, 520, 290]),
    ("200021086_00032_2", "faint_stamp", [680, 620, 430, 330]),
    ("200010454_00002_2", "near_seal_brown", [1655, 1810, 75, 195]),
    ("200010454_00002_2", "remote_brown", [710, 200, 450, 330]),
]
SPACES = ["rgb_additive", "encoded_od", "linear_srgb_od"]
WIDTHS = [1.5, 2.5, 4.0, 6.0, 9.0, 12.0]
CORE_BG = ["linear", "quadratic", "soft_edge", "sharp_edge", "broad_spot"]
CREASE_BG = ["crease_4", "crease_6", "crease_12"]
SETTINGS = dict(
    purpose="Stage 0/1 feasibility; known development; no accuracy test",
    median=3, half_profile=32, geometry_sigmas=[2., 4., 8.],
    geometry="Hessian of min(log(R/G), log(R/B)); no v6 color/distance gate",
    min_ridge=.003, roi_grid=16, max_roi_profiles=160,
    anchor_grid=24, max_anchor_profiles=384, anchor_roi_exclusion_margin=40,
    anchors="Frozen v2 red & robust_seed; exclude all target ROIs plus 40px",
    anchor_split="64px spatial tiles; (tile_x + 3 tile_y) % 4 == 0 held out",
    min_anchor_estimation=12, min_anchor_confirmation=4,
    anchor_direction="Median nonnegative normalized Gaussian contrast vectors",
    spaces=SPACES, ink_widths=WIDTHS,
    backgrounds=CORE_BG, crease_stress=CREASE_BG,
    noise="Flank first-difference MAD/sqrt(2), floor per color representation",
    noise_floors=dict(rgb_additive=.003, encoded_od=.005, linear_srgb_od=.01),
    cv="4 interleaved intensity folds; center geometry uses complete input",
    acceptable_mse=4., near_best_mse=.25, amplitude_snr_gate=1.,
    scenarios=["restricted_background", "colored_crease_allowed"],
    note="Crease models are no-ink alternative explanations; simultaneous "
         "crease-plus-ink is not solved. Ranges cover evaluated fits only.",
    outputs="Sampled locations and cross-sections, never a dense edit mask",
    insufficient_anchor_policy="Unexplained; never borrow pigment from another page",
    pixel_GT=False, neural_training=False, pretrained_inference=False,
)
T = np.arange(-32, 33, dtype=float)
U = T / 32.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read(path, mask=False):
    v = cv2.imread(str(path), 0 if mask else 1)
    if v is None:
        raise OSError(path)
    return v > 0 if mask else v


def write(path, v):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), v), path


def transform(rgb, space):
    f = rgb.astype(float) / 255.
    if space == "rgb_additive":
        return -f
    if space == "encoded_od":
        return -np.log((rgb.astype(float) + 1.) / 256.)
    f = np.where(f <= .04045, f / 12.92, ((f + .055) / 1.055) ** 2.4)
    return -np.log(f + 1. / 65535.)


def noise_scale(y, space):
    flank = np.abs(T) >= 22
    d = np.diff(y, axis=1)
    d = d[:, flank[:-1] & flank[1:], :]
    med = np.median(d, axis=1, keepdims=True)
    noise = 1.4826 * np.median(np.abs(d - med), axis=1) / np.sqrt(2.)
    return np.maximum(noise, SETTINGS["noise_floors"][space])


def geometry(bgr):
    z = transform(bgr[:, :, ::-1], "encoded_od")
    strength = np.minimum(z[:, :, 1] - z[:, :, 0], z[:, :, 2] - z[:, :, 0]).astype(np.float32)
    score = np.zeros(strength.shape, np.float32)
    angle = score.copy()
    for s in SETTINGS["geometry_sigmas"]:
        smooth = cv2.GaussianBlur(strength, (0, 0), s)
        xx, yy, xy = [cv2.Sobel(smooth, cv2.CV_32F, a, b, ksize=3, scale=s*s/4.)
                      for a, b in [(2, 0), (0, 2), (1, 1)]]
        root = np.sqrt((xx-yy)**2 + 4*xy**2)
        lo, hi = (xx+yy-root)/2., (xx+yy+root)/2.
        r = np.maximum(-lo, 0.) * np.exp(-2*(hi/(np.abs(lo)+1e-8))**2)
        r[(lo >= 0) | (np.abs(hi) > .5*np.abs(lo))] = 0
        a = .5*np.arctan2(2*xy, xx-yy) + np.pi/2.
        replace = r > score
        angle[replace], score[replace] = a[replace], r[replace]
    return score, angle


def select_points(score, angle, eligible, origin, grid, cap):
    points = []
    for y in range(0, score.shape[0], grid):
        for x in range(0, score.shape[1], grid):
            s = score[y:y+grid, x:x+grid]
            allowed = eligible[y:y+grid, x:x+grid]
            if not allowed.any():
                continue
            p = int(np.argmax(np.where(allowed, s, -1.)))
            dy, dx = np.unravel_index(p, s.shape)
            py, px = y+dy, x+dx
            if float(score[py, px]) >= SETTINGS["min_ridge"]:
                points.append([px+origin[0], py+origin[1], float(angle[py, px]), float(score[py, px])])
    points.sort(key=lambda p: (-p[3], p[1], p[0]))
    return np.asarray(points[:cap], float).reshape(-1, 4)


def sample_profiles(bgr, points):
    assert len(points)
    mx = points[:, 0, None] + np.cos(points[:, 2, None]) * T
    my = points[:, 1, None] + np.sin(points[:, 2, None]) * T
    assert mx.min() >= 0 and my.min() >= 0
    assert mx.max() < bgr.shape[1]-1 and my.max() < bgr.shape[0]-1
    rgb = cv2.remap(bgr, mx.astype(np.float32), my.astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)[:, :, ::-1]
    return rgb


def background(name):
    columns = [np.ones_like(U), U]
    if name == "quadratic":
        columns.append(U**2)
    elif name == "soft_edge":
        columns.append(np.tanh(T/8.))
    elif name == "sharp_edge":
        columns.append(np.tanh(T/2.))
    elif name == "broad_spot":
        columns.append(np.exp(-.5*(T/24.)**2))
    elif name.startswith("crease_"):
        columns.append(np.exp(-.5*(T/float(name.split("_")[1]))**2))
    return np.stack(columns, axis=1)


def fit(y, noise, q, bg, width, train):
    """Weighted RGB fit with one shared nonnegative pigment amplitude."""
    b = background(bg)
    bt = b[train]
    pinv = np.linalg.pinv(bt)
    residual = y[:, train] - np.einsum("tk,nkc->ntc", bt, np.einsum("kt,ntc->nkc", pinv, y[:, train]))
    amp = np.zeros(len(y))
    f = np.zeros_like(T) if width == 0 else np.exp(-.5*(T/width)**2)
    if width:
        fp = f[train] - bt @ (pinv @ f[train])
        denom = np.sum(fp**2) * np.sum((q[None, :]/noise)**2, axis=1)
        assert (denom > 1e-12).all(), "Ink profile aliases a background basis"
        numer = np.einsum("t,c,ntc,nc->n", fp, q, residual, 1./noise**2)
        amp = np.maximum(numer / denom, 0.)
    remainder = y[:, train] - amp[:, None, None]*f[train][None, :, None]*q[None, None, :]
    coef = np.einsum("kt,ntc->nkc", pinv, remainder)
    prediction = np.einsum("tk,nkc->ntc", b, coef) + amp[:, None, None]*f[None, :, None]*q[None, None, :]
    return amp, prediction


def models(y, noise, q):
    entries, scores, amps, predictions = [], [], [], []
    for bg in CORE_BG + CREASE_BG:
        # Colored crease is an alternative background-only explanation.
        for width in ([0.] if bg in CREASE_BG else [0.] + WIDTHS):
            se = np.zeros(len(y))
            for fold in range(4):
                test = np.arange(len(T)) % 4 == fold
                _, prediction = fit(y, noise, q, bg, width, ~test)
                se += np.sum(((prediction[:, test]-y[:, test])/noise[:, None, :])**2, axis=(1, 2))
            amp, prediction = fit(y, noise, q, bg, width, np.ones(len(T), bool))
            scores.append(se/(len(T)*3))
            amps.append(amp*np.sqrt(np.mean((q[None, :]/noise)**2, axis=1)))
            predictions.append(prediction)
            entries.append(dict(background=bg, ink_width=width, crease=bg in CREASE_BG))
    return entries, np.stack(scores, 1), np.stack(amps, 1), np.stack(predictions, 1)


def model_entries():
    return [dict(background=bg, ink_width=w, crease=bg in CREASE_BG)
            for bg in CORE_BG + CREASE_BG
            for w in ([0.] if bg in CREASE_BG else [0.] + WIDTHS)]


def color_direction(y, noise):
    """Independent-channel contrast; calibration only, not ink recognition."""
    scores, coefs = [], []
    b = background("linear")
    for width in WIDTHS:
        f = np.exp(-.5*(T/width)**2)
        design = np.column_stack([b, f])
        coef = np.einsum("kt,ntc->nkc", np.linalg.pinv(design), y)
        pred = np.einsum("tk,nkc->ntc", design, coef)
        scores.append(np.mean(((pred-y)/noise[:, None, :])**2, axis=(1, 2)))
        coefs.append(coef[:, -1])
    choice = np.argmin(np.stack(scores, 1), axis=1)
    coeff = np.stack(coefs, 1)[np.arange(len(y)), choice]
    norm = np.linalg.norm(coeff, axis=1)
    usable = (norm > 1e-8) & (coeff.min(axis=1) >= -.1*norm)
    usable &= (coeff[:, 1] > coeff[:, 0]) & (coeff[:, 2] > coeff[:, 0])
    unit = np.maximum(coeff, 0.)
    unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    return unit, usable, choice


def classify(scores, amplitudes, entries, allow_crease):
    active = np.asarray([allow_crease or not m["crease"] for m in entries])
    score = scores[:, :, active]
    amp = amplitudes[:, :, active]
    best = score.min(axis=2, keepdims=True)
    admissible = (score <= best + SETTINGS["near_best_mse"]) & (score <= SETTINGS["acceptable_mse"])
    enough = admissible.any(axis=2).all(axis=1)
    low = np.min(np.where(admissible, amp, np.inf), axis=(1, 2))
    high = np.max(np.where(admissible, amp, -np.inf), axis=(1, 2))
    low[~enough], high[~enough] = 0., 0.
    label = np.full(len(scores), "unexplained", dtype="U16")
    label[enough & (high < SETTINGS["amplitude_snr_gate"])] = "low_ink"
    label[enough & (high >= SETTINGS["amplitude_snr_gate"])] = "ambiguous"
    label[enough & (low >= SETTINGS["amplitude_snr_gate"])] = "ink_required"
    return label, low, high, admissible


def numerical_checks():
    q = np.array([.1, .5, .86]); q /= np.linalg.norm(q)
    base = np.array([.2, .4, .6])[None, None, :] + .03*U[None, :, None]
    ink = base + .35*np.exp(-.5*(T/6.)**2)[None, :, None]*q[None, None, :]
    noise = np.full((1, 3), .005)
    a, p = fit(ink, noise, q, "linear", 6., np.ones(len(T), bool))
    assert abs(a[0]-.35) < 1e-10 and np.max(np.abs(p-ink)) < 1e-10
    entries, s, amps, _ = models(ink, noise, q)
    score = np.repeat(s[:, None, :], 3, axis=1)
    amp = np.repeat(amps[:, None, :], 3, axis=1)
    assert classify(score, amp, entries, False)[0][0] == "ink_required"
    assert classify(score, amp, entries, True)[0][0] == "ambiguous"
    _, s, amps, _ = models(base, noise, q)
    assert classify(np.repeat(s[:, None, :], 3, axis=1),
                    np.repeat(amps[:, None, :], 3, axis=1), entries, False)[0][0] == "low_ink"
    inverted = base - .3*np.exp(-.5*(T/6.)**2)[None, :, None]*q[None, None, :]
    assert fit(inverted, noise, q, "linear", 6., np.ones(len(T), bool))[0][0] == 0
    test = np.arange(len(T)) % 4 == 0
    altered = ink.copy(); altered[:, test] += .2
    before = fit(ink, noise, q, "linear", 6., ~test)
    after = fit(altered, noise, q, "linear", 6., ~test)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    return dict(exact_known_pigment_recovery=True, colored_crease_ambiguity_detected=True,
                background_only_detected=True, nonnegative_constraint_checked=True,
                withheld_intensities_not_used_in_fold_fit=True)


def counts(labels, selection=None):
    labels = labels if selection is None else labels[selection]
    return {k: int(np.sum(labels == k)) for k in ["ink_required", "ambiguous", "low_ink", "unexplained"]}


def main():
    if (OUT/"complete.json").exists():
        raise RuntimeError("Completed experiment is frozen; use a new output directory")
    OUT.mkdir(parents=True, exist_ok=True)
    source = Path(__file__)
    prior = json.loads((V6/"manifest.json").read_text())
    cases = {r["id"]: r for r in prior["cases"] if r["id"] in {r[0] for r in ROIS}}
    dependencies = {str(p.relative_to(ROOT)): sha(p) for p in
                    [V6/"candidate_freeze.json", V6/"manifest.json", V6/"complete.json"]}
    for case in cases.values():
        for p in [ROOT/case["original"], V6/case["id"]/"original.png",
                  V6/case["id"]/"split_direction_edit_allowed.png"] + [
                      ROOT/case["reference"]/(k+".png") for k in ["red", "robust_seed", "edit_allowed"]]:
            dependencies[str(p.relative_to(ROOT))] = sha(p)
    manifest = dict(source_sha256=sha(source), settings=SETTINGS, rois=ROIS,
                    cases=cases, dependencies=dependencies, stage="Protocol before method output")
    dump(OUT/"manifest.json", manifest)
    shutil.copyfile(source, OUT/"source_snapshot.py")
    dump(OUT/"numerical_checks.json", numerical_checks())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, calibrations, cards = [], {}, []
    palette = dict(ink_required=(40, 170, 40), ambiguous=(0, 180, 245),
                   low_ink=(240, 120, 50), unexplained=(180, 60, 180))
    for page, case in cases.items():
        start = time.monotonic()
        original = read(ROOT/case["original"])
        assert np.array_equal(original, read(V6/page/"original.png"))
        median = cv2.medianBlur(original, 3)
        red = read(ROOT/case["reference"]/"red.png", True)
        robust = read(ROOT/case["reference"]/"robust_seed.png", True)
        anchor_mask = red & robust
        for p, _, box in ROIS:
            if p == page:
                x, y, w, h = box; pad = SETTINGS["anchor_roi_exclusion_margin"]
                anchor_mask[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad] = False
        anchor_mask[:40] = False; anchor_mask[-40:] = False
        anchor_mask[:, :40] = False; anchor_mask[:, -40:] = False
        ay, ax = np.nonzero(anchor_mask)
        if len(ax):
            x0, x1 = max(0, ax.min()-40), min(original.shape[1], ax.max()+41)
            y0, y1 = max(0, ay.min()-40), min(original.shape[0], ay.max()+41)
            score, angles = geometry(median[y0:y1, x0:x1])
            points = select_points(score, angles, anchor_mask[y0:y1, x0:x1], (x0, y0),
                                   SETTINGS["anchor_grid"], SETTINGS["max_anchor_profiles"])
        else:
            points = np.zeros((0, 4))
        anchor_rgb = sample_profiles(median, points) if len(points) else np.zeros((0, 65, 3), np.uint8)
        held = ((points[:, 0].astype(int)//64 + 3*(points[:, 1].astype(int)//64)) % 4) == 0
        (OUT/page).mkdir(parents=True, exist_ok=True)
        calibration, qs = dict(anchor_profiles=len(points), excluded_target_rois=True, spaces={}), {}
        for space in SPACES:
            y = transform(anchor_rgb, space); noise = noise_scale(y, space)
            directions, usable, _ = color_direction(y, noise)
            estimate = usable & ~held; confirm = usable & held
            if estimate.sum() < SETTINGS["min_anchor_estimation"] or confirm.sum() < SETTINGS["min_anchor_confirmation"]:
                calibration["spaces"][space] = dict(status="insufficient_automatic_anchors",
                    estimation_profiles=int(estimate.sum()), confirmation_profiles=int(confirm.sum()),
                    pigment_direction_rgb=None)
                np.savez_compressed(OUT/page/("anchors_"+space+".npz"), positions=points,
                                    rgb=anchor_rgb, direction=directions, usable=usable, held_out=held)
                continue
            q = np.median(directions[estimate], axis=0); q /= np.linalg.norm(q); qs[space] = q
            e, s, a, _ = models(y[confirm], noise[confirm], q)
            color_angles = np.degrees(np.arccos(np.clip(directions[confirm] @ q, -1, 1)))
            null = np.asarray([m["ink_width"] == 0 and not m["crease"] for m in e])
            ink = np.asarray([m["ink_width"] > 0 for m in e])
            calibration["spaces"][space] = dict(
                status="estimated",
                pigment_direction_rgb=q.tolist(), estimation_profiles=int(estimate.sum()),
                confirmation_profiles=int(confirm.sum()),
                held_direction_angle_degrees=np.quantile(color_angles, [.1, .5, .9]).tolist(),
                held_median_best_background_cv_mse=float(np.median(s[:, null].min(axis=1))),
                held_median_best_ink_cv_mse=float(np.median(s[:, ink].min(axis=1))),
                held_ink_model_adequate_fraction=float(np.mean(s[:, ink].min(axis=1) <= 4.)))
            np.savez_compressed(OUT/page/("anchors_"+space+".npz"), positions=points,
                                rgb=anchor_rgb, direction=directions, usable=usable,
                                held_out=held, estimated_q=q)
        np.savez_compressed(OUT/page/"anchor_geometry.npz", positions=points, rgb=anchor_rgb, held_out=held)
        dump(OUT/page/"calibration.json", calibration); calibrations[page] = calibration
        old_edit = read(ROOT/case["reference"]/"edit_allowed.png", True)
        v6_edit = read(V6/page/"split_direction_edit_allowed.png", True)
        for p, name, box in ROIS:
            if p != page:
                continue
            x, y, w, h = box; pad = 40
            lx, ly, rx, ry = x-pad, y-pad, x+w+pad, y+h+pad
            assert lx >= 0 and ly >= 0 and rx <= original.shape[1] and ry <= original.shape[0]
            score, angles = geometry(median[ly:ry, lx:rx])
            eligible = np.zeros(score.shape, bool); eligible[pad:pad+h, pad:pad+w] = True
            points_roi = select_points(score, angles, eligible, (lx, ly),
                                       SETTINGS["roi_grid"], SETTINGS["max_roi_profiles"])
            rgb = sample_profiles(median, points_roi)
            all_scores, all_amps, predictions, noises = [], [], {}, {}
            per_space = {}
            for space in SPACES:
                observed = transform(rgb, space); noise = noise_scale(observed, space)
                if space in qs:
                    entries, s, a, prediction = models(observed, noise, qs[space])
                else:
                    entries = model_entries()
                    s = np.full((len(rgb), len(entries)), 5.)
                    a = np.zeros_like(s); prediction = None
                all_scores.append(s); all_amps.append(a); predictions[space] = prediction; noises[space] = noise
                null = np.asarray([m["ink_width"] == 0 and not m["crease"] for m in entries])
                ink = np.asarray([m["ink_width"] > 0 for m in entries])
                per_space[space] = dict(status="estimated" if space in qs else "missing_pigment_direction",
                    median_best_background_cv_mse=float(np.median(s[:, null].min(axis=1))) if space in qs else None,
                    median_best_ink_cv_mse=float(np.median(s[:, ink].min(axis=1))) if space in qs else None)
            s = np.stack(all_scores, 1); a = np.stack(all_amps, 1)
            assert np.isfinite(s).all() and np.isfinite(a).all() and (a >= 0).all()
            px, py = points_roi[:, 0].astype(int), points_roi[:, 1].astype(int)
            old = old_edit[py, px]; current = v6_edit[py, px]
            destination = OUT/page/name; destination.mkdir(parents=True, exist_ok=True)
            write(destination/"original.png", original[y:y+h, x:x+w])
            row = dict(page=page, roi=name, xywh=box, profiles=len(points_roi),
                       sampled_v2_edit=int(old.sum()), sampled_v6_edit=int(current.sum()),
                       spaces=per_space, scenarios={}, pixel_GT=False)
            sampled = original[y:y+h, x:x+w].copy()
            for point in points_roi:
                cx, cy = int(point[0]-x), int(point[1]-y)
                cv2.circle(sampled, (cx, cy), 2, (255, 120, 0), -1)
            write(destination/"sampled_locations.png", sampled)
            stored = dict(points=points_roi, rgb=rgb, cv_scores=s, amplitude_snr=a,
                          sampled_v2_edit=old, sampled_v6_edit=current)
            for scenario, stress in [(SETTINGS["scenarios"][0], False), (SETTINGS["scenarios"][1], True)]:
                label, lo, hi, admissible = classify(s, a, entries, stress)
                row["scenarios"][scenario] = dict(all=counts(label), v2_edit_locations=counts(label, old),
                                                 v6_edit_locations=counts(label, current))
                stored[scenario+"_labels"] = label; stored[scenario+"_low"] = lo; stored[scenario+"_high"] = hi
                picture = original[y:y+h, x:x+w].copy()
                for point, key in zip(points_roi, label):
                    cv2.circle(picture, (int(point[0]-x), int(point[1]-y)), 3, palette[key], -1)
                write(destination/(scenario+".png"), picture)
            np.savez_compressed(destination/"profiles.npz", **stored)
            dump(destination/"diagnostics.json", row)
            # Representative fixed rule: strongest sampled ridge editable in v2, otherwise strongest overall.
            representative = int(np.flatnonzero(old)[0]) if old.any() else 0
            best_ink = np.argmin(np.where(ink, s[representative, 1], np.inf))
            best_null = np.argmin(np.where(null, s[representative, 1], np.inf))
            obs = transform(rgb[representative:representative+1], "encoded_od")[0]
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
            for c, ax_plot in enumerate(axes):
                ax_plot.plot(T, obs[:, c], color=["red", "green", "blue"][c], label="observed")
                if predictions["encoded_od"] is not None:
                    ax_plot.plot(T, predictions["encoded_od"][representative, best_null, :, c], "--", color="gray", label="background")
                    ax_plot.plot(T, predictions["encoded_od"][representative, best_ink, :, c], color="black", label="background + ink")
                else:
                    ax_plot.text(.05, .9, "No automatic pigment estimate", transform=ax_plot.transAxes)
                ax_plot.set_title("RGB"[c]); ax_plot.set_xlabel("cross-section position (native px)")
                ax_plot.set_ylabel("encoded optical density"); ax_plot.legend(fontsize=7)
            fig.suptitle(page+" / "+name+" / strongest v2-editable sampled ridge")
            fig.tight_layout(); fig.savefig(destination/"profile_fit.png", dpi=130); plt.close(fig)
            row["representative_profile"] = representative
            rows.append(row)
            rel = str(destination.relative_to(OUT))
            cards.append('<article><h2>'+html.escape(page+" / "+name)+'</h2><p>Sampled locations; no pixel GT.</p><div class="grid">'+
                         "".join('<figure><figcaption>'+html.escape(label)+'</figcaption><a href="'+rel+"/"+file+'"><img src="'+rel+"/"+file+'"></a></figure>'
                                 for label, file in [("Original", "original.png"), ("Sample locations", "sampled_locations.png"),
                                                     ("Restricted background", "restricted_background.png"), ("Colored crease allowed", "colored_crease_allowed.png")])+
                         '</div><a href="'+rel+'/profile_fit.png">Representative cross-section fit</a></article>')
            print(page, name, json.dumps(row["scenarios"], ensure_ascii=False), flush=True)
        print(f"{page}: finished in {time.monotonic()-start:.1f}s", flush=True)
        dump(OUT/"progress.json", dict(status="running", completed_pages=len(calibrations), total_pages=3))
    dump(OUT/"diagnostics.json", rows); dump(OUT/"calibrations.json", calibrations)
    dump(OUT/"model_dictionary.json", entries)
    (OUT/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Ink/background profile feasibility</title><style>body{font-family:sans-serif;margin:24px;background:#faf8f3}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{max-width:100%}article{margin:30px 0}a{color:#245889}</style><h1>Ink/background cross-section feasibility</h1><p>Known development: 3 pages / 5 rectangles. Points are sampled evidence, never a dense mask or accuracy score.</p><p>Green: ink required in evaluated models; amber: ambiguous; blue: low ink; purple: model inadequate. Dependence on background assumptions is part of the result.</p><a href="report.md">Report</a> · <a href="verification.json">Verification</a>'+''.join(cards)+'</html>')
    table = "\n".join("| "+r["roi"]+" | "+str(r["profiles"])+" | "+
                      " / ".join(str(r["scenarios"]["restricted_background"]["all"][k]) for k in palette)+" | "+
                      " / ".join(str(r["scenarios"]["colored_crease_allowed"]["all"][k]) for k in palette)+" |" for r in rows)
    (OUT/"report.md").write_text("# インク・背景分解：断面での成立確認 v0\n\n3ページ・5矩形は既知の開発資料。ニューラル学習・教師注釈・復元・新しい密な編集マスクは実施していない。\n\n各ページの確定朱とrobust seedから自動抽出した断面を使い、対象矩形と40px周囲を除いて色方向を推定した。64pxタイルで推定用と確認用を分ける。各断面の画素値も4分割し、当てはめに使わない値で再構成誤差を計算する。断面位置・方向の決定は原画像全体を利用するため、この誤差は独立データの精度ではない。\n\n色の表現3種類、背景5種類、インク幅6種類と背景のみを比較する。折り目対照では、追加の色付き線状背景も許容する。色付き折り目とインクを同時に分けるモデルは未実装。複数説明の寄与範囲は検討した有限候補の感度診断であり、厳密な上下限や校正済み確率ではない。\n\n| 矩形 | 自動断面数 | 通常背景：必要／曖昧／小／説明不足 | 色付き折り目を許容：同順 |\n|---|---:|---|---|\n"+table+"\n\n表は候補断面の件数。墨・背景が混在する矩形なので、件数をインクRecallや汚れPrecisionと解釈しない。正確な候補座標、色方向、断面値、CV誤差、寄与量はJSON／NPZに保存。\n\nモデルの成立と背景への依存は、出力を目視してvisual_findings.mdに追記する。別冊6ページへの適用は未実施。\n")
    for p, expected in dependencies.items():
        assert sha(ROOT/p) == expected, p
    assert sha(source) == manifest["source_sha256"]
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, val in attrs:
                if key in ["href", "src"] and val:
                    assert (OUT/val).is_file(), val
    dump(OUT/"verification.json", dict(status="pending"))
    Links().feed((OUT/"index.html").read_text())
    for row in rows:
        x, y, w, h = row["xywh"]; d = OUT/row["page"]/row["roi"]
        assert np.array_equal(read(d/"original.png"), read(ROOT/cases[row["page"]]["original"])[y:y+h, x:x+w])
        data = np.load(d/"profiles.npz")
        assert len(data["points"]) == row["profiles"]
        for scenario in SETTINGS["scenarios"]:
            assert sum(row["scenarios"][scenario]["all"].values()) == row["profiles"]
    dump(OUT/"verification.json", dict(status="passed", source_and_dependencies_unchanged=True,
        native_original_crops_verified=True, finite_nonnegative_estimates=True,
        sampled_label_counts_verified=True, numerical_checks=numerical_checks(),
        local_viewer_links_checked=True, browser_rendering_tested=False,
        pages=3, diagnostic_rectangles=5, dense_edit_mask_created=False,
        neural_training=False, pixel_GT=False, independent_accuracy_test=False))
    files = [p for p in OUT.rglob("*") if p.is_file() and "plot_cache" not in p.parts
             and p.name not in ["complete.json", "run.log", "progress.json"]]
    dump(OUT/"complete.json", dict(status="complete", outputs={
        str(p.relative_to(OUT)): sha(p) for p in sorted(files)}))
    dump(OUT/"progress.json", dict(status="verified", completed_pages=3, total_pages=3))
    print("Verified stage 0/1 complete", flush=True)


if __name__ == "__main__":
    main()
