"""Training-free 2D warm-ink/background feasibility with CSC-free initialization.

Only known development rectangles. Density and finite-model sensitivity are
diagnostics, not ink GT, certified bounds, or edit permission.
"""
from __future__ import annotations
import html
import json
import os
from pathlib import Path
import shutil
import time
import probe_ink_background_decomposition as base
import cv2
import numpy as np

ROOT = base.ROOT
OUT = ROOT/"tmp_work/ink_background_spatial_pilot_v3_20260930"
os.environ["MPLCONFIGDIR"] = str(OUT/"plot_cache")
cv2.setNumThreads(1)
SETTINGS = dict(
    analysis_downsample=2, crop_padding_native=48, median_native=3,
    observation="encoded OD_G-OD_R and OD_B-OD_R; approximate chromatic model",
    initialization="Two-sided warm contrasts on coherent ridges; original RGB only, no CSC masks",
    initialization_roi_exclusion_native=64, initialization_page_border_fraction=.05,
    initialization_grid_analysis=16, max_initialization_points=1200,
    initialization_min_ridge=.002, min_both_flanks=.003, min_mean_contrast=.008,
    initialization_split="native 128px tiles; (tile_x+3 tile_y)%4==0 confirmation",
    min_estimation_points=20, min_confirmation_points=8,
    q_quantiles=[.2, .5, .8], geometry_sigmas_analysis=[1., 2., 4.],
    lambda_background=36., lambda_ink=1., sparse_cost=.004,
    background_variants=["smooth", "edge_aware"],
    edge_aware_gradient_scale=.02, minimum_background_edge_weight=.03,
    ink_graph="tangent-weighted neighbor differences; orientation undefined -> isotropic .55",
    sparsity_geometry="cost multiplier 1-.85*ridge confidence; no_geometry uses 1",
    max_iterations=2000, kkt_relative_tolerance=5e-4,
    candidate_density=.015, display_density_max=.2,
    family="3 q directions x 2 backgrounds with geometry; median q without geometry for both backgrounds",
    scope="Local 2D color decomposition only; no neural training, pixel GT or restoration",
    inference="Ranges over tested regularizers, not calibrated uncertainty or certified bounds",
)


def colors(bgr):
    z = -np.log((bgr[:, :, ::-1].astype(np.float64)+1.)/256.)
    return z[:, :, 1:]-z[:, :, :1]


def reduced(bgr):
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, ((w+1)//2, (h+1)//2), interpolation=cv2.INTER_AREA)


def geometry(y):
    strength = np.minimum(y[:, :, 0], y[:, :, 1]).astype(np.float32)
    score = np.zeros(strength.shape, np.float32)
    normal = score.copy(); scale = np.ones_like(score); coherence = score.copy()
    for s in SETTINGS["geometry_sigmas_analysis"]:
        sm = cv2.GaussianBlur(strength, (0, 0), s)
        xx, yy, xy = [cv2.Sobel(sm, cv2.CV_32F, a, b, ksize=3, scale=s*s/4.)
                      for a, b in [(2, 0), (0, 2), (1, 1)]]
        disc = np.sqrt((xx-yy)**2+4*xy**2)
        lo, hi = (xx+yy-disc)/2, (xx+yy+disc)/2
        coh = np.clip(1.-np.abs(hi)/(np.abs(lo)+1e-9), 0., 1.)
        r = np.maximum(-lo, 0.)*coh
        r[(lo >= 0) | (np.abs(hi) > .5*np.abs(lo))] = 0
        angle = .5*np.arctan2(2*xy, xx-yy)+np.pi/2.
        keep = r > score
        score[keep], normal[keep], scale[keep], coherence[keep] = r[keep], angle[keep], s, coh[keep]
    confidence = score/(score+.01)*coherence
    return score, normal, scale, confidence


def initialize(original, page):
    h, w = original.shape[:2]
    small = reduced(cv2.medianBlur(original, 3)); y = colors(small)
    sh, sw = y.shape[:2]; sx, sy = w/sw, h/sh
    score, normal, scale, _ = geometry(y)
    px = (np.arange(sw)+.5)*sx-.5; py = (np.arange(sh)+.5)*sy-.5
    valid = ((px[None, :] >= .05*w) & (px[None, :] < .95*w) &
             (py[:, None] >= .05*h) & (py[:, None] < .95*h))
    for p, _, box in base.ROIS:
        if p == page:
            x0, y0, rw, rh = box; pad = 64
            valid &= ~((px[None, :] >= x0-pad) & (px[None, :] < x0+rw+pad) &
                       (py[:, None] >= y0-pad) & (py[:, None] < y0+rh+pad))
    points = []
    for gy in range(0, sh, 16):
        for gx in range(0, sw, 16):
            block = np.where(valid[gy:gy+16, gx:gx+16], score[gy:gy+16, gx:gx+16], -1.)
            dy, dx = np.unravel_index(int(np.argmax(block)), block.shape)
            iy, ix = gy+dy, gx+dx
            if score[iy, ix] >= .002 and valid[iy, ix]:
                points.append([ix, iy, float(normal[iy, ix]), float(score[iy, ix])])
    points.sort(key=lambda p: (-p[3], p[1], p[0]))
    points = np.asarray(points[:1200], float).reshape(-1, 4)
    if len(points) == 0:
        return dict(status="insufficient_contrasts", estimation_points=0, confirmation_points=0), {}
    ix, iy = points[:, 0].astype(int), points[:, 1].astype(int)
    radius = 3.*scale[iy, ix]
    nx, ny = np.cos(points[:, 2]), np.sin(points[:, 2])
    samples = []
    for sign in [-1, 1]:
        mx = (points[:, 0]+sign*radius*nx).astype(np.float32)[:, None]
        my = (points[:, 1]+sign*radius*ny).astype(np.float32)[:, None]
        samples.append(cv2.remap(y.astype(np.float32), mx, my, cv2.INTER_LINEAR)[:, 0])
    center = y[iy, ix]
    dl, dr = center-samples[0], center-samples[1]
    contrast = (dl+dr)/2.
    usable = ((dl > .003).all(1) & (dr > .003).all(1) &
              (contrast > .008).all(1) & (points[:, 3] >= .002))
    native = np.column_stack([px[ix], py[iy]])
    confirm = ((native[:, 0].astype(int)//128+3*(native[:, 1].astype(int)//128)) % 4) == 0
    estimate = usable & ~confirm; check = usable & confirm
    theta = np.arctan2(contrast[:, 1], contrast[:, 0])
    info = dict(source_ridge_points=len(points), estimation_points=int(estimate.sum()),
                confirmation_points=int(check.sum()), no_CSC_masks=True,
                all_target_ROIs_and_64px_margin_excluded=True)
    arrays = dict(native_points=native, contrast=contrast, usable=usable,
                  confirmation_split=confirm, angle_radians=theta)
    if estimate.sum() < 20 or check.sum() < 8:
        info["status"] = "insufficient_contrasts"
        return info, arrays
    angles = np.quantile(theta[estimate], [.2, .5, .8])
    qs = np.column_stack([np.cos(angles), np.sin(angles)])
    info.update(status="estimated", q_rgb_chromatic=qs.tolist(),
                angle_degrees=np.degrees(angles).tolist(),
                confirmation_absolute_angle_error_degrees=np.quantile(
                    np.abs(np.degrees(theta[check]-angles[1])), [.1, .5, .9]).tolist())
    return info, arrays


def graph_laplacian(x, wx, wy):
    if x.ndim == 3:
        wx, wy = wx[:, :, None], wy[:, :, None]
    dx = (x[:, 1:]-x[:, :-1])*wx
    dy = (x[1:]-x[:-1])*wy
    out = np.zeros_like(x)
    out[:, :-1] -= dx; out[:, 1:] += dx
    out[:-1] -= dy; out[1:] += dy
    return out


def graph_energy(x, wx, wy):
    if x.ndim == 3:
        wx, wy = wx[:, :, None], wy[:, :, None]
    return .5*(np.sum(wx*(x[:, 1:]-x[:, :-1])**2)+np.sum(wy*(x[1:]-x[:-1])**2))


def weights(y, use_geometry, background):
    h, w = y.shape[:2]
    bx = np.ones((h, w-1)); by = np.ones((h-1, w))
    if background == "edge_aware":
        bx = np.maximum(1./(1.+np.sum(np.diff(y, axis=1)**2, axis=2)/.02**2), .03)
        by = np.maximum(1./(1.+np.sum(np.diff(y, axis=0)**2, axis=2)/.02**2), .03)
    ax, ay = np.full_like(bx, .55), np.full_like(by, .55)
    sparse = np.ones((h, w))
    if use_geometry:
        _, normal, _, confidence = geometry(y)
        tx2 = confidence*np.sin(normal)**2+(1.-confidence)*.5
        ty2 = 1.-tx2
        ax = .1+.9*.5*(tx2[:, 1:]+tx2[:, :-1])
        ay = .1+.9*.5*(ty2[1:]+ty2[:-1])
        sparse = 1.-.85*confidence
    return bx, by, ax, ay, sparse


def objective(y, q, b, a, ws):
    bx, by, ax, ay, sparse = ws
    residual = b+a[:, :, None]*q-y
    return float(.5*np.sum(residual**2)+36.*graph_energy(b, bx, by)+
                 graph_energy(a, ax, ay)+.004*np.sum(sparse*a))


def gradient(y, q, b, a, ws):
    bx, by, ax, ay, _ = ws
    r = b+a[:, :, None]*q-y
    return r+36.*graph_laplacian(b, bx, by), np.sum(r*q, axis=2)+graph_laplacian(a, ax, ay)


def solve(y, q, ws, max_iterations=None):
    limit = SETTINGS["max_iterations"] if max_iterations is None else max_iterations
    b = cv2.GaussianBlur(y, (0, 0), 4.)
    a = np.maximum(np.sum((y-b)*q, axis=2)-.004*ws[-1], 0.)
    zb, za = b.copy(), a.copy(); momentum = 1.
    step = 1./(2.+8.*36.)
    old = objective(y, q, b, a, ws); history = [dict(iteration=0, objective=old)]
    restarts = 0; converged = False
    norm = max(float(np.max(np.abs(y))), .01)
    for iteration in range(1, limit+1):
        gb, ga = gradient(y, q, zb, za, ws)
        nb = zb-step*gb; na = np.maximum(za-step*(ga+.004*ws[-1]), 0.)
        current = objective(y, q, nb, na, ws)
        if current > old+1e-10*max(1., old):
            restarts += 1; momentum = 1.
            gb, ga = gradient(y, q, b, a, ws)
            nb = b-step*gb; na = np.maximum(a-step*(ga+.004*ws[-1]), 0.)
            current = objective(y, q, nb, na, ws)
        assert current <= old+1e-8*max(1., old)
        new_momentum = (1.+np.sqrt(1.+4.*momentum**2))/2.
        factor = (momentum-1.)/new_momentum
        zb, za = nb+factor*(nb-b), na+factor*(na-a)
        b, a, old, momentum = nb, na, current, new_momentum
        if iteration % 25 == 0 or iteration == limit:
            gb, ga = gradient(y, q, b, a, ws); ga += .004*ws[-1]
            kkt = max(float(np.max(np.abs(gb))),
                      float(np.max(np.abs(np.where(a > 1e-8, ga, np.minimum(ga, 0.))))))/norm
            history.append(dict(iteration=iteration, objective=current, relative_kkt=kkt))
            if kkt <= SETTINGS["kkt_relative_tolerance"]:
                converged = True
                break
    residual = y-b-a[:, :, None]*q
    return b, a, dict(converged=converged, iterations=iteration, relative_kkt=kkt,
                      residual_rmse=float(np.sqrt(np.mean(residual**2))),
                      objective=current, restarts=restarts, history=history)


def numerical_checks():
    rng = np.random.default_rng(20260930)
    y = rng.normal(.2, .04, (13, 17, 2))
    q = np.array([.6, .8]); b = rng.normal(.1, .02, y.shape); a = rng.uniform(.01, .03, y.shape[:2])
    ws = weights(y, True, "edge_aware")
    db, da = rng.normal(size=b.shape), rng.normal(size=a.shape)
    gb, ga = gradient(y, q, b, a, ws); ga += .004*ws[-1]
    eps = 1e-7
    finite = (objective(y, q, b+eps*db, a+eps*da, ws)-objective(y, q, b-eps*db, a-eps*da, ws))/(2*eps)
    analytic = float(np.sum(gb*db)+np.sum(ga*da))
    assert abs(finite-analytic)/max(1., abs(analytic)) < 1e-6
    yy, xx = np.mgrid[:40, :48]
    curve = .2*np.exp(-.5*((xx-24.-3*np.sin(yy/9.))/2.)**2)
    synthetic = np.stack([.2+.001*xx, .35+.001*yy], 2)+curve[:, :, None]*q
    bg, density, info = solve(synthetic, q, weights(synthetic, True, "smooth"))
    assert info["converged"] and np.isfinite(bg).all() and (density >= 0).all()
    assert np.median(density[curve > .15]) > np.median(density[curve < .001])+.01
    assert all(v["objective"] <= u["objective"]+1e-8 for u, v in zip(info["history"], info["history"][1:]))
    return dict(gradient_matches_finite_difference=True, nonnegative_density=True,
                synthetic_curve_has_larger_density_than_paper=True,
                synthetic_solver_converged=True, objective_monotone=True)


def heatmap(a, size):
    large = cv2.resize(a.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(np.rint(np.clip(large/.2, 0., 1.)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return color, large


def main():
    if (OUT/"complete.json").exists():
        raise RuntimeError("Completed pilot is frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    cases = json.loads((base.V6/"manifest.json").read_text())["cases"]
    cases = {r["id"]: r for r in cases if r["id"] in {p for p, _, _ in base.ROIS}}
    deps = {str(p.relative_to(ROOT)): base.sha(p) for p in
            [Path(base.__file__), base.V6/"manifest.json", base.V6/"candidate_freeze.json"]}
    for case in cases.values():
        for p in [ROOT/case["original"], base.V6/case["id"]/"original.png",
                  base.V6/case["id"]/"split_direction_edit_allowed.png"]:
            deps[str(p.relative_to(ROOT))] = base.sha(p)
    manifest = dict(settings=SETTINGS, source_sha256=base.sha(Path(__file__)), dependencies=deps,
                    rois=base.ROIS, protocol_before_outputs=True)
    base.dump(OUT/"manifest.json", manifest); shutil.copyfile(__file__, OUT/"source_snapshot.py")
    base.dump(OUT/"numerical_checks.json", numerical_checks())
    calibrations, results, cards = {}, [], []
    for page, case in cases.items():
        original = base.read(ROOT/case["original"])
        assert np.array_equal(original, base.read(base.V6/page/"original.png"))
        info, samples = initialize(original, page)
        calibrations[page] = info
        base.dump(OUT/page/"initialization.json", info)
        np.savez_compressed(OUT/page/"initialization_samples.npz", **samples)
        print(page, "initialization", json.dumps(info), flush=True)
        if info["status"] != "estimated":
            continue
        qs = np.asarray(info["q_rgb_chromatic"])
        for p, name, box in base.ROIS:
            if p != page:
                continue
            start = time.monotonic(); x, y0, w, h = box; pad = 48
            region = original[y0-pad:y0+h+pad, x-pad:x+w+pad]
            assert region.shape[:2] == (h+2*pad, w+2*pad)
            small = reduced(cv2.medianBlur(region, 3)); obs = colors(small)
            destination = OUT/page/name; destination.mkdir(parents=True, exist_ok=True)
            base.write(destination/"original.png", original[y0:y0+h, x:x+w])
            densities, fit_info = {}, {}
            for bg in ["smooth", "edge_aware"]:
                for use_geometry in [True, False]:
                    ws = weights(obs, use_geometry, bg)
                    for qi in ([0, 1, 2] if use_geometry else [1]):
                        label = bg+("_geometry_q" if use_geometry else "_no_geometry_q")+str(qi)
                        b, a, diagnostic = solve(obs, qs[qi], ws)
                        assert np.isfinite(a).all() and np.isfinite(b).all() and (a >= 0).all()
                        np.savez_compressed(destination/(label+".npz"), background=b.astype(np.float32),
                                            density=a.astype(np.float32), observation=obs.astype(np.float32), q=qs[qi])
                        display, native = heatmap(a, (w+2*pad, h+2*pad))
                        base.write(destination/(label+".png"), display[pad:pad+h, pad:pad+w])
                        densities[label] = native[pad:pad+h, pad:pad+w]
                        fit_info[label] = diagnostic
                        print(page, name, label, "iterations", diagnostic["iterations"], "kkt",
                              round(diagnostic["relative_kkt"], 7), "converged", diagnostic["converged"], flush=True)
            family_names = [key for key in densities if "_geometry_" in key]
            family = np.stack([densities[k] for k in family_names])
            amin, amax = family.min(0), family.max(0)
            stable = amin >= .015; sensitive = (amax >= .015) & ~stable
            crop = original[y0:y0+h, x:x+w]
            agreement = crop.copy()
            agreement[stable] = np.rint(.6*crop[stable]+.4*np.array([40, 180, 40])).astype(np.uint8)
            agreement[sensitive] = np.rint(.6*crop[sensitive]+.4*np.array([0, 170, 245])).astype(np.uint8)
            base.write(destination/"model_sensitivity.png", agreement)
            np.savez_compressed(destination/"native_diagnostics.npz", minimum_density=amin,
                                maximum_density=amax, all_models_above_gate=stable, some_models_above_gate=sensitive)
            old_edit = base.read(base.V6/page/"split_direction_edit_allowed.png", True)[y0:y0+h, x:x+w]
            row = dict(page=page, roi=name, xywh=box, fits=fit_info,
                       family_all_converged=all(fit_info[k]["converged"] for k in family_names),
                       all_models_above_gate_pixels=int(stable.sum()), some_models_above_gate_pixels=int(sensitive.sum()),
                       v6_editable_pixels=int(old_edit.sum()), stable_on_v6_edit_pixels=int((stable&old_edit).sum()),
                       analysis_shape=list(obs.shape[:2]), display_resized_to_native=True, pixel_GT=False,
                       note="Counts over interpolated diagnostic maps; never ink recall or edit permission")
            base.dump(destination/"diagnostics.json", row); results.append(row)
            rel = str(destination.relative_to(OUT))
            figures = [("原画像", "original.png"), ("形なし・なめらかな背景", "smooth_no_geometry_q1.png"),
                       ("形あり・なめらかな背景", "smooth_geometry_q1.png"),
                       ("形あり・境界を許す背景", "edge_aware_geometry_q1.png"),
                       ("条件による変化", "model_sensitivity.png")]
            cards.append('<article><h2>'+html.escape(page+" / "+name)+'</h2><div class="grid">'+
                         ''.join('<figure><figcaption>'+label+'</figcaption><a href="'+rel+"/"+f+'"><img src="'+rel+"/"+f+'"></a></figure>'
                                 for label, f in figures)+'</div></article>')
            print(page, name, "finished", round(time.monotonic()-start, 1), "seconds", flush=True)
            base.dump(OUT/"progress.json", dict(status="running", completed_rois=len(results), total_rois=5))
    base.dump(OUT/"initializations.json", calibrations); base.dump(OUT/"diagnostics.json", results)
    base.dump(OUT/"verification.json", dict(status="pending"))
    table = "\n".join("| "+r["roi"]+" | "+str(r["all_models_above_gate_pixels"])+" | "+
                      str(r["some_models_above_gate_pixels"])+" | "+str(r["family_all_converged"])+" |" for r in results)
    (OUT/"report.md").write_text("# 2次元のインク・背景分解：初回試作 v3\n\n既知3ページ・5矩形。単一画像の色差を背景と非負の暖色成分に同時分解する。初期化は対象矩形と64px周囲を除いた原画像の両側色差から行い、CSCマスクを使わない。色方向は自動候補内の推定で、インクという意味を保証しない。\n\n解析は1/2解像度で、表示は原寸へ補間。背景の平滑化、非負な色成分、筆画方向に沿う変化と線状応答による疎性コストを組み合わせた凸最適化。形あり／形なし、なめらかな背景／境界を許す背景、3つの色方向を比較する。損失の値を異なる背景モデル間の優劣として使わない。\n\n| 矩形 | 全6条件で寄与あり | 一部条件で寄与あり | 全6条件が収束 |\n|---|---:|---:|---|\n"+table+"\n\n件数は補間後の診断画素数で、GT精度ではない。緑は検討した全条件で寄与あり、黄は条件依存。どちらもインク確定や編集許可ではない。収束しない条件がある場合は範囲を暫定とする。元画像を修復せず、既存v6のマスクは変更しない。\n\n[比較ビューア](index.html)。実際の目視と次の判断は別の分析文書に保存する。\n")
    (OUT/"index.html").write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>2次元の色成分分解</title><style>body{font-family:sans-serif;margin:24px;background:#faf8f3}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}figure{margin:0}img{max-width:100%}article{margin:32px 0}p{line-height:1.6}</style><h1>2次元の色成分分解：初回</h1><p>解析は1/2解像度。色成分の寄与を共通スケール0〜0.2で表示（暗紫が0、黄が0.2以上）。緑＝全6条件で寄与あり、橙＝条件依存。分類精度や編集許可ではありません。</p><a href="report.md">条件と結果</a> · <a href="verification.json">検証</a>'+''.join(cards)+'</html>')
    for p, expected in deps.items():
        assert base.sha(ROOT/p) == expected, p
    assert base.sha(Path(__file__)) == manifest["source_sha256"]
    for r in results:
        x, y0, w, h = r["xywh"]
        assert np.array_equal(base.read(OUT/r["page"]/r["roi"]/"original.png"),
                              base.read(ROOT/cases[r["page"]]["original"])[y0:y0+h, x:x+w])
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, val in attrs:
                if key in ["src", "href"] and val:
                    assert (OUT/val).is_file(), val
    Links().feed((OUT/"index.html").read_text())
    verification = dict(status="passed", initialization_without_CSC_masks=True,
        source_and_dependencies_unchanged=True, original_native_crops_unchanged=True,
        pages=len(calibrations), completed_rois=len(results), expected_rois=5,
        all_solvers_converged=all(d["converged"] for r in results for d in r["fits"].values()),
        numerical_checks=numerical_checks(), local_viewer_links_checked=True,
        browser_rendering_tested=False, pixel_GT=False, neural_training=False,
        edit_permission_generated=False, analysis_downsample=2)
    base.dump(OUT/"verification.json", verification)
    files = [p for p in OUT.rglob("*") if p.is_file() and "plot_cache" not in p.parts
             and p.name not in ["complete.json", "run.log", "progress.json"]]
    base.dump(OUT/"complete.json", dict(status="complete", outputs={
        str(p.relative_to(OUT)): base.sha(p) for p in sorted(files)}))
    base.dump(OUT/"progress.json", dict(status="verified", completed_rois=len(results), total_rois=5))
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
