"""Follow-up: nonnegative piecewise-linear ink density, frozen geometry/q."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import probe_ink_background_decomposition as base
import probe_chromatic_ink_decomposition as chroma
import cv2
import numpy as np
from scipy.optimize import nnls

ROOT = base.ROOT
V1 = chroma.OUT
OUT = ROOT/"tmp_work/ink_background_flexible_pilot_v2_20260930"
os.environ["MPLCONFIGDIR"] = str(OUT/"plot_cache")
SETTINGS = dict(
    change="Add nonnegative piecewise-linear ink density instead of a single Gaussian",
    knot_spacing_native_pixels=[4, 8], endpoints="zero density at -32 and +32",
    frozen="v0 pigment and sampled geometry; v1 chromatic data and original hypotheses",
    background="Same 5 v0/v1 models",
    stress="Additional arbitrary-color piecewise profile as a no-ink background explanation",
    sufficient_anchor_rule="Unchanged; missing q remains unexplained",
    acceptable_mse=4., near_best_mse=.25, amplitude_snr_gate=1.,
    stage="Known-development chromatic reconstruction; no dense masks or accuracy",
    neural_training=False, pixel_GT=False)


def basis(step):
    knots = np.arange(-32+step, 32, step)
    return np.maximum(1.-np.abs(base.T[:, None]-knots[None, :])/step, 0.)


def flexible_fit(y, noise, q, bg, step, train, free_color=False):
    b, f = base.background(bg), basis(step)
    bt, ft = b[train], f[train]
    pinv = np.linalg.pinv(bt)
    if free_color:
        design = np.column_stack([bt, ft])
        coef = np.einsum("kt,ntc->nkc", np.linalg.pinv(design), y[:, train])
        full = np.column_stack([b, f])
        return np.zeros(len(y)), np.einsum("tk,nkc->ntc", full, coef)
    fp = ft-bt@(pinv@ft)
    residual = y[:, train]-np.einsum("tk,nkc->ntc", bt, np.einsum("kt,ntc->nkc", pinv, y[:, train]))
    density = np.zeros((len(y), f.shape[1]))
    for i in range(len(y)):
        design = (fp[:, None, :]*q[None, :, None]/noise[i][None, :, None]).reshape(-1, fp.shape[1])
        target = (residual[i]/noise[i]).ravel()
        density[i] = nnls(design, target, maxiter=10*f.shape[1])[0]
    profile = density@f.T
    rest = y[:, train]-profile[:, train, None]*q[None, None, :]
    coef = np.einsum("kt,ntc->nkc", pinv, rest)
    prediction = np.einsum("tk,nkc->ntc", b, coef)+profile[:, :, None]*q[None, None, :]
    snr = profile.max(axis=1)*np.sqrt(np.mean((q[None, :]/noise)**2, axis=1))
    return snr, prediction


EXTRAS = [dict(background=bg, ink_width=-step, crease=False, flexible_step=step, free_color=False)
          for bg in base.CORE_BG for step in [4, 8]] + [
    dict(background="quadratic", ink_width=0., crease=True, flexible_step=step, free_color=True)
    for step in [4, 8]]


def evaluate(y, noise, q):
    scores, amplitudes = [], []
    for m in EXTRAS:
        se = np.zeros(len(y))
        for fold in range(4):
            test = np.arange(65)%4 == fold
            _, prediction = flexible_fit(y, noise, q, m["background"], m["flexible_step"], ~test, m["free_color"])
            se += np.sum(((prediction[:, test]-y[:, test])/noise[:, None, :])**2, axis=(1, 2))
        a, _ = flexible_fit(y, noise, q, m["background"], m["flexible_step"], np.ones(65, bool), m["free_color"])
        scores.append(se/(65*2)); amplitudes.append(a)
    return np.stack(scores, 1), np.stack(amplitudes, 1)


def numerical_checks():
    q = np.array([.6, .8])
    density = np.zeros(basis(4).shape[1]); density[[4, 7, 8]] = [.12, .25, .1]
    profile = basis(4)@density
    y = np.array([.2, .4])[None, None, :]+.03*base.U[None, :, None]+profile[None, :, None]*q[None, None, :]
    a, p = flexible_fit(y, np.full((1, 2), .005), q, "linear", 4, np.ones(65, bool))
    assert np.max(np.abs(p-y)) < 1e-10 and a[0] > 1.
    _, alternate = flexible_fit(y, np.full((1, 2), .005), q, "linear", 4, np.ones(65, bool), True)
    assert np.max(np.abs(alternate-y)) < 1e-10
    test = np.arange(65)%4 == 0
    altered = y.copy(); altered[:, test] += .25
    before = flexible_fit(y, np.full((1, 2), .005), q, "linear", 4, ~test)
    after = flexible_fit(altered, np.full((1, 2), .005), q, "linear", 4, ~test)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    return dict(non_Gaussian_positive_density_recovered=True,
                arbitrary_color_background_can_explain_same_signal=True,
                withheld_intensities_excluded=True)


def main():
    if (OUT/"complete.json").exists():
        raise RuntimeError("Completed pilot is frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    old_rows = json.loads((V1/"diagnostics.json").read_text())
    old_cal = json.loads((base.OUT/"calibrations.json").read_text())
    dependencies = {str(p.relative_to(ROOT)): base.sha(p) for p in [
        Path(base.__file__), Path(chroma.__file__), V1/"manifest.json", V1/"complete.json",
        V1/"diagnostics.json", base.OUT/"calibrations.json"]}
    for r in old_rows:
        for name in ["profiles.npz", "original.png"]:
            p = V1/r["page"]/r["roi"]/name
            dependencies[str(p.relative_to(ROOT))] = base.sha(p)
    base.dump(OUT/"manifest.json", dict(settings=SETTINGS, source_sha256=base.sha(Path(__file__)),
                                       dependencies=dependencies, protocol_before_outputs=True))
    shutil.copyfile(__file__, OUT/"source_snapshot.py")
    base.dump(OUT/"numerical_checks.json", numerical_checks())
    entries = base.model_entries()+EXTRAS
    rows = []
    palette = dict(ink_required=(40, 170, 40), ambiguous=(0, 180, 245),
                   low_ink=(240, 120, 50), unexplained=(180, 60, 180))
    for old in old_rows:
        page, name = old["page"], old["roi"]
        data = np.load(V1/page/name/"profiles.npz")
        ss, aa = [], []
        for si, space in enumerate(base.SPACES):
            cal = old_cal[page]["spaces"][space]
            if cal["status"] != "estimated":
                s = np.full((len(data["rgb"]), len(EXTRAS)), 5.); a = np.zeros_like(s)
            else:
                q3 = np.asarray(cal["pigment_direction_rgb"]); q = q3[1:]-q3[0]
                observed = chroma.chromatic(base.transform(data["rgb"], space))
                noise = base.noise_scale(observed, space)
                s, a = evaluate(observed, noise, q)
            ss.append(np.column_stack([data["cv_scores"][:, si], s]))
            aa.append(np.column_stack([data["amplitude_snr"][:, si], a]))
        scores, amplitudes = np.stack(ss, 1), np.stack(aa, 1)
        assert np.isfinite(scores).all() and np.isfinite(amplitudes).all() and (amplitudes >= 0).all()
        # Existing colored creases plus the flexible colored-background stress are admitted together.
        row = dict(page=page, roi=name, xywh=old["xywh"], profiles=old["profiles"],
                   original_v1_scenarios=old["scenarios"], scenarios={})
        stored = dict(points=data["points"], rgb=data["rgb"], cv_scores=scores, amplitude_snr=amplitudes)
        original = base.read(V1/page/name/"original.png")
        destination = OUT/page/name; destination.mkdir(parents=True, exist_ok=True)
        base.write(destination/"original.png", original)
        x, y, _, _ = old["xywh"]
        for scenario, stress in [("restricted_background", False), ("flexible_colored_background", True)]:
            label, lo, hi, _ = base.classify(scores, amplitudes, entries, stress)
            row["scenarios"][scenario] = dict(all=base.counts(label))
            stored[scenario+"_labels"] = label; stored[scenario+"_low"] = lo; stored[scenario+"_high"] = hi
            picture = original.copy()
            for point, key in zip(data["points"], label):
                cv2.circle(picture, (int(point[0]-x), int(point[1]-y)), 3, palette[key], -1)
            base.write(destination/(scenario+".png"), picture)
        np.savez_compressed(destination/"profiles.npz", **stored)
        base.dump(destination/"diagnostics.json", row); rows.append(row)
        print(page, name, json.dumps(row["scenarios"], ensure_ascii=False), flush=True)
    base.dump(OUT/"diagnostics.json", rows); base.dump(OUT/"model_dictionary.json", entries)
    table = "\n".join("| "+r["roi"]+" | "+str(r["profiles"])+" | "+
                      " / ".join(str(r["scenarios"]["restricted_background"]["all"][k]) for k in palette)+" | "+
                      " / ".join(str(r["scenarios"]["flexible_colored_background"]["all"][k]) for k in palette)+" |" for r in rows)
    (OUT/"report.md").write_text("# 非負のインク量曲線による成立確認 v2\n\nv0/v1と同じ位置・色方向・色差を使用。一本のGaussianに加えて、4px／8px間隔の非負な区分線形インク量を背景と同時に推定した。雑音・受理条件は変更していない。\n\n背景の自由度への感度として、同じ細かさで任意の色の曲線を背景だけとして説明する仮説も追加した。両者の再構成が同じになる可能性を数値確認済み。有限候補の寄与範囲であり、厳密な上下限や確率ではない。\n\n| 矩形 | 断面数 | 通常背景：必要／曖昧／小／説明不足 | 色付きの柔軟な背景も許容：同順 |\n|---|---:|---|---|\n"+table+"\n\n既知開発資料のみ。色差の再構成であり、原画像の完全な分解やGT精度ではない。赤褐色筆記の初期化不足は解消していない。密な編集マスク・学習・手動画素GT・別冊への方法適用は実施していない。\n")
    for r in rows:
        p = OUT/r["page"]/r["roi"]
        new = np.load(p/"profiles.npz"); old = np.load(V1/r["page"]/r["roi"]/"profiles.npz")
        assert np.array_equal(new["points"], old["points"]) and np.array_equal(new["rgb"], old["rgb"])
        assert np.array_equal(new["cv_scores"][:, :, :38], old["cv_scores"])
        assert np.array_equal(base.read(p/"original.png"), base.read(V1/r["page"]/r["roi"]/"original.png"))
        for scenario in r["scenarios"]:
            assert sum(r["scenarios"][scenario]["all"].values()) == r["profiles"]
    for p, h in dependencies.items():
        assert base.sha(ROOT/p) == h, p
    manifest = json.loads((OUT/"manifest.json").read_text())
    assert base.sha(Path(__file__)) == manifest["source_sha256"]
    base.dump(OUT/"verification.json", dict(status="passed", pages=3, diagnostic_rectangles=5,
        original_geometry_pigment_unchanged=True, existing_v1_scores_exact=True,
        finite_nonnegative_estimates=True, numerical_checks=numerical_checks(),
        dense_edit_mask_created=False, pixel_GT=False, neural_training=False, independent_accuracy_test=False))
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name not in ["complete.json", "run.log"]]
    base.dump(OUT/"complete.json", dict(status="complete", outputs={
        str(p.relative_to(OUT)): base.sha(p) for p in sorted(files)}))
    print("Verified flexible pigment profile pilot complete", flush=True)


if __name__ == "__main__":
    main()
