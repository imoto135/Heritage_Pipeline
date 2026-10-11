"""Stage 0/1 follow-up: eliminate a shared intensity field before pigment fit.

Geometry and pigment initialization are frozen from profile v0. Only chromatic
profile reconstruction is tested; no full RGB reconstruction or dense edit mask.
"""
from __future__ import annotations
import hashlib
import html
import json
import os
from pathlib import Path
import shutil

import probe_ink_background_decomposition as base
import cv2
import numpy as np
ROOT = base.ROOT
V0 = base.OUT
OUT = ROOT / "tmp_work/ink_background_chromatic_pilot_v1_20260930"
os.environ["MPLCONFIGDIR"] = str(OUT / "plot_cache")
SETTINGS = dict(
    experiment="Change only profile reconstruction from three channels to two color differences",
    observations="d1=z_G-z_R, d2=z_B-z_R; shared scalar variation cancels in log spaces",
    rgb_additive_note="RGB channel differences are a comparator; do not have log multiplicative invariance",
    geometry="Frozen v0 profile centers, orientations, median RGB samples",
    pigment="Frozen page-specific v0 q, projected as [q_G-q_R, q_B-q_R]; not re-estimated",
    background_models=base.CORE_BG, crease_stress=base.CREASE_BG, ink_widths=base.WIDTHS,
    acceptance=dict(acceptable_mse=4., near_best_mse=.25, amplitude_snr_gate=1.),
    noise="Same flank MAD rule and floors, on color differences; diagonal approximation",
    limitation="Two differences share R noise. This is a sensitivity diagnostic, not calibrated probability.",
    target="Chromatic reconstruction only; achromatic texture and black ink are not resolved",
    insufficient_anchor_policy="Preserve missing initialization as unexplained",
    dense_edit_mask=False, pixel_GT=False, neural_training=False, independent_accuracy_test=False)


def chromatic(y):
    return y[:, :, 1:] - y[:, :, :1]


def evaluate(y, noise, q):
    scores, amps, predictions = [], [], []
    entries = base.model_entries()
    for m in entries:
        se = np.zeros(len(y))
        for fold in range(4):
            test = np.arange(len(base.T)) % 4 == fold
            _, prediction = base.fit(y, noise, q, m["background"], m["ink_width"], ~test)
            se += np.sum(((prediction[:, test]-y[:, test])/noise[:, None, :])**2, axis=(1, 2))
        amp, prediction = base.fit(y, noise, q, m["background"], m["ink_width"], np.ones(len(base.T), bool))
        scores.append(se/(len(base.T)*y.shape[2]))
        amps.append(amp*np.sqrt(np.mean((q[None, :]/noise)**2, axis=1)))
        predictions.append(prediction)
    return np.stack(scores, 1), np.stack(amps, 1), np.stack(predictions, 1)


def numerical_checks():
    q3 = np.array([.1, .5, .86]); q3 /= np.linalg.norm(q3)
    q = q3[1:]-q3[0]
    paper = np.array([.2, .4, .6])[None, None, :] + .03*base.U[None, :, None]
    observed = paper + .35*np.exp(-.5*(base.T/6.)**2)[None, :, None]*q3[None, None, :]
    texture = .09*np.sin(base.T/1.3)[None, :, None]
    y = chromatic(observed); y_texture = chromatic(observed+texture)
    assert np.max(np.abs(y-y_texture)) < 1e-14
    amp, pred = base.fit(y_texture, np.full((1, 2), .005), q, "linear", 6., np.ones(len(base.T), bool))
    assert abs(amp[0]-.35) < 1e-10 and np.max(np.abs(pred-y)) < 1e-10
    s, a, _ = evaluate(y_texture, np.full((1, 2), .005), q)
    repeated = (np.repeat(s[:, None], 3, 1), np.repeat(a[:, None], 3, 1))
    assert base.classify(*repeated, base.model_entries(), False)[0][0] == "ink_required"
    assert base.classify(*repeated, base.model_entries(), True)[0][0] == "ambiguous"
    return dict(shared_scalar_texture_cancels=True, known_pigment_recovered_after_texture=True,
                colored_crease_remains_ambiguous=True)


def main():
    if (OUT/"complete.json").exists():
        raise RuntimeError("Completed pilot is frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    old_rows = json.loads((V0/"diagnostics.json").read_text())
    old_cal = json.loads((V0/"calibrations.json").read_text())
    dependencies = {str(p.relative_to(ROOT)): base.sha(p) for p in [
        Path(base.__file__), V0/"manifest.json", V0/"complete.json",
        V0/"diagnostics.json", V0/"calibrations.json"]}
    for r in old_rows:
        for name in ["profiles.npz", "original.png"]:
            p = V0/r["page"]/r["roi"]/name
            dependencies[str(p.relative_to(ROOT))] = base.sha(p)
    for page in old_cal:
        for space in base.SPACES:
            p = V0/page/("anchors_"+space+".npz")
            dependencies[str(p.relative_to(ROOT))] = base.sha(p)
    manifest = dict(settings=SETTINGS, source_sha256=base.sha(Path(__file__)), dependencies=dependencies,
                    protocol_before_method_output=True)
    base.dump(OUT/"manifest.json", manifest)
    shutil.copyfile(__file__, OUT/"source_snapshot.py")
    base.dump(OUT/"numerical_checks.json", numerical_checks())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    entries = base.model_entries()
    null = np.array([e["ink_width"] == 0 and not e["crease"] for e in entries])
    ink = np.array([e["ink_width"] > 0 for e in entries])
    calibrations, rows, cards = {}, [], []
    for page, old_page in old_cal.items():
        page_result = {}
        for space in base.SPACES:
            cal = old_page["spaces"][space]
            if cal["status"] != "estimated":
                page_result[space] = dict(status="insufficient_automatic_anchors")
                continue
            data = np.load(V0/page/("anchors_"+space+".npz"))
            selected = data["usable"] & data["held_out"]
            observed = chromatic(base.transform(data["rgb"][selected], space))
            noise = base.noise_scale(observed, space)
            q3 = np.asarray(cal["pigment_direction_rgb"]); q = q3[1:]-q3[0]
            s, _, _ = evaluate(observed, noise, q)
            page_result[space] = dict(status="estimated", confirmation_profiles=len(observed),
                projected_pigment=q.tolist(),
                median_best_background_cv_mse=float(np.median(s[:, null].min(1))),
                median_best_ink_cv_mse=float(np.median(s[:, ink].min(1))),
                ink_model_adequate_fraction=float(np.mean(s[:, ink].min(1) <= 4.)))
        calibrations[page] = page_result
    palette = dict(ink_required=(40, 170, 40), ambiguous=(0, 180, 245),
                   low_ink=(240, 120, 50), unexplained=(180, 60, 180))
    for old in old_rows:
        page, name = old["page"], old["roi"]
        previous = V0/page/name; destination = OUT/page/name
        destination.mkdir(parents=True, exist_ok=True)
        original = base.read(previous/"original.png")
        base.write(destination/"original.png", original)
        data = np.load(previous/"profiles.npz")
        rgb, points = data["rgb"], data["points"]
        ss, aa, preds, per_space = [], [], {}, {}
        for space in base.SPACES:
            cal = old_cal[page]["spaces"][space]
            if cal["status"] != "estimated":
                s = np.full((len(rgb), len(entries)), 5.); a = np.zeros_like(s)
                preds[space] = None
                per_space[space] = dict(status="missing_pigment_direction")
            else:
                observed = chromatic(base.transform(rgb, space)); noise = base.noise_scale(observed, space)
                q3 = np.asarray(cal["pigment_direction_rgb"]); q = q3[1:]-q3[0]
                s, a, pred = evaluate(observed, noise, q); preds[space] = pred
                per_space[space] = dict(status="estimated",
                    median_best_background_cv_mse=float(np.median(s[:, null].min(1))),
                    median_best_ink_cv_mse=float(np.median(s[:, ink].min(1))))
            ss.append(s); aa.append(a)
        s, a = np.stack(ss, 1), np.stack(aa, 1)
        assert np.isfinite(s).all() and np.isfinite(a).all() and (a >= 0).all()
        row = dict(page=page, roi=name, xywh=old["xywh"], profiles=len(points), spaces=per_space,
                   scenarios={}, original_v0_scenarios=old["scenarios"])
        stored = dict(points=points, rgb=rgb, cv_scores=s, amplitude_snr=a)
        x, y, _, _ = old["xywh"]
        for scenario, stress in [(base.SETTINGS["scenarios"][0], False), (base.SETTINGS["scenarios"][1], True)]:
            label, lo, hi, _ = base.classify(s, a, entries, stress)
            row["scenarios"][scenario] = dict(all=base.counts(label),
                v2_edit_locations=base.counts(label, data["sampled_v2_edit"]),
                v6_edit_locations=base.counts(label, data["sampled_v6_edit"]))
            stored[scenario+"_labels"] = label; stored[scenario+"_low"] = lo; stored[scenario+"_high"] = hi
            picture = original.copy()
            for point, key in zip(points, label):
                cv2.circle(picture, (int(point[0]-x), int(point[1]-y)), 3, palette[key], -1)
            base.write(destination/(scenario+".png"), picture)
        np.savez_compressed(destination/"profiles.npz", **stored)
        base.dump(destination/"diagnostics.json", row)
        representative = old["representative_profile"]
        observed = chromatic(base.transform(rgb[representative:representative+1], "encoded_od"))[0]
        best_null = int(np.argmin(np.where(null, s[representative, 1], np.inf)))
        best_ink = int(np.argmin(np.where(ink, s[representative, 1], np.inf)))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for c, ax in enumerate(axes):
            ax.plot(base.T, observed[:, c], color=["green", "blue"][c], label="observed color difference")
            if preds["encoded_od"] is not None:
                ax.plot(base.T, preds["encoded_od"][representative, best_null, :, c], "--", color="gray", label="background")
                ax.plot(base.T, preds["encoded_od"][representative, best_ink, :, c], color="black", label="background + ink")
            else:
                ax.text(.05, .9, "No automatic pigment estimate", transform=ax.transAxes)
            ax.set_title(["OD_G - OD_R", "OD_B - OD_R"][c]); ax.legend(fontsize=7)
            ax.set_xlabel("native-pixel cross-section"); ax.set_ylabel("chromatic difference")
        fig.suptitle(page+" / "+name+" / same representative as v0")
        fig.tight_layout(); fig.savefig(destination/"profile_fit.png", dpi=130); plt.close(fig)
        rows.append(row)
        rel = str(destination.relative_to(OUT))
        cards.append('<article><h2>'+html.escape(page+" / "+name)+'</h2><div class="grid">'+
                     "".join('<figure><figcaption>'+label+'</figcaption><a href="'+rel+"/"+file+'"><img src="'+rel+"/"+file+'"></a></figure>'
                             for label, file in [("Original", "original.png"), ("Restricted background", "restricted_background.png"),
                                                 ("Colored crease allowed", "colored_crease_allowed.png")])+
                     '</div><a href="'+rel+'/profile_fit.png">Chromatic fit</a></article>')
        print(page, name, json.dumps(row["scenarios"], ensure_ascii=False), flush=True)
    base.dump(OUT/"calibrations.json", calibrations); base.dump(OUT/"diagnostics.json", rows)
    table = "\n".join("| "+r["roi"]+" | "+str(r["profiles"])+" | "+
                      " / ".join(str(r["scenarios"]["restricted_background"]["all"][k]) for k in palette)+" | "+
                      " / ".join(str(r["scenarios"]["colored_crease_allowed"]["all"][k]) for k in palette)+" |" for r in rows)
    (OUT/"report.md").write_text("# 共通の明暗成分を除いた色差の成立確認 v1\n\nv0と同じ3ページ・5矩形・自動断面・色材推定を使用。背景や受理条件を緩めず、再構成の対象をRGB3成分から色差2成分へ変更した。対数色の共通の明暗成分は差で消える。RGB差は比較用で同じ不変性はない。\n\nこれは色差の再構成であり、原RGB・墨・紙の模様を完全に復元した結果ではない。雑音の共分散は近似で、CV値は校正済み確率ではない。色付き折り目は朱と同じ色差を作ることがある。\n\n| 矩形 | 同じ自動断面数 | 通常背景：必要／曖昧／小／説明不足 | 色付き折り目許容：同順 |\n|---|---:|---|---|\n"+table+"\n\n件数はGT精度ではない。赤褐色筆記ページの種不足はそのまま推定不能として残す。新しい密な編集マスク・ニューラル学習・手動画素GT・別冊への方法適用は行っていない。\n")
    base.dump(OUT/"verification.json", dict(status="pending"))
    (OUT/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Chromatic ink/background feasibility</title><style>body{font-family:sans-serif;margin:24px;background:#faf8f3}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{max-width:100%}article{margin:32px 0}a{color:#245889}</style><h1>Chromatic ink/background feasibility</h1><p>Known development, same sampled cross-sections and pigment as v0. Chromatic reconstruction only. Points are never a dense mask.</p><p>Green: ink required in evaluated models; amber: ambiguous; blue: low ink; purple: model inadequate or no pigment estimate.</p><a href="report.md">Report</a> · <a href="verification.json">Verification</a>'+''.join(cards)+'</html>')
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, val in attrs:
                if key in ["href", "src"] and val:
                    assert (OUT/val).is_file(), val
    Links().feed((OUT/"index.html").read_text())
    for r in rows:
        path = OUT/r["page"]/r["roi"]
        assert np.array_equal(base.read(path/"original.png"), base.read(V0/r["page"]/r["roi"]/"original.png"))
        old = np.load(V0/r["page"]/r["roi"]/"profiles.npz")
        new = np.load(path/"profiles.npz")
        assert np.array_equal(new["points"], old["points"]) and np.array_equal(new["rgb"], old["rgb"])
        for scenario in base.SETTINGS["scenarios"]:
            assert sum(r["scenarios"][scenario]["all"].values()) == r["profiles"]
    for path, h in dependencies.items():
        assert base.sha(ROOT/path) == h, path
    assert base.sha(Path(__file__)) == manifest["source_sha256"]
    base.dump(OUT/"verification.json", dict(status="passed", pages=3, diagnostic_rectangles=5,
        source_dependencies_unchanged=True, v0_original_samples_geometry_parity=True,
        pigment_not_reestimated=True, numerical_checks=numerical_checks(),
        finite_nonnegative_estimates=True, local_viewer_links_checked=True,
        browser_rendering_tested=False, dense_edit_mask_created=False, pixel_GT=False,
        neural_training=False, independent_accuracy_test=False))
    files = [p for p in OUT.rglob("*") if p.is_file() and "plot_cache" not in p.parts
             and p.name not in ["complete.json", "run.log"]]
    base.dump(OUT/"complete.json", dict(status="complete", outputs={
        str(p.relative_to(OUT)): base.sha(p) for p in sorted(files)}))
    print("Verified chromatic profile pilot complete", flush=True)


if __name__ == "__main__":
    main()
