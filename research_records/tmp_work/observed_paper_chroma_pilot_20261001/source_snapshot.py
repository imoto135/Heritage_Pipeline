"""Transfer chromatic paper residuals with frozen observed L donor coordinates."""
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_distributed_paper_guidance as prior
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT = prior.ROOT
OUT = ROOT/'tmp_work/observed_paper_chroma_pilot_20261001'
SETTINGS = dict(mask_reference_applied_L_fixed=True, base='saved distributed sigma32 guide + observed L seam',
    source='exact saved L seam x/y maps; every source pixel inside known paper',
    chroma_signal='known-paper-only normalized Gaussian sigma6 residual, minus global known-residual mean',
    clip_ab=[1.5, 2.0], fade='existing distance feather: first inside 0, full at distance5',
    newly_accepted_only=True, previous_applied_preserved=True, development_ROIs=34,
    damage_and_DDRM_signal=False, random_noise=False, pixel_GT=False)


def observed_chroma(lab, known, applied, sx, sy):
    assert not np.any(known&applied)
    selected = (sx >= 0)&(sy >= 0)
    assert not np.any(selected&~applied)
    assert known[sy[selected], sx[selected]].all()
    mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 6)
    low = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), 6)/np.maximum(mass[:, :, None], 1e-6)
    residual = lab[:, :, 1:]-low[:, :, 1:]
    center = residual[known].mean(axis=0, dtype=np.float64).astype(np.float32) if known.any() else np.zeros(2, np.float32)
    result = np.zeros_like(lab)
    result[selected, 1:] = np.clip(residual[sy[selected], sx[selected]]-center,
        -np.array([1.5, 2.0], np.float32), np.array([1.5, 2.0], np.float32))
    distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
    result *= (np.clip((distance-1)/4., 0, 1)*applied)[:, :, None]
    assert not result[:, :, 0].any()
    return result, center


def main():
    check_bundle(prior.OUT); check_bundle(prior.prior.OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((prior.OUT/'diagnostics.json').read_text()); originals = {}
    for source in prior.prior.frozen.MARGIN.SOURCES + [prior.prior.new.INPUT]:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']: originals[page['page']] = page['original']
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        **{str((folder/name).relative_to(ROOT)): sha(folder/name)
           for folder in [prior.OUT, prior.prior.OUT] for name in ['manifest.json', 'complete.json']},
        **{path: sha(ROOT/path) for path in originals.values()}}
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py'); results = []; poison = []; current = None
    for row in rows:
        rid = row['id']; key = row['page']; native = prior.OUT/'native'/rid; src = prior.prior.OUT/'native'/rid
        if current != key: original = read(ROOT/originals[key]); current = key
        x, y, w, h = row['xywh']; cx, cy, cw, ch = row['context_xywh']
        context = original[cy:cy+ch, cx:cx+cw]; s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
        base = read(native/'guide32.png'); crop = context[s]; value = base.copy()
        extra = read(src/'newly_applied_context.png', True); applied = read(src/'applied8_context.png', True)
        known = read(src/'reference_context.png', True); dest = OUT/'native'/rid
        center = [0., 0.]; maxima = [0., 0.]
        if extra.any():
            sx = np.load(src/'source_x.npy', allow_pickle=False); sy = np.load(src/'source_y.npy', allow_pickle=False)
            lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            delta, center_value = observed_chroma(lab, known, applied, sx, sy)
            field = np.load(native/'guide32/result_field_Lab.npy', allow_pickle=False)
            changed = field+delta
            assert np.array_equal(field[:, :, 0], changed[:, :, 0])
            value = prior.prior.frozen.compose(changed, extra, base, s)
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest/'chroma_delta_Lab.npy', delta, allow_pickle=False)
            np.save(dest/'result_field_Lab.npy', changed, allow_pickle=False)
            center = center_value.tolist(); maxima = np.abs(delta[extra, 1:]).max(axis=0).tolist()
            if extra[s].any():
                poisoned = lab.copy(); poisoned[~known] = (250, 123, -225)
                repeat = observed_chroma(poisoned, known, applied, sx, sy)
                assert np.array_equal(delta, repeat[0]); poison.append(rid)
        assert not np.any(np.any(value != base, axis=2)&~extra[s])
        for name, image in dict(original=crop, base=base, chroma=value, newly_applied=extra[s]).items(): write(dest/(name+'.png'), image)
        figure([crop, base, value], ['原画像', '分布紙色＋明度模様', '同じ紙色・模様＋色模様'],
            OUT/'comparisons'/(rid+'.png'), row['title']+' · 観測紙模様の色成分だけ',
            f'固定の新規補間範囲 {int(extra[s].sum()):,}画素。明度の場は変更0。',
            '紙色・マスク・明度模様・参照座標を固定。色模様も観測紙面から。DDRM再推論なし。')
        results.append(dict(**row, chroma_mean_center=center, maximum_delta_ab=maxima,
            changed_vs_L_only=int(np.any(value != base, axis=2).sum())))
        print('Observed chromatic paper:', len(results), '/34', rid, flush=True)
    for relative, digest in dependencies.items(): assert sha(ROOT/relative) == digest
    dump(OUT/'diagnostics.json', results)
    dump(OUT/'verification.json', dict(status='passed', outside_extra_changes=0, L_field_exactly_preserved=True,
        observed_source_known_only=True, nonreference_poison_cases=poison, ROIs=34, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.json', '.py'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: known chroma with fixed L donors, 34 ROIs', flush=True)


if __name__ == '__main__': main()
