"""Fixed-mask dirt renderer ablation; no stroke repair or mask expansion.

Known development case, frozen geometric mask. This does not establish an
automatic dirt detector or recover writing hidden by dirt.
"""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from build_npj_v6_comparison import ROOT, read, write, sha, dump, overlay
from package_saved_npj_role_separated_review import figure
from probe_saved_npj_role_separated_postprocess import harmonic

OUT = ROOT / 'tmp_work/npj_dirt_clean_reference_20261002'
CONTROL = ROOT / 'tmp_work/npj_dirt_control_20261002'
OLD = ROOT / 'tmp_work/saved_npj_ruled_margin_pilot_v2_20261002'
SETTINGS = dict(mask='Unchanged frozen ruled-margin margin_update mask, not guarded_update',
                scope='Whole 620x620 control; no crop ROI supplied to renderer',
                paper_reference='Outside update, inside frozen margin, outside mandatory keep; Lab L >= max(50, eligible median L - 8), local 65px closing drop <= 4',
                reference_smoothing_sigma=2,
                solver='4-neighbour BGR Laplace; zero flux at non-paper neighbours; minimum 16 clean boundary pairs per 4-connected component',
                comparison='Same mask: all-neighbour harmonic vs clean-paper-only smoothed boundary',
                stroke_repair=False, mask_expansion=False, adopted=False,
                limitations=['Known case used to design mask and thresholds; no independent validation',
                             'No material GT; lower margin might include writing or paper structure',
                             'Inside writing frame remains unchanged; texture is not generated',
                             'Fresh control is current local v2 code with CSC v1 mask, not historical replay'])


def paper_reference(base, update, margin, mandatory):
    light = cv2.cvtColor(base.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[:, :, 0]
    broad = cv2.morphologyEx(light, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65)))
    eligible = margin & ~update & ~mandatory
    bright = eligible & (light >= 50)
    threshold = max(50., float(np.median(light[bright])) - 8) if bright.any() else 100.
    return eligible & (light >= threshold) & (broad - light <= 4), threshold


def fill(base, update, known):
    assert not (update & known).any()
    h, w = update.shape
    weight = cv2.GaussianBlur(known.astype(np.float64), (0, 0), 2)
    smooth = cv2.GaussianBlur(np.where(known[:, :, None], base, 0).astype(np.float64), (0, 0), 2)
    smooth /= np.maximum(weight[:, :, None], 1e-12)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(update.astype(np.uint8), connectivity=4)
    out = base.copy()
    applied = np.zeros_like(update)
    used = np.zeros_like(update)
    notes = []
    for label in range(1, n):
        component = labels == label
        yy, xx = np.nonzero(component)
        count = len(yy)
        ids = np.full(update.shape, -1, np.int32)
        ids[yy, xx] = np.arange(count)
        diagonal = np.zeros(count, np.float64)
        rhs = np.zeros((count, 3), np.float64)
        rows, cols, values = [], [], []
        refs = []
        local_used = np.zeros_like(used)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = yy + dy, xx + dx
            valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            ii = np.flatnonzero(valid)
            oy, ox = ny[valid], nx[valid]
            inside = component[oy, ox]
            active = ii[inside]
            diagonal[active] += 1
            rows.append(active)
            cols.append(ids[oy[inside], ox[inside]])
            values.append(np.full(len(active), -1., np.float64))
            boundary = ~inside & known[oy, ox]
            active = ii[boundary]
            diagonal[active] += 1
            samples = smooth[oy[boundary], ox[boundary]]
            rhs[active] += samples
            refs.append(samples)
            local_used[oy[boundary], ox[boundary]] = True
        pairs = sum(len(a) for a in refs)
        note = dict(component=label, pixels=count, boundary_pairs=pairs, applied=pairs >= 16)
        if pairs < 16:
            notes.append(note)
            continue
        rows.append(np.arange(count)); cols.append(np.arange(count)); values.append(diagonal)
        matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(count, count)).tocsr()
        solved = spsolve(matrix, rhs)
        assert np.isfinite(solved).all()
        residual = float(np.abs(matrix @ solved - rhs).max())
        assert residual < 1e-6
        samples = np.concatenate(refs)
        assert (solved >= samples.min(0) - 1e-6).all() and (solved <= samples.max(0) + 1e-6).all()
        out[component] = np.clip(np.rint(solved), 0, 255).astype(np.uint8)
        applied |= component
        used |= local_used
        note.update(max_residual=residual, reference_BGR_min=samples.min(0).tolist(), reference_BGR_max=samples.max(0).tolist())
        notes.append(note)
    assert np.array_equal(out[~applied], base[~applied])
    return out, applied, used, notes


def main(freeze):
    assert not (OUT / 'complete.json').exists(), 'Frozen; use a new version'
    if freeze:
        manifest = json.loads((OUT / 'manifest.json').read_text())
        for p, digest in manifest['dependencies'].items():
            assert sha(ROOT / p) == digest, p
        assert (OUT / 'report.md').exists(), 'Inspect images and write assessment first'
        pngs = sorted(OUT.glob('*.png'))
        with zipfile.ZipFile(OUT / 'comparison_pngs.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in pngs:
                z.write(p, p.name)
        with zipfile.ZipFile(OUT / 'comparison_pngs.zip') as z:
            assert z.testzip() is None
            for p in pngs:
                assert z.read(p.name) == p.read_bytes()
        dump(OUT / 'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
        print('Frozen', len(pngs), 'comparison PNGs')
        return
    assert not OUT.exists(), 'Use a new output name'
    OUT.mkdir(parents=True)
    deps = {}
    def dependency(p):
        deps[str(p.relative_to(ROOT))] = sha(p)
        return p
    for p in [Path(__file__), Path(__file__).parent / 'build_npj_v6_comparison.py',
              Path(__file__).parent / 'package_saved_npj_role_separated_review.py',
              Path(__file__).parent / 'probe_saved_npj_role_separated_postprocess.py',
              CONTROL / 'manifest.json', CONTROL / 'inference_verified.json', OLD / 'complete.json']:
        dependency(p)
    original = read(dependency(CONTROL / 'input/hnsd_raw_control.png'))
    historical_input = read(dependency(OLD / 'native/crop_hnsd/saved_input.png'))
    bases = dict(current_control=read(dependency(CONTROL / 'output/hnsd_raw_control/result/hnsd_raw_control.png')),
                 historical_saved=read(dependency(OLD / 'native/crop_hnsd/saved_npj.png')))
    masks = {k: read(dependency(OLD / 'native/crop_hnsd' / (k + '.png')), True)
             for k in ['margin_update', 'margin', 'mandatory']}
    update, margin, mandatory = [masks[k] for k in ['margin_update', 'margin', 'mandatory']]
    assert not (update & ~margin).any() and not (update & mandatory).any()
    records = []
    for fid, base in bases.items():
        naive, _ = harmonic(base, update)
        known, threshold = paper_reference(base, update, margin, mandatory)
        cleaned, applied, used, notes = fill(base, update, known)
        # Renderer isolation: untrusted inner/boundary values cannot leak into its fill.
        poison = base.copy(); poison[~known] = [255, 0, 255]
        tested, again, used_again, _ = fill(poison, update, known)
        assert np.array_equal(tested[applied], cleaned[applied])
        assert np.array_equal(again, applied) and np.array_equal(used_again, used)
        assert not (used & ~known).any()
        assert np.array_equal(cleaned[~margin | mandatory], base[~margin | mandatory])
        native = OUT / 'native' / fid
        for name, a in dict(original=original, baseline=base, all_boundary=naive, clean_reference=cleaned,
                            update=update, applied=applied, reference=known, used_reference=used).items():
            write(native / (name + '.png'), a)
        roi = np.s_[300:510, 170:520]
        labels = ['原画像', '現在コードの対照' if fid == 'current_control' else '過去の保存npj', '同マスク・全境界参照', '同マスク・紙面だけ参照']
        footer = ['枠外の既存マスクを固定。文字・枠線・朱候補は基準画像のまま。筆画補完なし。',
                  '右列は暗い境界を参照から除外した試作。枠内の黒い塊やマスク外の断片は残る。',
                  '既知の開発例。現在対照はローカルv2コード＋保存CSC v1で、過去結果の再現ではない。']
        figure([original[roi], base[roi], naive[roi], cleaned[roi]], labels, OUT / (fid + '_dark_stain.png'), '枠線に接する暗部：マスク固定で参照だけ比較', footer)
        figure([original, base, naive, cleaned], labels, OUT / (fid + '_overview.png'), '文字を補完せず、枠外の暗部だけ処理', footer, overview=True)
        rendered = overlay(original, [('edit', applied), ('blue', used), ('hold', mandatory)])
        figure([original, rendered], ['原画像', '編集／使用紙面／保持'], OUT / (fid + '_scope.png'), '黄＝編集、青＝実使用紙面、紫＝保持',
               ['基準値が黒い枠線・残留暗部からは補間値を供給しない。',
                '範囲は凍結した既存候補。汚れの材質や余白に文字がないことの正解ではない。'])
        L0 = cv2.cvtColor(base.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[:, :, 0]
        L1 = cv2.cvtColor(cleaned.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[:, :, 0]
        records.append(dict(id=fid, update_pixels=int(update.sum()), applied_pixels=int(applied.sum()),
                            held_pixels=int((update & ~applied).sum()), reference_pixels=int(known.sum()),
                            used_reference_pixels=int(used.sum()), reference_L_threshold=threshold,
                            changed_pixels=int(np.any(cleaned != base, axis=2).sum()),
                            update_dark_L_lt30_before=int((applied & (L0 < 30)).sum()),
                            update_dark_L_lt30_after=int((applied & (L1 < 30)).sum()),
                            solves=notes, outside_update_exact=True, nonmargin_exact=True,
                            mandatory_exact=True, excluded_boundary_poison_invariant=True))
    figure([original, historical_input, bases['historical_saved'], bases['current_control']],
           ['原画像・今回の入力', '過去の保存入力', '過去の保存npj', '現在コードの対照'],
           OUT / 'input_and_baseline.png', '入力の濃淡差と、条件記録付き対照を区別',
           ['今回対照は原画素の620px切り出し、保存CSC v1、現在のローカルv2コード、既存紙背景。',
            '異なる背景・コードの可能性がある過去結果との画素差を、入力濃淡だけの効果とは扱わない。'], overview=True)
    control_reviews = []
    for fid, roi, name in [('crop_hnsd', [30, 50, 160, 260], 'text_rule_control'),
                           ('known_200010454_v2', [1330, 1800, 430, 550], 'red_seal_control'),
                           ('page_200017197', [100, 30, 650, 380], 'paper_pattern_control')]:
        folder = OLD / 'native' / fid
        base = read(dependency(folder / 'saved_npj.png'))
        ref = read(dependency(folder / 'reference.png'))
        u = read(dependency(folder / 'margin_update.png'), True)
        m = read(dependency(folder / 'margin.png'), True)
        keep = read(dependency(folder / 'mandatory.png'), True)
        known, threshold = paper_reference(base, u, m, keep)
        cleaned, applied, _, _ = fill(base, u, known)
        x, y, w, h = roi; s = np.s_[y:y+h, x:x+w]
        assert not u[s].any() and np.array_equal(cleaned[s], base[s])
        figure([ref[s], base[s], cleaned[s]], ['原画像', '保存npj', '汚れ試作後'], OUT / (name + '.png'), '処理で変えないための対照：' + name,
               ['この範囲は追加変更0。文字の修復や朱印の原画像への復元は行わない。',
                'マスクが対象を選ばないことによる保持。汚れ除去成功や検出精度の評価ではない。'])
        control_reviews.append(dict(id=fid, xywh=roi, changed_pixels=0, fixed_mask_noop=True))
    dump(OUT / 'diagnostics.json', dict(cases=records, preservation_controls=control_reviews))
    dump(OUT / 'verification.json', dict(panels_exact=True, outside_update_exact=True,
         nonmargin_exact=True, mandatory_exact=True, boundary_poison_invariant=True,
         solver_finite_bounded_residual_verified=True, masks_unchanged=True, stroke_repair=False))
    dump(OUT / 'manifest.json', dict(settings=SETTINGS, dependencies=deps))
    shutil.copyfile(__file__, OUT / 'source_snapshot.py')
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    main(parser.parse_args().freeze)
