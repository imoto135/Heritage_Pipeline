"""Check separate rim/tone factors and package native comparisons as PNGs."""
import gc
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import probe_paper_rim_lowfrequency_v2 as probe
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_blending_review import boundary_metric, check_bundle
from package_paper_seam_texture_review import verify_texture

ROOT = probe.ROOT
PILOT = probe.OUT
OUT = ROOT / 'tmp_work/paper_rim_lowfrequency_review_20261001'
INITIAL = ROOT / 'tmp_work/paper_rim_lowfrequency_pilot_20261001'
MODES = probe.MODES


def mask_views(original, old, added, guarded, below):
    previous = original.copy(); candidate = original.copy()
    contours, _ = cv2.findContours(old.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(previous, contours, -1, (255, 90, 0), 1)
    cv2.drawContours(candidate, contours, -1, (255, 90, 0), 1)
    for selected, color in [(added, (0, 180, 255)), (guarded, (180, 40, 160)), (below, (120, 120, 120))]:
        candidate[selected] = np.rint(.45*candidate[selected]+.55*np.array(color)).astype(np.uint8)
    return previous, candidate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for directory in [PILOT, probe.prior.OUT, probe.prior.prior.OUT, probe.MASKS]: check_bundle(directory)
    check_bundle(probe.MARGIN.OUT, 'candidate_manifest.json', 'candidate_complete.json')
    data = json.loads((PILOT/'diagnostics.json').read_text())
    rows = data['regions']; lookup = {r['id']: r for r in rows}
    assert len(rows) == len(lookup) == 22
    initial_matches = []
    # The interrupted v1 remains untouched; its generated arrays/images match v2.
    for path in (INITIAL/'native').rglob('*'):
        if path.is_file():
            target = PILOT/'native'/path.relative_to(INITIAL/'native')
            assert target.exists() and sha(path) == sha(target), path
            initial_matches.append(str(path.relative_to(INITIAL)))
    # Constant known paper must induce no nontrivial guidance correction.
    lab = np.full((40, 40, 3), (90, 3, 12), np.float32)
    active = np.zeros((40, 40), bool); active[12:28, 12:28] = True
    lab[active] = 0
    for sigma in [24, 48]:
        correction = probe.tone_guidance(lab, ~active, active, sigma)[0]
        assert np.abs(correction).max() < 1e-4
    pages, books = set(), set(); checks = []; provenance = []; residuals = []
    figure = probe.prior.prior.previous.prior_ddrm.figure
    for source in probe.MARGIN.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key = page['page']; pages.add(key); books.add(page['book'])
            original = read(ROOT/page['original'])
            v6 = {k: read(source/'v6_masks'/key/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra']}
            features, _ = probe.MARGIN.load_features(key, original, v6)
            directory = PILOT/'pages'/key
            old = read(directory/'old_edit.png', True); edit = read(directory/'rim_edit.png', True)
            added = read(directory/'rim_added.png', True); core = read(directory/'core.png', True)
            guard = read(directory/'guard.png', True); domain = read(directory/'domain.png', True)
            halo = read(directory/'halo.png', True)
            guarded = read(directory/'guarded.png', True); below = read(directory/'below_drop.png', True)
            assert np.array_equal(old, read(probe.MASKS/'pages'/key/'completed/edit.png', True))
            assert np.array_equal(core, read(probe.MARGIN.OUT/'candidate_pages'/key/'features/accepted_core.png', True)
                                  | read(probe.MASKS/'pages'/key/'features/completed_core.png', True))
            assert np.array_equal(domain, read(probe.MASKS/'pages'/key/'features/safe_domain.png', True))
            mandatory = read(probe.MASKS/'pages'/key/'features/warm_anchor_guard.png', True)
            mandatory |= read(probe.MASKS/'pages'/key/'features/long_line_guard.png', True) | v6['red'] | v6['extra']
            assert np.array_equal(guard, mandatory)
            assert not np.any(added & (old | guard | ~domain))
            assert (features['light_drop'][added] >= 2).all()
            assert np.array_equal(edit, old | added)
            # All proposals and exclusions partition the bounded outside neighborhood.
            neighborhood = halo & domain & ~old
            assert np.array_equal(added | guarded | below, neighborhood)
            assert not np.any(guarded & below) and not np.any(added & (guarded | below))
            assert np.array_equal(guarded, neighborhood & guard)
            assert np.array_equal(below, neighborhood & ~guard & (features['light_drop'] < 2))
            expected_halo = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
            assert np.array_equal(halo, expected_halo)
            before, proposal = mask_views(original, old, added, guarded, below)
            figure([original, before, proposal], ['原画像', '青輪郭＝前回編集', '黄＝追加・紫＝保護・灰＝色差不足'],
                   OUT/'whole_page_masks'/(key+'.png'), key+' · 全ページの輪郭候補',
                   f'全ページ追加 {int(added.sum()):,}画素。表示はマスク診断。復元出力は固定ROIだけ。',
                   '既に受理した損傷成分の6px近傍だけ。本文・イラストの新しい候補探索はしない。')
            for roi in page['rois']:
                rid = roi['id']; row = lookup[rid]; native = PILOT/'native'/rid
                cx, cy, cw, ch = row['context_xywh']; x, y, w, h = row['xywh']
                cs = (slice(cy, cy+ch), slice(cx, cx+cw)); s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
                context = original[cs]; crop = context[s]
                known = read(native/'reference_context.png', True)
                rknown = read(native/'rim_reference_context.png', True)
                applied = read(native/'old_applied_context.png', True)
                rapplied = read(native/'rim_applied_context.png', True)
                active = read(native/'rim_eligible_context.png', True)
                old_native = probe.prior.prior.OUT/'native'/rid
                for name, prior_name in [('old_edit', 'edit'), ('old_applied', 'applied'), ('reference_context', 'reference_context'), ('old_applied_context', 'applied_context')]:
                    assert sha(native/(name+'.png')) == sha(old_native/(prior_name+'.png'))
                assert np.array_equal(read(native/'rim_edit.png', True), edit[cs][s])
                assert np.array_equal(read(native/'rim_added.png', True), added[cs][s])
                assert np.array_equal(known & ~edit[cs], rknown)
                assert not np.any(rknown & edit[cs])
                assert np.array_equal(active, read(old_native/'eligible_context.png', True) | added[cs])
                base = read(native/'base.png')
                assert sha(native/'base.png') == sha(probe.prior.OUT/'native'/rid/'combined.png')
                assert np.array_equal(read(native/'original.png'), crop)
                lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
                base_field, original_applied = probe.saved_base_field(old_native)
                assert np.array_equal(original_applied, applied)
                distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
                for sigma in [24, 48]:
                    mode = 'tone'+str(sigma)
                    correction = np.load(native/(mode+'_correction_Lab.npy'), allow_pickle=False)
                    guide = np.load(native/(mode+'_guide_Lab.npy'), allow_pickle=False)
                    hguide = np.load(native/(mode+'_harmonic_guide_Lab.npy'), allow_pickle=False)
                    mass = np.load(native/(mode+'_reference_mass.npy'), allow_pickle=False)
                    if applied.any():
                        expected_mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
                        expected_guide = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), sigma) / np.maximum(expected_mass[:, :, None], 1e-6)
                        assert np.array_equal(mass, expected_mass) and np.array_equal(guide, expected_guide)
                        error, used = verify_field(hguide, applied, known, guide)
                        assert np.array_equal(used, read(native/(mode+'_used.png'), True))
                        residuals.append(dict(id=rid, mode=mode, residual=error))
                    weight = np.clip((distance-1)/8., 0, 1) * applied * (mass >= .05)
                    expected = np.clip(guide-hguide, -np.array([3, 1.5, 1.5], np.float32), np.array([3, 1.5, 1.5], np.float32)) * weight[:, :, None]
                    assert np.array_equal(correction, expected)
                    assert not correction[~applied].any()
                    assert not correction[applied & (distance <= 1)].any()
                    value = read(native/(mode+'.png'))
                    converted = cv2.cvtColor((base_field+correction)[s], cv2.COLOR_LAB2BGR)
                    converted = np.rint(np.clip(converted, 0, 1)*255).astype(np.uint8)
                    assert np.array_equal(value[applied[s]], converted[applied[s]])
                    assert not np.any(np.any(value != base, axis=2) & ~applied[s])
                    assert not np.any(np.any(value != crop, axis=2) & ~old[cs][s])
                rim = read(native/'rim.png')
                if added[cs].any():
                    proof = verify_texture(lab, rknown, rapplied, native)
                    provenance.append(dict(id=rid, **proof))
                    fields, recomputed, used, notes = probe.prior.prior.solve_fields(lab, rknown, active)
                    assert np.array_equal(recomputed, rapplied)
                    assert np.array_equal(used, read(native/'rim_used_context.png', True))
                    smooth = probe.prior.prior.reference_lowpass(lab, rknown)
                    for name, target in [('boundary_raw', lab), ('boundary_smooth', smooth)]:
                        error, _ = verify_field(fields[name], rapplied, rknown, target)
                        residuals.append(dict(id=rid, mode='rim_'+name, residual=error))
                    dist = cv2.distanceTransform(rapplied.astype(np.uint8), cv2.DIST_L2, 5)
                    bw = np.exp(-(np.maximum(dist-1, 0)/8.)**2) * rapplied
                    reconstructed = fields['boundary_smooth'] + (fields['boundary_raw']-fields['boundary_smooth']) * bw[:, :, None]
                    reconstructed[:, :, 0] += np.load(native/'texture_seam.npy', allow_pickle=False) * np.clip((dist-1)/4., 0, 1) * rapplied
                    saved = np.load(native/'rim_field_Lab.npy', allow_pickle=False)
                    assert np.array_equal(saved, reconstructed)
                    converted = np.rint(np.clip(cv2.cvtColor(saved[s], cv2.COLOR_LAB2BGR), 0, 1)*255).astype(np.uint8)
                    assert np.array_equal(rim[rapplied[s]], converted[rapplied[s]])
                else:
                    assert np.array_equal(rim, base) and np.array_equal(rapplied, applied)
                assert not np.any(np.any(rim != base, axis=2) & ~rapplied[s])
                assert not np.any(np.any(rim != crop, axis=2) & ~edit[cs][s])
                for mode in ['base']+MODES:
                    value = read(native/(mode+'.png')); count, median = boundary_metric(value, applied[s], rknown[s])
                    stats = row['diagnostics'][mode]
                    assert count == stats['fixed_old_boundary_pairs']
                    recorded = stats['fixed_old_boundary_median_Lab']
                    assert (median is None and recorded is None) or np.isclose(median, recorded, rtol=0, atol=1e-6)
                    L = cv2.cvtColor(value.astype(np.float32)/255, cv2.COLOR_BGR2LAB)[:, :, 0]
                    assert stats['old_applied_L_le_80'] == int(((L <= 80)&applied[s]).sum())
                    assert stats['common_added_L_le_80'] == int(((L <= 80)&added[cs][s]).sum())
                old_scope = applied[s]
                checks.append(dict(id=rid, added=int(added[cs][s].sum()), added_applied=int((added[cs]&rapplied)[s].sum()),
                    added_unfilled=int((added[cs]&~rapplied)[s].sum()),
                    old_mask_newly_applied=int((rapplied&~applied&old[cs])[s].sum()),
                    old_applied_lost=int((applied&~rapplied)[s].sum()),
                    removed_reference_pixels=int((known&edit[cs]).sum()),
                    tone24_boundary_changed_pixels=int(np.any(read(native/'tone24.png') != base, axis=2)[old_scope & (distance[s] <= 1)].sum())))
                before, proposal = mask_views(crop, old[cs][s], added[cs][s], guarded[cs][s], below[cs][s])
                figure([crop, before, proposal], ['原画像', '青輪郭＝前回編集', '黄＝追加・紫＝保護・灰＝色差不足'],
                       OUT/'mask_diagnostic'/(rid+'.png'), row['title']+' · 回収と除外の内訳',
                       f'追加候補 {row["rim_added_pixels"]:,} / 補間 {row["rim_added_applied_pixels"]:,}画素。表示色は診断用。',
                       '色差不足や保護部分は残る。追加候補数を検出精度・成功数として数えない。')
                print('Verified rim/tone:', rid, flush=True)
            del original, features, v6, old, edit, added, core, guard, domain, halo, guarded, below
            gc.collect()
    assert len(pages) == 14 and len(books) == 11
    unchanged = [r['id'] for r in rows if all(r['diagnostics'][m]['changed_vs_base'] == 0 for m in MODES)]
    assert len(unchanged) == 18
    overview = []
    for filename, keys, directory, group, title, notes in [
        ('01_rim_and_paper_color.png', ['new_margin_hole', 'hole_edge_new'], PILOT, 'comparisons', '薄い輪郭と緩やかな紙色を別々に比較',
         ['輪郭回収で薄い縁は減る。紙色だけの効果は小さく、補修跡は残る。',
          '原画像・前回・輪郭のみ・紙色σ24・紙色σ48の順。両要素を混ぜない。']),
        ('02_rim_masks.png', ['new_margin_hole', 'hole_edge_new'], OUT, 'mask_diagnostic', '輪郭の追加候補と残る除外部分',
         ['青＝前回の編集輪郭。黄＝追加、紫＝保護、灰＝色差不足。',
          'マスクは全ページから推定。これは損傷・文字の画素正解ではない。']),
        ('05_controls.png', ['new_frame_text', 'new_header_ink', 'new_mono_text'], PILOT, 'comparisons', '枠・見出し・本文の対照',
         ['追加0・紙色補間0。今回の3条件は前回画像と完全一致。']),
        ('06_stamp_and_paper.png', ['new_brown_stamp', 'brown_false_red_new'], PILOT, 'comparisons', '印影と広い紙色ムラの対照',
         ['印影には新たに110画素の候補、107画素を追加補間。全資料への採用は保留。',
          '広い茶色い紙面は今回も224画素だけ対象。全体のシミは未解決。'])]:
        stack([directory/group/(rid+'.png') for rid in keys], OUT/filename, title, notes)
        overview.append(filename)
    crops = json.loads((ROOT/'tmp_work/known_paper_harmonic_fill_review_20261001/diagnostics.json').read_text())['closeup_xywh']
    for filename, rid in [('03_new_hole_closeup.png', 'new_margin_hole'), ('04_old_hole_closeup.png', 'hole_edge_new')]:
        x, y, w, h = crops[rid]; native = PILOT/'native'/rid
        figure([read(native/(mode+'.png'))[y:y+h, x:x+w] for mode in ['original', 'base']+MODES],
               ['原画像', '前回：境界＋紙模様', '輪郭回収のみ', '紙色のみ・σ24', '紙色のみ・σ48'],
               OUT/filename, rid+' · 前回と同じ切り出し',
               '輪郭列は参照と模様候補も編集マスクに合わせて再計算。紙色列は固定。',
               '同じ範囲を最近傍で拡大。輪郭だけで完全な修復と数えない。')
        overview.append(filename)
    rid = 'new_margin_hole'; native = PILOT/'native'/rid
    historical = probe.prior.prior.previous.prior_ddrm.OUT/'native'/rid
    figure([read(path) for path in [native/'original.png', historical/'npj.png', historical/'v6.png', native/'base.png', native/'rim.png']],
           ['原画像', 'npj → DDRM', 'v6 → DDRM', '前回：紙色＋模様', '今回：輪郭回収'],
           OUT/'07_npj_v6_context.png', '保存済みnpj・固定v6との位置づけ',
           '保存済みDDRMとの比較。今回列は紙色補間・観測模様で、DDRM単独の結果ではない。',
           'マスクと復元方式の両方が異なる。元npj全工程の再現ではない。')
    overview.append('07_npj_v6_context.png')
    native = PILOT/'native/new_brown_stamp'
    selected = read(native/'old_applied.png', True) | read(native/'rim_added.png', True)
    yy, xx = np.nonzero(selected); H, W = selected.shape
    x0, y0 = max(0, int(xx.min())-16), max(0, int(yy.min())-16)
    x1, y1 = min(W, int(xx.max())+17), min(H, int(yy.max())+17)
    before, proposal = mask_views(read(native/'original.png'), read(native/'old_edit.png', True),
        read(native/'rim_added.png', True), np.zeros_like(selected), np.zeros_like(selected))
    figure([im[y0:y1, x0:x1] for im in [read(native/'original.png'), read(native/'base.png'), read(native/'rim.png'), before, proposal]],
           ['原画像', '以前の点の消失', '今回の輪郭回収', '青＝前回の編集', '黄＝今回の追加'],
           OUT/'08_stamp_closeup.png', '印影対照 · 追加編集を別に確認',
           '以前の変化への利用者の許容を、今回の110画素の正解判定には使わない。',
           '今回の追加は点の周辺。文字なしの自動判定は未実証。最近傍拡大。')
    overview.append('08_stamp_closeup.png')
    items = [(OUT/name, 'overview/'+name) for name in overview]
    items.extend((PILOT/'comparisons'/(r['id']+'.png'), 'comparisons/'+r['id']+'.png') for r in rows)
    items.extend((OUT/'mask_diagnostic'/(r['id']+'.png'), 'mask_diagnostic/'+r['id']+'.png') for r in rows)
    items.extend((OUT/'whole_page_masks'/(page+'.png'), 'whole_page_masks/'+page+'.png') for page in sorted(pages))
    archive_path = OUT/'paper_rim_lowfrequency_comparison_images.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(items) == 66 and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    check_lookup = {r['id']: r for r in checks}
    rim_table = '\n'.join(f'| `{rid}` | {lookup[rid]["rim_added_pixels"]:,} | {lookup[rid]["rim_added_applied_pixels"]:,} | {check_lookup[rid]["added_unfilled"]} | {check_lookup[rid]["old_mask_newly_applied"]} |' for rid in ['hole_edge_new', 'new_margin_hole', 'new_brown_stamp'])
    tone_table = '\n'.join('| `'+rid+'` | '+' | '.join('/'.join(f'{v:.3f}' for v in lookup[rid]['tone_correction_max'][m]) for m in ['tone24', 'tone48'])+' |' for rid in ['hole_edge_new', 'new_margin_hole', 'new_brown_stamp'])
    page_table = '\n'.join(f'| [{r["page"]}](whole_page_masks/{r["page"]}.png) | {r["added_pixels"]:,} | {r["guarded_pixels"]:,} | {r["below_drop_pixels"]:,} |' for r in data['pages'] if r['added_pixels'])
    all_table = '\n'.join(f'| [{r["title"]}](../{PILOT.name}/comparisons/{r["id"]}.png) · `{r["id"]}` | {r["rim_added_pixels"]:,} | {r["rim_added_applied_pixels"]:,} | {r["old_applied_pixels"]:,} |' for r in rows)
    report = f'''# 残る輪郭と緩やかな紙色ムラを分けて比較する

2026-10-01。**輪郭回収の条件では、虫食い2例の薄い縁が減った。固定マスクで紙色だけを変える条件の効果は小さく、輪郭や塗り跡は残る。輪郭回収は印影対照にも110画素の候補を追加したため、全資料へ採用せず開発候補として保存する。**

前回の「境界＋観測紙模様」を基準に、同じ14ページ・11冊・22箇所で比較した。輪郭マスクだけを変える条件と、固定マスクの紙色だけを変える2条件（σ24／σ48）を分離し、両方を混ぜた条件は作らなかった。DDRMの追加推論・再学習・乱数模様はない。

## 比較PNG

- [輪郭と紙色の分離比較](01_rim_and_paper_color.png)。原画像、前回、輪郭のみ、紙色σ24、紙色σ48。
- [回収・保護・色差不足のマスク診断](02_rim_masks.png)。黄＝追加、紫＝保護、灰＝色差不足、青線＝前回の編集。
- [新しい穴の拡大](03_new_hole_closeup.png)、[既存の穴の拡大](04_old_hole_closeup.png)。前回と同じ切り出し・最近傍拡大。
- [枠・見出し・本文](05_controls.png)、[印影と広い紙色ムラ](06_stamp_and_paper.png)、[印影の追加箇所の拡大](08_stamp_closeup.png)。
- [保存済みnpj・固定v6との位置づけ](07_npj_v6_context.png)。保存済みDDRM列と最新の紙色補間列ではマスクと復元方式の両方が違う。元npj全工程の再現ではない。
- [全66PNG ZIP](paper_rim_lowfrequency_comparison_images.zip)。概要8枚、全22箇所の復元比較、全22箇所のマスク診断、全14ページのマスク比較。全ページ画像はマスクの診断で、復元出力は固定ROIだけ。

## 輪郭を回収する条件

保存された上余白の受理済み暗色成分と上方の全成分回収コアを種にする。各コアの6px近傍だけで、元の紙色推定との差Lが2以上の画素を候補に追加する。以前はコアの3px近傍・色差4以上だった。本文の枠から離した既存の安全領域、長い横線の保護、朱・暖色接続保護を維持し、元の編集マスクはすべて残す。ROI名や「穴」の意味ラベルで種を作っていない。

これは距離と色差を同時に緩めた**輪郭回収の1条件**であり、どちらの変更だけが効いたかの比較ではない。6px近傍と色差2未満の部分も診断PNGへ残した。

| 全ページの変更 | 追加候補 | 近傍内で保護 | 近傍内で色差2未満 |
| --- | ---: | ---: | ---: |
{page_table}

ほか11ページの全編集マスクは前回と一致した。既存の穴のページでは全ページ4,266画素の候補に対し固定ROIでは1,387画素。ROI外の候補も全ページPNGに示すが、そこでの復元品質は今回評価していない。

追加した候補を紙色補間へ渡し、以前と同じ4近傍の調和補間、直接／平滑境界の混合、観測明度パッチの継ぎ目合成を適用する。参照は以前の紙面参照から新しい編集画素を除くので、補間場と模様候補も変わる。**輪郭列はマスク変更に伴う復元全体の比較で、既存内部の紙色・パッチ配置まで固定した比較ではない。** 新しいマスクでDDRMを推論した結果でもない。参照不足の小成分は前回出力を保持する。

| ROI | 追加候補 | 追加して補間 | 追加候補だが未補間 | 旧編集内で新たに補間 |
| --- | ---: | ---: | ---: | ---: |
{rim_table}

追加候補数を検出精度や損傷の正解数と数えない。既存の穴は1,335画素、新しい穴は2,150画素を追加補間し、薄い茶色い輪郭が減った。塗り跡は残り、新しい穴には保護された192画素と、近傍内でも色差が小さい3,375画素がある。これらがすべて損傷とは確定していないので、残存損傷の正解数とも数えない。

印影対照では以前からの点の消失に加えて110画素が候補となり、107画素を追加補間した。点の周辺への新しい変更であり、以前の変化についての利用者の許容を、今回の追加の正解判定として使わない。大きな印影の形は概観で維持されるが、位置・色・太さだけで文字なしを認定できる状態ではない。この例を比較から除外しない。

## マスクを固定して紙色だけを変える条件

前回の編集マスク、補間範囲、紙面参照、紙模様の場と配置をそのまま使う。参照紙面だけを正規化Gaussianでσ24px／48pxに平滑化し、緩やかな紙色場を作る。その境界値の調和補間を引いた差を、前回の補間場へ足す。これで境界に整合する範囲で紙色の曲がりを試す。

補正は最初の内側画素では0、距離9px以上で全強度とする。参照重み0.05未満では適用せず、Lは±3、a/bは±1.5以内に制限する。元の損傷画素・DDRM模様を補正に取り込まず、紙模様を選び直すこともない。

| ROI | σ24の補正最大絶対値 L/a/b | σ48の補正最大絶対値 L/a/b |
| --- | --- | --- |
{tone_table}

今回の紙色補正は小さく、PNGでの改善も控えめだった。輪郭自体は変わらず、σ24／48のどちらでも平坦な塗り跡と薄い縁が残る。紙色補正を追加採用せず、前回の条件を比較基準に残す。参照から推定した紙色場と近づくことを、失われた紙色の真正な復元や自然さの独立評価と扱わない。

穴2例の旧補間範囲では、全条件でL≤80の画素は0を維持した。固定の追加候補範囲では、前回のL≤80は既存の穴3画素、新しい穴38画素、輪郭条件では両方0。これも明度の観測であり、薄い縁の除去精度ではない。印影の紙面は元から灰色なので同じしきい値を品質評価に使わない。

全18箇所は今回の3条件とも前回と完全一致した。広い上端の茶色い紙面は今回も224画素だけ紙色補間する状態で、大部分のシミは未解決。本文に重なる汚れ、細い未検出の裂け、イラストの問題も今回の成功に含めない。イラストは対象外。

## 全22箇所

| ROI | 輪郭追加候補 | 追加補間 | 固定の旧補間画素 |
| --- | ---: | ---: | ---: |
{all_table}

## 確認と次の段階

全ページの入力・ソース・保存マスク・前回出力のハッシュと、元マスクの包含、保護・領域条件、追加と除外の内訳を確認した。紙色条件では固定マスクと参照が前回と同一で、固定模様を使った合成式と元解像度の出力が一致した。参照だけのGaussian場を別に再計算し、その境界の調和場に4近傍の差を直接足して方程式残差を確認した。定数紙面に紙色補正が生じないことも確認した。

輪郭条件では新しい編集画素が参照パッチに入らないこと、元画素座標と記録された配置パッチの模様値の一致、同じ配置からの重複平均の再構成を確認した。補間場を再計算したうえで残差を別の近傍列挙で確認し、記録済み模様を足した場が保存場と完全一致した。補間不足の画素は消したと数えず、上表へ記録した。[確認JSON](verification.json)。

初回は、紙色のみの条件にも輪郭用の拡張マスクで損傷画素の置換テストをかけており、紙色条件では正当な参照画素まで変えたため検証で停止した。初回のソース・途中出力を残し、各条件のマスクで検証するv2へ修正した。復元式・パラメータは変更せず、初回に保存された全ネイティブファイル{len(initial_matches)}件はv2とハッシュ同一だった。4適用例で、それぞれの編集画素を任意色に変えても補正・復元場は完全一致した。

全22例の編集範囲外は原画素、各条件の適用範囲外は前回画像と一致した。境界対と明度を別に計算し、ZIP66PNGの内容ハッシュも確認した。今回の全例は調整を重ねた開発例で、画素GT・未使用冊の精度はない。

**次の優先は、新しい紙色補正を重ねることより、輪郭回収の条件を未使用冊で確かめること。** 輪郭回収の候補と前回基準を保存し、原画像から文字のない損傷と本文・印影の対照を先に選ぶ。固定した6px・色差2の条件をそのまま適用し、本文や印影の追加編集も同じPNGで確認する。紙色σ24／48は大きな改善を示していないため追加採用しない。位置・参照条件の文字なし判定や、新手法の確立はまだ実証していない。

[固定条件](../{PILOT.name}/manifest.json)、[全診断](../{PILOT.name}/diagnostics.json)、[原寸出力](../{PILOT.name}/native)、[実装v2](../../scripts/benchmark/probe_paper_rim_lowfrequency_v2.py)、[初回ソース](../../scripts/benchmark/probe_paper_rim_lowfrequency.py)、[確認・梱包](../../scripts/benchmark/package_paper_rim_lowfrequency_review.py)、[前回の境界・紙模様](../paper_seam_texture_review_20261001/report.md)。
'''
    (OUT/'report.md').write_text(report)
    dump(OUT/'diagnostics.json', dict(scope=checks, harmonic_residuals=residuals, donor_provenance=provenance,
        closeup_xywh=crops, stamp_closeup_xywh=[x0, y0, x1-x0, y1-y0]))
    dump(OUT/'verification.json', dict(status='passed', pages=14, books=11, regions=22,
        mask_factors_separate=True, old_edit_preserved=True, mandatory_guards_preserved=True,
        tone_masks_references_texture_fixed=True, constant_paper_no_tone_correction=True,
        direct_discrete_residual_max=max(r['residual'] for r in residuals),
        donor_provenance=True, independent_overlap_reconstruction=True,
        initial_native_files_unchanged=len(initial_matches), factor_specific_poison_masks=True,
        outside_each_edit_changed_pixels=0, outside_each_applied_changed_pixels=0,
        unchanged_regions=unchanged, zip_PNG_count=66, zip_hashes=True))
    dependencies = {str((folder/name).relative_to(ROOT)): sha(folder/name)
                    for folder in [PILOT, probe.prior.OUT, probe.prior.prior.OUT, probe.MASKS] for name in ['manifest.json', 'complete.json']}
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=dependencies,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: separate mask/tone factors, references, fields, 66 PNGs', flush=True)


if __name__ == '__main__': main()
