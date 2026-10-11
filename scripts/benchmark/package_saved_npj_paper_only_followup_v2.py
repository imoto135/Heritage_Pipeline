"""PNG-only primary delivery of fixed paper-only saved npj follow-up."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, read, write, sha, dump, overlay
from package_saved_npj_inference_v6_review import check
from package_saved_npj_role_separated_review import figure

PILOT = ROOT/'tmp_work/saved_npj_paper_only_followup_pilot_20261002'
OUT = ROOT/'tmp_work/saved_npj_paper_only_followup_review_v2_20261002'
NOTES = ROOT/'tmp_work/saved_npj_paper_only_followup_notes_v2_20261002.md'
SCOPE = ROOT/'tmp_work/saved_npj_hnsd_input_scope_diagnostic_20261002'


def main(render_only):
    assert not (OUT/'complete.json').exists(), 'Review already frozen'
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = check(PILOT); check(SCOPE); records = []; page_records = []
    for c in manifest['cases']:
        fid = c['id']; src = PILOT/'native'/fid
        a = {k:read(src/(k+'.png')) for k in ['original', 'saved_npj', 'paper_only']}
        roles = {k:read(src/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra', 'paper_update']}
        base = a['saved_npj']; result = a['paper_only']; update = roles['paper_update']
        assert np.array_equal(a['original'], read(ROOT/c['input']))
        assert np.array_equal(base, read(ROOT/c['saved_npj']))
        assert np.array_equal(result[~update], base[~update])
        for k in ['protect', 'hold', 'red', 'extra']:
            assert not (update & roles[k]).any()
            assert np.array_equal(result[roles[k]], base[roles[k]])
        # Reconstruct the frozen mask without relying on a stored pass flag.
        keep = roles['protect'] | roles['hold'] | roles['red'] | roles['extra']
        guard = cv2.dilate(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        expected = roles['edit'] & ~guard
        expected[[0, -1], :] = False; expected[:, [0, -1]] = False
        assert np.array_equal(expected, update)
        yy, xx = np.nonzero(update)
        residual = 4*result[yy, xx].astype(np.int32)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            residual -= result[yy+dy, xx+dx].astype(np.int32)
        maxres = int(np.abs(residual).max()) if len(yy) else 0
        assert maxres <= 4
        changed = np.any(result != base, axis=2)
        L = cv2.cvtColor(base.astype(np.float32)/255., cv2.COLOR_BGR2LAB)[:, :, 0]
        dark_update = update & (L < 40)
        page_records.append(dict(id=fid, updated_pixels=int(update.sum()), changed_pixels=int(changed.sum()),
                                 npj_L_below_40_in_update=int(dark_update.sum()), rounded_harmonic_max_residual=maxres,
                                 protected_exact_saved_npj=True, outside_update_exact_saved_npj=True))
        input_label = '原画像・保存入力一致' if c['reference_input_exact'] else '保存入力（参照と濃淡が異なる）'
        labels = [input_label, '実際の保存npj', '保存npj＋紙面補修のみ']
        figure(list(a.values()), labels, OUT/('overview_'+fid+'.png'), c['page']+' / '+c['baseline_label'],
               ['保存npjと閾値は固定。文字の接続補完は使わず、紙面更新外は保存npjのまま。',
                '保存入力との対応を確認。参照原画像と違う入力では、その保存入力から分類する。',
                '既知冊の追加例と別保存版。GT精度・独立した新冊評価とは扱わない。'], overview=True)
        for roi in c['rois']:
            x, y, w, h = roi['xywh']; s = np.s_[y:y+h, x:x+w]
            counts = {k:int(v[s].sum()) for k, v in roles.items()}
            real_changes = int(changed[s].sum())
            rendering = figure([v[s] for v in a.values()], labels, OUT/(roi['id']+'.png'), roi['title'],
                               [f'紙面更新 {counts["paper_update"]:,}／実変更 {real_changes:,}画素。文字補完0。',
                                '保護・朱・保留は保存npj一致。ROI全体や材質GTの一致とは区別する。',
                                '原画像／入力・版・更新の有無を記録。変化なしを汚れ除去の成功と数えない。'])
            for k, v in a.items(): write(OUT/'native'/roi['id']/(k+'.png'), v[s])
            for k, v in roles.items(): write(OUT/'native'/roi['id']/(k+'.png'), v[s])
            records.append(dict(**roi, frame=fid, page=c['page'], reference_input_exact=c['reference_input_exact'],
                                mask_pixels=counts, changed_pixels=real_changes,
                                exact_saved_npj=bool(np.array_equal(result[s], base[s])), **rendering))
        if not c['reference_input_exact']:
            reference = read(ROOT/c['reference_original'])
            if c['reference_xywh'] is not None:
                x, y, w, h = c['reference_xywh']; reference = reference[y:y+h, x:x+w]
            assert reference.shape == a['original'].shape
            difference = np.abs(reference.astype(np.int16)-a['original'].astype(np.int16))
            assert float(difference.mean()) == c['reference_input_mae']
            assert int(difference.max()) == c['reference_input_max_difference']
            figure([reference, a['original'], base], ['参照原画像', '実推論の保存入力', '実際の保存npj'],
                   OUT/('input_provenance_'+fid+'.png'), c['page']+'：保存入力の濃淡を区別',
                   [f'参照との差：平均絶対差 {difference.mean():.4f}/255、最大 {difference.max()}。',
                    'この比較の分類元と観測元は中央の保存入力。過去の濃淡変換の全条件は不明。',
                    '参照原画像のマスクを一致しない入力へ流用しない。npj出力は再生成しない。'], overview=True)
        if fid == 'crop_hnsd':
            painted = overlay(a['original'], [('blue', roles['protect']), ('hold', roles['hold']), ('edit', update)])
            figure([a['original'], painted, result], ['実推論の保存入力', '分類と紙面更新範囲の表示', '保存npj＋紙面補修のみ'],
                   OUT/'13_hnsd_mask_limitation.png', '濃い塊の失敗：保存入力上の編集0',
                   ['中央は分類表示で復元画像ではない。青＝保護、紫＝保留、黄＝紙面更新。',
                    'この入力では黄の紙面更新が0。保護は文字GTではなく、塊が残ることを明示。',
                    '復元器へ渡る範囲がなければ除去できない。参照原画像と濃淡が異なる条件。'])
    dump(OUT/'diagnostics.json', dict(regions=records, pages=page_records, figure_count=19,
         selection_before_candidate=True, stroke_update_disabled=True, pixel_GT=False))
    assert len(list(OUT.glob('*.png'))) == 19
    if render_only:
        print('Rendered 19 draft PNGs: 12 regions, 4 overviews, 2 provenance pairs, 1 mask limitation.', flush=True)
        return
    assert NOTES.exists()
    report = '''# 保存npj＋紙面補修のみ：固定条件による追加検証

2026-10-02。前回の役割分離試作から紙面補修だけを固定した。文字の短い接続補完は採用せず、元npjの再推論・新DDRM・ACE/CSC再計算もない。実保存出力を基準として、新たな紙面補修を全保存入力範囲へ適用した。

## 入力と版

4保存入力・12ROI。200019865_00003_1と200017197_00008_1は、この後処理の前回2入力には含まれない別ページ。200010454_00002_2は既知ページのCSC v2保存版を朱印対照として使った。hnsd001_011_cropは既知ページの別の保存入力で、参照原画像と濃淡が大きく異なる条件。4枚の独立した未使用冊の評価ではない。すべてイラストを含まないページから選び、暖色の紙模様は汚れGTとしない。

'''
    for c in manifest['cases']:
        report += f'- {c["page"]}：{c["baseline_label"]}。保存入力と参照の一致={c["reference_input_exact"]}、平均絶対差={c["reference_input_mae"]:.4f}/255。\n'
        report += f'  保存出力：`{c["saved_npj"]}`。保存入力：`{c["input"]}`。\n'
    report += '''
保存入力と参照が違う2例は、保存入力を実際の観測元とし、そこから固定v6分類を計算した。今回の比較列も保存入力として明示する。濃淡の異なる原画像の既存マスクを流用せず、原画像と全画素一致した評価へ混ぜない。過去の全変換・実行ソース・全設定の完全再現は未確認だが、実保存結果を固定比較へ利用する。異なる保存版を旧v1の結果として混ぜない。

## 固定した条件

固定foreground・色／形状・保護・v6 split_directionを保存入力全体で適用。元入力と全画素一致した200010454だけ凍結済み全ページマスクを再利用した。今回のROI座標や意味ラベルは分類へ渡さない。ページサイズと保存入力の切り出し範囲は入力条件であり、既存の原ページ全体の分類と同じマスクになる保証はない。

紙面更新はv6 editからprotect／hold／朱／extra周囲2pxと外縁1pxを除外。参照は保存npjの直近の周囲画素で、4近傍BGR調和補間。前回2入力を再計算し、抽出した紙面だけのマスクと結果が全画素一致した。閾値変更・パラメータ探索・新しい色補正・文字補完は行わない。保持対象と更新外は原画像ではなく保存npjの画素を維持する。

選定は保存入力・従来の実出力を見て新候補より前に固定した。改善例だけを選んだ独立評価ではなく、問題候補と対照を含む開発追加確認である。

## 目視所見

'''+NOTES.read_text()+'''

## 全12ROI

'''
    for r in records:
        report += f'- [{r["title"]}]({r["id"]}.png)：紙面更新 {r["mask_pixels"]["paper_update"]:,}、実変更 {r["changed_pixels"]:,}画素。保存npj全画素一致={r["exact_saved_npj"]}。\n'
    report += '''
## 検証と解釈

紙面条件の前回2入力との完全な再計算一致、凍結コード・分類条件・入力／保存出力のハッシュ、AST適応コードの既存スナップショット一致を確認。各新保存入力から固定分類し、実入力の更新内を異色へ置換して紙面補間結果の不変性を確認した。表示範囲とは独立に紙面マスク規則を再構成し一致、量子化した4近傍方程式の残差も確認。保護・朱・保留・追加暖色保留・更新外の保存npj一致、全パネルの整数反復／明示縮小一致、ZIP全19PNGのCRCとバイト、全ハッシュを確認。

保存npjのL<40が更新へ入る数も診断したが、これは筆画GTではない。保持範囲外の薄い実筆画を消さない保証、汚れの除去率、人の読解精度、全文書での優越は未確認。変化0の失敗例を成功率へ数えない。まだ採用しない。

[全19PNG ZIP](comparison_pngs.zip)、[検証](verification.json)、[入力対応と全体](diagnostics.json)、[濃い塊で編集が動かない診断](13_hnsd_mask_limitation.png)。
'''
    (OUT/'report.md').write_text(report)
    deps = {str((PILOT/name).relative_to(ROOT)):sha(PILOT/name)
            for name in ['manifest.json', 'complete.json', 'source_snapshot.py', 'verification.json']}
    for name in ['manifest.json', 'complete.json', 'verification.json', 'source_snapshot.py']:
        deps[str((SCOPE/name).relative_to(ROOT))] = sha(SCOPE/name)
    for p in [Path(__file__), NOTES, Path(__file__).parent/'package_saved_npj_role_separated_review.py',
              Path(__file__).parent/'build_npj_v6_comparison.py',
              Path(__file__).parent/'package_saved_npj_inference_v6_review.py']:
        deps[str(p.relative_to(ROOT))] = sha(p)
    dump(OUT/'manifest.json', dict(dependencies=deps, source_sha256=sha(Path(__file__)), primary_artifact='PNG', pixel_GT=False))
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    dump(OUT/'verification.json', dict(status='passed', pages=page_records, regions=records,
         baseline_uses_real_saved_npj=True, mask_rule_independently_reconstructed=True,
         rounded_harmonic_equation_checked=True, protected_exact_saved_npj=True,
         outside_update_exact_saved_npj=True, panels_exact=True, archive_png_count=19,
         stroke_update_disabled=True, pixel_GT=False, adopted=False))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')): z.write(p, p.name)
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert z.testzip() is None and len(z.namelist()) == 19
        for name in z.namelist(): assert z.read(name) == (OUT/name).read_bytes()
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('Packaged 19 PNGs: fixed paper-only follow-up, failures and controls.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--render-only', action='store_true')
    main(parser.parse_args().render_only)
