"""Verify and report original-first four-book fixed-parameter validation."""
import gc
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import restore_unseen_margin_paper as paper
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_blending_review import check_bundle
from package_paper_seam_texture_review import verify_texture

ROOT = paper.ROOT
INPUT = paper.INPUT
MASKS = paper.masks.OUT
PAPER = paper.OUT
OUT = INPUT/'review'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for directory in [MASKS, PAPER]: check_bundle(directory)
    metadata = json.loads((INPUT/'comparison_manifest.json').read_text())
    for relative, digest in metadata['dependencies'].items(): assert sha(ROOT/relative) == digest
    for relative, digest in json.loads((INPUT/'comparison_complete.json').read_text())['outputs'].items(): assert sha(INPUT/relative) == digest
    assert sha(INPUT/'comparison_runner_snapshot.py') == metadata['source_sha256']
    selected = json.loads((INPUT/'baseline_review.json').read_text())
    rows = json.loads((PAPER/'diagnostics.json').read_text()); lookup = {r['id']: r for r in rows}
    mask_rows = json.loads((MASKS/'diagnostics.json').read_text())
    assert len(rows) == len(lookup) == 12
    books = {p['book'] for p in selected['cases']}
    assert len(books) == 4 and not (books & set(selected['excluded_books']))
    residuals = []; provenance = []; checked = []
    for page in selected['cases']:
        key = page['page']; original = read(ROOT/page['original'])
        assert sha(ROOT/page['original']) == selected['inputs'][page['original']]
        full = MASKS/'pages'/key
        v6_edit = read(full/'v6/edit.png', True); complete = read(full/'completed/edit.png', True)
        rim = read(full/'rim/edit.png', True); added = read(full/'rim/added.png', True)
        domain = read(full/'features/safe_domain.png', True); guard = read(full/'features/rim_guard.png', True)
        assert np.array_equal(rim, complete | added) and not np.any(added & complete)
        assert not np.any(v6_edit & ~complete) and not np.any(added & (guard | ~domain))
        for mode in ['completed', 'rim']:
            for protected in ['protect', 'hold', 'red']:
                assert not np.any(read(full/mode/'edit.png', True) & read(full/mode/(protected+'.png'), True))
        full_modes = dict(npj=read(ROOT/page['npj_degradation'], True), v6=v6_edit, completed=complete, rim=rim)
        for roi in page['rois']:
            rid = roi['id']; row = lookup[rid]
            x, y, w, h = row['xywh']; cx, cy, cw, ch = row['context_xywh']
            cs = (slice(cy, cy+ch), slice(cx, cx+cw)); s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
            context = original[cs]; crop = context[s]
            assert np.array_equal(read(PAPER/'native'/rid/'original.png'), crop)
            for mode in paper.MODES:
                native = PAPER/'native'/rid/mode
                edit = read(native/'edit.png', True); eligible = read(native/'eligible_context.png', True)
                known = read(native/'reference_context.png', True); applied = read(native/'applied_context.png', True)
                value = read(native/'output.png'); target = full_modes[mode][cs]
                assert np.array_equal(edit, target[s])
                assert np.array_equal(eligible, target & domain[cs] & ~guard[cs])
                assert not np.any(known & target) and not np.any(applied & ~eligible)
                assert np.array_equal(read(native/'applied.png', True), applied[s])
                assert not np.any(np.any(value != crop, axis=2) & ~applied[s])
                assert not np.any(np.any(value != crop, axis=2) & ~edit)
                stats = row['diagnostics'][mode]
                assert stats['applied_pixels'] == int(applied[s].sum())
                assert stats['changed_pixels'] == int(np.any(value != crop, axis=2).sum())
                assert stats['eligible_pixels'] == int(eligible[s].sum())
                if applied.any():
                    lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
                    fields = np.load(native/'harmonic_fields_Lab.npz', allow_pickle=False)
                    smooth = paper.frozen.prior.prior.reference_lowpass(lab, known)
                    for name, boundary in [('boundary_raw', lab), ('boundary_smooth', smooth)]:
                        error, used = verify_field(fields[name], applied, known, boundary)
                        assert np.array_equal(used, read(native/'used_context.png', True))
                        residuals.append(dict(id=rid, mode=mode, field=name, residual=error))
                    donors = json.loads((native/'donors.json').read_text())
                    if donors['components']:
                        provenance.append(dict(id=rid, mode=mode, **verify_texture(lab, known, applied, native)))
                    else:
                        assert len(donors['patch_xy']) < 4
                        assert not np.load(native/'texture_seam.npy', allow_pickle=False).any()
                        assert (np.load(native/'source_x.npy', allow_pickle=False) == -1).all()
                    dist = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
                    weight = np.exp(-(np.maximum(dist-1, 0)/8.)**2) * applied
                    combined = fields['boundary_smooth'] + (fields['boundary_raw']-fields['boundary_smooth']) * weight[:, :, None]
                    combined[:, :, 0] += np.load(native/'texture_seam.npy', allow_pickle=False) * np.clip((dist-1)/4., 0, 1) * applied
                    saved = np.load(native/'result_field_Lab.npy', allow_pickle=False)
                    assert np.array_equal(saved, combined)
                    converted = np.rint(np.clip(cv2.cvtColor(saved[s], cv2.COLOR_LAB2BGR), 0, 1)*255).astype(np.uint8)
                    assert np.array_equal(value[applied[s]], converted[applied[s]])
                else: assert np.array_equal(value, crop)
            checked.append(rid)
            print('Verified fixed new book:', rid, flush=True)
        del original, v6_edit, complete, rim, added, domain, guard, full_modes
        gc.collect()
    overview = []
    for filename, keys, folder, group, title, notes in [
        ('01_fixed_mask_targets.png', ['unseen_thin_crack', 'unseen_brown_spot', 'unseen_fold'], MASKS, 'comparisons', '未使用冊へ条件を固定して適用：分類',
         ['細い裂けは回収0。斑点と折れ付近に追加候補が出た。正解の成功数ではない。']),
        ('02_same_paper_restorer.png', ['unseen_thin_crack', 'unseen_brown_spot', 'unseen_fold'], PAPER, 'comparisons', '同じ紙面補間を適用：未解決も残す',
         ['斑点は参照境界不足で全条件未補間。折れ付近は薄くなるが不均一な塗り跡が残る。',
          'DDRMではない。未補間の編集画素も原画像。本文全体の復元結果ではない。']),
        ('03_text_seal_controls.png', ['unseen_dense_text', 'unseen_blue_text', 'unseen_seal'], MASKS, 'comparisons', '濃い本文・薄い本文・印影の対照',
         ['今回の余白・全成分・輪郭の追加0。位置・形の条件で対象外。']),
        ('04_blank_paper_controls.png', ['unseen_blank', 'unseen_upper_paper'], PAPER, 'comparisons', '空白余白の対照',
         ['v6系は今回の紙面補間0で原画素。npjマスクでは一部を補間した。']),
        ('05_original_selection.png', [], INPUT, '', '元画像から先に固定した新しい4冊', [])]:
        if keys:
            stack([folder/group/(rid+'.png') for rid in keys], OUT/filename, title, notes)
        else:
            shutil_path = INPUT/'original_top_details.png'
            from shutil import copyfile
            copyfile(shutil_path, OUT/filename)
        overview.append(filename)
    items = [(OUT/name, 'overview/'+name) for name in overview]
    for directory, group, label in [(INPUT, 'classification_comparisons', 'npj_v6'), (MASKS, 'comparisons', 'fixed_masks'), (PAPER, 'comparisons', 'paper_fill')]:
        items.extend((directory/group/(r['id']+'.png'), label+'/'+r['id']+'.png') for r in rows)
    items.extend((MASKS/'comparisons'/(p['page']+'_whole.png'), 'whole_page_masks/'+p['page']+'.png') for p in selected['cases'])
    archive_path = OUT/'fixed_new_books_comparison_images.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(items) == 45 and archive.testzip() is None
        for path, name in items: assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    masks_lookup = {r['id']: r for r in mask_rows['regions']}
    table = '\n'.join(f'| [{r["title"]}](../paper_fill/comparisons/{r["id"]}.png) | {masks_lookup[r["id"]]["margin_added"]:,} | {masks_lookup[r["id"]]["rim_added"]:,} | '+ ' | '.join(f'{r["diagnostics"][m]["applied_pixels"]:,}' for m in paper.MODES)+' |' for r in rows)
    report = f'''# 未使用4冊で輪郭回収と紙面補間を条件固定で検証

2026-10-01。**現在の条件は太い斑点・折れ付近の一部を候補に戻すが、細い裂けを回収できず、斑点は参照境界不足で補間できなかった。折れ付近は薄くなる一方、不均一な塗り跡が残る。未知資料への完成した改善とは言えない。**

利用者の「止めるまで改善をできるだけ続ける」に従って継続中。まず新しい冊に対して条件を変えず適用した結果を保存し、検出と参照不足を次の改善課題に分ける。

## 原画像を先に選んだ範囲

これまでの調整とv6診断に使った26冊を除外し、原画像2,939ページの上部の茶色い成分を候補抽出した。得点は確認用であり損傷の正解ではない。11冊の候補画像を目視で確認し、イラスト6ページと表紙1ページを除外。本文4ページ・4冊・12箇所を、新しいv6・余白・輪郭マスクを見る前に[選定JSON](../baseline_review.json)へ固定した。[原画像上部の確認PNG](05_original_selection.png)。学習データとの独立性は不明。

細い裂け状の跡・折れ・斑点は外観から選んだ候補で、物理原因・画素GTは未確定。本文、枠、空白余白、印影を同時に選んだ。斑点付近の黒い小さな点も正体を確定していないため、インク保存／損傷除去の正解率を計算しない。

## 比較PNG

- [分類マスクの問題例](01_fixed_mask_targets.png)。原画像、保存npj、固定v6、全成分回収、固定の輪郭回収。
- [同じ紙面復元器の比較](02_same_paper_restorer.png)。原画像、npjマスク、v6、全成分、輪郭の順。
- [本文・印影](03_text_seal_controls.png)、[空白余白](04_blank_paper_controls.png)。
- [全45PNG ZIP](fixed_new_books_comparison_images.zip)。概要5枚、全12箇所のnpj/v6分類、全12箇所の候補分類、全12箇所の紙面補間、全4ページの分類。

## 固定条件の結果

前回の受理済みコアの6px近傍・紙色差L≥2を、そのまま新4冊へ適用した。太いコアの受理条件はピーク半径6px、茶色種32画素、面積128以上などで従来どおり。本文枠の上に限定した安全領域と、朱・暖色接続・長い横線の保護を維持した。ROI名では候補を限定していない。

| ROI | 旧余白の追加 | 輪郭の追加 | npjマスクの補間 | v6の補間 | 全成分の補間 | 輪郭の補間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

細い裂けの暗色成分は面積151／365画素、茶色種107／317画素、ピーク半径4.0／4.394px。面積と茶色種の条件を満たすが、ピーク6pxの条件で除外される。現在の輪郭回収は既に受理したコアの周りしか広げないため、この未検出を補えない。原画像の裂け状の跡がv6の保護・保留に残り、改善0を成功として数えない。

斑点の箇所は余白で1,223画素、輪郭でさらに123画素を候補に戻したが、紙面補間は全条件0。大きな成分は5,081画素になっても参照に隣接する境界対が12組しかなく、現在の16組以上の条件を満たさない。紙面参照自体は文脈内に12万画素以上あるため、全体に参照がないという意味ではない。直接隣接する境界の不足と、断片化した候補の小成分を区別する。

折れ付近は旧余白4,743画素・輪郭2,593画素を候補に追加し、同じ紙面補間で適用範囲が75→5,853→8,648画素と増えた。縦の暗い部分が薄くなる一方、茶色い不均一な塗り跡や保護された縁が残る。これは折れ・ページ端の局所診断で、失われた紙形状や真正な模様の復元でも、損傷の画素精度でもない。

新しい本文・印影・空白の対照では、余白・全成分・輪郭による追加0。本文・印影は主に安全領域の外であり、自動的に文字と損傷を区別できた証明ではない。青灰色本文や印影の細部を今の紙面補間で改善したとも扱わない。

## 復元比較の意味

この段階は**DDRMではなく、4種類のマスクを同じ紙色補間・観測明度模様へ渡す診断**。固定した余白安全領域と保護幅だけで補間し、未補間の編集画素も原画像を保持する。過去14ページでは範囲外に保存DDRMが残っていたが、今回の新冊にはそれを仮装しない。元npjの全処理の再現、本文全体の復元、npj/v6のDDRM性能の評価ではない。

マスクに合わせて参照と模様候補が変わる。全マスクの外から条件を満たす紙色を取り、4近傍の調和補間と既存の境界補正、参照24px角パッチの明度模様を使った。小成分や16組未満の境界は補間せず、原画素を保持した。

## 確認と継続改善

選定前後の入力・固定コード・マスク・保存出力のハッシュを確認した。各マスクの包含と保護、全ページの位置条件、全48条件の補間範囲外と編集範囲外の原画素一致を確認した。保存された場に4近傍の差を直接足して残差を確認し、参照元画素と観測模様、配置記録からの重複平均、Lab合成式を別に確認した。編集対象の原画素を任意の色に変えた不変も実行済み。[確認JSON](verification.json)。ZIP45PNGの内容ハッシュも一致。

条件固定の初回結果はこのフォルダに維持する。**次は、薄い裂けを落とす太さ条件と、斑点を補間できない境界対数条件を別々に変更して比較する。** 新しい4冊の結果を使って調整した時点で開発側に移し、今回の固定条件の結果を上書きしない。過去の全14ページ・22箇所も含めて本文・印影・原因不明の曲線への追加を確認し、成功例だけを選ばない。継続指示が有効なので、ここで作業を止めず次の比較へ進む。

[分類の固定条件](../fixed_masks/manifest.json)、[紙面補間の固定条件](../paper_fill/manifest.json)、[分類の全診断](../fixed_masks/diagnostics.json)、[復元の全診断](../paper_fill/diagnostics.json)、[確認・梱包スクリプト](../../../scripts/benchmark/package_unseen_margin_review.py)。
'''
    (OUT/'report.md').write_text(report)
    dump(OUT/'verification.json', dict(status='passed', pages=4, books=4, regions=checked,
        original_selection_first=True, parameters_fixed=True, training_independence_unknown=True,
        all_unapplied_pixels_original=True, outside_edit_changed_pixels=0,
        direct_discrete_residual_max=max(r['residual'] for r in residuals), observed_texture_provenance=True,
        zip_PNG_count=45, zip_hashes=True, pixel_GT=False, DDRM=False))
    dump(OUT/'diagnostics.json', dict(harmonic_residuals=residuals, donor_provenance=provenance))
    dependencies = {str((folder/name).relative_to(ROOT)): sha(folder/name)
        for folder in [MASKS, PAPER] for name in ['manifest.json', 'complete.json']}
    dump(OUT/'manifest.json', dict(source_sha256=sha(Path(__file__)), dependencies=dependencies,
        outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Passed: four new books, all 48 paper variants, 45 PNGs', flush=True)


if __name__ == '__main__': main()
