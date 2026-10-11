"""Package the two development probes and fixed-DDRM check without rerunning them.

Verify full-page mask arithmetic and native output pixels independently of the
inference runners. Keep failed examples and factor ablations in the PNG package.
"""
import csv
import gc
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import dump, sha, read, FONT_PATH
import probe_defect_context_preprocessing as base

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/defect_context_review_20261001'
V1 = ROOT / 'tmp_work/defect_context_preprocessing_pilot_v1_20261001'
V2 = ROOT / 'tmp_work/defect_context_preprocessing_pilot_v2_20261001'
DDRM = ROOT / 'tmp_work/defect_context_ddrm_pilot_v2_20261001'
OLD_DDRM = ROOT / 'tmp_work/npj_v6_ddrm_edit_comparison_20261001'
SOURCES = base.SOURCES

# Provisional visual review; no pixel truth or human endorsement is implied.
NOTES = {
    'dense_stain': ('追加は周縁35画素。中心はv6で既に編集対象。', '明るい平坦な塗り跡・境界が残る。解決と数えない。'),
    'thin_black_water': ('ROI内追加0。', 'v6と同一出力。薄い本文を維持するが影・水染みは残る。'),
    'brown_as_red': ('ROI内追加0。', 'v6と同一出力。既存の編集境界・色差が残る。'),
    'thin_black_loop': ('ROI内追加0。', 'v6と同じ原画素。npj側の消失・変形を避ける状態を維持。'),
    'thin_black_text': ('本文対照に63画素追加。完全な無変更ではない。', '主要筆画は概観で維持。小さな変更の正誤は未判定。'),
    'warm_stroke': ('v1では実筆画も編集候補化。v2の追加は5,195→81画素。', '主要な暖色筆画を概観で維持。残る81画素の正誤は未判定。'),
    'control_black': ('本文対照に35画素追加。', '主要筆画は概観で維持。小さな変更の正誤は未判定。'),
    'dense_stain_new': ('濃い黄褐色・オリーブ色部分に追加0。色差条件の対象外。', 'v6と同一出力。本文・枠に重なる汚れは残り、未解決。'),
    'brown_false_red_new': ('紙面上端の茶色い部分に12,411画素追加。', '新しい暗い線・文字状の断片が発生。広い色ムラも残り、復元成功と数えない。'),
    'hole_edge_new': ('文字のない穴周辺の濃い部分を4,960画素、編集側に戻す。', '濃い中心は薄くなるが茶色い縁・不均一な塗り跡が残る。自然な修復は未達。'),
    'control_text_new': ('本文対照に24画素追加。', '主要筆画は概観で維持。小さな変更の正誤は未判定。'),
    'brown_near_text': ('茶色い部分に15,426画素追加。直下の黒い横画は保護。', '茶色い中心に原画像で確認できない文字状の線を生成。復元成功と数えない。'),
    'dark_text_control': ('本文対照に35画素追加。', '主要筆画は概観で維持。小さな変更の正誤は未判定。'),
    'faint_left_text': ('ROI内追加0。', 'v6と同一出力。元の薄い本文を維持。'),
    'damage_control': ('損傷の周縁310画素を追加。濃い中性の縁は保護が残る。', '大きな損傷の線はほぼ残る。未解決。このIDは対照でなく問題例。'),
}


def check_archive(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for rel, digest in manifest['dependencies'].items():
        assert sha(ROOT / rel) == digest, (directory.name, rel)
    if directory == V2:
        assert sha(V2 / 'source_snapshot.py') == manifest['source_sha256']
        assert sha(V2 / 'runner_snapshot.py') == manifest['settings']['runner_source_sha256']
    elif directory == V1:
        assert sha(V1 / 'source_snapshot.py') == manifest['source_sha256']
    complete = json.loads((directory / 'complete.json').read_text())['outputs']
    for rel, digest in complete.items():
        assert sha(directory / rel) == digest, (directory.name, rel)
    return len(complete)


def stack(paths, dest, title, notes):
    images = [Image.open(p).convert('RGB') for p in paths]
    width = max(im.width for im in images)
    head = 130
    canvas = Image.new('RGB', (width, head + sum(im.height + 20 for im in images)), (250, 248, 243))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 12), title, font=ImageFont.truetype(FONT_PATH, 30), fill='black')
    font = ImageFont.truetype(FONT_PATH, 23)
    for i, note in enumerate(notes):
        draw.text((24, 58 + i * 31), note, font=font, fill='black')
    y = head
    for im in images:
        canvas.paste(im, (0, y)); y += im.height + 20
    canvas.save(dest)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {d.name: check_archive(d) for d in [V1, V2, DDRM, OLD_DDRM]}
    factors = []; pages = []; roi_rows = []; books = set()
    ddrm_rows = {r['id']: r for r in json.loads((DDRM / 'diagnostics.json').read_text())}
    repeated_contexts = 0
    for source in SOURCES:
        for page in json.loads((source / 'baseline_review.json').read_text())['cases']:
            id_page = page['page']; books.add(Path(page['original']).parent.name)
            original = read(ROOT / page['original'])
            v6 = {k: read(source / 'v6_masks' / id_page / (k + '.png'), True)
                  for k in ['edit', 'protect', 'hold', 'red', 'extra']}
            masks = {k: read(V2 / 'pages' / id_page / 'context_stroke' / (k + '.png'), True)
                     for k in ['edit', 'added', 'protect', 'hold', 'red']}
            old = read(V1 / 'pages' / id_page / 'context_stroke/added.png', True)
            guard = read(V2 / 'pages' / id_page / 'features/neutral_ridge_guard_4px.png', True)
            anchor = read(V2 / 'pages' / id_page / 'features/warm_anchor_guard.png', True)
            variants = dict(v1=old, anchor_only=old & ~anchor,
                            ridge_only=old & ~guard, combined=old & ~anchor & ~guard)
            assert np.array_equal(variants['combined'], masks['added'])
            assert np.array_equal(masks['edit'], v6['edit'] | masks['added'])
            assert np.array_equal(masks['protect'], v6['protect'] & ~masks['added'])
            assert np.array_equal(masks['hold'], v6['hold'] & ~masks['added'])
            assert np.array_equal(masks['red'], v6['red'])
            assert not np.any(masks['edit'] & (masks['protect'] | masks['hold'] | masks['red'] | v6['extra']))
            assert original.shape[:2] == masks['edit'].shape
            whole = dict(page=id_page, width=original.shape[1], height=original.shape[0],
                         v6_edit_pixels=int(v6['edit'].sum()), added_pixels=int(masks['added'].sum()),
                         added_fraction=float(masks['added'].mean()))
            pages.append(whole)
            for roi in page['rois']:
                rid = roi['id']; x, y, w, h = roi['xywh']; s = (slice(y, y+h), slice(x, x+w))
                counts = {name: int(value[s].sum()) for name, value in variants.items()}
                factors.append(dict(id=rid, page=id_page, **counts))
                crop = original[s]; native = DDRM / 'native' / rid
                result = read(native / 'candidate.png')
                edit = read(native / 'candidate_edit.png', True)
                added = read(native / 'added.png', True)
                changed = np.any(result != crop, axis=2)
                assert np.array_equal(crop, read(native / 'original.png'))
                assert np.array_equal(edit, masks['edit'][s])
                assert np.array_equal(added, masks['added'][s])
                assert np.array_equal(changed, read(native / 'candidate_changed.png', True))
                assert not np.any(changed & ~edit)
                assert int(changed.sum()) == ddrm_rows[rid]['candidate_changed_pixels']
                assert counts['combined'] == ddrm_rows[rid]['added_edit_pixels']
                if ddrm_rows[rid]['reused_exact_same_context_mask']:
                    x0, y0 = max(0, x-64), max(0, y-64)
                    x1, y1 = min(original.shape[1], x+w+64), min(original.shape[0], y+h+64)
                    assert np.array_equal(masks['edit'][y0:y1, x0:x1], v6['edit'][y0:y1, x0:x1])
                    assert np.array_equal(result, read(OLD_DDRM / 'native' / rid / 'v6.png'))
                    repeated_contexts += 1
                roi_rows.append(dict(id=rid, page=id_page, xywh=roi['xywh'],
                    baseline_failure=roi['baseline_failure'],
                    v1_added_pixels=counts['v1'], v2_added_pixels=counts['combined'],
                    v6_changed_pixels=ddrm_rows[rid]['v6_changed_pixels'],
                    candidate_changed_pixels=int(changed.sum()),
                    mask_review=NOTES[rid][0], restoration_review=NOTES[rid][1]))
                if rid in ['warm_stroke', 'hole_edge_new', 'brown_near_text']:
                    images = [crop]
                    for value in variants.values():
                        mode = dict(edit=v6['edit'][s] | value[s], protect=v6['protect'][s] & ~value[s],
                                    hold=v6['hold'][s] & ~value[s], red=v6['red'][s])
                        images.append(base.paint(crop, mode))
                    base.figure(images, ['原画像', '試作v1', '暖色接続保護のみ', '墨ガード拡張のみ', '修正版v2'],
                        OUT / 'factor_ablation' / (rid + '.png'), roi['title'] + ' · 保護要素の切り分け',
                        f'追加候補：v1 {counts["v1"]:,} / 接続保護 {counts["anchor_only"]:,} / 墨ガード {counts["ridge_only"]:,} / 両方 {counts["combined"]:,}（精度ではない）')
            del original, v6, masks, old, guard, anchor, variants
            gc.collect()
    assert len(roi_rows) == 15 and len(pages) == 11
    dump(OUT / 'visual_review.json', dict(
        reviewer='assistant provisional visual inspection', pixel_GT=False,
        independent_validation=False, adopted=False, sampler_or_weight_changes=False,
        cases=roi_rows, whole_pages=pages))
    for filename, rows in [('factor_ablation.csv', factors), ('whole_page_counts.csv', pages)]:
        with (OUT / filename).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    stack([V2/'comparisons'/f'{id}.png' for id in ['hole_edge_new', 'brown_near_text']],
          OUT/'01_mask_targets.png', '穴・茶色い部分の分類比較：原画像 / 保存npj / 固定v6 / 試作',
          ['編集候補に戻る部分は増えた。欠損の縁の検出・実筆画との分離は未完成。', '黄は編集候補で、正解ラベルではありません。'])
    stack([DDRM/'comparisons'/f'{id}.png' for id in ['hole_edge_new', 'brown_near_text']],
          OUT/'02_ddrm_targets.png', '同じDDRMでの確認：濃い部分は薄くなるが復元は未達',
          ['上：穴の茶色い縁と塗り跡が残る。下：原画像で確認できない文字状の線が生成。', '前処理の改善を、自然な紙面の復元成功として数えません。'])
    stack([V2/'comparisons'/f'{id}.png' for id in ['damage_control', 'dense_stain_new']],
          OUT/'03_remaining_masks.png', '未解決の分類例：別冊の裂け・本文に重なる黄褐色の汚れ',
          ['濃い損傷の縁は筆画ガード側に残る。オリーブ色の汚れは色差条件を満たさない。'])
    stack([DDRM/'comparisons'/f'{id}.png' for id in ['warm_stroke', 'thin_black_loop', 'faint_left_text']],
          OUT/'04_stroke_controls.png', '筆画の確認：暖色筆画・細い黒筆画・薄い本文',
          ['主要な筆画は概観で維持。暖色の追加81画素を含め、無誤編集の保証ではありません。'])
    stack([V1/'comparisons/warm_stroke.png', V2/'comparisons/warm_stroke.png'],
          OUT/'05_initial_regression.png', '初回の失敗も保存：暖色の実筆画が編集候補になる',
          ['上：試作v1で筆画を侵食。下：接続保護＋墨ガード拡張後、追加5,195→81画素。'])
    stack([DDRM/'comparisons/brown_false_red_new.png'],
          OUT/'06_paper_artifact.png', '文字のない紙面でもDDRMが暗い線・文字状の断片を生成',
          ['紙面上端と左端に新しい暗い線が発生。広い茶色いムラも残り、復元成功と数えない。'])

    table = '\n'.join(f'| [{r["id"]}](../{V2.name}/comparisons/{r["id"]}.png) | {r["v1_added_pixels"]:,} | {r["v2_added_pixels"]:,} | {r["mask_review"]} | [{r["restoration_review"]}](../{DDRM.name}/comparisons/{r["id"]}.png) |' for r in roi_rows)
    whole_table = '\n'.join(f'| [{r["page"]}](../{V2.name}/comparisons/{r["page"]}_whole.png) | {r["v6_edit_pixels"]:,} | {r["added_pixels"]:,} | {r["added_fraction"]:.2%} |' for r in pages)
    warm = next(r for r in factors if r['id'] == 'warm_stroke')
    report = f'''# 虫食い状の損傷・シミの前処理試作と同条件DDRM確認

2026-10-01。**前処理で誤保護を減らす手掛かりは得たが、虫食い・シミの解決やv6を超える新手法の確立には至っていない。** 既存の固定11ページ・{len(books)}冊・15箇所（問題11、本文対照4）を使い、全ページから推定する試作を2段階で実行した。修正版の全15箇所を、前回と同じDDRMで確認した。画像を見て調整した開発例であり、独立した別冊評価でも画素GTによる精度でもない。イラストとして除外済みのページは追加していない。

## 比較画像

- [穴・茶色い部分の分類PNG](01_mask_targets.png)：原画像、保存済みnpj、固定v6、修正版。
- [同じDDRM後のPNG](02_ddrm_targets.png)：穴の縁の残存と新しい文字状の生成。
- [別冊の未解決分類PNG](03_remaining_masks.png)、[筆画のDDRM確認PNG](04_stroke_controls.png)。
- [初回試作の筆画侵食と修正版PNG](05_initial_regression.png)、[文字のない紙面での生成失敗PNG](06_paper_artifact.png)。
- [全比較画像ZIP](defect_context_comparison_images.zip)。全ページ分類11枚、ROI分類15枚、同じDDRM15枚、色のみとの比較15枚、要素を切り分けた3枚、初回失敗1枚、概要6枚。合計66枚。

分類PNGの青＝保護、赤＝朱として保護、黄＝編集候補、紫＝保留。黄の増加は処理対象面積であり、精度や改善率ではない。DDRM PNGには分類色を重ねていない。

## 試作の内容

試作v1では、近傍の明るい紙から推定した紙色との差、茶色い領域の連結・太さ、黒い線状応答を使った。原画像のページ全体と固定v6のマスクだけを推定入力とし、ROI座標・問題ラベルで処理を限定していない。固定v6の編集候補に新しい候補を加え、その部分だけ保護・保留から外す。v6の朱候補と追加暖色保留は保持する。

ただし、赤褐色の実筆画にも黄が入り、`warm_stroke`で5,195画素を追加した。これを失敗例として保存した。色と局所形状だけで汚れと暖色筆画を分離できたとは言えない。

修正版v2では、v6の朱候補・追加暖色保留につながる保護／保留領域全体と2pxの周囲を守り、黒い線状応答の周囲の保護を2pxから4pxへ広げた。アンカーに接続する汚れも残しうる保守的な仮説であり、領域が実インクだという証明ではない。ここでの「文脈」は局所紙色・連結・線状応答を指し、文書の意味や損傷の物理原因を認識したものではない。

[要素別CSV](factor_ablation.csv)ではv1から「接続保護のみ」「墨ガード拡張のみ」「両方」を同じ全ページマスクから作り、効果を分けた。暖色筆画の追加候補はv1 {warm['v1']:,}、接続保護のみ {warm['anchor_only']:,}、墨ガード拡張のみ {warm['ridge_only']:,}、両方 {warm['combined']:,}画素。[暖色の要素別PNG](factor_ablation/warm_stroke.png)、[穴](factor_ablation/hole_edge_new.png)、[文字に接する茶色](factor_ablation/brown_near_text.png)。これらは分類比較で、各アブレーションのDDRMは実行していない。

## 全15箇所の所見

| 箇所・分類PNG | v1追加画素 | v2追加画素 | 分類の所見 | 同条件DDRM後の所見・PNG |
| --- | ---: | ---: | --- | --- |
{table}

穴の中心、文字の直上の茶色い部分の誤保護は減る。しかし、穴の濃い縁を全部検出できていない。黒い損傷の縁は本物の黒筆画と同様に線状応答が強く、保護されて残る。本文に重なるオリーブ色のシミは今回の茶色色差条件を満たさず、追加0で未解決。物理的な虫食い・シミ・焼損の原因は画像から確定していない。

復元では、茶色い部分を編集対象に戻すと、新しい文字状の線が生成される例がある。元の文字を正しく取り戻したという根拠はなく、成功として数えない。文字用事前分布と紙面修復という接続の適合性、残存する境界・編集範囲の不確かさは別々に検証すべきで、今回だけで生成原因を特定してはいない。

## ページ全体での編集範囲

切り出し以外も自動推定している。特に穴のあるページでは追加が336,882画素、紙面上端が茶色いページでは125,344画素となり、側端の変色等にも広がる。全ページの縮小比較は確認したが、全追加画素の正誤を判定したものではない。本文対照4箇所にも24〜63画素の追加があり、無誤編集とは言えない。DDRMは15箇所の周辺64pxを含む局所実験であり、全ページ復元を完了したわけではない。

| ページ・全体分類PNG | 固定v6編集画素 | v2追加画素 | ページ面積に対する追加割合 |
| --- | ---: | ---: | ---: |
{whole_table}

## DDRM条件と検証

前回の[条件と結果](../{OLD_DDRM.name}/report.md)を固定。保存済みnpjの`04_degradation`、固定v6の`edit`、修正版の`edit`を同じinpaintingの欠損領域に接続した。**旧npjの黒文字再生成と朱の合成を含む全工程の再現ではない。** npj/v6出力はハッシュ確認した前回の保存結果を再利用した。

同じ原画像、`model_char/ckpt_epoch0350.pth`、RGB [-1,1]、20段、sigma=0.05、etaB=1、etaA/etaC=0.5。128pxパッチ、重なり32px、周辺64px、同じ画像座標のノイズ場、同じ重み付き合成。文字用xt抑制は全方式で無効。編集範囲外は原画素へ戻す。sampler・学習済み重みは変更していない。ROIの周辺も含めてマスクが完全一致する{repeated_contexts}箇所は、v6保存出力を完全一致で再利用した。ROI内追加0でも周辺でマスクが変わる場合は再推論した。

入力・保存済み比較・ソース・重み・全出力のSHA-256、全11ページのマスク演算、朱／暖色保留の維持、編集と保護・保留の非重複、全15箇所の原寸出力・編集範囲外の不変を検証。同じマスクとノイズでDDRM出力が画素単位で一致することも実行時に確認した。追加候補画素数と出力差は一致するとは限らず、マスク変更によって以前から編集対象だった画素の生成も変わる。

[独立した整合性検証](verification.json)、[DDRM実行時の検証](../{DDRM.name}/verification.json)、[全例の原寸データ](../{DDRM.name}/native)、[暫定目視所見](visual_review.json)。いずれも精度・真正性の検証結果ではない。

## 次に進める課題

固定v6を比較基準として残し、この試作は採用しない。次は前処理の課題を、**文字のない紙面の損傷の検出**と**文字に接する汚れの安全な分離**に分ける。前者では黒い損傷の縁・紙色に近いシミ、後者では薄い黒／赤褐色筆画の侵食を同時に評価する。今回の15箇所は開発側に置き、少数の手動正解領域と未使用の別冊を確保してから改善率を測る。

紙面修復で文字状の線を生成しない条件も、マスクの正しさと分けて検証する。現段階で拡散モデル本体の学習変更を始める根拠にはせず、まず編集領域が紙面か文字を含むかの接続仕様と評価を整理する。重なりが不明な箇所は保留する。前処理を続ける研究上の理由は残るが、前処理だけで全損傷が自然に復元できるとは結論しない。

実装：[初回試作](../../scripts/benchmark/probe_defect_context_preprocessing.py)、[接続保護を加えた修正版](../../scripts/benchmark/probe_defect_context_anchors.py)、[同条件DDRM確認](../../scripts/benchmark/check_defect_context_ddrm.py)、[検証・画像整理](../../scripts/benchmark/package_defect_context_review.py)。旧npj、固定v6、過去の比較結果と利用者の既存変更は更新していない。
'''
    (OUT / 'report.md').write_text(report, encoding='utf-8')
    items = []
    for path in sorted(OUT.glob('0*.png')):
        items.append((path, 'overview/' + path.name))
    for page in pages:
        p = V2 / 'comparisons' / (page['page'] + '_whole.png')
        items.append((p, 'mask_whole/' + p.name))
    for row in roi_rows:
        rid = row['id']
        for source_dir, prefix in [(V2/'comparisons', 'mask_roi'), (DDRM/'comparisons', 'ddrm_roi'), (V2/'ablation', 'color_ablation')]:
            p = source_dir / (rid + '.png'); items.append((p, prefix + '/' + p.name))
    for p in sorted((OUT/'factor_ablation').glob('*.png')):
        items.append((p, 'factor_ablation/' + p.name))
    items.append((V1/'comparisons/warm_stroke.png', 'initial_failure/warm_stroke.png'))
    assert len(items) == 66
    package = OUT / 'defect_context_comparison_images.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, name in items: archive.write(path, name)
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 66
        import hashlib
        for path, name in items:
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha(path)
    dump(OUT/'verification.json', dict(status='passed', checked_output_hashes=hashes,
        pages=11, books=len(books), regions=15, full_page_mask_arithmetic=True,
        native_ROI_pixels=True, outside_edit_changed_pixels=0,
        exact_reused_v6_contexts=repeated_contexts, factor_ablation=True,
        ZIP_PNG_count=66, ZIP_CRC_and_pixel_file_hashes=True, report_links_exist=True,
        pixel_GT=False, independent_validation=False, adopted=False))
    for match in re.finditer(r'\]\(([^)]+)\)', report):
        assert (OUT / match.group(1)).exists(), match.group(1)
    dump(OUT/'manifest.json', dict(
        dependencies={str((d/'manifest.json').relative_to(ROOT)):sha(d/'manifest.json')
                      for d in [V1, V2, DDRM, OLD_DDRM]},
        source_sha256=sha(Path(__file__)),
        ZIP_contents={name:sha(path) for path, name in items}))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)):sha(p)
        for p in OUT.rglob('*') if p.is_file() and p.name != 'complete.json'}))
    print('Review package verified: 11 full pages / 15 ROIs / 66 PNGs', flush=True)
    print('Warm stroke factor ablation:', warm, flush=True)


if __name__ == '__main__':
    main()
