"""Verify native DDRM outputs, assemble PNG sheets and a reviewable report."""
import csv
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import dump, sha, FONT_PATH

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/npj_v6_ddrm_edit_comparison_20261001'
GROUPS = {
    '01_stain_ddrm.png': ['dense_stain', 'brown_as_red', 'brown_false_red_new'],
    '02_stroke_ddrm.png': ['thin_black_water', 'thin_black_loop', 'warm_stroke', 'faint_left_text'],
    '03_remaining_ddrm.png': ['dense_stain_new', 'hole_edge_new', 'brown_near_text', 'damage_control'],
    '04_controls_ddrm.png': ['thin_black_text', 'control_black', 'control_text_new', 'dark_text_control'],
}


def sheet(ids, findings, path):
    images = []
    for id in ids:
        with Image.open(OUT / 'comparisons' / (id + '.png')) as im:
            images.append(im.convert('RGB'))
    width = max(im.width for im in images)
    canvas = Image.new('RGB', (width, 100 + sum(im.height+98 for im in images)), (250,248,243))
    draw = ImageDraw.Draw(canvas); font = ImageFont.truetype(FONT_PATH,27)
    small = ImageFont.truetype(FONT_PATH,23)
    draw.text((28,16), 'DDRM後の比較：同じモデル・設定・空間ノイズ', font=font, fill='black')
    draw.text((28,56), '原画像 / npj編集候補でDDRM / 固定v6編集候補でDDRM', font=small, fill='black')
    top=100
    for id,im in zip(ids,images):
        canvas.paste(im,(0,top));top+=im.height
        draw.text((28,top+16),findings[id]['short_observation'],font=small,fill='black')
        top+=98
    canvas.save(path)


def verify():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for rel,digest in manifest['dependencies'].items(): assert sha(ROOT/rel)==digest,rel
    for rel,digest in json.loads((OUT/'complete.json').read_text())['outputs'].items():
        assert sha(OUT/rel)==digest,rel
    rows=json.loads((OUT/'diagnostics.json').read_text())
    assert len(rows)==15 and len({r['page'] for r in rows})==11
    specs={}
    for folder in ['npj_failure_collection_20261001','npj_failure_followup_20261001']:
        source=ROOT/'tmp_work'/folder
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            for roi in page['rois']: specs[roi['id']]=(page,roi,source)
    for r in rows:
        path=OUT/'native'/r['id'];original=cv2.imread(str(path/'original.png'))
        page,roi,source=specs[r['id']];x,y,w,h=roi['xywh']
        expected=cv2.imread(str(ROOT/page['original']))[y:y+h,x:x+w]
        assert np.array_equal(original,expected)
        mask_paths={'npj':ROOT/page['npj_degradation'],
                    'v6':source/'v6_masks'/page['page']/'edit.png'}
        for method in ['npj','v6']:
            output=cv2.imread(str(path/(method+'.png')))
            edit=cv2.imread(str(path/(method+'_edit.png')),0)>0
            expected_edit=cv2.imread(str(mask_paths[method]),0)[y:y+h,x:x+w]>0
            assert np.array_equal(edit,expected_edit)
            changed=np.any(output!=original,axis=2)
            saved_changed=cv2.imread(str(path/(method+'_changed.png')),0)>0
            assert np.array_equal(changed,saved_changed)
            assert not (changed&~edit).any()
            assert int(changed.sum())==r[method+'_changed_pixels']
            assert int(edit.sum())==r[method+'_edit_pixels']
            if not edit.any(): assert np.array_equal(output,original)
    return rows


def main():
    rows=verify()
    review=json.loads((OUT/'visual_review.json').read_text())
    findings={r['id']:r for r in review['findings']}
    assert set(findings)=={r['id'] for r in rows}
    for name,ids in GROUPS.items(): sheet(ids,findings,OUT/name)
    archive=OUT/'npj_v6_ddrm_comparison_images.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for name in GROUPS: z.write(OUT/name,'overview/'+name)
        for p in sorted((OUT/'comparisons').glob('*.png')): z.write(p,'individual/'+p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==19
    table='\n'.join(f'| [{r["id"]}](comparisons/{r["id"]}.png) | {r["npj_changed_pixels"]:,} | {r["v6_changed_pixels"]:,} | {findings[r["id"]]["short_observation"]} |' for r in rows)
    report='''# 同じDDRMにnpjとv6の編集マスクを渡した比較

[汚れの比較PNG](01_stain_ddrm.png) / [筆画の比較PNG](02_stroke_ddrm.png) / [残る問題の比較PNG](03_remaining_ddrm.png) / [本文対照PNG](04_controls_ddrm.png) / [全19枚ZIP](npj_v6_ddrm_comparison_images.zip)

今回の画像は実際にDDRMで推論した結果。前回のTeleaによる局所補間とは別の実験。前回から固定している11ページ・15箇所をすべて使い、モデル・入力・解像度・設定・合成を固定し、npjとv6の編集候補マスクだけを変えた。イラストページは評価に含めていない。

## マスクをどう接続したか

保存済みnpj v1の04_degradation（白=編集候補）と固定v6のeditを、DDRM inpaintingの欠損領域として使用した。欠損領域以外は同じ原画像を観測値とし、生成後に元の画素へ戻す。v6のprotect・hold・redは欠損領域に含めない。赤・保留の別途の色調変更は行っていない。両方ともTeleaの事前穴埋め・背景参照は使っていない。

既存npjパイプラインの黒文字マスクは文字の再生成を指示するため、v6 editをそこへ直接渡していない。この実験は**編集候補を欠損に対応させたDDRM比較で、元のnpj全工程や黒文字の再生成経路の再現ではない**。モデルへ渡す値も学習時に合わせてRGB [-1,1]としている。旧コードの文字用xt抑制は両方とも無効。したがって、この結果から元のnpjパイプライン全体よりv6が優れるとは結論しない。

## 固定条件

- 重み：model_charのckpt_epoch0350.pth。同じUNet（基底幅64、残差ブロック2、attention 16/8）をstrict load。
- DDRM：リポジトリのefficient_generalized_steps。1000段線形beta=0.0001〜0.02から20段（0,50,…,950）、sigma0=0.05、etaB=1、etaA=etaC=0.5。
- 原寸、矩形周辺64px、128pxパッチ、重なり32px、共通の窓重みで合成。端は共通の反射パディング、パディング領域は既知。マスク膨張・解像度変更・新しい閾値調整なし。
- float32、同じGPU、seed=999からページ・座標ごとのseedを固定。観測ノイズ・初期ノイズ・各段のノイズは同じ画像座標のGaussian場から生成し、各マスクのSVD座標へ対応させた。欠損数が異なるため、単なる乱数seedのリセットだけに依存していない。
- 編集候補が空のパッチは原画像を維持。DDRMの生出力と、編集範囲だけを合成した最終出力を両方保存した。

## 今回の所見

'''+review['summary']+'''

すべて補助者の暫定目視で、画素GT・正解復元画像による精度ではない。v6一式と旧npjの違いには土台の分類処理の変更も含まれる。前回の分類比較でv6固有の追加暖色保留があったのは15箇所中warm_strokeだけであり、他の結果をv6固有の追加保留の成果とは数えない。

## 全15箇所の比較

変更画素数は診断値。精度・汚れ除去率ではない。

| 個別比較PNG | npj変更画素 | v6変更画素 | 暫定目視 |
| --- | ---: | ---: | --- |
'''+table+'''

## 判断と次の検証

この条件では、編集候補に実筆画を入れるとDDRMでも筆画の形が変わり、v6で編集を避ける意義がある。一方、保護を広げたままでは汚れ・穴も残る。編集候補に戻せた汚れにも境界や塗り跡が生じるため、分類と復元の両方に課題がある。v6の全面置換や新手法の有効性はまだ示していない。

次は本文・汚れ・損傷が接する固定例の分離を優先し、薄い輪状筆画・払い・赤褐色筆画・通常本文を対照として残す。復元側の調整を試す場合はこの20段・同じマスクの結果を基準として保存し、マスクを同時に変更しない。これらの例で調整したら開発例と明記し、別冊の未使用ページで確認する。

## 保存と検証

native/<id>に原画像、DDRM生出力、最終出力、編集マスク、変更マスクを保存。入力・重み・実装のSHA256、マスクの原寸一致、編集範囲外の不変、同じマスク・同じ乱数での完全再現、SVDの画素対応、PNGとZIPを確認した。[実行条件](manifest.json)、[実行検証](verification.json)、[成果物検証](final_verification.json)、[個別所見](visual_review.json)。

実行：`/home/imoto/miniconda3/envs/diffusion/bin/python -B scripts/benchmark/compare_npj_v6_ddrm_edits.py`

比較画像のまとめ：`/home/imoto/miniconda3/envs/diffusion/bin/python -B scripts/benchmark/package_npj_v6_ddrm_comparison.py`
'''
    (OUT/'report.md').write_text(report)
    (OUT/'README.md').write_text('''# 実際のDDRM後のnpj・v6比較

原画像／npj編集候補でDDRM／固定v6編集候補でDDRMの順でPNGを作成。

分類マスクの重ね描き、Telea補間、元npj全工程の再現と区別する。条件と所見は[report.md](report.md)。画像は4枚のまとめと15枚の個別比較。ZIPはnpj_v6_ddrm_comparison_images.zip。
''')
    dump(OUT/'final_verification.json',dict(status='passed',pages=11,regions=15,
        input_code_weight_hashes=True,native_output_mask_changes=True,outside_edit_changes=0,
        archive_integrity=True,archive_pngs=19,primary_artifact='PNG',
        package_source_sha256=sha(Path(__file__)),archive_sha256=sha(archive),
        report_sha256=sha(OUT/'report.md'),review_sha256=sha(OUT/'visual_review.json'),
        overview_hashes={name:sha(OUT/name) for name in GROUPS},
        pixel_GT=False,historical_npj_pipeline_reproduction=False))
    print('DDRM PNGs packaged and verified: 4 overview + 15 individual images.')


if __name__=='__main__': main()
