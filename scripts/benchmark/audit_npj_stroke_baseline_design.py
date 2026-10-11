"""Audit a diagnostic inpainting adapter against local npj reconstruction."""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_npj_v6_comparison import ROOT, FONT_PATH, dump, sha
import package_npj_v6_ddrm_comparison as adapter

OUT = ROOT / 'tmp_work/npj_baseline_design_audit_20261002'
PAGE = '200010454_00002_2'
OLD = ROOT / 'data/split_dataset_csc_mask/test/200010454' / PAGE
ORIGINAL = ROOT / 'data/split_dataset/test/200010454' / (PAGE + '.jpg')
HISTORICAL = ROOT / 'tmp_work/speed_test_single/ddrm_out_v2fix' / PAGE / 'result' / (PAGE + '.png')
HISTORICAL_INPUT = ROOT / 'tmp_work/speed_test_single/input' / (PAGE + '.jpg')
LOG = ROOT / 'tmp_work/speed_test_single/ddrm_run_v2fix.log'
FOCI = [('01_black_stroke_baseline_check', [1330, 855, 150, 135]),
        ('02_faint_loop_baseline_check', [1325, 710, 140, 145])]


def read(path, flag=1):
    a = cv2.imread(str(path), flag)
    assert a is not None, path
    return a


def figure(images, labels, path, notes):
    h, w = images[0].shape[:2]
    assert all(im.shape == images[0].shape for im in images)
    scale = 4
    pw, ph = w * scale, h * scale
    top, gap, pad = 115, 20, 20
    canvas = Image.new('RGB', (len(images) * pw + (len(images) - 1) * gap + 2 * pad, ph + 245), '#faf8f2')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, 24)
    small = ImageFont.truetype(FONT_PATH, 20)
    draw.text((pad, 12), '比較設計の監査：同じ箇所で入力と処理を照合', font=font, fill='black')
    draw.text((pad, 48), PAGE + ' / 倍率4・最近傍', font=small, fill='black')
    for i, (im, label) in enumerate(zip(images, labels)):
        px = pad + i * (pw + gap)
        draw.text((px, 83), label, font=small, fill='black')
        canvas.paste(Image.fromarray(im[..., ::-1]).resize((pw, ph), Image.Resampling.NEAREST), (px, top))
    for i, note in enumerate(notes):
        draw.text((pad, ph + 130 + i * 33), note, font=small, fill='black')
    canvas.save(path)
    saved = np.asarray(Image.open(path).convert('RGB'))
    for i, im in enumerate(images):
        px = pad + i * (pw + gap)
        expected = np.repeat(np.repeat(im[..., ::-1], scale, axis=0), scale, axis=1)
        assert np.array_equal(saved[top:top+ph, px:px+pw], expected)


def overlay(image, mask, bgr):
    result = image.copy()
    result[mask] = np.rint(image[mask] * .45 + np.asarray(bgr) * .55).astype('uint8')
    return result


def main():
    assert not (OUT / 'complete.json').exists(), 'Audit already frozen'
    OUT.mkdir(parents=True, exist_ok=True)
    adapter.verify()
    original = read(ORIGINAL)
    assert np.array_equal(read(HISTORICAL_INPUT), original)
    old_diagnostic = read(adapter.OUT / 'native/thin_black_loop/npj.png')
    saved = read(HISTORICAL)
    assert saved.shape == original.shape
    log = LOG.read_text()
    command = next(line.strip() for line in log.splitlines() if 'Command being timed:' in line)
    assert 'split_dataset_csc_mask_v2/test/200010454' in command
    assert '--back_ground ./background/isemonogatari.png' in command
    assert '/bg_patch.png' in log
    settings = json.loads((adapter.OUT / 'manifest.json').read_text())['settings']
    assert settings['masking'] is False
    assert settings['historical_npj_pipeline_reproduction'] is False
    assert '04_degradation' in settings['baseline_missing']
    assert settings['outside_edit'] == 'exact original pixels after raw DDRM blend'
    sources = [
        'modules/ddrm_npj/scripts/pipeline_ddrm.py',
        'modules/ddrm_npj/scripts/pipeline_ddrm_page.py',
        'modules/ddrm_npj/scripts/generate_mask_csc.py',
        'modules/ddrm_npj/codes/Inpainting.py',
        'modules/ddrm_npj/codes/OrigDataset.py',
        'modules/ddrm_npj/functions/denoising.py',
        'modules/ddrm_npj_v2/scripts/pipeline_ddrm.py',
        'scripts/benchmark/compare_npj_v6_ddrm_edits.py',
    ]
    local_pipeline = (ROOT / sources[0]).read_text()
    assert "'02_text_black.png'" in local_pipeline
    assert 'self.masking = True' in local_pipeline
    assert 'out_bgr[red_mask] = img_bgr[red_mask]' in local_pipeline
    local_inference = (ROOT / sources[3]).read_text()
    assert 'path = args.back_ground' in local_inference
    assert 'test_dataset = SingleDataset(' in local_inference
    assert 'y_0 = H_funcs.H(x_orig)' in local_inference
    local_kernel = (ROOT / sources[5]).read_text()
    assert 'if args.masking:' in local_kernel
    assert 'xt = xt*mask.to(args.device)' in local_kernel
    masks = {'black': read(OLD / '02_text_black.png', 0) < 127,
             'red': read(OLD / '03_text_red.png', 0) < 127,
             'degradation': read(OLD / '04_degradation.png', 0) > 0,
             'background': read(OLD / '05_background.png', 0) > 0}
    assert np.all(sum(a.astype('uint8') for a in masks.values()) == 1)
    v6edit_path = ROOT / 'tmp_work/npj_failure_collection_20261001/v6_masks' / PAGE / 'edit.png'
    v6edit = read(v6edit_path, 0) > 0
    dependencies = {str(p.relative_to(ROOT)): sha(p) for p in
                    [ORIGINAL, HISTORICAL, HISTORICAL_INPUT, LOG, v6edit_path,
                     adapter.OUT / 'native/thin_black_loop/npj.png', adapter.OUT / 'manifest.json',
                     Path(__file__)] + [ROOT / p for p in sources] + list(OLD.glob('*.png'))}
    rows = []
    for name, (x, y, w, h) in FOCI:
        s = np.s_[y:y+h, x:x+w]
        diagnostic = old_diagnostic[y-600:y-600+h, x-1290:x-1290+w]
        original_crop = original[s]
        figure([original_crop, diagnostic, saved[s]],
               ['原画像', '前の診断：劣化マスク→穴埋め', '保存DDRM出力（CSC v2マスク）'], OUT / (name + '.png'),
               ['中央は通常のnpj工程ではない。右も別条件の保存出力で、旧v1再現ではない。',
                '右のログ：黒文字マスク／背景参照。元入力は原画像と全画素一致。',
                'この3列を優劣評価に使わず、前の診断と通常処理の取り違えを確認する。'])
        figure([original_crop, overlay(original_crop, masks['black'][s], (255, 100, 20)),
                overlay(original_crop, masks['degradation'][s], (0, 220, 255)),
                overlay(original_crop, v6edit[s], (0, 220, 255))],
               ['原画像', '保存旧v1：黒文字候補（青）', '保存旧v1：劣化候補（黄）', '固定v6：編集候補（黄）'],
               OUT / (name + '_mask_roles.png'),
               ['通常のnpjは黒文字候補を文字再生成へ渡す。前の診断は劣化候補を穴埋めへ渡した。',
                '分類候補は筆画GTではない。範囲の意味と復元への接続は別に確認する。',
                'v6編集0の原画像一致だけで、文字再生成を行う通常のnpjへの優位は確定しない。'])
        rows.append(dict(name=name, page_xywh=[x,y,w,h],
                         old_v1_classes={k:int(a[s].sum()) for k,a in masks.items()},
                         v6_edit_pixels=int(v6edit[s].sum()), saved_input_exact_original=True,
                         saved_output_method='Historical local DDRM with CSC v2 masks; not v1',
                         saved_run_command=command, saved_run_source_snapshot_available=False))
    report = '''# npj筆画比較の設計監査と結論の訂正

2026-10-02。利用者の「npjでここまで掠れていなかった気がする」「設計は正しい？」を受けて監査した。

**前に提示した中央列は通常のnpj出力ではない。編集マスクの診断として計算の整合性は確認しているが、通常のnpjに対する筆画保持の有効性を確定する比較設計にはなっていない。前回答の「明瞭な根拠」は、この実験の範囲を越えて読める説明だったため訂正する。**

## 処理の違い

| 要素 | 現在のローカルnpj通常工程 | 前に提示した編集マスク診断 |
| --- | --- | --- |
| 渡したマスク | 02_text_blackの黒文字候補を文字再生成対象へ接続 | 04_degradationの劣化候補を欠損へ接続 |
| 途中のマスク処理 | masking=True、時刻50以上でxtへマスクを掛ける | masking=False |
| 観測の参照 | args.back_groundをSingleDatasetへ渡す。保存ログは切り出したbg_patchを記録 | 同じ原画像をRGB[-1,1]へ変換して観測する |
| 合成 | パッチ全出力を合成し、朱候補を原画素から戻す。ページ段階は別に存在する | 編集範囲内だけ生成を採用し、範囲外は原画素へ戻す。ページ段階なし |
| 比較の意味 | マスクに基づく文字再生成を含む | 編集候補に筆画を入れたときの穴埋め挙動 |

データの範囲も異なる。ローカルSingleDataset(train=False)はToTensorで[0,1]を渡し、sample_sequence内のdata_transformはコメントアウトされている。診断は[-1,1]を使用した。これはモデルの学習範囲に合わせた診断設計だが、従来条件の再現にはならない。どちらの範囲が推論として妥当かの検証と、従来出力の再現を同時に変更しない。

同じ重み・段数を使っていても、マスクの意味、観測、途中処理、合成を変更すると通常のnpjとの比較にはならない。v6の編集候補が0の切り出しは原画像と必ず一致するため、再生成を行う通常方式との優位をその一致だけで決めない。

## 保存出力との照合

[黒筆画の保存出力照合](01_black_stroke_baseline_check.png)、[薄いループの照合](02_faint_loop_baseline_check.png)、[黒筆画のマスク用途](01_black_stroke_baseline_check_mask_roles.png)、[ループのマスク用途](02_faint_loop_baseline_check_mask_roles.png)。

同じページの過去のddrm_out_v2fix出力を見つけ、入力がdata/split_datasetの原画像と全画素一致すること、出力の寸法、ログのmask_dir（CSC v2）・背景指定・bg_patch読込を確認した。保存出力は前の診断と見た目が異なり、こちらにも筆画の欠けがある。したがって「通常のnpjならこのページは欠けない」とも、この保存結果だけでは断定しない。CSC v2マスクの別条件出力であり、旧v1の忠実再現や、利用者が覚えている出力と同じものという保証はない。過去の実行ソースのスナップショットも確認できていない。

## 確認結果と判断の修正

- 前の診断に誤ったマスク反転や表示のすり替えは見つからなかった。11ページ15ROIの保存入力・出力・マスク・編集外一致を再確認した。
- 問題は、通常のnpjとは異なる用途の診断を、筆画保持の優位性を確定する根拠へ使ったこと。通常のnpjに対する有効性は未確定に戻す。
- 旧分類で筆画が劣化候補に入ることは独立した前処理の課題候補として残す。候補分類の誤りと、元npj最終出力の文字消失を同一視しない。
- v6黒筆画の例は以前の保護処理に由来する。v6固有の追加暖色保留の効果とも分ける。

## 次の公平な比較

利用者が念頭に置くnpj出力の版・工程・参照・合成を特定し、その条件で保存出力を再現する。元画像と出力は同じ座標・倍率・無補正で比較し、パッチだけ／ページ段階後／追加合成後を混ぜない。

前処理の差を評価するときは、同じ意味の文字・朱・保護／背景／劣化を定義して同じ通常復元工程へ渡す。旧npj、既存保護v2、追加暖色保留v6を分け、観測や合成は固定する。v6のprotectは材質未確定の保護も含むため、すべてを黒文字再生成マスクへそのまま変換しない。対応を確認した少数筆画と汚れ対照で、文字の欠け・生成変形・汚れ残りを別々に見る。

背景馴染みや新しい輪郭条件の探索より、このベースラインの整備と有効性確認を優先する。今回、新しい拡散推論や既存コード変更は行っていない。元の凍結比較は削除・書換えず、監査結果を別に保存した。
'''
    (OUT / 'report.md').write_text(report)
    dump(OUT / 'verification.json', dict(status='passed', design_suitable_for_historical_npj_superiority=False,
         prior_diagnostic_integrity=True, current_local_pipeline_source_checks=True,
         historical_input_exact_original=True, historical_saved_mask_version='CSC v2',
         historical_run_fully_reproduced=False, normal_npj_vs_v6_effectiveness='unconfirmed',
         records=rows, all_comparison_panels_exact=True, new_inference=False))
    dump(OUT / 'manifest.json', dict(dependencies=dependencies, source_sha256=sha(Path(__file__)),
         correction='Prior edit-mask diagnosis is not a historical npj superiority test'))
    shutil.copyfile(__file__, OUT / 'source_snapshot.py')
    with zipfile.ZipFile(OUT / 'comparison_pngs.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')):
            z.write(p, p.name)
    with zipfile.ZipFile(OUT / 'comparison_pngs.zip') as z:
        assert z.testzip() is None and len(z.namelist()) == 4
    dump(OUT / 'complete.json', dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
