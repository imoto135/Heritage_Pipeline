"""Native, integer-nearest zooms of fixed-mask DDRM stroke comparisons."""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import package_npj_v6_ddrm_comparison as previous
from build_npj_v6_comparison import ROOT, FONT_PATH, dump, sha

OUT = ROOT / 'tmp_work/stroke_preservation_focus_20261002'
SPECS = [
    dict(name='01_black_stroke_loss', case='thin_black_loop', crop=[40, 255, 150, 135],
         title='黒筆画の欠落が大きい箇所',
         note='中央の折れ・右へ伸びる筆画を比較。v6列は原画素を保持。'),
    dict(name='02_faint_loop_breaks', case='thin_black_loop', crop=[35, 110, 140, 145],
         title='薄いループの途切れ',
         note='左辺・下側・右上の細い線の連続性を比較。文字の読みは未確定。'),
    dict(name='03_warm_long_stroke', case='warm_stroke', crop=[40, 170, 305, 270],
         title='赤褐色の長い払い',
         note='左下へ伸びる線の端・途中の欠落と、周辺の灰色生成跡を比較。'),
    dict(name='04_small_difference_control', case='faint_left_text', crop=[100, 100, 120, 160],
         title='差が小さい別ページも保持',
         note='この例ではnpjでも細い線が多く残る。明瞭な例だけで改善率を決めない。'),
]


def read(path, mask=False):
    value = cv2.imread(str(path), 0 if mask else 1)
    assert value is not None, path
    return value > 0 if mask else value


def main():
    assert not (OUT / 'complete.json').exists(), 'Frozen delivery already exists'
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {r['id']: r for r in previous.verify()}
    all_verified = json.loads((previous.OUT / 'verification.json').read_text())
    repeat = json.loads((previous.OUT / 'repeat_verification.json').read_text())
    operator = json.loads((previous.OUT / 'operator_verification.json').read_text())
    dependencies = {}
    records = []
    font = ImageFont.truetype(FONT_PATH, 24)
    small = ImageFont.truetype(FONT_PATH, 20)
    for spec in SPECS:
        row = rows[spec['case']]
        source = previous.OUT / 'native' / spec['case']
        x, y, w, h = spec['crop']
        scale = min(4, max(1, 660 // w))
        s = np.s_[y:y+h, x:x+w]
        images = {}
        for name in ['original', 'npj', 'v6']:
            p = source / (name + '.png')
            dependencies[str(p.relative_to(ROOT))] = sha(p)
            images[name] = read(p)[s].copy()
            assert images[name].shape == (h, w, 3)
            target = OUT / 'native' / spec['name'] / (name + '.png')
            target.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(target), images[name])
        changed = {}
        for name in ['npj', 'v6']:
            mask_path = source / (name + '_edit.png')
            dependencies[str(mask_path.relative_to(ROOT))] = sha(mask_path)
            edit = read(mask_path, True)[s]
            difference = np.any(images[name] != images['original'], axis=2)
            assert not (difference & ~edit).any()
            changed[name] = int(difference.sum())
            assert cv2.imwrite(str(OUT / 'native' / spec['name'] / (name + '_edit.png')), edit.astype('uint8') * 255)
        baseline_folder = 'npj_failure_followup_20261001' if spec['case'] == 'faint_left_text' else 'npj_failure_collection_20261001'
        extra_path = ROOT / 'tmp_work' / baseline_folder / 'v6_masks' / row['page'] / 'extra.png'
        dependencies[str(extra_path.relative_to(ROOT))] = sha(extra_path)
        rx, ry, rw, rh = row['xywh']
        extra = read(extra_path, True)[ry:ry+rh, rx:rx+rw][s]
        if spec['case'] == 'thin_black_loop':
            assert changed['v6'] == 0 and not extra.any()
        pad, gap, top, bottom = 20, 20, 110, 120
        panel_w, panel_h = w * scale, h * scale
        canvas = Image.new('RGB', (pad * 2 + panel_w * 3 + gap * 2, top + panel_h + bottom), '#faf8f2')
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 12), spec['title'] + ' / ' + row['page'], font=font, fill='black')
        draw.text((pad, 45), f'保存DDRM出力の同位置を{scale}倍表示（最近傍・色補正なし）', font=small, fill='black')
        labels = ['原画像', 'npj編集候補 → 同じDDRM', 'v6編集候補 → 同じDDRM']
        panel_boxes = []
        for i, (name, label) in enumerate(zip(images, labels)):
            px = pad + i * (panel_w + gap)
            draw.text((px, 78), label, font=small, fill='black')
            rgb = cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb).resize((panel_w, panel_h), Image.Resampling.NEAREST)
            canvas.paste(im, (px, top))
            panel_boxes.append([px, top, panel_w, panel_h])
        draw.text((pad, top + panel_h + 12), spec['note'], font=small, fill='black')
        draw.text((pad, top + panel_h + 42), f'この切り出しの変更画素：npj {changed["npj"]:,} / v6 {changed["v6"]:,}（筆画GTの数ではない）', font=small, fill='black')
        draw.text((pad, top + panel_h + 72), '重み・条件・空間ノイズは固定。編集候補の比較で、元npj全工程の再現ではありません。', font=small, fill='black')
        output = OUT / (spec['name'] + '.png')
        canvas.save(output)
        # Check saved comparison pixels with independent array repetition.
        saved = np.asarray(Image.open(output).convert('RGB'))
        for name, box in zip(images, panel_boxes):
            px, py, pw, ph = box
            expected = np.repeat(np.repeat(images[name][..., ::-1], scale, axis=0), scale, axis=1)
            assert np.array_equal(saved[py:py+ph, px:px+pw], expected)
        records.append(dict(**spec, page=row['page'], parent_xywh=row['xywh'],
                            page_xywh=[rx+x, ry+y, w, h], scale=scale,
                            changed_pixels=changed, v6_extra_hold_pixels=int(extra.sum()),
                            v6_exact_original=bool(np.array_equal(images['v6'], images['original'])),
                            panel_pixels_exact_integer_nearest=True))
    dependencies[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    for filename in ['manifest.json', 'complete.json', 'verification.json', 'repeat_verification.json', 'operator_verification.json', 'visual_review.json']:
        p = previous.OUT / filename
        dependencies[str(p.relative_to(ROOT))] = sha(p)
    dump(OUT / 'verification.json', dict(status='passed', inherited_all_15_roi_checks=True,
         records=records, execution_verification=all_verified, repeat_verification=repeat,
         operator_verification=operator, pixel_GT=False, human_stroke_confirmation='pending',
         historical_npj_pipeline_reproduction=False, new_diffusion_inference=False))
    dump(OUT / 'manifest.json', dict(dependencies=dependencies, source_sha256=sha(Path(__file__)),
         selection='Existing output reviewed to select easy-to-see illustrative zooms; not independent evaluation',
         mask_and_model_changes=False, settings=json.loads((previous.OUT / 'manifest.json').read_text())['settings']))
    report = '''# 筆画保持の差を先に確認する比較

原画像／保存旧npjの編集候補でDDRM／固定v6の編集候補でDDRMを、同じ位置・同じ整数倍率・最近傍で拡大した。新しい推論や条件調整はない。

1. [黒筆画の欠落](01_black_stroke_loss.png)：中央の折れと右へ伸びる筆画の欠落が目立つ。v6は原画素と完全一致。
2. [薄いループ](02_faint_loop_breaks.png)：線の途切れを比較する。v6は原画素と完全一致。
3. [赤褐色の長い払い](03_warm_long_stroke.png)：長い線の端・途中と、周囲の灰色生成跡を比較。
4. [差の小さい例](04_small_difference_control.png)：別ページではnpjでも細い線が多く残る。

01と02は同じページ・同じ親ROIであり、独立した2例として集計しない。出力を見て見やすい箇所を選んだ提示用画像で、未知資料の評価ではない。各列の切り出し元、座標、変更画素、追加暖色保留をverification.jsonに記録した。

## この比較で確かめたい有効性

まず、原画像にある比較箇所が実筆画であり、npj列の欠落・変形を避ける必要があるかを利用者に確認する。確認できれば「この固定条件で、実筆画を編集候補から外すことが原形保持に有効」という局所的な根拠になる。v6が原画素を保持すること自体は画素一致で確認済み。失われた文字を復元した成果ではない。

黒筆画01・02にはv6固有の追加暖色保留が0で、改善はv6以前の保護処理に由来する。暖色03は追加保留が作用するが、旧npjとの差には土台の処理も含まれるため、v6固有の寄与の確定にはv2との比較が別に必要。

この実験は旧npjの04_degradationを欠損へ渡す、編集マスクを統一したDDRM比較。元npj全工程・黒文字再生成経路への優位は確定しない。ACEとCSCの寄与もこの比較では分離していない。

## 次の検証の判断基準

筆画保持の確認を優先し、背景の馴染み調整は混ぜない。少数の確認済み筆画の範囲で誤編集と出力変化を記録する。v2とv6を分け、固定条件で別冊の薄い筆画・暖色筆記と汚れ対照を選ぶ。本文の保持と汚れを余分に残す傾向を同時に確認し、処理が動かない例や変更画素数だけを全体の精度としない。

保存済み11ページ15ROI全体の入力・コード・重み・出力ハッシュ、原画像の全ページからの切り出し一致、編集外一致を再確認した。新PNGの各パネルは元画素の整数反復と完全一致、ZIPの全画素を確認した。人の筆画確認は未実施。
'''
    (OUT / 'report.md').write_text(report)
    shutil.copyfile(__file__, OUT / 'source_snapshot.py')
    archive = OUT / 'comparison_pngs.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png')):
            z.write(p, p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == 4
    dump(OUT / 'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print(json.dumps(records, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
