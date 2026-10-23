"""Report every predeclared translation; no inference or training."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

import diagnose_binarization_translation as exp
from compare_v5_aligned_fade_control import dump, sha
from apply_ragan_real_regions import panel


def main():
    plan = exp.check()
    out = exp.OUT
    result = exp.read(out/'results.json')
    verification = exp.read(out/'verification.json')
    assert verification['results_sha256'] == sha(out/'results.json')
    rows = result['rows']
    # Keep source pixels at native resolution, add white canvas so Japanese
    # labels on the small120px stain crop do not overlap adjacent panels.
    for cp in plan['checkpoints']:
        for case in plan['cases']:
            n, k = cp['condition'], case['id']
            x0, y0, x1, y1 = case['figure_box']
            crop = np.s_[y0:y1, x0:x1]
            raw = np.asarray(Image.open(out/'inputs'/k/'original.png'))[crop]
            def padded(a):
                if a.ndim == 2:
                    a = np.repeat(a[:, :, None], 3, axis=2)
                h, w = a.shape[:2]
                b = np.full((max(256, h), max(256, w), 3), 255, np.uint8)
                b[:h, :w] = a
                assert np.array_equal(b[:h, :w], a)
                return b
            items = [('原画像', padded(raw))]
            for r in rows:
                if r['condition'] == n and r['case'] == k:
                    a = np.asarray(Image.open(exp.ROOT/r['mask_file']))[crop]
                    dx, dy = r['shift_xy']
                    items.append((f"x{dx} y{dy} 墨{r['core_r2']} 黒{r['paper_ink']}", padded(a)))
            title = k[:3] + (' 実タイル' if 'actual' in k else ' 中央512') + ' / ' + n.replace('ragan_', '')
            panel(out/'figures'/f'{n}_{k}.png', title, items)
    verification['comparison_figures'] = {f: sha(out/f) for f in result['figures']}
    verification['figure_renderer_sha256'] = sha(__file__)
    verification['figure_padding_only_no_resampling'] = True
    dump(out/'verification.json', verification)
    with (out/'全条件.csv').open('w', newline='') as f:
        fields = ['condition', 'case', 'shift_xy', 'core_n', 'core_exact', 'core_r2',
                  'paper_n', 'paper_ink', 'core_disagreement', 'paper_disagreement',
                  'selected_probability_max_abs', 'selected_probability_mean_abs']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    text = [
        '# 学習なし：入力移動による判定変動', '',
        '保存済み生成器4個を固定し、3入力×9配置＝108条件をCPUで推論した。'
        '重み更新、最適化器、GAN、追加注釈、閾値探索は実行していない。閾値は全条件0.5。', '',
        '## 何を変えたか', '',
        '同じ512角のRGB・Otsu・Sobelの5チャンネルを一緒に循環移動し、出力確率を逆移動して元の注釈座標で採点した。'
        '移動は原点、上下左右1画素、上下左右32画素の9種類。32は生成器の縮小率に合わせた値で、結果から選んでいない。'
        '補助画像は再計算せず、全入力値と各チャンネルの画素集合を維持した。'
        '注釈から半径32の正方形内では、5チャンネルの局所内容が移動前後で完全一致し、継ぎ目と端をまたがないことを検証した。', '',
        '循環移動は遠方の継ぎ目と画像端に接する内容も変える。'
        'したがって純粋な位置のみ、縮小格子のみ、特定層のみの原因とは同定しない。'
        '1画素と32画素の比較は、配置・縮小の位相・遠方境界への感度を調べる診断である。', '',
        'S03/S07は前の診断と同じ学習原画像で、未知画像の性能ではない。'
        'S03は中央512、S07は中央512と実タイル原点(1920,0)を使った。'
        'S03の暫定墨芯115点、各入力の非文字576画素は既存注釈をそのまま使用した。'
        '被覆r2は半径2以内の黒画素の存在を測り、字形の正解率ではない。', '',
        '## 全配置の結果', '',
        '表の範囲は全4方向の最小～最大であり、最良配置を採用・選択するものではない。'
        '「反転」は原点との二値判定の相違数。', '',
        '|モデル|入力|原点 墨r2/紙黒|1画素 墨r2範囲|1画素 紙黒範囲|32画素 墨r2範囲|32画素 紙黒範囲|',
        '|---|---|---|---:|---:|---:|---:|']
    groups = []
    for cp in plan['checkpoints']:
        for case in plan['cases']:
            n, k = cp['condition'], case['id']
            rr = [r for r in rows if r['condition'] == n and r['case'] == k]
            z = next(r for r in rr if r['shift_xy'] == [0, 0])
            one = [r for r in rr if sum(abs(x) for x in r['shift_xy']) == 1]
            aligned = [r for r in rr if sum(abs(x) for x in r['shift_xy']) == 32]
            def span(rs, key):
                a = [r[key] for r in rs]
                return f'{min(a)}–{max(a)}'
            text.append(f"|{n}|{k}|{z['core_r2']}/{z['paper_ink']}|{span(one,'core_r2')}|{span(one,'paper_ink')}|{span(aligned,'core_r2')}|{span(aligned,'paper_ink')}|")
            groups.append(dict(condition=n, case=k,
                one_pixel_mean_selected_disagreement=sum(r['selected_disagreement'] for r in one)/4,
                aligned32_mean_selected_disagreement=sum(r['selected_disagreement'] for r in aligned)/4))
    text.extend(['', '## 配置ごとの値と原寸PNG', ''])
    for cp in plan['checkpoints']:
        for case in plan['cases']:
            n, k = cp['condition'], case['id']
            text += [f'### {n} / {k}', '', f'[原画像＋全9出力](figures/{n}_{k}.png)', '',
                     '|移動(x,y)|墨芯厳密|墨芯r2|非文字黒化|墨芯反転|非文字反転|',
                     '|---|---:|---:|---:|---:|---:|']
            for r in rows:
                if r['condition'] == n and r['case'] == k:
                    text.append(f"|{r['shift_xy']}|{r['core_exact']}/{r['core_n']}|{r['core_r2']}/{r['core_n']}|{r['paper_ink']}/{r['paper_n']}|{r['core_disagreement']}|{r['paper_disagreement']}|")
            text.append('')
    text += ['## 検証', '',
             '108保存マスクを確率から再構成し、別の畳み込みによる半径2計算で再採点した。'
             '原点の全12条件は、前回の中央512または実タイル出力の保存確率と完全一致した。'
             '4モデルで同一入力の再推論も完全一致し、推論前後の全モデル状態ハッシュは不変。'
             'PNGは原画像を並べ、拡大補間を使わず原寸で保存した。', '',
             '[全条件CSV](全条件.csv)・[凍結計画](plan.json)・[検証記録](verification.json)。']
    (out/'比較結果.md').write_text('\n'.join(text)+'\n')
    dump(out/'group_summary.json', groups)
    snapshot = out/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for script in [Path(exp.__file__), Path(__file__)]:
        shutil.copy2(script, snapshot/script.name)
    dump(out/'report_check.json', dict(
        script_sha256=sha(__file__), inference_script_sha256=sha(exp.__file__),
        documents={name: sha(out/name) for name in ['比較結果.md','全条件.csv','group_summary.json']},
        verification_sha256=sha(out/'verification.json'), no_training=True))
    print('Wrote all-condition report and12 original-image comparisons')


if __name__ == '__main__':
    main()
