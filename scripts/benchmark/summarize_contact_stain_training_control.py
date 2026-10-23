"""Save the exploratory union readout and a compact Japanese research note."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
from run_contact_stain_training_control import OUT
from compare_v5_aligned_fade_control import sha, dump, rgb, mask, save
import apply_ragan_real_regions as real


def group(rows, case=None):
    return {name: [r for r in rows if r['method'] == name and (case is None or r.get('case') == case)]
            for name in sorted({r['method'] for r in rows})}


def pair(rows, core='source_core_r2', core_n='source_core_n', paper='source_remote_paper_ink', paper_n='source_remote_paper_n'):
    return f"{sum(r[core] for r in rows)}/{sum(r[core_n] for r in rows)}・{sum(r[paper] for r in rows)}/{sum(r[paper_n] for r in rows)}"


def main():
    first = json.loads((OUT / 'evaluation/results.json').read_text())
    later = json.loads((OUT / 'howe_ragan_union_transfer/results.json').read_text())
    real_result = json.loads((OUT / 'real_transfer/results.json').read_text())
    dest = OUT / 'real_transfer/union_exploratory'
    assert not (dest / 'results.json').exists()
    scoring_spec = importlib.util.spec_from_file_location('contact_union_partial_scoring', ROOT / 'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring = importlib.util.module_from_spec(scoring_spec)
    sys.modules[scoring_spec.name] = scoring
    scoring_spec.loader.exec_module(scoring)
    part = ROOT / 'tmp_work/classical_partial_labels_20261021'
    regions = json.loads((part / 'source_regions.json').read_text())
    annotations = {r['id']: r for r in json.loads((part / 'annotations_source_only.json').read_text())['regions']}
    cases = [c for c in json.loads((ROOT / 'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']
             if c['id'] in ('D01', 'D02', 'D03', 'L07')]
    union_rows = []
    for case in cases:
        x, y, w, h = case['roi']
        howe = mask(ROOT / 'tmp_work/classical_limit_audit_20261021/crops' / case['id'] / 'howe_fixed.png')
        ragan = mask(OUT / 'real_transfer/masks' / case['id'] / 'ragan_contact.png')
        union = howe | ragan
        dest_mask = dest / 'masks' / f'{case["id"]}.png'
        save(dest_mask, union)
        source = rgb(ROOT / case['source'])[y:y+h, x:x+w]
        panel = dest / 'figures' / f'{case["id"]}_原本比較.png'
        real.panel(panel, case['id'] + ' / 探索的なHoweとRA-GANの和集合', [
            ('原画像', source), ('Howe', np.where(howe, 0, 255).astype(np.uint8)),
            ('RA-GAN 接触汚れ学習', np.where(ragan, 0, 255).astype(np.uint8)),
            ('Howe ∪ RA-GAN', np.where(union, 0, 255).astype(np.uint8))])
        for region in [r for r in regions if r['case'] == case['id']]:
            rx, ry, rw, rh = region['roi']
            labels, paths = scoring.raster(region, annotations[region['id']])
            local = union[ry-y:ry-y+rh, rx-x:rx-x+rw]
            union_rows.append(dict(case=case['id'], region=region['id'], method='howe_or_ragan_contact',
                prediction_sha256=sha(dest_mask), **scoring.measure(local, labels, paths)))
    dump(dest / 'results.json', dict(rows=union_rows, posthoc=True, known_development_examples=True,
        provisional_partial_source_labels=True, formal_accuracy=False))
    for r in union_rows:
        print(r['region'], 'union', r['ink_r2'], '/', r['ink_n'], 'nonink',
              r['nonink_false'], '/', r['nonink_n'], flush=True)
    def real_line(region, method):
        row = next((r for r in real_result['rows'] + union_rows if r['region'] == region and r['method'] == method))
        return f"{row['ink_r2']}/{row['ink_n']}・{row['nonink_false']}/{row['nonink_n']}"
    primary = group(first['rows'], 'contact_stain')
    transfer = group(later['rows'])
    note = f'''# 文字接触汚れ学習の対照実験（探索）

2026-10-22。結果は**単一本からの学習・同一本内の検証**、原本由来の暗部骨格と遠方紙面による代理評価である。専門家GTではなく、DKDSの印章合成でもない。RA-GANは公開実装に整合させた432更新のパイロットで、論文値の再現ではない。

## 入力と学習

96学習パッチ×3強度の288枚だけを、従来の茶色半透明汚れから暗い乗算汚れへ入れ替えた。教師・他576枚の入力・学習回数・初期値・切り出し乱数・検証入力は各モデルの旧/新条件で一致する。新汚れは全パッチで原本暗部骨格と紙面に重なる。従来の汚れも強度0.75では96枚すべてで暗部骨格と交差していたため、**「旧汚れは文字に当たらなかった」とは言えない**。新旧の差は汚れ色・合成式・配置と強度の組合せで、各因子の独立効果は未分離。[入力監査](input_audit.json)、[旧汚れ監査](original_stain_overlap_audit.json)、[実験条件](protocol.json)。

## 未学習ページ群の合成接触汚れ

値は「汚れに重なった原本暗部骨格の半径2画素被覆 / 対象骨格・汚れ内の遠方紙面を黒判定 / 対象紙面」。最初の2パッチ（00171と00283のp00）は学習前に固定した。和集合はこの結果とD03を見てから追加した探索結果である。

| 方法 | 最初の2パッチ |
|---|---:|
| Howe | {pair(primary['howe_fixed'])} |
| Sauvola | {pair(primary['sauvola_w61_k020'])} |
| U-Net 旧 q≥0.5 | {pair(primary['unet_original_q50'])} |
| U-Net 新 q≥0.5 | {pair(primary['unet_contact_q50'])} |
| RA-GAN 旧 q≥0.5 | {pair(primary['ragan_original_q50'])} |
| RA-GAN 新 q≥0.5 | {pair(primary['ragan_contact_q50'])} |

同じ2ページ群の追加6パッチを、和集合の規則・閾値を変えずに確認した。ただし**ページ群は独立ではない**。

| 方法 | 追加6パッチ |
|---|---:|
| Howe | {pair(transfer['howe_fixed'])} |
| RA-GAN 新 | {pair(transfer['ragan_contact_q50'])} |
| Howe ∪ RA-GAN 新 | {pair(transfer['howe_or_ragan_contact_q50'])} |

[最初の2例の原本比較](evaluation/figures/200017197_00171_1_p00_contact_stain_原本比較.png)、[追加6例の一覧](howe_ragan_union_transfer/results.json)。RA-GAN新条件は合成汚れで墨線を多く保ち、旧条件の「汚れ全面を黒」の挙動を大幅に減らした。ただし紙の誤黒化は残る。和集合はHowe単体より線を拾うが、RA-GANの誤黒化も継承する。

## 既知の実損傷への移行

値は原本のみから作成済みの高確信筆画中心の半径2画素被覆・選択済み非文字内部の黒判定。注釈の外にある誤検出は数えない。D03はすでに開発に利用した既知ページ。

| D03領域 | Howe | U-Net新 | RA-GAN新 | Howe ∪ RA-GAN新 |
|---|---:|---:|---:|---:|
| 上側の接触 | {real_line('contact_upper','howe_fixed')} | {real_line('contact_upper','unet_contact')} | {real_line('contact_upper','ragan_contact')} | {real_line('contact_upper','howe_or_ragan_contact')} |
| 下側の接触 | {real_line('contact','howe_fixed')} | {real_line('contact','unet_contact')} | {real_line('contact','ragan_contact')} | {real_line('contact','howe_or_ragan_contact')} |

[D03の原本比較](real_transfer/figures/D03_原本比較.png)、[和集合の原本比較](real_transfer/union_exploratory/figures/D03_原本比較.png)、[全実損傷の暫定測定](real_transfer/results.json)。RA-GAN新条件の単体は上側の実文字を削る。和集合は既存の部分注釈では補えるが、**非注釈の染み縁などに偽陽性が残る可能性**を原本比較で確認する必要がある。D01の薄文字はHowe 181/225に対しRA-GAN新50/225であり、現時点で置換できない。

## 次の判定条件

現状は合成接触汚れの改善が明確で、実損傷での全面的な改善は未証明。次にすべきは、(1) 同一原本から専門家確認の墨芯・紙・不明マスクを作る、(2) 別本の**未使用**接触汚れ・退色を評価前に固定する、(3) Howe・RA-GAN単体・和集合を閾値固定で測る、(4) 汚れ色/配置/強度を一因子ずつ変えることである。薄文字と染みを同時に評価し、改善が一方の犠牲ならモデル採用を保留する。
'''
    (OUT / '研究結果_文字接触汚れ学習_20261022.md').write_text(note)
    print('wrote research note', flush=True)


if __name__ == '__main__':
    main()
