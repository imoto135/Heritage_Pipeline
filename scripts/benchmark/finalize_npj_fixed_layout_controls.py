"""Package the fixed original-only layout controls after visual review."""
import json
from pathlib import Path
import re
import shutil
import zipfile

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, read, sha, dump

OUT=ROOT/'tmp_work/npj_layout_fixed_screening_20261004'


def main():
    assert not (OUT/'complete.json').exists(), 'Already frozen'
    selection=json.loads((OUT/'selection.json').read_text())
    for rel,digest in selection['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(OUT/'source_snapshot.py')==selection['dependencies']['scripts/benchmark/check_npj_fixed_layout_controls.py']
    cases=json.loads((OUT/'diagnostics.json').read_text())['cases']
    assert sum(c['layout']['chosen'] is None for c in cases)==4
    heading=next(c for c in cases if c['page']=='200008316_00020_1')
    assert heading['heading_witness']['dark_pixels_outside_envelope']==0
    checks=[]
    for c in selection['cases']:
        original=read(ROOT/c['original']);H,W=original.shape[:2]
        ratio=min(540/W,900/H,1);dw,dh=round(W*ratio),round(H*ratio)
        canvas=read(OUT/(c['page']+'_layout.png'))
        shown=cv2.resize(original,(dw,dh),interpolation=cv2.INTER_AREA)
        assert np.array_equal(canvas[120:120+dh,24:24+dw],shown)
        native=OUT/'native'/c['page']
        keep=read(native/'interior.png',True);outer=read(native/'exterior.png',True)
        assert not (keep&outer).any()
        record=next(r for r in cases if r['page']==c['page'])
        assert int(outer.sum())==record['exterior_pixels']
        checks.append(dict(page=c['page'],original_panel_exact=True,
                           native_masks_disjoint=True,exterior_count_exact=True))
    dump(OUT/'visual_review.json',dict(
        review='All five complete original/layout panels and the heading witness reviewed',
        observations={
          '200006665_00047_2':'Real text page without complete enclosing rules; detector abstains',
          '200005598_00013_1':'Thick nested frame touches page boundary; detector abstains',
          '200022050_00012_1':'Dense text on brown paper with near-edge rules; detector abstains; paper material uncertain',
          '200008316_00020_1':'Observed heading above body frame; detector abstains, heading stays within full keep',
          '200021712_00042_1':'Enclosing main table is supported; dense binding-side content remains materially uncertain'},
        implication='Four abstentions limit coverage; one fitted table does not prove safe dirt editing',
        full_page_GT=False,no_output_repair_or_success_claim=True))
    table=['| 原ページ | 対照 | 枠判定 |','| --- | --- | --- |']
    for c in cases:
        box=c['layout']['chosen']['box'] if c['layout']['chosen'] else None
        table.append(f"| [{c['page']}]({c['page']}_layout.png) | {c['role']} | {box or '条件不成立・見送り'} |")
    report=f'''# 固定した全ページ配置判定を追加5ページで確認

2026-10-04。開発8例の観測横線幅版を凍結して追加5ページに設定変更なしで適用した。4ページは条件不成立、1ページは表を囲む外包が成立。枠のない本文、ページ境界に近い二重枠、茶色い密な本文、枠外見出しで適用範囲が狭い。**枠判定だけに依存する汚れ除去を一般採用する根拠にはならない。**

{chr(10).join(table)}

[枠外に実際の見出しがある対照](heading_outside_frame_witness.png)は、枠が不成立なので全保持になった。見出し領域内の暗い画素が外包外へ出た数は0。見出しを認識できた証明ではなく、見送りによる保持。もし本文枠だけを信頼して枠外を全編集するなら、このような文字の存在を別に扱う必要がある。

同冊別ページ200021712の主要な表は保持外包に入る。ただし右側の密集像は綴じ・裏写り等との区別が未確定。青は「紙面内で本文外包の外」であって汚れ確定領域でも補修マスクでもない。

## 評価範囲

今回は原ページの配置診断のみ。対応する保存npjを代用せず、新規DDRM・補修・色変換は行っていない。汚れ除去結果や本文消失率を測定したものではない。5ページは今回の配置開発8例には含まれないが、以前の原画像走査／v6で見られているものを含む。未使用冊の独立精度検証とは呼ばない。役割と見出し領域を検出前に記録し、閾値変更なし。5ページとも原画像で文字ページと確認、イラストを除外。

原画像パネルの再照合、マスクの非重複、範囲数、凍結版と入力の依存ハッシュを確認した。内部整合は配置GTを意味しない。

## 次に確認する課題

虫食いの黒い縁を除去できた[局所比較](../npj_full_page_layout_v4_20261004/focus_wormhole_rim.png)は手掛かりとして残す。枠検出の閾値をこの5例に合わせて緩める段階には進まない。

次は実保存npjに残る、枠内の汚れと欠損縁を対象にする。原画像で観測できる紙の欠損／汚れ／実際の文字、保存npj上の残留、補修後の変化を別々に記録し、v6の黒保護・holdに入った汚れを選べるかを検証する。文字は保持の対照、補修対象は汚れ。材質がわからない綴じ・裏写りは成功例に数えない。

[全6比較PNG ZIP](comparison_pngs.zip)・[原画像5ページの走査](original_screening.png)・[固定条件と入力](selection.json)・[独立画素確認](independent_verification.json)・[目視所見](visual_review.json)。
'''
    (OUT/'report.md').write_text(report)
    dump(OUT/'independent_verification.json',dict(cases=checks,
         source_dependencies_unchanged=True,layout_abstentions=4,
         heading_outer_dark_pixels=0,no_inference_or_edit=True,
         no_damage_accuracy_or_success_rate_claim=True))
    for rel in re.findall(r'\]\(([^)]+)\)',report):
        if rel=='comparison_pngs.zip':continue
        assert (OUT/rel).exists(),rel
    shutil.copyfile(__file__,OUT/'finalizer_snapshot.py')
    pngs=sorted(p for p in OUT.glob('*.png') if p.name!='original_screening.png')
    assert len(pngs)==6
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,p.name)
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert z.testzip() is None
        for p in pngs:assert z.read(p.name)==p.read_bytes()
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p)
         for p in sorted(OUT.rglob('*')) if p.is_file()}))
    for rel,digest in json.loads((OUT/'complete.json').read_text())['outputs'].items():
        assert sha(OUT/rel)==digest,rel
    print('Frozen additional five pages, six comparison PNG; no restoration inference',flush=True)


if __name__=='__main__':main()
