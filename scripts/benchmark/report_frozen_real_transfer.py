"""Reviewed report: preserve original readout, quarantine R07, add frozen R08."""
import json, re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import frozen_real_transfer as f
O=f.OUT;D=O/'additional_control'
FAMILIES={'isolated_spot':'孤立した斑点','contact_stain':'文字に接するシミ','faint_stroke':'薄い筆画','clean_control':'比較的きれいな対照','isolated_dark_mark':'由来未確認の黒い非文字物（参考）'}
def main():
    p=f.load();extra=json.loads((D/'protocol.json').read_text());review=json.loads((O/'annotation_review.json').read_text())
    assert review['invalid_cases']==['R07']
    for name,h in extra['files'].items():assert f.sha(f.ROOT/name)==h,name
    initial=json.loads((O/'combined_results.json').read_text());add=json.loads((D/'results.json').read_text())
    assert add['protocol_sha256']==f.sha(D/'protocol.json')
    rows=initial['rows']+add['rows'];valid=[r for r in rows if r['case']!='R07'];primary=[r for r in valid if not r['secondary']]
    cases=[c for c in p['cases'] if c['id']!='R07']+[extra['case']]
    for c in p['cases']+[extra['case']]:
        assert f.sha(f.ROOT/c['source'])==c['source_sha256'],c['source']
        x,y,w,h=c['roi'];crop=f.rgb(f.ROOT/c['source'])[y:y+h,x:x+w]
        original=D/'R08_original.png' if c['id']=='R08' else O/'inputs'/f"{c['id']}_original.png"
        assert np.array_equal(crop,f.rgb(original))
    checked=[]
    for r in rows:
        if r['case']=='R08':lp=D/'labels.png';mp=D/'masks'/f"{r['method']}.png"
        else:lp=O/'labels'/f"{r['case']}.png";mp=O/'evaluation'/r['method']/'masks'/f"{r['case']}.png"
        label=np.asarray(Image.open(lp));pred=np.asarray(Image.open(mp))==0
        assert f.sha(mp)==r['prediction_sha256']
        distance=ndi.distance_transform_edt(~pred) if pred.any() else np.full(pred.shape,np.inf)
        for k,v in [('ink_exact',int((pred&(label==1)).sum())),('ink_r2',int(((distance<=2)&(label==1)).sum())),('nonink_false',int((pred&(label==2)).sum()))]:assert r[k]==v,(r['case'],r['method'],k)
        checked.append(dict(case=r['case'],method=r['method'],numerical_rescore=True,annotation_valid=r['case']!='R07',mask_sha256=f.sha(mp)))
    # Rebuild every endorsed figure; invalid R07 is deliberately absent.
    figures=[]
    for c in cases:
        key=c['id'];raw=f.rgb(D/'R08_original.png' if key=='R08' else O/'inputs'/f'{key}_original.png');items=[('原画像',raw)]
        for m in f.NAMES:
            r=next(r for r in valid if r['case']==key and r['method']==m);mp=D/'masks'/f'{m}.png' if key=='R08' else O/'evaluation'/m/'masks'/f'{key}.png'
            items.append((f.TITLES[m]+f" 墨{r['ink_r2']}/{r['ink_n']} 非文字{r['nonink_false']}",np.asarray(Image.open(mp))))
        target=O/'reviewed_figures'/f'{key}_原画像と固定手法比較.png';f.real.panel(target,key+' '+FAMILIES[c['family']]+' / 暫定注釈・原寸',items);figures.append(str(target.relative_to(O)))
    groups={}
    for family in ['isolated_spot','contact_stain','faint_stroke','clean_control']:
        groups[family]=[]
        for m in f.NAMES:
            selected=[r for r in primary if r['method']==m and r['family']==family]
            groups[family].append(dict(method=m,**{k:sum(r[k] for r in selected) for k in ['ink_exact','ink_r2','ink_n','nonink_false','nonink_n']}))
    for r in rows:r['annotation_status']='invalid_coordinates' if r['case']=='R07' else 'conditional_secondary' if r['secondary'] else 'provisional_valid'
    f.dump(O/'reviewed_results.json',dict(rows=rows,primary_cases=sorted({r['case'] for r in primary}),conditional_secondary=['R02'],invalid_cases=['R07'],groups=groups,figures=figures,formal_accuracy=False))
    lines=['# 実損傷への転移：注釈確認後の比較','',
      '主比較は6冊・6ページのR01/R03/R04/R05/R06/R08。別に由来未確認の黒い非文字物R02を参考表示する。R07は注釈の座標不備が原画像で確認されたため、全手法の数値を無効とした。元注釈・出力・初回集計は保存し、別ページR08を原画像で注釈・確認・凍結してから追加推論した。[注釈監査記録](annotation_review.json)。','',
      'Howeは固定条件で原寸ページ全体を処理し、学習モデルは固定512分割・stride384のページ格子から切り出した。4重みは14周時点で固定し、追加学習・閾値調整・良い初期化だけの選別はしていない。公開Gは教師あり生成器で、GANの効果やRA-GAN原論文の再現を示す実験ではない。','',
      '墨は厳密一致r0・半径2画素r2の中心線被覆、非文字は選択した内部領域の黒化画素。両方を別々に読む。Codexの疎な暫定注釈で、専門家GT・文字認識率・文書全体の精度ではない。','',
      '| 課題 | 手法 | 墨 r0 / r2 / 全点 | 選択非文字の黒化 / 全点 |','|---|---|---:|---:|']
    for family,rs in groups.items():
        for r in rs:lines.append(f"| {FAMILIES[family]} | {f.TITLES[r['method']]} | {r['ink_exact']} / {r['ink_r2']} / {r['ink_n']} | {r['nonink_false']} / {r['nonink_n']} |")
    lines+=['','## 原画像付きの全例','']
    for c in cases:
        key=c['id'];anno='additional_control/R08_原画像と部分注釈.png' if key=='R08' else f'annotations/{key}_原画像と部分注釈.png'
        lines += [f"### {key} {FAMILIES[c['family']]} / {c['page']}",'',f'[原寸比較](reviewed_figures/{key}_原画像と固定手法比較.png)、[部分注釈]({anno})。','', '| 手法 | 墨 r0 / r2 / 全点 | 非文字の黒化 / 全点 |','|---|---:|---:|']
        for m in f.NAMES:
            r=next(r for r in valid if r['case']==key and r['method']==m);lines.append(f"| {f.TITLES[m]} | {r['ink_exact']} / {r['ink_r2']} / {r['ink_n']} | {r['nonink_false']} / {r['nonink_n']} |")
        lines.append('')
    lines+=['## 適用範囲と監査','',
      '全ページに過去の自動スクリーニング記録とCSC処理があり、完全な未使用資料ではない。2194ページを照合し、既知の比較・目視記録などの記載があるページを保守的に除外した。台帳・自動特徴計算の許可リストを区別した候補は914ページ・11冊。その原画像から目的を持って課題例を選んだ。冊自体は過去にプロジェクトで扱っており、未見冊の無作為テストとはしない。今回の6冊は、4重みの学習冊Makuraとは異なる。','',
      'R01–R06は原画像による注釈を凍結してから推論した。R07の座標不備は出力値が表示された後の原本オーバーレイ確認で判明したため、除外判断も完全盲検ではない。全手法へ同じ無効判定を適用し、元データを保存した。R08はその後に別ページを選定し、当該ページの推論出力を見る前に注釈を確認・凍結した。','',
      'R02は非文字に見える黒い物体だが、破損・背景混入等も考えられ、由来と保存対象としての扱いが未確認。主比較に混ぜない。今回、大きな炭状斑点の新しい確実な例は得られていない。既知D04を別の開発例として残す。','',
      '注釈は選択部分のみで、例えばR03の公開Gに見える広い紙模様の黒化は、狭い非文字注釈の数値だけでは全量を表せない。PNGを併せて確認する。2初期化は不確実性区間ではなく、全資料の勝者や優越性の統計的証明は行わない。','',
      '[最初の凍結条件](protocol.json)、[追加対照の凍結条件](additional_control/protocol.json)、[ページ記録の監査](refined_exposure_audit_v2.json)、[再採点とハッシュ検証](reviewed_verification.json)。','']
    (O/'比較結果_注釈確認後.md').write_text('\n'.join(lines))
    notice='> **注釈監査による無効判定：R07の数値と、それを含むclean_control集計は使わない。確定範囲は[注釈確認後の比較](比較結果_注釈確認後.md)を参照。**\n\n'
    old=O/'比較結果_新しい実画像.md';s=old.read_text()
    if not s.startswith('> **注釈監査'):old.write_text(notice+s)
    digest=f.sha(O/'protocol.json');extra_digest=f.sha(D/'protocol.json')
    f.dump(O/'reviewed_verification.json',dict(utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=digest,additional_protocol_sha256=extra_digest,
      fixed_files_checked=len(p['frozen_files'])+len(extra['files']),rescored_masks=checked,primary_cases=6,primary_rows=30,secondary_rows=5,invalid_rows_retained=5,
      model_state_unchanged_all5_results=True,figures={name:f.sha(O/name) for name in figures},annotation_validity_is_manual_and_provisional=True,
      full_source_hashes_checked=8,native_original_crops_checked=8,dense_GT=False,fully_blind=False))
    print('Reviewed report:6 primary cases,1 secondary,1 invalid;40 masks independently rescored',flush=True)
if __name__=='__main__':main()
