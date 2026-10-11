"""Development abstention ablation on frozen renders, without new inference.

Component vetoes protect and protect|hold are predeclared, never tuned per
case. Keep previous layout pixels and reject only added contact components.
The new-book cases have already been seen: this is development, not held out.
"""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, read, write, sha, dump
from finalize_npj_full_page_layout import render
from validate_npj_dirt_transfer import verified_bundle

OLD=ROOT/'tmp_work/npj_contact_dirt_guard_v2_20261004'
NEW=ROOT/'tmp_work/npj_dirt_new_book_20261004'
OUT=ROOT/'tmp_work/npj_contact_component_veto_20261004'


def main():
    assert not OUT.exists(), 'Use a new output name'
    verified_bundle(OLD);verified_bundle(NEW)
    old=json.loads((OLD/'manifest.json').read_text())['cases']
    new=json.loads((NEW/'selection.json').read_text())['cases']
    cases=[];dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for folder,items in [(OLD,old),(NEW,new)]:
        dependencies[str((folder/'complete.json').relative_to(ROOT))]=sha(folder/'complete.json')
        for c in items:
            n=folder/'native'/c['id']
            role_dir=ROOT/c['v6_roles'] if folder==OLD else NEW/'classification/v6_masks'/c['page']
            contact_dir='ridge_guard' if folder==OLD else 'contact_guard'
            has_prior=(n/'combined.png').exists()
            cases.append(dict(id=('old_' if folder==OLD else 'new_')+c['id'],
                source_case=c['id'],page=c['page'],xywh=c['xywh'],native=str(n.relative_to(ROOT)),
                roles=str(role_dir.relative_to(ROOT)),role=c.get('role',c.get('reason','known development case')),
                contact_dir=contact_dir,has_prior=has_prior,
                prior_path=('previous.png' if folder==OLD else 'full_page_layout/repaired.png') if has_prior else 'baseline.png'))
            sources=[n/'combined.png',n/'additional_applied.png'] if has_prior else []
            for p in [n/'original.png',n/'baseline.png',n/contact_dir/'repaired.png',n/contact_dir/'applied.png',
                      n/cases[-1]['prior_path'],*sources,
                      role_dir/'protect.png',role_dir/'hold.png']:
                dependencies[str(p.relative_to(ROOT))]=sha(p)
    OUT.mkdir()
    dump(OUT/'manifest.json',dict(cases=cases,dependencies=dependencies,
        predeclared_modes=['protect_component_veto','protect_hold_component_veto'],
        fixed_rule='8-connected component of additional applied mask; reject whole component if any pixel overlaps named fixed-v6 role',
        no_new_inference=True,no_reference_or_renderer_changes=True,
        thresholds_not_tuned=True,development_after_transfer_failure=True,
        independent_test=False,adopted=False,pixel_GT=False))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    records=[]
    for c in cases:
        n=ROOT/c['native'];target=OUT/'native'/c['id'];x,y,w,h=c['xywh'];sl=np.s_[y:y+h,x:x+w]
        original=read(n/'original.png');base=read(n/'baseline.png')
        prior=read(n/c['prior_path'])
        combined=read(n/'combined.png') if c['has_prior'] else read(n/c['contact_dir']/'repaired.png')
        additional=read(n/'additional_applied.png',True) if c['has_prior'] else read(n/c['contact_dir']/'applied.png',True)
        contact=read(n/c['contact_dir']/'repaired.png')
        check=prior.copy();check[additional]=contact[additional]
        assert np.array_equal(combined,check)
        roles={k:read(ROOT/c['roles']/(k+'.png'),True)[sl] for k in ['protect','hold']}
        count,labels,stats,_=cv2.connectedComponentsWithStats(additional.astype(np.uint8),8)
        results={};mode_records={}
        for name,role in [('protect_component_veto',roles['protect']),
                          ('protect_hold_component_veto',roles['protect']|roles['hold'])]:
            overlap=np.bincount(labels.ravel(),weights=role.ravel(),minlength=count)
            reject=overlap>0;reject[0]=True
            kept=additional&~reject[labels]
            after=prior.copy();after[kept]=contact[kept]
            assert np.array_equal(after[~kept],prior[~kept])
            assert not (kept&role).any()
            # Recompute the whole-component decision independently, one component at a time.
            audit=np.zeros_like(kept)
            for i in range(1,count):
                component=labels==i
                if not (component&role).any():audit|=component
            assert np.array_equal(audit,kept)
            L=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
            afterL=cv2.cvtColor(after.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
            changed=np.any(after!=prior,2)
            for k,a in dict(kept=kept,rejected=additional&~kept,repaired=after).items():
                write(target/name/(k+'.png'),a)
            mode_records[name]=dict(kept_additional=int(kept.sum()),
                rejected_additional=int((additional&~kept).sum()),
                changed_from_prior=int(changed.sum()),
                additional_dark_to_bright=int((kept&(L<30)&(afterL>60)).sum()),
                rejected_components=int(reject[1:].sum()),total_components=count-1)
            results[name]=after
        arrays=[original,base,combined,results['protect_component_veto'],results['protect_hold_component_veto']]
        render(OUT/(c['id']+'_comparison.png'),arrays,
            ['原画像','同じnpj基準','前回：枠外＋線保持','黒保護に触れる成分を見送る','黒保護／holdに触れる成分を見送る'],
            c['page']+'：保護への接触で追加補修を見送る開発比較',
            ['各条件は全17範囲で共通。前回枠外補修は全画素維持。追加の推論・補間なし。',
             '0編集は汚れ除去成功ではない。変更したマスクは文字生成・筆画補完に使わない。'],overview=True)
        for k,a in dict(original=original,baseline=base,prior=prior,previous_combined=combined,
                         additional=additional,protect=roles['protect'],hold=roles['hold']).items():
            write(target/(k+'.png'),a)
        records.append(dict(id=c['id'],page=c['page'],previous_additional=int(additional.sum()),modes=mode_records))
        print(c['id'],int(additional.sum()),{k:v['kept_additional'] for k,v in mode_records.items()},flush=True)
    dump(OUT/'diagnostics.json',dict(cases=records))
    # Audit all three components of the transfer experiment, including the residual edge case.
    transfer=json.loads((NEW/'independent_verification.json').read_text())
    controls=[]
    for component in transfer['all_combination_changed_components']:
        cid='new_'+component['id'];n=OUT/'native'/cid;cx,cy,cw,ch=component['bbox']
        bx,by,bw,bh=component['display_xywh'];cut=np.s_[by:by+bh,bx:bx+bw]
        arrays=[read(n/'original.png'),read(n/'baseline.png'),read(n/'previous_combined.png')]
        arrays.extend(read(n/m/'repaired.png') for m in ['protect_component_veto','protect_hold_component_veto'])
        filename=cid+'_component_%02d.png'%component['component']
        render(OUT/filename,[a[cut] for a in arrays],
               ['原画像','同じnpj基準','前回の追加補修','黒保護接触なら見送る','黒保護／hold接触なら見送る'],
               cid+'：転用失敗の全変更成分',
               ['この位置は結果確認後の回帰対照。新しい独立評価ではない。',
                '筆画の再生成なし。npj基準へ戻すことで削除を防げるかを確認。'])
        box=np.s_[cy:cy+ch,cx:cx+cw]
        controls.append(dict(id=cid,component=component['component'],filename=filename,
            changed_in_bbox={m:int(np.any(read(n/m/'repaired.png')[box]!=arrays[1][box],2).sum())
                             for m in ['protect_component_veto','protect_hold_component_veto']}))
    dump(OUT/'regression_controls.json',dict(controls=controls,pixel_GT=False,
        source_components=transfer['all_combination_changed_components']))
    report(records,controls)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    figures=sorted(OUT.glob('*.png'))
    with zipfile.ZipFile(OUT/'comparison_png.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in figures:z.write(p,p.name)
    with zipfile.ZipFile(OUT/'comparison_png.zip') as z:
        assert sorted(z.namelist())==sorted(p.name for p in figures)
        for p in figures:
            assert np.array_equal(cv2.imdecode(np.frombuffer(z.read(p.name),np.uint8),1),read(p))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        comparison_png_count=len(figures),cases=17,no_new_inference=True,
        whole_component_veto_independently_checked=True,previous_layout_exact=True,
        adopted=False,independent_test=False,pixel_GT=False))


def report(records,controls):
    lines=['# 保護に触れる成分を見送る：筆画削除と汚れ残留を別々に確認', '',
        '2026-10-04。前回の線保持が茶色い本文を取りこぼしたため、追加補修の8連結成分が固定v6のprotect、'
        'またはprotect|holdに1画素でも触れたら、成分を丸ごと見送る2条件を全17範囲へ同じように適用した。'
        'これは結果を見た後の開発比較であり、新しい独立評価ではない。前回の枠外補修は全画素維持する。'
        '前回の組合せがない実保存3対照はnpj基準に線保持補修を加えたものを起点にする。'
        '追加DDRM・学習・補間・文字生成・色補正・閾値の調整はない。', '',
        '同じ基準で出力の一部を見送るため、見送り箇所はnpj基準／前回枠外補修へ戻る。'
        '筆画を生成して直す処理ではない。既存v6のprotectも汚れを含み得るので、見送り量を改善量と数えない。', '',
        '| 範囲 | 前回追加 | protect接触見送り後 | protect／hold接触見送り後 |',
        '| --- | ---: | ---: | ---: |']
    for r in records:
        lines.append('| [%s](%s_comparison.png) | %d | %d | %d |'%
            (r['id'],r['id'],r['previous_additional'],r['modes']['protect_component_veto']['kept_additional'],
             r['modes']['protect_hold_component_veto']['kept_additional']))
    lines.extend(['', '新しい冊で見つかった全変更3成分：', ''])
    for c in controls:
        lines.append('- [%s・成分%d](%s)：変更画素 %s'%
                     (c['id'],c['component'],c['filename'],str(c['changed_in_bbox'])))
    lines.extend(['', '原画像の筆画形と汚れの材質の判定は目視が必要。黒画素減少やマスク保存の検証だけで'
        '汚れ精度・筆画の完全保持は決められない。小さな縁の変更が残る条件は未解決の回帰例として保持する。', '',
        '[全比較PNG](comparison_png.zip)／[固定条件](manifest.json)／[画素数](diagnostics.json)／'
        '[全変更成分の比較記録](regression_controls.json)', '',
        '## 前回報告の数え方の訂正', '',
        '[別冊転用の凍結報告](../npj_dirt_new_book_20261004/report.md)の本文に「2つの欠損範囲と3つの本文範囲」とあるが、'
        '正しくは本文範囲4つ（暖色上端・暖色下端・薄い本文・白黒本文）。保存した8範囲のJSONと比較PNGは4本文範囲を含み、'
        'いずれも組合せ変更0という結果は変わらない。凍結成果物を変更せずここに訂正を記録する。', '',
        '元の26PNGの比較と今回の20PNGは役割が異なる。前者は固定条件の2新規冊への転用、'
        '後者は転用後に考えた見送り条件を既知の11範囲と転用6範囲へ適用した開発比較。'
        '冊数・独立評価数を足し合わせない。'])
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
