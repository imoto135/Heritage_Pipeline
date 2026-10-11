"""Audit fixed saved-output transfer and show every edited component."""
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from build_npj_v6_comparison import ROOT,read,sha,dump,overlay
from finalize_npj_full_page_layout import render,panel_audit
import validate_npj_ridge_saved_ranges as run

OUT=run.OUT


def main():
    assert not (OUT/'complete.json').exists()
    selection=json.loads((OUT/'selection.json').read_text())
    diagnostics={c['id']:c for c in json.loads((OUT/'diagnostics.json').read_text())['cases']}
    audits=[];components=[];regions=[]
    for c in selection['cases']:
        n=OUT/'native'/c['id'];x,y,w,h=c['xywh'];sl=np.s_[y:y+h,x:x+w]
        original=read(n/'original.png');base=read(n/'baseline.png')
        assert np.array_equal(read(ROOT/c['original']),read(ROOT/c['input']))
        assert np.array_equal(original,read(ROOT/c['original'])[sl])
        assert np.array_equal(base,read(ROOT/c['saved_npj'])[sl])
        mandatory=read(n/'mandatory.png',True);old_guard=read(n/'old_guard.png',True)
        roles={k:read(ROOT/c['roles']/(k+'.png'),True)[sl] for k in ['protect','hold','red','extra']}
        assert np.array_equal(mandatory,roles['red']|roles['extra']|(cv2.imread(str(ROOT/c['npj_red']),0)[sl]<127))
        applied={};results={};method_checks={}
        for mode in ['frame','v6','old_contact',*run.ridge.MODES]:
            f=n/mode;m={k:read(f/(k+'.png'),True) for k in ['update','domain','reference','applied','used_reference']}
            result=read(f/'repaired.png')
            assert not (m['applied']&~m['update']).any()
            assert not (m['reference']&(m['update']|mandatory|~m['domain'])).any()
            assert not (m['used_reference']&~m['reference']).any()
            assert np.array_equal(result[mandatory|~m['applied']],base[mandatory|~m['applied']])
            fn=run.paper.paper_reference if mode in ['frame','v6'] else run.reference.paper_reference
            known,t=fn(base,m['update'],m['domain'],mandatory)
            assert np.array_equal(known,m['reference'])
            assert t==diagnostics[c['id']]['methods'][mode]['reference_L_threshold']
            if mode in run.ridge.MODES:
                guard=read(n/(mode+'.png'),True)
                assert not (m['applied']&(guard|old_guard)).any()
                assert np.array_equal(m['update'],read(n/'old_contact/update.png',True)&~guard)
            if mode=='frame':assert not (m['applied']&read(n/'frame_interior.png',True)).any()
            if mode=='v6':assert not (m['applied']&(roles['protect']|roles['hold'])).any()
            applied[mode]=m['applied'];results[mode]=result
            method_checks[mode]=dict(reference_recomputed=True,roles_disjoint=True,
                outside_applied_exact=True,mandatory_exact=True,
                changed=int(np.any(result!=base,2).sum()))
            assert method_checks[mode]['changed']==diagnostics[c['id']]['methods'][mode]['changed']
        combinations={}
        for mode in ['old_contact',*run.ridge.MODES]:
            add=applied[mode]&~applied['frame'];combined=results['frame'].copy();combined[add]=results[mode][add]
            assert np.array_equal(add,read(n/mode/'additional.png',True))
            assert np.array_equal(combined,read(n/mode/'combined.png'))
            combinations[mode]=combined
        arrays=[original,base,results['v6'],*combinations.values()]
        panel_audit(OUT/(c['id']+'_comparison.png'),arrays,True)
        panel_audit(OUT/(c['id']+'_baseline.png'),[original,base],True)
        for r in json.loads((OUT/'witnesses_before_run.json').read_text())['regions']:
            if r['id']!=c['id']:continue
            bx,by,bw,bh=r['xywh'];cut=np.s_[by:by+bh,bx:bx+bw]
            assert 0<=bx and 0<=by and bx+bw<=w and by+bh<=h
            filename=c['id']+'_'+r['name']+'.png'
            render(OUT/filename,[a[cut] for a in arrays],
                ['原画像','実保存npj','固定v6紙面補修','従来の線保持＋枠外','色を問わず線保持＋枠外','保護接続線の保持＋枠外'],
                c['id']+'：'+r['name'],
                ['原画像と保存基準で事前に指定した範囲。閾値調整・追加推論なし。',
                 '同じ原画素を整数倍表示。汚れや筆画の画素GTではない。'])
            regions.append(dict(**r,filename=filename,changed={k:int(np.any(v[cut]!=base[cut],2).sum()) for k,v in combinations.items()}))
        # Display all changes for the fixed candidate, including tiny components.
        change=np.any(combinations['all_colour_ridge']!=base,2)
        count,labels,stats,_=cv2.connectedComponentsWithStats(change.astype(np.uint8),8)
        for i in range(1,count):
            cx,cy,cw,ch,area=map(int,stats[i]);bx=max(0,cx-35);by=max(0,cy-35)
            ex=min(w,cx+cw+35);ey=min(h,cy+ch+35);cut=np.s_[by:ey,bx:ex]
            filename=c['id']+'_all_change_%02d.png'%i
            render(OUT/filename,[original[cut],base[cut],overlay(base[cut],[('blue',change[cut])]),
                combinations['all_colour_ridge'][cut],combinations['protect_anchored_ridge'][cut]],
                ['原画像','実保存npj','青＝全変更','色を問わず線保持','保護接続線の保持'],
                c['id']+'：変更成分%d／%d'%(i,count-1),
                ['変更成分を省略せず全て表示。位置は出力後の監査用で判定には使わない。',
                 '黒い点の減少を汚れ精度と数えない。原画像の筆画・別の像・紙面と照合。'])
            components.append(dict(id=c['id'],component=i,bbox=[cx,cy,cw,ch],pixels=area,filename=filename))
        audits.append(dict(id=c['id'],actual_input_exact_original=True,saved_crop_exact=True,
            methods=method_checks,combined_equals_old=np.array_equal(combinations['all_colour_ridge'],combinations['old_contact']),
            candidate_changed=int(change.sum())))
    dump(OUT/'verification.json',dict(cases=audits,regions=regions,all_changed_components=components,
        method_thresholds_unchanged=True,independent_book_test=False,pixel_GT=False,
        full_new_inference=False,actual_saved_outputs=True))
    shutil.copyfile(__file__,OUT/'audit_source_snapshot.py')
    print('Verified',len(audits),'cases;',len(components),'changed components;',sum(c['pixels'] for c in components),'pixels',flush=True)


if __name__=='__main__':main()
