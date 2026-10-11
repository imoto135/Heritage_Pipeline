"""Same frozen 16-pair paper restorer, plus original-pixel abstention."""
import gc
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import probe_paper_boundary_pair_gate_v2 as fill
import probe_paper_tone_boundary as paper
import validate_unseen_directional_margin_candidate as mask
from collect_saved_npj_failures import saved_masks
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=mask.ROOT
INPUT=mask.INPUT
OUT=INPUT/'paper_fill'
MODES=['npj','v6','candidate']


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(mask.OUT)
    selected=json.loads((INPUT/'baseline_review.json').read_text())
    deps={str(Path(m.__file__).relative_to(ROOT)):sha(Path(m.__file__)) for m in [fill,paper,mask]}
    deps[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    for n in ['manifest.json','complete.json']:deps[str((mask.OUT/n).relative_to(ROOT))]=sha(mask.OUT/n)
    deps[str((INPUT/'baseline_review.json').relative_to(ROOT))]=sha(INPUT/'baseline_review.json')
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=deps,pages=4,ROIs=13,settings=dict(
        mask_modes=MODES+['abstention'],minimum_boundary_pairs=16,context_px=64,original_elsewhere=True,
        new_DDRM=False,adaptive_texture=False,chromatic_rim=False,low_order_surface=False,pixel_GT=False,
        hold='replace baseline paper-applied & frozen full-page hold with original; other pixels bitwise unchanged',
        npj='saved degradation mask through same pure-paper scope, not full original npj pipeline'))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=[];poison=[]
    for pg in selected['cases']:
        key=pg['page'];original=read(ROOT/pg['original']);H,W=original.shape[:2];directory=mask.OUT/'pages'/key
        domain=read(directory/'domain.png',True);guard=read(directory/'guard.png',True);held=read(directory/'paper_hold.png',True)
        v6={k:read(INPUT/'v6_masks'/key/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        npj=saved_masks(pg);edit=read(directory/'edit.png',True)
        values=dict(npj=dict(edit=npj['edit'],protect=npj['black'],red=npj['red'],hold=np.zeros_like(edit)),v6=v6,
            candidate=dict(edit=edit,protect=v6['protect']&~edit,red=v6['red'],hold=v6['hold']&~edit,extra=v6['extra']))
        references={};eligible={}
        for mode,maps in values.items():
            protected=guard.copy()
            for n in ['protect','hold','red','extra']:protected|=maps.get(n,np.zeros_like(edit))
            references[mode]=paper.clean_reference(original,maps['edit'],protected&~maps['edit'],domain&~guard)[1]
            eligible[mode]=maps['edit']&domain&~guard
            write(OUT/'reference_pages'/key/(mode+'_known.png'),references[mode]);write(OUT/'reference_pages'/key/(mode+'_eligible.png'),eligible[mode])
        assert np.array_equal(references['candidate'],read(directory/'known.png',True))
        for roi in pg['rois']:
            rid=roi['id'];x,y,w,h=roi['xywh'];cx,cy=max(0,x-64),max(0,y-64);cw,ch=min(W,x+w+64)-cx,min(H,y+h+64)-cy
            cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w));context=original[cs];crop=context[s]
            lab=cv2.cvtColor(context.astype(np.float32)/255,cv2.COLOR_BGR2LAB);outputs={};stats={};cache=[]
            for mode in MODES:
                known=references[mode][cs];active=eligible[mode][cs];local=OUT/'native'/rid/mode
                cached=next((r for r in cache if np.array_equal(known,r['known']) and np.array_equal(active,r['active'])),None)
                result=cached['result'] if cached is not None else fill.field_for(lab,known,active,16)
                fields,applied,used,notes,field,positions,qnotes,seam,average,sx,sy=result
                cache.append(dict(known=known,active=active,result=result))
                value=fill.frozen.compose(field,applied,crop,s);outputs[mode]=value
                assert not np.any(np.any(value!=crop,axis=2)&~applied[s])
                assert not np.any(applied&~active) and not np.any(known&active)
                if applied[s].any() and cached is None:
                    bad=lab.copy();bad[~known]=(230,-190,140);repeat=fill.field_for(bad,known,active,16)
                    assert np.array_equal(field,repeat[4]) and np.array_equal(applied,repeat[1]);poison.append(dict(id=rid,mode=mode))
                for name,item in dict(output=value,original=crop,reference_context=known,eligible_context=active,applied_context=applied,used_context=used,edit=values[mode]['edit'][cs][s]).items():write(local/(name+'.png'),item)
                local.mkdir(parents=True,exist_ok=True);np.savez_compressed(local/'harmonic_fields_Lab.npz',**fields)
                for name,item in dict(result_field_Lab=field,texture_seam=seam,texture_average=average,source_x=sx,source_y=sy).items():np.save(local/(name+'.npy'),item,allow_pickle=False)
                dump(local/'components.json',notes);dump(local/'donors.json',dict(context_xywh=[cx,cy,cw,ch],patch_xy=positions,components=qnotes))
                stats[mode]=dict(eligible=int(active[s].sum()),applied=int(applied[s].sum()),changed_vs_original=int(np.any(value!=crop,axis=2).sum()),references=int(known.sum()),patches=len(positions))
                if mode=='candidate':candidate_applied=applied.copy()
            local_hold=held[cs];reverted=candidate_applied&local_hold;effective=candidate_applied&~local_hold
            value=outputs['candidate'].copy();value[reverted[s]]=crop[reverted[s]];outputs['abstention']=value
            assert np.array_equal(value[local_hold[s]],crop[local_hold[s]])
            assert np.array_equal(value[~local_hold[s]],outputs['candidate'][~local_hold[s]])
            assert not np.any(references['candidate']&held)
            local=OUT/'native'/rid/'abstention'
            for name,item in dict(output=value,paper_hold_context=local_hold,reverted_context=reverted,effective_applied_context=effective,reference_context=references['candidate'][cs]).items():write(local/(name+'.png'),item)
            stats['abstention']=dict(held=int(local_hold[s].sum()),reverted=int(reverted[s].sum()),remaining_applied=int(effective[s].sum()),changed_vs_candidate=int(np.any(value!=outputs['candidate'],axis=2).sum()))
            figure([crop]+[outputs[m] for m in MODES+['abstention']],['原画像','npjマスク × 紙面補間','固定v6 × 紙面補間','固定の余白前処理 × 紙面補間','方向一致を原画素へ保留'],OUT/'comparisons'/(rid+'.png'),roi['title']+' · '+key,
                '同じ資格紙面・16境界組・観測明度模様。紙面外と未補間は原画像。','DDRMではなく紙面だけの診断。npj全工程の再現ではない。')
            rows.append(dict(id=rid,page=key,title=roi['title'],xywh=roi['xywh'],context_xywh=[cx,cy,cw,ch],diagnostics=stats))
            dump(OUT/'progress.json',dict(completed=len(rows),total=13,latest=rid));print('New directional paper:',len(rows),'/13',rid,stats,flush=True)
        del original,v6,npj,domain,guard,held,values,references,eligible;gc.collect()
    for path,h in deps.items():assert sha(ROOT/path)==h
    dump(OUT/'diagnostics.json',rows)
    dump(OUT/'verification.json',dict(status='passed',pages=4,ROIs=13,nonreference_poison=poison,unapplied_original=True,
        hold_original=True,outside_hold_exact_candidate=True,reference_unchanged_for_hold=True,new_DDRM=False,pixel_GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:new4-books same paper restorer and abstention13 PNGs',flush=True)


if __name__=='__main__':main()
