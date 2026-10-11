"""Compare retained-applied versus full eligible masks in the same ROI context."""
import gc
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import probe_chromatic_rim_reference_gap_factorial as prior
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/chromatic_rim_full_eligible_scope_pilot_20261001'
MODES=['full_original_eligible','full_rim_eligible']
SETTINGS=dict(pages=29,ROIs=90,classification='same whole-page frozen candidate, direction holds, chromatic6px unheld-core addition',
    numerical_context_px=64,comparison='current saved output, retained-applied plus rim6, full old eligible, full old eligible plus rim6',
    original_eligible='whole-page saved effective eligible with original direction hold removed',
    rim_eligible='original eligible OR frozen unheld-core chromatic6 addition',
    reference='same old known-paper, minus rim added for rim arm; held pixels never become references',
    restorer='same16-pair64-area raw/smooth boundary8 and old24px fineL seam',
    fallback='retain current baseline when new solve unsupported; preserve all previous applied pixels',
    new_scope='previously unfilled eligible pixels may become accepted; record separately from rim additions',
    full_page_colour_solve=False,colour_gap_or_absolute_patch_or_energy=False,new_DDRM=False,pixel_GT=False,development=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [prior.OUT,prior.prior.diagnostic.previous.OUT,prior.prior.diagnostic.fresh.OUT]:check_bundle(folder)
    data=json.loads((prior.OUT/'diagnostics.json').read_text());rows=data['regions'];deps={str(Path(p).relative_to(ROOT)):sha(Path(p)) for p in [__file__,prior.prior.fill.__file__]}
    for folder in [prior.OUT,prior.prior.diagnostic.previous.OUT,prior.prior.diagnostic.fresh.OUT]:
        for n in ['manifest.json','complete.json']:deps[str((folder/n).relative_to(ROOT))]=sha(folder/n)
    manifest=dict(settings=SETTINGS,dependencies=deps,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');results=[];poison=[];current=None;page_records=[]
    for row in rows:
        key=row['page'];rid=row['id'];source=prior.OUT/'native'/rid
        if current!=key:
            original=read(ROOT/row['original_source']);full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB);current=key
            oldpage=prior.prior.diagnostic.previous.OUT/'pages'/key
            if oldpage.exists():
                eligible=read(oldpage/'effective_eligible.png',True);fullknown=read(oldpage/'known.png',True);hold=read(oldpage/'paper_hold.png',True)
            else:
                fresh=prior.prior.diagnostic.fresh;reference=fresh.OUT/'reference_pages'/key
                hold=read(fresh.mask.OUT/'pages'/key/'paper_hold.png',True)
                eligible=read(reference/'candidate_eligible.png',True)&~hold;fullknown=read(reference/'candidate_known.png',True)
            fulladded=read(prior.OUT/'pages'/key/'added.png',True);assert not np.any(fullknown&(eligible|hold)) and not np.any(fulladded&hold)
            page_records.append(dict(page=key,original_eligible=int(eligible.sum()),old_known=int(fullknown.sum()),hold=int(hold.sum()),added=int(fulladded.sum())))
            pd=OUT/'pages'/key
            for n,a in dict(original_eligible=eligible,known=fullknown,hold=hold,added=fulladded,rim_eligible=eligible|fulladded).items():write(pd/(n+'.png'),a)
            gc.collect()
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        lab=full_lab[cs];crop=read(source/'original.png');base=read(source/'baseline.png');narrow=read(source/'rim6.png')
        oldapp=read(source/'original_applied_context.png',True);known=read(source/'original_reference_context.png',True);assert np.array_equal(known,fullknown[cs])
        oldeligible=eligible[cs];add=fulladded[cs];local_hold=hold[cs];assert not np.any(oldapp&~oldeligible)
        dest=OUT/'native'/rid;dest.mkdir(parents=True,exist_ok=True);outputs={};stats={}
        for mode in MODES:
            md=dest/mode;md.mkdir(parents=True,exist_ok=True);active=oldeligible if mode=='full_original_eligible' else oldeligible|add
            reference=known if mode=='full_original_eligible' else known&~add
            accepted=np.zeros_like(oldapp);used=np.zeros_like(oldapp);notes=[];positions=[];qnotes=[]
            if active[s].any() or oldapp[s].any():
                result=prior.prior.fill.field_for(lab,reference,active,16)
                fields,accepted,used,notes,field,positions,qnotes,seam,average,sx,sy=result
                value=prior.prior.fill.frozen.compose(field,accepted,base,s)
                assert not np.any(accepted&~active) and not np.any(accepted&local_hold)
                bad=lab.copy();bad[~reference]=(240,-210,140);repeat=prior.prior.fill.field_for(bad,reference,active,16)
                assert np.array_equal(field,repeat[4]) and np.array_equal(accepted,repeat[1]);poison.append(dict(id=rid,mode=mode))
                np.savez_compressed(md/'harmonic_fields_Lab.npz',**fields)
                for n,a in dict(result_field_Lab=field,texture_seam=seam,texture_average=average,source_x=sx,source_y=sy).items():np.save(md/(n+'.npy'),a,allow_pickle=False)
                dump(md/'donors.json',dict(patch_xy=positions,components=qnotes))
            else:value=base.copy()
            physical=oldapp|accepted;newold=accepted&oldeligible&~oldapp;newrim=accepted&add
            for n,a in dict(output=value,eligible_context=active,reference_context=reference,accepted_context=accepted,physical_context=physical,
                used_context=used,new_original_eligible_context=newold,new_rim_context=newrim,fallback_old_context=oldapp&~accepted).items():write(md/(n+'.png'),a)
            dump(md/'components.json',notes)
            assert np.array_equal(value[~accepted[s]],base[~accepted[s]]) and np.array_equal(value[local_hold[s]],crop[local_hold[s]])
            stats[mode]=dict(eligible_ROI=int(active[s].sum()),accepted_ROI=int(accepted[s].sum()),old_applied_ROI=int(oldapp[s].sum()),
                newly_accepted_original_ROI=int(newold[s].sum()),newly_accepted_rim_ROI=int(newrim[s].sum()),fallback_old_ROI=int((oldapp&~accepted)[s].sum()),
                changed_vs_base=int(np.any(value!=base,axis=2).sum()),changed_vs_narrow_rim=int(np.any(value!=narrow,axis=2).sum()),donor_patches=len(positions))
            outputs[mode]=value
        for n,a in dict(original=crop,baseline=base,narrow_rim=narrow,original_applied_context=oldapp,original_eligible_context=oldeligible,
            original_reference_context=known,added_context=add,paper_hold_context=local_hold).items():write(dest/(n+'.png'),a)
        figure([crop,base,narrow]+[outputs[m] for m in MODES],['原画像','現行・以前の受理を保持','受理範囲＋輪郭6px','全編集資格・輪郭0','全編集資格＋輪郭6px'],OUT/'comparisons'/(rid+'.png'),row['title']+' · 復元へ渡す資格範囲を確認',
            '分類と周辺64px・16組・元24px模様は固定。以前の未補間候補の再受理も別記録する。',
            '全ページで候補分類、色場は同じROI周辺条件。gap・絶対パッチ・模様量補正は併用なし。')
        results.append({**row,'full_scope_diagnostics':stats});dump(OUT/'progress.json',dict(completed=len(results),total=90,latest=rid))
        print('Full eligible scope:',len(results),'/90',rid,[(m,stats[m]['newly_accepted_original_ROI'],stats[m]['changed_vs_narrow_rim']) for m in MODES],flush=True)
    dump(OUT/'diagnostics.json',dict(pages=page_records,regions=results))
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,whole_page_eligibility_reference_checked=True,
        nonreference_poison_invariance=poison,all_old_physical_pixels_retained_on_failure=True,held_original_pixels_exact=True,
        new_scope_recorded_separately_from_rim=True,full_page_colour_solve=False,new_DDRM=False,pixel_GT=False))
    for p,h in deps.items():assert sha(ROOT/p)==h
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete: full eligible paper scope90 ROIs',flush=True)


if __name__=='__main__':main()
