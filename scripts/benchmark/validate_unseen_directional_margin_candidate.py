"""Fixed reusable margin mask and directional abstention on four new books."""
import gc
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import frozen_margin_preprocessing_candidate as candidate
import probe_paper_tone_boundary as paper
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'tmp_work/unseen_directional_margin_validation_20261001'
OUT=INPUT/'fixed_masks'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selected=json.loads((INPUT/'baseline_review.json').read_text())
    assert selected['candidate_settings']==candidate.SETTINGS
    for path,h in selected['candidate_sources'].items():assert sha(ROOT/path)==h
    for path,h in selected['inputs'].items():assert sha(ROOT/path)==h
    for path,h in json.loads((INPUT/'comparison_manifest.json').read_text())['dependencies'].items():assert sha(ROOT/path)==h
    for path,h in json.loads((INPUT/'comparison_complete.json').read_text())['outputs'].items():assert sha(INPUT/path)==h
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),**selected['candidate_sources'],**selected['inputs']}
    for n in ['baseline_review.json','visual_selection.json','comparison_manifest.json','comparison_complete.json']:
        deps[str((INPUT/n).relative_to(ROOT))]=sha(INPUT/n)
    for folder in ['frozen_margin_candidate_parity_20261001','margin_original_feature_recomputation_20261001','margin_reference_recomputation_20261001']:
        path=ROOT/'tmp_work'/folder/'verification.json';assert json.loads(path.read_text())['status']=='passed';deps[str(path.relative_to(ROOT))]=sha(path)
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=deps,settings=candidate.SETTINGS,pages=4,ROIs=13,
        original_selection_first=True,parameters_tuned_after_outputs=False,restoration=False,pixel_GT=False)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');pages=[];rows=[]
    for i,pg in enumerate(selected['cases'],1):
        key=pg['page'];original=read(ROOT/pg['original'])
        v6={k:read(INPUT/'v6_masks'/key/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        _,features,_=candidate.rim.MARGIN.second.infer(original,v6)
        masks,notes=candidate.infer(original,v6,features)
        protected=v6['protect']|v6['hold']|v6['red']|v6['extra']|masks['guard']
        known=paper.clean_reference(original,masks['edit'],protected&~masks['edit'],masks['domain']&~masks['guard'])[1]
        labels,direction_notes,hold,retained=candidate.infer_hold(original,masks,known)
        dest=OUT/'pages'/key
        for name,value in {**masks,'known':known,'paper_hold':hold,'retained':retained,'effective_edit':masks['edit']&~hold}.items():write(dest/(name+'.png'),value)
        dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'labels.npz',labels=labels)
        dump(dest/'components.json',notes);dump(dest/'direction.json',direction_notes)
        assert not np.any(known&hold)
        assert not np.any((masks['edit']&~v6['edit'])&(~masks['domain']|masks['guard']))
        record=dict(page=key,rule_y=notes['completion']['rule_y'],safe_domain=int(masks['domain'].sum()),
            core=int(masks['core'].sum()),new_strong_core=int(masks['strong_core'].sum()),
            added_vs_v6=int((masks['edit']&~v6['edit']).sum()),candidate_eligible=int((masks['edit']&masks['domain']&~masks['guard']).sum()),
            known=int(known.sum()),held_pixels=int(hold.sum()),matched_components=[n for n in direction_notes if n['provisional_matched']])
        pages.append(record)
        for roi in [dict(id=key+'_whole',title='ページ全体',xywh=[0,0,original.shape[1],original.shape[0]])]+pg['rois']:
            rid=roi['id'];x,y,w,h=roi['xywh'];s=(slice(y,y+h),slice(x,x+w));crop=original[s]
            before=crop.copy();before[v6['edit'][s]]=(0,200,255)
            current=before.copy();current[(masks['edit']&~v6['edit'])[s]]=(255,120,0)
            after=current.copy();after[hold[s]]=(210,30,200)
            figure([crop,before,current,after],['原画像','黄＝固定v6編集','青＝固定前処理の追加','紫＝方向一致の補修保留'],OUT/'comparisons'/(rid+'.png'),roi['title']+' · '+key,
                '原画像から全ページで同じ条件を適用。方向の保留をROIへ依存させない。','マスク比較。白黒・印影・紙模様も残す。物理原因GTなし。')
            rows.append(dict(id=rid,page=key,xywh=roi['xywh'],added_vs_v6=int((masks['edit']&~v6['edit'])[s].sum()),held_pixels=int(hold[s].sum())))
        print('Fixed directional candidate:',i,'/4',record,flush=True)
        del original,v6,features,masks,known,labels,hold,retained;gc.collect()
    for path,h in deps.items():assert sha(ROOT/path)==h
    dump(OUT/'diagnostics.json',dict(pages=pages,regions=rows))
    dump(OUT/'verification.json',dict(status='passed',pages=4,ROIs=13,original_selection_first=True,full_page_inference=True,
        ROI_gating=False,parameters_tuned_after_outputs=False,guards_preserved=True,held_pixels_nonreference=True,pixel_GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete: frozen reusable masks and abstention4 new books /17 PNGs',flush=True)


if __name__=='__main__':main()
