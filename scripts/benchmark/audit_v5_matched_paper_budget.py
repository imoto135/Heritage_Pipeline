"""Oracle threshold tradeoff on one held-out synthetic image; diagnostic only."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'tmp_work/makura_v5_aligned_fade_control_20261021'
EXP=ROOT/'tmp_work/makura_v5_exposure_loss_control_20261021'
OUT=EXP/'probability_audit'
CASE=ROOT/'tmp_work/targeted_thin_fade_20261021/cases/alpha_0.70'
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
sys.path.insert(0,str(ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from compare_v5_aligned_fade_control import sha,dump,rgb,coverage
from audit_v5_probability_thresholds import ink_probability
from ragan_pilot_support import make_model

def gray(p):return np.asarray(Image.open(p).convert('L'))
def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25,device)
    patch='200017197_00172_0_p00'
    trace=gray(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(patch+'_high_centers.png'))>0
    target=gray(ROOT/'tmp_work/makura_supervised_smoke_20261021/labels_v2'/patch/'paper_target.png')
    valid=gray(ROOT/'tmp_work/makura_supervised_smoke_20261021/labels_v2'/patch/'valid.png')>0
    paper=valid&(target==255);howe=gray(CASE/'howe_fixed.png')==0
    howe_fp=int((howe&paper).sum());howe_trace=coverage(howe,trace)
    assert howe_fp==2 and howe_trace['r0']==42
    rows=[];inp=rgb(CASE/'input.png')
    for stem,kind in (('unet_v3','unet'),('ragan_public_v3','ragan')):
        for factor in ('baseline','exposure','core_loss'):
            folder=BASE/(stem+'_v5') if factor=='baseline' else EXP/(stem+'_'+factor)
            path=folder/'models'/kind/'last.pt';model=make_model(kind)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
            q=ink_probability(model.to(device).eval(),kind,inp,device)
            order=np.sort(q[paper])[::-1]
            budgets=[]
            for allowed in (0,howe_fp,int(.001*paper.sum()),int(.01*paper.sum()),int(.03*paper.sum())):
                threshold=float(np.nextafter(order[allowed],np.float32(np.inf))) if allowed<len(order) else 0.
                pred=q>=threshold
                fp=int((pred&paper).sum());assert fp<=allowed
                budgets.append(dict(allowed_paper_false=allowed,threshold=threshold,actual_paper_false=fp,
                    trace=coverage(pred,trace)))
            rows.append(dict(model=stem,factor=factor,checkpoint_sha256=sha(path),
                input_sha256=sha(CASE/'input.png'),budgets=budgets))
    dump(OUT/'matched_paper_budget.json',dict(rows=rows,howe=dict(paper_false=howe_fp,trace=howe_trace),
        paper_n=int(paper.sum()),formal_accuracy=False,
        note='Oracle threshold chosen on the same held-out diagnostic image and provisional paper mask; upper bound only, not a deployable calibration'))
    print('Howe',howe_trace['r0'],'/',howe_trace['n'],'paper false',howe_fp,'/',paper.sum())
    for row in rows:
        print(row['model'],row['factor'],[(b['allowed_paper_false'],b['trace']['r0'],b['actual_paper_false']) for b in row['budgets']])
if __name__=='__main__':main()
