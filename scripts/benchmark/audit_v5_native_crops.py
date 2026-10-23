"""Check the v5 fine-stroke failure at native 256-pixel inference size."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/makura_v5_aligned_fade_control_20261021'
CORES=ROOT/'tmp_work/makura_additional_thin_cores_20261021'
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from compare_v5_aligned_fade_control import coverage,sha
from score_makura_heldout_thin_transfer import neural_mask
from ragan_pilot_support import make_model

def read(p):return np.asarray(Image.open(p).convert('RGB'))
def trace(p):return np.asarray(Image.open(p).convert('L'))>0
def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25,device)
    models={}
    for stem,kind in (('unet_v3','unet'),('ragan_public_v3','ragan')):
        for teacher in ('v4','v5'):
            path=OUT/(stem+'_'+teacher)/'models'/kind/'last.pt'
            model=make_model(kind)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
            models[(stem,teacher)]=(model.to(device).eval(),kind)
    cases=[]
    for row in json.loads((CORES/'audit.json').read_text())['rows']:
        patch=row['patch'];cases.append((patch,OUT/'inputs'/patch/'fade_0.80.png',CORES/'masks'/(patch+'_trace.png')))
    patch='200017197_00172_0_p00'
    cases.append(('heldout_alpha_0.70',ROOT/'tmp_work/targeted_thin_fade_20261021/cases/alpha_0.70/input.png',
        ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(patch+'_high_centers.png')))
    rows=[]
    for case,path,tpath in cases:
        inp=read(path);t=trace(tpath);y,x=np.where(t)
        x0=int(np.clip(round((x.min()+x.max())/2)-128,0,256))
        y0=int(np.clip(round((y.min()+y.max())/2)-128,0,256))
        tc=t[y0:y0+256,x0:x0+256]
        assert tc.sum()==t.sum()
        for (stem,teacher),(model,kind) in models.items():
            pred=neural_mask(model,kind,inp[y0:y0+256,x0:x0+256],device)
            rows.append(dict(case=case,model=stem,teacher=teacher,crop_xy=[x0,y0],
                input_sha256=sha(path),**coverage(pred,tc)))
    dest=OUT/'evaluation'/'native_256_audit.json'
    dest.write_text(json.dumps(dict(rows=rows,formal_accuracy=False),ensure_ascii=False,indent=2)+'\n')
    for row in rows:print(row['case'],row['model'],row['teacher'],row['r0'],'/',row['n'],'r2',row['r2'])
if __name__=='__main__':main()
