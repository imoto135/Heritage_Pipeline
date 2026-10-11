"""Independent direct adapter/channel audit for the new comparison run."""
import sys,json
from pathlib import Path
import numpy as np,torch
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_npj_literature_models import load,model_call
from run_direct_literature_baselines import invoke

def main():
    dest=OUT/'direct_literature_baseline_audit';dest.mkdir(exist_ok=True)
    class StainIdentity(torch.nn.Module):
        def forward(self,x):return [x,x]
    class DocIdentity(torch.nn.Module):
        def forward(self,x):return x[:,:3]
    # Explicit BGR byte array; PIL receives reversed RGB once only.
    bgr=np.array([[[0,0,255],[255,0,0],[0,255,0],[40,120,220]]],np.uint8)
    for name,model in [('stain',StainIdentity()),('docres_appearance_native',DocIdentity())]:
        inp=bgr if name=='stain' else np.concatenate([bgr,bgr],2)
        v=invoke(model,inp,name,'cpu');result=np.rint(v*255).astype(np.uint8)
        assert np.array_equal(result,bgr);Image.fromarray(result[:,:,::-1]).save(dest/(name+'_identity.png'))
        assert np.array_equal(np.asarray(Image.open(dest/(name+'_identity.png'))),bgr[:,:,::-1])
    # Same original native 256 crop; compare the previous independently audited
    # author-tensor adapter with the new adapter before PNG conversion.
    c=next(r for r in json.loads((OUT/'cases_frozen.json').read_text())['cases'] if r['case_id']=='dev_08')
    rgb=np.asarray(Image.open(ROOT/c['original']).convert('RGB'))[620:876,680:936];a=rgb[:,:,::-1].copy()
    model,resources=load('stain','cuda:0')
    old=model_call(model,a,'stain','cuda:0');new=invoke(model,a,'stain','cuda:0');err=float(np.max(np.abs(old-new)))
    assert err<=1e-6
    rows=dict(identity_BGR_RGB_PNG_exact=True,StainRestorer_direct_old_adapter_max_float_error=err,
        resources={str(f):file_hash(f) for f in resources},script_sha256=file_hash(__file__),
        new_adapter_sha256=file_hash(OUT/'run_direct_literature_baselines.py'),
        old_adapter_sha256=file_hash(ROOT/'scripts/benchmark/run_npj_literature_models.py'),
        author_appearance_precision_note='Float16 author-geometry forward, float32 output conversion before uint8 truncation. Author code truncates a float16 NumPy array; exact bitwise author benchmark equality is not claimed.',
        docres_weights='Community mirror, agreed hash against second mirror; identity to author OneDrive copy not independently established',
        task='New thin-red/brown comparison, not official paired benchmark reproduction',real_GT=False)
    (dest/'verification.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print('Direct color adapters independently verified; Stain max error',err,flush=True)

if __name__=='__main__':main()
