"""Repeat one frozen near configuration with strict deterministic math."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json,shutil
from pathlib import Path
import torch
import run_paper_location_control as location
from deterministic_reflection_support import install
from compare_v5_aligned_fade_control import dump,sha
ROOT=location.ROOT;PARENT=location.OUT;OLD=location.OLD;base=location.base
OUT=ROOT/'tmp_work/deterministic_training_repeat_20261023';ARMS=['repeat0','repeat1']
def parent_check():
    location.previous.check();p=json.loads((PARENT/'protocol.json').read_text())
    for n,h in p['frozen_files'].items():assert sha(ROOT/n)==h,n
    return p
def freeze():
    assert not OUT.exists();p=parent_check();OUT.mkdir();files=[PARENT/'protocol.json',Path(__file__),ROOT/'scripts/benchmark/deterministic_reflection_support.py',ROOT/'tmp_work/training_update_repeatability_v3_20261023/results.json']
    for arm in ARMS:
        d=OUT/'paper_masks'/arm;d.mkdir(parents=True)
        for src in sorted((PARENT/'paper_masks/near').glob('*.png')):
            dest=d/src.name;shutil.copy2(src,dest);assert sha(src)==sha(dest);files.append(dest)
    p.update(arms=ARMS,kinds=['ragan'],seeds=[1],purpose='Two identical repeats of same frozen near-teacher setup, seed1. Tests end-to-end numerical reproducibility, not two independent seeds or a new paper-location ablation.',
        numerics=dict(deterministic_algorithms=True,cudnn_deterministic=True,cudnn_benchmark=False,matmul_tf32=False,cudnn_tf32=False,cublas_workspace=':4096:8',reflection='Slicing/flip/concatenation before identical Conv2d; parameter and forward reflection values unchanged'),
        primary='Compare final model and Adam tensors plus all156 readout masks. Do not choose the better repeat.',limits='Numeric math flags and reflection backward summation differ from older runs. Metrics not a single-factor performance comparison to older runs.',
        frozen_files={str(f.relative_to(ROOT)):sha(f) for f in files})
    dump(OUT/'protocol.json',p);(OUT/'source_snapshot').mkdir()
    for f in [Path(__file__),ROOT/'scripts/benchmark/deterministic_reflection_support.py']:shutil.copy2(f,OUT/'source_snapshot'/f.name)
    print('Frozen2 identical864-update repeats, strict deterministic, same seed1',flush=True)
def check():
    parent_check();p=json.loads((OUT/'protocol.json').read_text())
    for n,h in p['frozen_files'].items():assert sha(ROOT/n)==h,n
    return p
def train(arm,device):
    check();torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original=location.hooked
    def deterministic_hooked(kind):
        model,state=original(kind);names=install(model);assert names;return model,state
    location.OUT=OUT;location.check=check;location.hooked=deterministic_hooked
    location.train('ragan',1,arm,device)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','train']);a.add_argument('--arm',choices=ARMS);a.add_argument('--device',default='cuda:0');v=a.parse_args();freeze() if v.action=='freeze' else train(v.arm,v.device)
