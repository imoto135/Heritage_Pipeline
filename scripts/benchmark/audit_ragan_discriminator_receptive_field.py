"""Analytic receptive field and actual critic outputs at the training/inference sizes."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from diagnose_ragan_discriminator import OUT,CHECKPOINT
from ragan_pilot_support import reference_classes
from compare_v5_aligned_fade_control import dump


def main():
    dest=OUT/'receptive_field_audit.json';assert not dest.exists()
    torch.set_num_threads(2);_,_,D=reference_classes();model=D(4).eval()
    layers=[(f'discriminator.{i}.conv2d',m.conv2d) for i,m in enumerate(model.discriminator)]
    layers.append(('shrink.0',model.shrink[0]))
    r=1;jump=1;rows=[]
    for name,conv in layers:
        kernel=conv.kernel_size[0];stride=conv.stride[0];dilation=conv.dilation[0]
        r+=(kernel-1)*dilation*jump;jump*=stride
        rows.append(dict(name=name,kernel=kernel,stride=stride,padding=conv.padding[0],
            theoretical_receptive_field=r,output_jump=jump))
    observed=[]
    with torch.inference_mode():
        for size in (256,512):
            output=model(torch.zeros(1,1,size,size),torch.zeros(1,3,size,size))
            observed.append(dict(input_hw=size,critic_output_shape=list(output.shape),scores_per_image=output.shape[1]))
    assert r==323 and observed[0]['scores_per_image']==1 and observed[1]['scores_per_image']==25
    dump(dest,dict(layers=rows,observed=observed,
        training_size=256,training_scores_per_image=1,theoretical_rf=323,
        note='Theoretical footprint includes padded coordinates. One training score per crop; this does not prove inability to detect local defects.',
        source='local frozen author Discriminator.py and SNConv.py; no architecture changes'))
    print('D receptive field',r,'scores',observed,flush=True)


if __name__=='__main__':main()
