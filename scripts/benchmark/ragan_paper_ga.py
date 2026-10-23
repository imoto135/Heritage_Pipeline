"""Equation-based RA-GAN attention candidate, NOT an official reproduction.

Paper Sec. 3.4, Eq. 9-12 (p.5): compute W from I and Wcat*I.
The paper does not give an explicit feature-output equation after W. This
candidate applies W multiplicatively to I, with no extra residual/1x1 conv.
This last choice is an explicit interpretation, not an author-confirmed fix.
The reference checkout and earlier experiment scripts stay unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ragan_pilot_support import ROOT, make_model, predict, auxiliaries

OUT=ROOT/'tmp_work/ragan_paper_alignment_20261021'


class PaperGuidedAttention(nn.Module):
    def __init__(self, channels, output_mode='multiply'):
        super().__init__()
        assert output_mode in ('multiply','residual_gate')
        self.output_mode=output_mode
        self.spatial=nn.Sequential(nn.Conv2d(2,2,1),nn.ReLU(),nn.Conv2d(2,1,7,padding=3))
        self.channel=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,groups=channels),
                                   nn.ReLU(),nn.Conv2d(channels,channels,1))
        self.pointwise=nn.Conv2d(2*channels,channels,1)
        self.depthwise=nn.Conv2d(channels,channels,5,padding=2,groups=channels)

    def weights(self,image):
        spatial=self.spatial(torch.cat((image.mean(1,keepdim=True),image.amax(1,keepdim=True)),1))
        channel=self.channel(image.mean((2,3),keepdim=True))
        mixed=spatial+channel
        # Interleave [I0, (Wcat*I)0, I1, (Wcat*I)1, ...], Eq.12 S.
        paired=torch.stack((image,mixed*image),dim=2).flatten(1,2)
        return torch.sigmoid(self.depthwise(self.pointwise(paired)))

    def forward(self,image,legacy_unused=None):
        weights=self.weights(image)
        feature=image*weights if self.output_mode=='multiply' else image*(1+weights)
        return feature,weights


def candidate_generator(output_mode='multiply'):
    model=make_model('ra_no_gan')
    for name in ('ga1','ga2','ga3','ga4'):
        setattr(model,name,PaperGuidedAttention(64,output_mode=output_mode))
    return model


def audit(device_name):
    torch.set_num_threads(2);torch.manual_seed(20261021)
    OUT.mkdir(exist_ok=True,parents=True)
    module=PaperGuidedAttention(8)
    x=torch.randn(2,8,24,24,requires_grad=True)
    weights=module.weights(x)
    # Independent direct expression for Eq.9-12, including interleave order.
    spatial=module.spatial(torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True).values],1))
    channel=module.channel(x.mean((2,3),keepdim=True))
    mixed=(spatial+channel)*x
    expected_pair=torch.cat([v for i in range(8) for v in (x[:,i:i+1],mixed[:,i:i+1])],1)
    expected=torch.sigmoid(module.depthwise(module.pointwise(expected_pair)))
    torch.testing.assert_close(weights,expected,rtol=0,atol=0)
    output,returned=module(x,x)
    torch.testing.assert_close(output,x*weights,rtol=0,atol=0)
    assert returned.min()>0 and returned.max()<1 and output.shape==x.shape
    output.square().mean().backward()
    branches={name:sum(float(p.grad.abs().sum()) for p in part.parameters()) for name,part in
              [('spatial',module.spatial),('channel',module.channel),('pointwise',module.pointwise),('depthwise',module.depthwise)]}
    assert all(v>0 for v in branches.values())
    original=weights.detach().clone()
    with torch.no_grad():module.spatial[-1].bias.add_(1.)
    assert not torch.equal(original,module.weights(x).detach())
    device=torch.device(device_name)
    if device.type=='cuda':
        torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.3,device)
    torch.manual_seed(20261021);model=candidate_generator().to(device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-5)
    before={name:p.detach().clone() for name,p in model.named_parameters() if name.startswith(('ga1.','ga2.','ga3.','ga4.'))}
    parameter_ids={name:id(p) for name,p in model.named_parameters()}
    histories=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        # Native random engineering input; not a synthetic damage quality test.
        image=torch.rand(1,5,256,256,device=device)
        prediction=predict(model,'ra_no_gan',image)
        assert prediction.shape==(1,1,256,256) and torch.isfinite(prediction).all()
        loss=(prediction-.8).square().mean();loss.backward()
        gradients={name:sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters())
                   for name in ('ga1','ga2','ga3','ga4')}
        assert all(value>0 for value in gradients.values())
        optimizer.step();histories.append(dict(step=step+1,loss=float(loss),ga_gradient_l1=gradients))
    assert parameter_ids=={name:id(p) for name,p in model.named_parameters()}
    unchanged=[name for name,p in model.named_parameters() if name in before and torch.equal(before[name],p)]
    changed=[name for name,p in model.named_parameters() if name in before and not torch.equal(before[name],p)]
    assert len(changed)==48 and not unchanged
    model.eval()
    with torch.inference_mode():
        shapes=[]
        for size in (256,512):
            p=predict(model,'ra_no_gan',torch.rand(1,5,size,size,device=device))
            assert p.shape==(1,1,size,size) and torch.isfinite(p).all()
            shapes.append(list(p.shape))
    report=dict(paper='RA-GAN, DOI 10.1016/j.knosys.2026.115860',page=5,equations=[9,10,11,12],
        expression_matches=True,pair_order='I0,mixed0,I1,mixed1,...',
        weight_range=[float(returned.min()),float(returned.max())],branch_gradients=branches,
        perturbing_attention_changes_weights=True,full_generator_updates=histories,
        updated_attention_tensors=len(changed),unchanged_attention_tensors=unchanged,
        parameter_registration_stable=True,generator_shapes=shapes,
        official_reproduction=False,quality_training=False,
        interpretation='Return I*W as feature; the paper defines W but does not explicitly specify this final output equation',
        assumptions=['Channels preserved without reduction bottleneck in Eq.10',
            'Zero padding for spatial/depthwise convolutions','Channel interleave interprets S',
            'GGC/GFP and discriminator still use the prior local reference adapter',
            'No claim that all other reference/paper discrepancies have been resolved'])
    (OUT/'paper_ga_candidate_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['updated_attention_tensors','parameter_registration_stable','generator_shapes','official_reproduction']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');args=parser.parse_args();audit(args.device)
