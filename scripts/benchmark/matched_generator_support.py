"""Common supervised objective for one-logit UNet and public product-head G."""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from ragan_pilot_support import make_model,predict
from run_ragan_stable_bce_control import product_logit,stable_loss
from ragan_local_region_support import local_terms


def logit_of(kind,raw):
    return raw.float() if kind=='unet' else product_logit(raw)


def objective(paper,target,valid,logits,positive,negative):
    p=paper.float().clamp(1e-6,1-1e-6);t=target.float();v=valid.float();n=v.sum().clamp_min(1)
    l1=((p-t).abs()*v).sum()/n
    bce=(F.binary_cross_entropy_with_logits(logits,t,reduction='none')*v).sum()/n
    ink=(1-p)*v;truth=(1-t)*v;intersection=(ink*truth).sum()
    dice=1-(2*intersection+1e-6)/(ink.sum()+truth.sum()+1e-6)
    jaccard=1-(intersection+1e-6)/(ink.sum()+truth.sum()-intersection+1e-6)
    global_loss=10*l1+bce+dice+jaccard
    core,paper_loss=local_terms(logits,positive,negative)
    return global_loss+core+paper_loss,dict(global_loss=global_loss,normalized_ink_BCE=core,normalized_paper_BCE=paper_loss)


def hooked(kind):
    model=make_model(kind);state={}
    head=model.output if kind=='unet' else model.corase_c6s.gate01.gate01
    head.register_forward_hook(lambda module,args,out:state.update(raw=out))
    return model,state


def checks():
    torch.manual_seed(31)
    raw=torch.randn(2,2,9,9,requires_grad=True)
    a,b=raw.chunk(2,1);p=torch.sigmoid(a)*torch.sigmoid(b)
    t=torch.randint(0,2,p.shape).float();v=(torch.rand_like(p)>.2).float()
    positive=(t==0)&(v>0)&(torch.rand_like(p)>.8)
    negative=(t==1)&(v>0)&(torch.rand_like(p)>.8)
    z=product_logit(raw)
    new,_=objective(p,t,v,z,positive,negative)
    ci,cp=local_terms(z,positive,negative)
    old=stable_loss(p,t,v,raw)+ci+cp
    torch.testing.assert_close(new,old,rtol=2e-6,atol=1e-6)
    torch.testing.assert_close(torch.autograd.grad(new,raw,retain_graph=True)[0],torch.autograd.grad(old,raw)[0],rtol=2e-5,atol=1e-7)
    # An identical paper probability/logit gives an identical objective,
    # independent of which model's output parameterization produced it.
    z=z.detach().requires_grad_(True)
    pu=torch.sigmoid(z)
    direct,_=objective(pu,t,v,z,positive,negative)
    reference,_=objective(p.detach(),t,v,z,positive,negative)
    torch.testing.assert_close(direct,reference,rtol=3e-6,atol=2e-6)
    altered=torch.where(v>0,t,1-t)
    unchanged,_=objective(pu,altered,v,z,positive,negative)
    torch.testing.assert_close(direct,unchanged,rtol=0,atol=0)
    g=torch.autograd.grad(direct,z)[0];assert (g[v==0]==0).all()
    for kind in ['unet','ragan']:
        shape=(1,1 if kind=='unet' else 2,3,3)
        r=torch.full(shape,30.,requires_grad=True)
        logit=logit_of(kind,r)
        prob=torch.sigmoid(r) if kind=='unet' else torch.sigmoid(r[:,:1])*torch.sigmoid(r[:,1:])
        valid=torch.ones_like(prob);zeros=torch.zeros_like(prob);empty=torch.zeros_like(prob,dtype=torch.bool)
        loss,_=objective(prob,zeros,valid,logit,empty,empty)
        grad=torch.autograd.grad(loss,r)[0]
        assert torch.isfinite(loss) and (grad>0).all()
    return dict(RA_objective_matches_previous=True,RA_gradients_match_previous=True,
        same_probability_and_logit_same_loss=True,unknown_targets_invariant=True,unknown_gradient_zero=True,
        saturated_wrong_ink_nonzero_BCE_gradient_both_heads=True)
