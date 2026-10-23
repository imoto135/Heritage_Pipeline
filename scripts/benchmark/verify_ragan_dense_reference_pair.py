"""CPU-only objective/gradient/input/guard checks, with zero optimizer steps."""
import json
import subprocess
from types import SimpleNamespace
import torch
from torch.nn import functional as F
import run_ragan_dense_reference_pair as run
from ragan_author_dense_support import GPUUnavailable,gpu_window,author_objectives,make_models,state_digest,tensor_digest,gradient_norm
from compare_v5_aligned_fade_control import dump,sha


def main():
    plan=run.check();out=run.OUT;assert not (out/'verification.json').exists()
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    guard=[]
    for label,code,stdout,stderr,allowed in [
        ('empty',0,'','',True),('self_only',0,'99\n99\n','',True),('foreign_even_on_other_gpu',0,'99\n123\n','',False),
        ('driver_error',9,'','driver unavailable',False),('unrecognized',0,'N/A\n','',False)]:
        try:gpu_window(SimpleNamespace(returncode=code,stdout=stdout,stderr=stderr),own_pid=99);passed=True
        except GPUUnavailable:passed=False
        assert passed==allowed,label;guard.append(dict(case=label,allowed=passed))
    checked=run.validate_data();assert checked==2440
    ds,rows=run.datasets();schedule=run.read(out/'schedule.json')
    first_hashes=[tensor_digest(run.batch_data(ds['train'],e)) for e in schedule[:64]]
    replay=[tensor_digest(run.batch_data(ds['train'],e)) for e in schedule[:64]]
    assert first_hashes==replay
    # A single fixture for literal arithmetic checks; these are tensors, not
    # updates to any model, and use the actual author's Loss_Doc below.
    g,d,loss_doc=make_models(plan['seed']);initial_g=state_digest(g)
    g2,d2,_=make_models(plan['seed']);assert state_digest(g2)==initial_g and state_digest(d2)==state_digest(d)
    del g2,d2
    objective=author_objectives()
    rs=torch.tensor([[.2,.7]],requires_grad=True);fs=torch.tensor([[.1,.4]],requires_grad=True);answers=iter([rs,fs])
    dl,_,_=objective['d'](lambda *a:next(answers),torch.zeros(1),torch.zeros(1),torch.zeros(1))
    literal_d=F.relu(1-rs.mean())+F.relu(1+fs.mean());torch.testing.assert_close(dl,literal_d,rtol=0,atol=0)
    p=torch.linspace(.05,.95,256).reshape(1,1,16,16).requires_grad_();t=torch.ones_like(p);t[:,:,5:9,4:12]=0
    rgb=torch.zeros(1,3,16,16)
    fake=lambda pp,rr:torch.sigmoid(pp.mean((2,3)))
    total,l1,bce,dice,jaccard,adv=objective['g'](fake,loss_doc,rgb,p,t)
    a,b,c,e=loss_doc(rgb,p,t);literal_g=10*a+b+c+e+.1*F.binary_cross_entropy(fake(p,rgb),torch.ones_like(fake(p,rgb)))
    torch.testing.assert_close(total,literal_g,rtol=0,atol=0)
    grad1=torch.autograd.grad(total,p,retain_graph=True)[0];grad2=torch.autograd.grad(literal_g,p)[0]
    torch.testing.assert_close(grad1,grad2,rtol=1e-6,atol=1e-7)
    # Native256 real denseGT input, raw unmodified author G/D, no adapters.
    batch=run.batch_data(ds['train'],schedule[0]);x,o,s,target=[a[:1] for a in batch]
    gp=list(g.parameters());dp=list(d.parameters());g_before=tensor_digest(gp);d_before=tensor_digest(dp)
    registered_before={n:id(v) for n,v in g.named_parameters()}
    g.train();d.train();pred=g(x,o,s);assert tuple(pred.shape)==(1,1,256,256) and torch.isfinite(pred).all()
    dloss,_,_=objective['d'](d,target,x,pred);dnorm=gradient_norm(dloss,dp,retain_graph=False)
    gl,l1,bce,dice,jaccard,advloss=objective['g'](d,loss_doc,x,pred,target)
    snorm=gradient_norm(10*l1+bce+dice+jaccard,gp)
    anorm=gradient_norm(.1*advloss,gp,retain_graph=False)
    assert all(torch.isfinite(v).item() for v in [dloss,gl,advloss]) and min(dnorm,snorm,anorm)>0
    assert tensor_digest(gp)==g_before and tensor_digest(dp)==d_before
    assert all(p.grad is None for p in gp+dp)
    replacements=[n for n,v in g.named_parameters() if registered_before[n]!=id(v)]
    assert len(replacements)==4 and all(n.endswith('diff_conv.weight') for n in replacements)
    assert not torch.cuda.is_initialized()
    run.inspect();readiness=run.read(out/'latest_readiness.json')
    dump(out/'verification.json',dict(protocol_sha256=sha(out/'protocol.json'),verification_script_sha256=sha(__file__),
        no_optimizer_steps=True,no_CUDA_initialization=True,objectives_match_literal_source=True,
        literal_G_gradient_matches=True,identical_initial_G_and_D=True,matched_augmented_batch_hashes=first_hashes,
        replayed_batches=64,verified_materialized_files=checked,gpu_guard_checks=guard,
        native256_untrained_gradient_probe=dict(D_gradient_l2=dnorm,G_supervised_gradient_l2=snorm,weighted_G_adversarial_gradient_l2=anorm,
            ratio=anorm/snorm,interpretation='Single random-initialization CPU forward/derivative check, not learning or evidence of GAN efficacy.'),
        original_optimized_parameters_unchanged=True,expected_author_derived_parameter_replacements=replacements,
        spectral_normalization_buffers_may_update_without_learning=True,current_gpu_readiness=readiness))
    print('Verified literal objectives/gradients,64paired augmented batches,2440files,5GPUguard cases; optimizer steps0; CUDA not initialized',flush=True)


if __name__=='__main__':main()
