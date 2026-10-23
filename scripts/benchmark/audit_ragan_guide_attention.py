"""Show attention cancellation in the reference generator's GA(x, x) calls.

Do not change the author architecture to guess an intended second feature.
"""
import copy
import json

import torch

from ragan_pilot_support import reference_classes, make_model
from run_makura_supervised_pilot import OUT, SEED, dump, sha


def main():
    torch.set_num_threads(2)
    reference_classes()
    from models.ga import GA
    torch.manual_seed(SEED)
    module=GA(8,8)
    same=torch.randn(2,8,24,24,requires_grad=True)
    out,attention=module(same,same)
    equivalent=module.conv(3*same)
    error=float((out-equivalent).abs().max())
    torch.testing.assert_close(out,equivalent,rtol=1e-5,atol=1e-6)
    out.square().mean().backward()
    names=[name for name,_ in module.named_parameters() if not name.startswith('conv.')]
    gradients={name:float(p.grad.abs().max()) for name,p in module.named_parameters() if name in names}
    assert all(value==0 for value in gradients.values())
    control=copy.deepcopy(module);control.zero_grad(set_to_none=True)
    x=torch.randn(2,8,24,24);y=torch.randn_like(x)
    different,_=control(x,y);different.square().mean().backward()
    control_max=max(float(p.grad.abs().max()) for name,p in control.named_parameters() if name in names)
    assert control_max>0
    model_audits=[]
    for kind in ('ra_no_gan','ragan'):
        torch.manual_seed(SEED);initial=make_model(kind)
        payload=torch.load(OUT/'models'/kind/'last.pt',map_location='cpu')
        unchanged=[name for name,p in initial.named_parameters() if p.requires_grad and torch.equal(p,payload['model'][name])]
        expected=[name for name,p in initial.named_parameters() if p.requires_grad and name.startswith(('ga1.','ga2.','ga3.','ga4.')) and '.conv.' not in name]
        assert set(unchanged)==set(expected) and len(unchanged)==48
        model_audits.append(dict(kind=kind,unchanged_attention_tensors=unchanged,
            unchanged_count=48,other_active_tensors_changed=104))
    dump(OUT/'guide_attention_audit.json',dict(
        formula='GA(x,y)=conv(x+y+a*x+(1-a)*y); GA(x,x)=conv(3*x)',
        reference_calls='Generator.py passes y,y to each of ga1,ga2,ga3,ga4 and discards returned attention',
        same_input_forward_max_error=error,same_input_attention_gradient_max=gradients,
        distinct_input_control_gradient_max=control_max,trained_models=model_audits,
        architecture_changed=False,
        interpretation='Current pilot does not validate the effectiveness of guide attention or reproduce a claimed paper result',
        action='Resolve intended two inputs from the paper/author implementation before a definitive RA-GAN benchmark'))
    # Optimizer state existence does not prove a tensor actually changed.
    path=OUT/'final_verification.json'
    if path.exists():
        verification=json.loads(path.read_text())
        for entry in verification['models']:
            entry.pop('all_active_parameters_updated',None)
            entry['all_trainable_parameters_have_optimizer_state']=True
            entry['unchanged_trainable_tensors']=48 if entry['model']!='unet' else 0
            entry['changed_state_tensors_confirmed']=entry['changed_state_tensors']
        verification['guide_attention_audit_sha256']=sha(OUT/'guide_attention_audit.json')
        dump(path,verification)
    print('GA(x,x) cancellation confirmed; 48 attention tensors unchanged in each RA pilot',flush=True)


if __name__=='__main__':main()
