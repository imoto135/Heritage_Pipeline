"""Do frozen critics correct actual generator false positives/negatives? No G updates."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from diagnose_ragan_discriminator import OUT,CHECKPOINT,CONDITIONS,dataset,get_discriminator
from ragan_pilot_support import make_model,predict,auxiliaries,critic_input,supervised_loss
from compare_v5_aligned_fade_control import sha,dump


def main():
    dest=OUT/'native_generator_feedback.json';assert not dest.exists()
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    generator=make_model('ragan').to(device);generator.load_state_dict(payload['model']);generator.eval()
    named=[(n,p) for n,p in generator.named_parameters() if p.requires_grad]
    data=dataset('val',device);entries=[];batch_context=[]
    for condition in ['initial']+list(CONDITIONS):
        objective='sigmoid_scalar_hinge' if condition=='initial' else CONDITIONS[condition]['objective']
        discriminator=get_discriminator(payload,objective,device)
        checkpoint=CHECKPOINT
        if condition!='initial':
            checkpoint=OUT/condition/'last.pt'
            discriminator.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['discriminator'])
        discriminator.eval()
        for p in discriminator.parameters():p.requires_grad_(False)
        for start in range(0,len(data['rows']),4):
            inp_rgb=np.stack([np.load(OUT/r['cache'])['image'] for r in data['rows'][start:start+4]])
            image=torch.from_numpy(np.stack([auxiliaries(a) for a in inp_rgb])).permute(0,3,1,2).float().to(device)/255
            single_inputs=[torch.from_numpy(auxiliaries(a)).permute(2,0,1).unsqueeze(0).float().to(device)/255 for a in inp_rgb]
            # GGC.theta is reduced across batch as well as space in the public code.
            # Cached generator fakes were made one image at a time: preserve that context.
            if condition=='initial':
                with torch.inference_mode():
                    batched=predict(generator,'ragan',image)
                    single=torch.cat([predict(generator,'ragan',x) for x in single_inputs])
                    delta=(batched-single).abs()
                    batch_context.append(dict(batch_start=start,max_probability_difference=float(delta.max()),
                        mean_probability_difference=float(delta.mean()),
                        threshold_flips=int(((batched<.5)!=(single<.5)).sum()),pixels=single.numel()))
            p=torch.cat([predict(generator,'ragan',x) for x in single_inputs])
            target=data['real'][start:start+4];valid=data['valid'][start:start+4]
            torch.testing.assert_close(p,data['generator_soft'][start:start+4],rtol=1e-4,atol=1e-5)
            sup=supervised_loss(p,target,valid)
            score=discriminator(critic_input(p,valid),image[:,:3])
            adv=-.1*score.mean() if objective=='logit_pointwise_hinge' else .1*F.binary_cross_entropy(score,torch.ones_like(score))
            agp=torch.autograd.grad(adv,p,retain_graph=True)[0]
            sgp=torch.autograd.grad(sup,p,retain_graph=True)[0]
            sg=torch.autograd.grad(sup,[p for _,p in named],retain_graph=True,allow_unused=True)
            ag=torch.autograd.grad(adv,[p for _,p in named],allow_unused=True)
            sn=sum(float(g.double().square().sum()) for g in sg if g is not None)**.5
            an=sum(float(g.double().square().sum()) for g in ag if g is not None)**.5
            entry=dict(condition=condition,batch_start=start,critic_checkpoint_sha256=sha(checkpoint),
                generator_adv_grad_norm=an,generator_supervised_grad_norm=sn,
                weighted_GAN_to_supervised_parameter_grad_ratio=an/max(sn,1e-30),errors=[])
            fp=(p<.5)&(target>.5)&(valid>.5)
            fn=(p>=.5)&(target<.5)&(valid>.5)
            for key,region,sign in [('false_positive_ink',fp,-1),('false_negative_ink',fn,1)]:
                values=agp[region];correct=sign*values>0
                entry['errors'].append(dict(type=key,n=int(values.numel()),
                    desired_gradient_pixels=int(correct.sum()),
                    desired_direction_fraction=float(correct.float().mean()) if values.numel() else None,
                    mean_desired_signed_gradient=float((sign*values).mean()) if values.numel() else None,
                    mean_abs_gradient=float(values.abs().mean()) if values.numel() else None))
            entries.append(entry)
        print(condition,'native feedback evaluated',flush=True)
        del discriminator
    dump(dest,dict(rows=entries,generator_checkpoint_sha256=sha(CHECKPOINT),val_groups=8,
        weighted_adversarial_coefficient=.1,threshold=.5,source_teacher_provisional=True,
        no_generator_updates=True,parameter_gradient_ratios_are_local_diagnostics=True,
        no_restoration_improvement_claim=True))
    dump(OUT/'generator_batch_context_audit.json',dict(rows=batch_context,
        reason='public GGC theta_net(image).mean() reduces batch and spatial dimensions',
        batch_and_input_memory_layout_both_change=True,
        attribution_of_observed_difference_to_theta_alone_not_established=True,
        same_model_eval_mode=True,same_RGB=True,threshold=.5,
        primary_feedback_audit_uses_single_image_forward_to_match_cached_fakes=True))


if __name__=='__main__':main()
