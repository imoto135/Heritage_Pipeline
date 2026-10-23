"""Measure supervised/adversarial generator gradients without modifying checkpoints."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_convergence_audit import OUT
from ragan_pilot_support import make_model,predict,auxiliaries,supervised_loss,critic_input,reference_classes
from compare_v5_aligned_fade_control import rgb,sha,dump


def main():
    dest=OUT/'gradient_audit.json';assert not dest.exists()
    frozen=json.loads((OUT/'fixed_batch_fit/protocol.json').read_text())
    parent=ROOT/'tmp_work/makura_contact_stain_train_control_20261022/ragan_contact'
    records=[json.loads(s) for s in (parent/'train_manifest.jsonl').read_text().splitlines()]
    images=[];targets=[];valids=[]
    for info in frozen['records']:
        row=next(r for r in records if r['input_sha256']==info['input_sha256'])
        x,y=info['crop_xy'];image=rgb(ROOT/row['input'])[y:y+256,x:x+256]
        images.append(auxiliaries(image))
        targets.append(np.asarray(Image.open(ROOT/row['labels']['target']).convert('L'))[y:y+256,x:x+256])
        valids.append(np.asarray(Image.open(ROOT/row['labels']['valid']).convert('L'))[y:y+256,x:x+256]>0)
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    image=torch.from_numpy(np.stack(images)).permute(0,3,1,2).float().to(device)/255
    target=torch.from_numpy(np.stack(targets)).unsqueeze(1).float().to(device)/255
    valid=torch.from_numpy(np.stack(valids)).unsqueeze(1).float().to(device)
    rows=[]
    for condition in ('ragan_original','ragan_contact'):
        for step in (432,2160):
            ck=OUT/condition/'models/ragan'/f'step_{step:05d}.pt'
            payload=torch.load(ck,map_location='cpu',weights_only=False)
            model=make_model('ragan').to(device);model.load_state_dict(payload['model']);model.eval()
            _,_,D=reference_classes();discriminator=D(4).to(device)
            discriminator.load_state_dict(payload['discriminator']);discriminator.eval()
            for p in discriminator.parameters():p.requires_grad_(False)
            p=predict(model,'ragan',image)
            sup=supervised_loss(p,target,valid)
            fake=discriminator(critic_input(p,valid),image[:,:3])
            adv=.1*torch.nn.functional.binary_cross_entropy(fake.clamp(1e-6,1-1e-6),torch.ones_like(fake))
            named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
            params=[p for _,p in named]
            sg=torch.autograd.grad(sup,params,retain_graph=True,allow_unused=True)
            ag=torch.autograd.grad(adv,params,allow_unused=True)
            sn=an=dot=0.;attention=[]
            for (name,_),s,a in zip(named,sg,ag):
                if s is not None:sn+=float(s.double().square().sum())
                if a is not None:an+=float(a.double().square().sum())
                if s is not None and a is not None:dot+=float((s.double()*a.double()).sum())
                if name.startswith(('ga1.','ga2.','ga3.','ga4.')) and '.conv.' not in name:
                    attention.append(dict(name=name,supervised_max=0. if s is None else float(s.abs().max()),
                        adversarial_max=0. if a is None else float(a.abs().max())))
            with torch.inference_mode():real=discriminator(critic_input(target,valid),image[:,:3])
            rows.append(dict(condition=condition,step=step,checkpoint_sha256=sha(ck),
                supervised_loss=float(sup.detach()),weighted_adversarial_loss=float(adv.detach()),
                supervised_grad_norm=sn**.5,weighted_adversarial_grad_norm=an**.5,
                weighted_adversarial_to_supervised_norm=(an/max(sn,1e-30))**.5,
                grad_cosine=dot/max((sn*an)**.5,1e-30),
                critic_real=real.flatten().cpu().tolist(),critic_fake=fake.detach().flatten().cpu().tolist(),
                attention_gradients=attention))
            print(condition,step,'grad ratio',rows[-1]['weighted_adversarial_to_supervised_norm'],
                'critic means',float(real.mean()),float(fake.detach().mean()),flush=True)
            del model,discriminator,payload,p;torch.cuda.empty_cache()
    dump(dest,dict(rows=rows,batch='four fixed training crops; mixed clean and dark stain',
        frozen_batch_protocol_sha256=sha(OUT/'fixed_batch_fit/protocol.json'),
        eval_mode=True,no_updates=True,gradient_norms_are_local_diagnostic_not_global_GAN_effect=True))


if __name__=='__main__':main()
