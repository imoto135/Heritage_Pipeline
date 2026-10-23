"""Can the public-code generator fit four frozen clean/contact training crops?"""
from __future__ import annotations
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
from ragan_pilot_support import make_model, predict, auxiliaries, supervised_loss, critic_input, reference_classes
from compare_v5_aligned_fade_control import sha, dump, rgb, save
from evaluate_ridge_gate_independent_val import proxies
from run_contact_stain_training_control import OUT as PARENT
import apply_ragan_real_regions as real

OUT = ROOT / 'tmp_work/ragan_convergence_audit_20261022/fixed_batch_fit'
SEED = 20261021


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    patches = ['200017197_00103_1_p00', '200017197_00214_1_p00']
    rows = [json.loads(s) for s in (PARENT / 'ragan_contact/train_manifest.jsonl').read_text().splitlines()]
    anchors = {r['patch']:r['anchor_xy'] for r in json.loads((PARENT / 'input_audit.json').read_text())['rows']}
    data = []
    for patch in patches:
        for family in ('identity','stain_overlap'):
            r = next(r for r in rows if r['patch_id']==patch and r['damage_family']==family
                and (family=='identity' or float(r['damage_strength'])==.75))
            cx, cy = anchors[patch]
            x, y = int(np.clip(cx-128,0,256)), int(np.clip(cy-128,0,256))
            source = rgb(ROOT / r['clean'])[y:y+256,x:x+256]
            image = rgb(ROOT / r['input'])[y:y+256,x:x+256]
            target = np.asarray(Image.open(ROOT/r['labels']['target']).convert('L'))[y:y+256,x:x+256]
            valid = np.asarray(Image.open(ROOT/r['labels']['valid']).convert('L'))[y:y+256,x:x+256]>0
            _, core, remote, _ = proxies(source)
            data.append(dict(id=patch+'_'+family, source=source, image=image, target=target, valid=valid,
                core=core, remote=remote, record=r, crop_xy=[x,y]))
    dump(OUT/'protocol.json',dict(seed=SEED,patches=patches,training_examples=4,
        steps=400,fixed_batch_size=4,lr=3e-5,threshold=.5,
        conditions=['generator_only','gan'],same_generator_initialization=True,
        no_validation_or_generalization_claim=True,provisional_labels=True,
        records=[dict(id=d['id'],input_sha256=d['record']['input_sha256'],crop_xy=d['crop_xy'],
            label_target_sha256=d['record']['labels']['target_sha256'],
            label_valid_sha256=d['record']['labels']['valid_sha256']) for d in data]))
    torch.set_num_threads(2)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25,device)
    image=torch.from_numpy(np.stack([auxiliaries(d['image']) for d in data])).permute(0,3,1,2).float().to(device)/255
    target=torch.from_numpy(np.stack([d['target'] for d in data])).unsqueeze(1).float().to(device)/255
    valid=torch.from_numpy(np.stack([d['valid'] for d in data])).unsqueeze(1).float().to(device)
    histories={};outputs={}
    for condition in ('generator_only','gan'):
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
        model=make_model('ragan').to(device)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-5)
        discriminator=None
        if condition=='gan':
            _,_,D=reference_classes();discriminator=D(4).to(device)
            d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=1e-6)
        history=[];start=time.monotonic()
        for step in range(401):
            if step in (0,100,200,400):
                model.eval()
                with torch.inference_mode():
                    probability=predict(model,'ragan',image)
                    pred=probability<.5;truth=target<.5;mask=valid>.5
                    tp=int((pred&truth&mask).sum());fp=int((pred&~truth&mask).sum());fn=int((~pred&truth&mask).sum())
                    history.append(dict(step=step,ink_f1_to_provisional=2*tp/max(1,2*tp+fp+fn),
                        tp=tp,fp=fp,fn=fn,supervised_loss=float(supervised_loss(probability,target,valid)),
                        seconds=time.monotonic()-start))
                print(condition,history[-1],flush=True)
                if step==400:
                    outputs[condition]=pred[:,0].cpu().numpy()
                    torch.save(dict(model=model.state_dict(),step=400),OUT/(condition+'.pt'))
                    break
            model.train();optimizer.zero_grad(set_to_none=True)
            probability=predict(model,'ragan',image)
            adversarial=probability.new_zeros(())
            if discriminator is not None:
                discriminator.train()
                for p in discriminator.parameters():p.requires_grad_(True)
                d_optimizer.zero_grad(set_to_none=True)
                real_value=discriminator(critic_input(target,valid),image[:,:3]).mean()
                fake_value=discriminator(critic_input(probability.detach(),valid),image[:,:3]).mean()
                d_loss=torch.relu(1-real_value)+torch.relu(1+fake_value)
                d_loss.backward();d_optimizer.step()
                discriminator.eval()
                for p in discriminator.parameters():p.requires_grad_(False)
                judgment=discriminator(critic_input(probability,valid),image[:,:3]).clamp(1e-6,1-1e-6)
                adversarial=torch.nn.functional.binary_cross_entropy(judgment,torch.ones_like(judgment))
            loss=supervised_loss(probability,target,valid)+.1*adversarial
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        histories[condition]=history
        del model,optimizer,discriminator
        torch.cuda.empty_cache()
    measures=[]
    for i,d in enumerate(data):
        display=[('原本',d['source']),('学習入力',d['image']),('暫定教師',d['target'])]
        for condition in outputs:
            pred=outputs[condition][i]
            path=OUT/'masks'/d['id']/(condition+'.png');save(path,pred)
            measures.append(dict(id=d['id'],condition=condition,
                source_core_n=int(d['core'].sum()),source_core_exact=int((pred&d['core']).sum()),
                source_remote_paper_n=int(d['remote'].sum()),source_remote_paper_ink=int((pred&d['remote']).sum())))
            display.append((condition,np.where(pred,0,255).astype(np.uint8)))
        real.panel(OUT/'figures'/(d['id']+'_原本比較.png'),'固定4画像への当てはめ / '+d['id'],display)
    dump(OUT/'results.json',dict(histories=histories,proxy_metrics=measures,
        same_four_training_images=True,provisional_teacher=True,formal_accuracy=False))


if __name__=='__main__':main()
