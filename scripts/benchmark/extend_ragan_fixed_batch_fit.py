"""Capacity-only follow-up: supervise the same four crops beyond 400 updates."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_convergence_audit import OUT
from compare_v5_aligned_fade_control import sha,dump,rgb,save
from ragan_pilot_support import make_model,predict,auxiliaries,supervised_loss
import apply_ragan_real_regions as real


def main(fresh=False):
    parent=OUT/'fixed_batch_fit';dest=OUT/('fixed_batch_capacity_fresh' if fresh else 'fixed_batch_capacity_extension')
    assert not dest.exists();dest.mkdir()
    protocol=json.loads((parent/'protocol.json').read_text())
    manifest=ROOT/'tmp_work/makura_contact_stain_train_control_20261022/ragan_contact/train_manifest.jsonl'
    rows=[json.loads(s) for s in manifest.read_text().splitlines()];data=[]
    for entry in protocol['records']:
        row=next(r for r in rows if r['input_sha256']==entry['input_sha256'])
        x,y=entry['crop_xy'];source=rgb(ROOT/row['clean'])[y:y+256,x:x+256]
        image=rgb(ROOT/row['input'])[y:y+256,x:x+256]
        target=np.asarray(Image.open(ROOT/row['labels']['target']).convert('L'))[y:y+256,x:x+256]
        valid=np.asarray(Image.open(ROOT/row['labels']['valid']).convert('L'))[y:y+256,x:x+256]>0
        data.append(dict(id=entry['id'],source=source,image=image,target=target,valid=valid))
    ck=parent/'generator_only.pt'
    max_steps=2000 if fresh else 1600
    parent_updates=0 if fresh else 400
    dump(dest/'protocol.json',dict(parent_checkpoint_sha256=None if fresh else sha(ck),parent_updates=parent_updates,
        additional_updates_max=max_steps,optimizer_reset=not fresh,lr=3e-5,GAN=False,same_fixed_four_crops=True,
        stop_at_provisional_training_f1=.995,evaluate_every=200,
        interpretation='capacity probe only; fresh seed and continuous optimizer' if fresh else 'capacity probe only; optimizer reset prevents direct duration-only comparison',
        provisional_teacher=True,formal_accuracy=False))
    torch.set_num_threads(2);torch.manual_seed(20261021)
    device=torch.device('cuda:1');torch.cuda.set_device(device)
    image=torch.from_numpy(np.stack([auxiliaries(d['image']) for d in data])).permute(0,3,1,2).float().to(device)/255
    target=torch.from_numpy(np.stack([d['target'] for d in data])).unsqueeze(1).float().to(device)/255
    valid=torch.from_numpy(np.stack([d['valid'] for d in data])).unsqueeze(1).float().to(device)
    model=make_model('ragan').to(device)
    if not fresh:model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model'])
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-5)
    history=[];start=time.monotonic();outputs=None
    for step in range(max_steps+1):
        if step%200==0:
            model.eval()
            with torch.inference_mode():
                p=predict(model,'ragan',image);pred=p<.5;truth=target<.5;mask=valid>.5
                tp=int((pred&truth&mask).sum());fp=int((pred&~truth&mask).sum());fn=int((~pred&truth&mask).sum())
                f1=2*tp/max(1,2*tp+fp+fn)
                history.append(dict(additional_step=step,total_G_updates=parent_updates+step,
                    provisional_training_f1=f1,tp=tp,fp=fp,fn=fn,loss=float(supervised_loss(p,target,valid)),
                    seconds=time.monotonic()-start))
                outputs=pred[:,0].cpu().numpy()
            dump(dest/'history.json',history);print(history[-1],flush=True)
            if step==max_steps or (step>0 and f1>=.995):break
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=supervised_loss(predict(model,'ragan',image),target,valid)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),additional_step=step),dest/'last.pt')
    for i,d in enumerate(data):
        save(dest/'masks'/(d['id']+'.png'),outputs[i])
        real.panel(dest/'figures'/(d['id']+'_原本比較.png'),'固定4画像の容量確認 / '+d['id'],[
            ('原本',d['source']),('学習入力',d['image']),('暫定教師',d['target']),
            ('生成器のみ 継続',np.where(outputs[i],0,255).astype(np.uint8))])
    dump(dest/'completion.json',dict(history=history,optimizer_reset=not fresh,
        criterion_reached=history[-1]['provisional_training_f1']>=.995,
        generalization_not_tested=True,formal_accuracy=False))


if __name__=='__main__':main(fresh='--fresh' in sys.argv)
