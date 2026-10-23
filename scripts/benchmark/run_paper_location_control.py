"""Matched source-paper location ablation; all paired models rerun."""
import argparse,json,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import run_near_paper_teacher_control as previous
from matched_generator_support import hooked,objective,logit_of
from ragan_pilot_support import predict,auxiliaries
from audit_ragan_inference_context import digest_state
from compare_v5_aligned_fade_control import rgb,dump,sha
ROOT=previous.ROOT;OLD=previous.OLD;NEAR=previous.OUT;OUT=ROOT/'tmp_work/paper_location_control_20261023';base=previous.base
ARMS=['near','far'];STEPS=864
def freeze():
    assert not (OUT/'protocol.json').exists();previous.check();prep=json.loads((OUT/'label_preparation.json').read_text());assert prep['total_pixels_each']==5315
    files=[NEAR/'protocol.json',OUT/'label_preparation.json',Path(__file__),ROOT/'scripts/benchmark/prepare_paper_location_control.py']
    for r in prep['rows']:
        assert r['near_n']==r['far_n']
        for arm in ARMS:files.append(OUT/'paper_masks'/arm/f"{r['id']}.png")
    for seed in [0,1]:files.append(OLD/'plans'/f'mixed_seed{seed}.json')
    dump(OUT/'source_review.json',dict(all8_source_originals_and_near_far_overlays_reviewed=True,new_predictions_seen=False,prior_results_seen=True,source_paper_labels='Provisional, not expertGT. No source RGB or old labels changed.'))
    files.append(OUT/'source_review.json')
    p=dict(utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,kinds=['unet','ragan'],seeds=[0,1],steps=864,lr=3e-5,gradient_clip=1.,saved_steps=[432,864],primary_step=864,
        changed_factor='Spatial placement of additional mean paper BCE mask: near annotated source strokes versus far. Identical per-image counts, source contrast strata and brightness bins32. coefficient1 both.',
        common='All8 conditions rerun from same original14epoch model and Adam3024, identical main RGB/plans,16 background windows,8 mixed windows,old1159 ink/14728 paper labels. Separate main/bkg/mixed forwards; same one-step gradient accumulation.',
        extra_paper_pixels=5315,extra_paper_exposures=5315*108,mixed_ink_exposures=1159*108,mixed_paper_exposures_both=14728*108,threshold=.5,no_GAN=True,no_new_damage=True,formal_accuracy=False,
        limits='Far means >=40px from annotated source ink, not every unannotated glyph. Source-only contrast proxy, not expert GT. Spatial mask shape/scattering differs too. Known development cases, no blind test.',
        frozen_files={str(f.relative_to(ROOT)):sha(f) for f in files})
    dump(OUT/'protocol.json',p)
    (OUT/'source_snapshot').mkdir(exist_ok=True)
    for f in [Path(__file__),ROOT/'scripts/benchmark/prepare_paper_location_control.py']:shutil.copy2(f,OUT/'source_snapshot'/f.name)
    print('Frozen8 runs: matched near/far5315 pixels,574020 exposures perarm',flush=True)
def check():
    previous.check();p=json.loads((OUT/'protocol.json').read_text())
    for n,h in p['frozen_files'].items():assert sha(ROOT/n)==h,n
    return p
def train(kind,seed,arm,device_name):
    p=check();dest=OUT/'runs'/f'{kind}_seed{seed}_{arm}';assert not dest.exists();dest.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(20261021+seed);device=torch.device(device_name);torch.cuda.set_device(device)
    ck=base.old.OUT/f'{kind}_seed{seed}'/'step_03024.pt';payload=torch.load(ck,map_location='cpu',weights_only=False)
    model,state=hooked(kind);model=model.to(device);model.load_state_dict(payload['model']);initial=digest_state(model)
    opt=torch.optim.Adam([q for q in model.parameters() if q.requires_grad],lr=p['lr']);opt.load_state_dict(payload['optimizer']);del payload
    assert {int(s['step']) for s in opt.state.values() if 'step' in s}=={3024}
    for name,h in json.loads((base.old.OUT/'source_hashes.json').read_text()).items():assert sha(ROOT/name)==h
    data=base.old.Data(base.old.full.read_rows('train'));plan=json.loads((base.OUT/'plans'/f'main_seed{seed}.json').read_text());bplan=json.loads((base.OUT/'plans'/f'background_seed{seed}.json').read_text());mplan=json.loads((OLD/'plans'/f'mixed_seed{seed}.json').read_text())
    def inp(raw):return torch.from_numpy(auxiliaries(raw).copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
    bgs=[r for r in json.loads((base.OUT/'background_candidates.json').read_text())['candidates'] if r['arm']=='multibook'];backgrounds=[inp(rgb(ROOT/r['crop'])) for r in bgs]
    cs=json.loads((OLD/'selected_sources.json').read_text())['cases'];mixed=[]
    for c in cs:
        lab=np.asarray(Image.open(OLD/'labels'/f"{c['id']}.png"));mixed.append((inp(rgb(ROOT/c['crop'])),torch.from_numpy(lab==1).to(device),torch.from_numpy(lab==2).to(device),torch.from_numpy(np.asarray(Image.open(OUT/'paper_masks'/arm/f"{c['id']}.png"))>0).to(device)))
    weight=1;history=[];start=time.monotonic();ink_n=paper_n=extra_n=0
    for step in range(STEPS):
        samples=[data.sample(e) for e in plan[step*4:step*4+4]];x,t,v,pos,neg=[torch.stack([a[j] for a in samples]).to(device) for j in range(5)]
        model.train();opt.zero_grad(set_to_none=True);prob=predict(model,kind,x);main,parts=objective(prob,t,v,logit_of(kind,state['raw']),pos,neg);assert torch.isfinite(main);main.backward()
        predict(model,kind,backgrounds[bplan[step]]);z=logit_of(kind,state['raw']);bg_loss=F.softplus(-z[:,:,64:192,64:192]).mean();assert torch.isfinite(bg_loss);bg_loss.backward()
        mx,ip,pp,npaper=mixed[mplan[step]];predict(model,kind,mx);z=logit_of(kind,state['raw'])[0,0]
        paper_loss=F.softplus(-z[pp]).mean();ink_loss=F.softplus(z[ip]).mean();extra_loss=F.softplus(-z[npaper]).mean();loss=paper_loss+ink_loss+weight*extra_loss;assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip']);assert torch.isfinite(norm);opt.step();ink_n+=int(ip.sum());paper_n+=int(pp.sum());extra_n+=int(npaper.sum())*weight
        history.append(dict(step=step+1,main=float(main.detach()),background=float(bg_loss.detach()),mixed_paper=float(paper_loss.detach()),mixed_ink_unweighted=float(ink_loss.detach()),ink_weight=1,extra_weight=weight,extra_paper_unweighted=float(extra_loss.detach()),gradient_norm=float(norm),active=sum(e['active'] for e in plan[step*4:step*4+4])))
        if (step+1)%216==0:
            tail=history[-216:];progress=dict(step=step+1,means={key:float(np.mean([r[key] for r in tail])) for key in ['main','background','mixed_paper','mixed_ink_unweighted']},seconds=time.monotonic()-start);dump(dest/'progress.json',progress);print(dest.name,progress,flush=True)
        if step+1 in p['saved_steps']:
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),kind=kind,seed=seed,arm=arm,step=3024+step+1,additional_steps=step+1,protocol_sha256=sha(OUT/'protocol.json'),discriminator=None),dest/f'step_{step+1:05d}.pt')
    assert ink_n==p['mixed_ink_exposures'] and paper_n==p['mixed_paper_exposures_both'] and extra_n==p['extra_paper_exposures']*weight
    dump(dest/'loss_components.json',history);dump(dest/'completion.json',dict(kind=kind,seed=seed,arm=arm,initial_state_sha256=initial,initial_checkpoint_sha256=sha(ck),initial_optimizer_steps=[3024],final_optimizer_steps=sorted({int(s['step']) for s in opt.state.values() if 'step' in s}),
      main_plan_sha256=sha(base.OUT/'plans'/f'main_seed{seed}.json'),background_plan_sha256=sha(base.OUT/'plans'/f'background_seed{seed}.json'),mixed_plan_sha256=sha(OLD/'plans'/f'mixed_seed{seed}.json'),strong_fade_draws=sum(e['active'] for e in plan),mixed_ink_exposures=ink_n,extra_paper_exposures=extra_n,mixed_paper_exposures=paper_n,background_draws=864,mixed_draws=864,seconds=time.monotonic()-start,formal_accuracy=False))
    print('Completed',dest.name,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','train']);a.add_argument('--kind',choices=['unet','ragan']);a.add_argument('--seed',type=int,choices=[0,1]);a.add_argument('--arm',choices=ARMS);a.add_argument('--device',default='cuda:0');args=a.parse_args();freeze() if args.action=='freeze' else train(args.kind,args.seed,args.arm,args.device)
