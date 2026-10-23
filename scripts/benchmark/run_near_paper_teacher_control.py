"""Same-input paired near-paper BCE ablation, restarting original14epoch Adam."""
import argparse,json,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import run_mixed_paper_ink_control as previous
from matched_generator_support import hooked,objective,logit_of
from ragan_pilot_support import predict,auxiliaries
from audit_ragan_inference_context import digest_state
from compare_v5_aligned_fade_control import rgb,dump,sha
ROOT=previous.ROOT;OLD=previous.OUT;OUT=ROOT/'tmp_work/near_boundary_teacher_control_20261023';base=previous.base
ARMS=['without_near','with_near'];STEPS=864

def freeze():
    assert not (OUT/'protocol.json').exists();previous.check()
    prep=json.loads((OUT/'label_preparation.json').read_text());cs=json.loads((OLD/'selected_sources.json').read_text())['cases']
    files=[OLD/'protocol.json',OUT/'label_preparation.json',Path(__file__),ROOT/'scripts/benchmark/prepare_near_paper_teacher.py']
    for c in cs:
        f=OUT/'near_paper'/f"{c['id']}.png";near=np.asarray(Image.open(f))>0;lab=np.asarray(Image.open(OLD/'labels'/f"{c['id']}.png"));assert near.any() and not (near&(lab>0)).any();files.append(f)
    for seed in [0,1]:files += [OLD/'plans'/f'mixed_seed{seed}.json']
    dump(OUT/'source_review.json',dict(original_and_overlay_reviewed_all8=True,new_prediction_seen=False,development_prior_results_seen=True,
        notes='Initial very sparse source-only mask (deficit6,corridor12) revised before training to deficit12,corridor20, all8 overlays reviewed. B03 only14 near pixels; do not interpret as complete boundary labels. No expert denseGT.'))
    files.append(OUT/'source_review.json')
    p=dict(utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,kinds=['unet','ragan'],seeds=[0,1],steps=864,lr=3e-5,gradient_clip=1.,saved_steps=[432,864],primary_step=864,
        changed_factor='Weight0 versus1 mean paper BCE on identical additional source-supported near-paper masks. Old mixed paper+ink terms both weight1.',
        common='Identical original14epoch model+Adam3024, main batch4,16 background RGB,8 mixed RGB,plans,1159 ink and14728 far-paper labels; all8 paired runs rerun; no cached control substituted.',
        near_paper_pixels=sum(r['near_paper_n'] for r in prep['counts']),near_paper_exposures=sum(r['near_paper_n'] for r in prep['counts'])*108,
        mixed_ink_exposures=1159*108,mixed_paper_exposures_both=14728*108,threshold=.5,no_GAN=True,no_new_damage=True,formal_accuracy=False,
        supervision_status='Provisional contrast-guided partial negative labels, source review only; not dense expertGT. Known development cases. Not RA-GAN reproduction.',
        frozen_files={str(f.relative_to(ROOT)):sha(f) for f in files})
    dump(OUT/'protocol.json',p)
    for f in [Path(__file__),ROOT/'scripts/benchmark/prepare_near_paper_teacher.py']:
        (OUT/'source_snapshot').mkdir(exist_ok=True);shutil.copy2(f,OUT/'source_snapshot'/f.name)
    print('Frozen8 runs;',p['near_paper_pixels'],'near-paper pixels;',p['near_paper_exposures'],'exposures',flush=True)
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
        lab=np.asarray(Image.open(OLD/'labels'/f"{c['id']}.png"));mixed.append((inp(rgb(ROOT/c['crop'])),torch.from_numpy(lab==1).to(device),torch.from_numpy(lab==2).to(device),torch.from_numpy(np.asarray(Image.open(OUT/'near_paper'/f"{c['id']}.png"))>0).to(device)))
    weight=int(arm=='with_near');history=[];start=time.monotonic();ink_n=paper_n=near_n=0
    for step in range(STEPS):
        samples=[data.sample(e) for e in plan[step*4:step*4+4]];x,t,v,pos,neg=[torch.stack([a[j] for a in samples]).to(device) for j in range(5)]
        model.train();opt.zero_grad(set_to_none=True);prob=predict(model,kind,x);main,parts=objective(prob,t,v,logit_of(kind,state['raw']),pos,neg);assert torch.isfinite(main);main.backward()
        predict(model,kind,backgrounds[bplan[step]]);z=logit_of(kind,state['raw']);bg_loss=F.softplus(-z[:,:,64:192,64:192]).mean();assert torch.isfinite(bg_loss);bg_loss.backward()
        mx,ip,pp,npaper=mixed[mplan[step]];predict(model,kind,mx);z=logit_of(kind,state['raw'])[0,0]
        paper_loss=F.softplus(-z[pp]).mean();ink_loss=F.softplus(z[ip]).mean();near_loss=F.softplus(-z[npaper]).mean();loss=paper_loss+ink_loss+weight*near_loss;assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip']);assert torch.isfinite(norm);opt.step();ink_n+=int(ip.sum());paper_n+=int(pp.sum());near_n+=int(npaper.sum())*weight
        history.append(dict(step=step+1,main=float(main.detach()),background=float(bg_loss.detach()),mixed_paper=float(paper_loss.detach()),mixed_ink_unweighted=float(ink_loss.detach()),ink_weight=1,near_weight=weight,near_paper_unweighted=float(near_loss.detach()),gradient_norm=float(norm),active=sum(e['active'] for e in plan[step*4:step*4+4])))
        if (step+1)%216==0:
            tail=history[-216:];progress=dict(step=step+1,means={key:float(np.mean([r[key] for r in tail])) for key in ['main','background','mixed_paper','mixed_ink_unweighted']},seconds=time.monotonic()-start);dump(dest/'progress.json',progress);print(dest.name,progress,flush=True)
        if step+1 in p['saved_steps']:
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),kind=kind,seed=seed,arm=arm,step=3024+step+1,additional_steps=step+1,protocol_sha256=sha(OUT/'protocol.json'),discriminator=None),dest/f'step_{step+1:05d}.pt')
    assert ink_n==p['mixed_ink_exposures'] and paper_n==p['mixed_paper_exposures_both'] and near_n==p['near_paper_exposures']*weight
    dump(dest/'loss_components.json',history);dump(dest/'completion.json',dict(kind=kind,seed=seed,arm=arm,initial_state_sha256=initial,initial_checkpoint_sha256=sha(ck),initial_optimizer_steps=[3024],final_optimizer_steps=sorted({int(s['step']) for s in opt.state.values() if 'step' in s}),
      main_plan_sha256=sha(base.OUT/'plans'/f'main_seed{seed}.json'),background_plan_sha256=sha(base.OUT/'plans'/f'background_seed{seed}.json'),mixed_plan_sha256=sha(OLD/'plans'/f'mixed_seed{seed}.json'),strong_fade_draws=sum(e['active'] for e in plan),mixed_ink_exposures=ink_n,near_paper_exposures=near_n,mixed_paper_exposures=paper_n,background_draws=864,mixed_draws=864,seconds=time.monotonic()-start,formal_accuracy=False))
    print('Completed',dest.name,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','train']);a.add_argument('--kind',choices=['unet','ragan']);a.add_argument('--seed',type=int,choices=[0,1]);a.add_argument('--arm',choices=ARMS);a.add_argument('--device',default='cuda:0');args=a.parse_args();freeze() if args.action=='freeze' else train(args.kind,args.seed,args.arm,args.device)
