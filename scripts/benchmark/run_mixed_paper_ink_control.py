"""Pair paper-only and paper+ink labels with identical auxiliary RGB contexts."""
import argparse,json,shutil,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import run_background_diversity_control as base
from matched_generator_support import hooked,objective,logit_of
from ragan_pilot_support import predict,auxiliaries
from audit_ragan_inference_context import digest_state
from compare_v5_aligned_fade_control import rgb,dump,sha
from prepare_mixed_paper_ink_labels import labels as make_labels
ROOT=base.ROOT;OUT=ROOT/'tmp_work/mixed_paper_ink_control_20261022';ARMS=['paper_only','paper_ink'];STEPS=864
def freeze():
    assert not (OUT/'protocol.json').exists();base.check()
    cs=json.loads((OUT/'selected_sources.json').read_text())['cases'];aa={r['id']:r for r in json.loads((OUT/'annotations_source_only.json').read_text())['regions']};counts=[];files=[]
    evalcases=json.loads((ROOT/'tmp_work/frozen_real_transfer_20261022/protocol.json').read_text())['cases']+[json.loads((ROOT/'tmp_work/frozen_real_transfer_20261022/additional_control/protocol.json').read_text())['case']]
    evalbooks={c['book'] for c in evalcases};bg=json.loads((base.OUT/'background_candidates.json').read_text())['candidates'];bgs=[r for r in bg if r['arm']=='multibook'];assert {c['book'] for c in cs}=={r['book'] for r in bgs} and len(cs)==8
    for c in cs:
        assert c['book'] not in evalbooks and 'split_dataset_jogai' in c['source']
        assert c['source'] not in {r['source'] for r in bg}
        assert sha(ROOT/c['source'])==c['source_sha256'] and sha(ROOT/c['crop'])==c['crop_sha256']
        x,y,w,h=c['roi'];assert np.array_equal(rgb(ROOT/c['source'])[y:y+h,x:x+w],rgb(ROOT/c['crop']))
        lab,_=make_labels(c,aa[c['id']]);path=OUT/'labels'/f"{c['id']}.png";assert np.array_equal(lab,np.asarray(Image.open(path)))
        assert (lab==1).any() and (lab==2).any();counts.append(dict(id=c['id'],ink=int((lab==1).sum()),paper=int((lab==2).sum())))
        files += [ROOT/c['source'],ROOT/c['crop'],path]
    assert sum(r['ink'] for r in counts)==1159 and sum(r['paper'] for r in counts)==14728
    dump(OUT/'source_review.json',dict(utc=datetime.now(timezone.utc).isoformat(),original_overlays_visually_reviewed=True,predictions_seen=False,
      selection='Reject source-ranked fibers/ghosts/frame lines; choose definite foreground. Review all8 original+label overlays before training. Remove B08 uncertain tail; source support filter excludes12 unsupported points.',
      label_status='Provisional training labels by Codex, not expertGT; ink centerlines only, includes a strong-stroke scan control; uncertain structures unknown',counts=counts))
    for seed in [0,1]:
        rng=np.random.default_rng(20261024+seed);plan=np.concatenate([rng.permutation(8) for _ in range(108)]).tolist();assert set(Counter(plan).values())=={108}
        path=OUT/'plans'/f'mixed_seed{seed}.json';dump(path,plan);files.append(path)
        files += [base.OUT/'plans'/f'main_seed{seed}.json',base.OUT/'plans'/f'background_seed{seed}.json']
    for n in ['selected_sources.json','annotations_source_only.json','label_preparation.json','source_review.json']:files.append(OUT/n)
    files += [base.OUT/'protocol.json',base.OUT/'background_candidates.json']
    for name in ['run_mixed_paper_ink_control.py','screen_mixed_paper_ink_sources.py','prepare_mixed_paper_ink_labels.py']:
        f=ROOT/'scripts/benchmark'/name;files.append(f);target=OUT/'source_snapshot'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(f,target)
    dump(OUT/'protocol.json',dict(utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,kinds=['unet','ragan'],seeds=[0,1],steps=STEPS,lr=3e-5,gradient_clip=1.,
      start='Same original14epoch step3024 model and restoredAdam per kind/seed across2arms; do not warmstart selected18epoch outputs',
      common='Same original batch4 main RGB/targets/crops/loss,3456draws,692strong-fade; same16 multibook paper contexts and masks,864draws; same8 mixed text+paper RGB and paper masks,864draws',
      sole_changed_factor='Weight0 versus1 of source-supported ink-center mean BCE on identical mixed text-paper forwards. Paper and unknown masks identical.',
      mixed_loss='mean softplus(-logit) on selected paper + ink_weight*mean softplus(logit) on selected ink, coefficient1each; outside unknown',
      background_loss='Existing central128 mean paperBCE coefficient1 unchanged. Background, mixed and original batch are separate forwards; one summed-gradient clip and one Adam step.',
      input_context='Native256 RGB+per-crop Otsu/Sobel; batch4main, batch1paper, batch1mixed; same ordering and memory layout in botharms',
      primary_step=864,saved_steps=[432,864],threshold=.5,no_best_checkpoint_selection=True,no_new_damage=True,no_GAN=True,
      counts=counts,ink_teacher_pixels=1159,ink_exposures_paper_only=0,ink_exposures_paper_ink=1159*108,mixed_paper_exposures_both=14728*108,background_paper_exposures_both=864*128*128,
      evaluation='Inherited26cases, native512primary and fixed256grids0/16; R02secondary and R07invalid; known development provisional annotations, not independent test',
      frozen_files={str(f.relative_to(ROOT)):sha(f) for f in files},formal_accuracy=False))
    print('Frozen8 paired runs: identical text+paper inputs, only ink supervision differs',flush=True)
def check():
    base.check();p=json.loads((OUT/'protocol.json').read_text())
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
    data=base.old.Data(base.old.full.read_rows('train'));plan=json.loads((base.OUT/'plans'/f'main_seed{seed}.json').read_text());bplan=json.loads((base.OUT/'plans'/f'background_seed{seed}.json').read_text());mplan=json.loads((OUT/'plans'/f'mixed_seed{seed}.json').read_text())
    def inp(raw):return torch.from_numpy(auxiliaries(raw).copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
    bgs=[r for r in json.loads((base.OUT/'background_candidates.json').read_text())['candidates'] if r['arm']=='multibook'];backgrounds=[inp(rgb(ROOT/r['crop'])) for r in bgs]
    cs=json.loads((OUT/'selected_sources.json').read_text())['cases'];mixed=[]
    for c in cs:
        lab=np.asarray(Image.open(OUT/'labels'/f"{c['id']}.png"));mixed.append((inp(rgb(ROOT/c['crop'])),torch.from_numpy(lab==1).to(device),torch.from_numpy(lab==2).to(device)))
    weight=int(arm=='paper_ink');history=[];start=time.monotonic();ink_n=paper_n=0
    for step in range(STEPS):
        samples=[data.sample(e) for e in plan[step*4:step*4+4]];x,t,v,pos,neg=[torch.stack([a[j] for a in samples]).to(device) for j in range(5)]
        model.train();opt.zero_grad(set_to_none=True);prob=predict(model,kind,x);main,parts=objective(prob,t,v,logit_of(kind,state['raw']),pos,neg);assert torch.isfinite(main);main.backward()
        predict(model,kind,backgrounds[bplan[step]]);z=logit_of(kind,state['raw']);bg_loss=F.softplus(-z[:,:,64:192,64:192]).mean();assert torch.isfinite(bg_loss);bg_loss.backward()
        mx,ip,pp=mixed[mplan[step]];predict(model,kind,mx);z=logit_of(kind,state['raw'])[0,0]
        paper_loss=F.softplus(-z[pp]).mean();ink_loss=F.softplus(z[ip]).mean();loss=paper_loss+weight*ink_loss;assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip']);assert torch.isfinite(norm);opt.step();ink_n+=int(ip.sum())*weight;paper_n+=int(pp.sum())
        history.append(dict(step=step+1,main=float(main.detach()),background=float(bg_loss.detach()),mixed_paper=float(paper_loss.detach()),mixed_ink_unweighted=float(ink_loss.detach()),ink_weight=weight,gradient_norm=float(norm),active=sum(e['active'] for e in plan[step*4:step*4+4])))
        if (step+1)%216==0:
            tail=history[-216:];progress=dict(step=step+1,means={key:float(np.mean([r[key] for r in tail])) for key in ['main','background','mixed_paper','mixed_ink_unweighted']},seconds=time.monotonic()-start);dump(dest/'progress.json',progress);print(dest.name,progress,flush=True)
        if step+1 in p['saved_steps']:
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),kind=kind,seed=seed,arm=arm,step=3024+step+1,additional_steps=step+1,protocol_sha256=sha(OUT/'protocol.json'),discriminator=None),dest/f'step_{step+1:05d}.pt')
    assert ink_n==p['ink_exposures_'+arm] and paper_n==p['mixed_paper_exposures_both']
    dump(dest/'loss_components.json',history);dump(dest/'completion.json',dict(kind=kind,seed=seed,arm=arm,initial_state_sha256=initial,initial_checkpoint_sha256=sha(ck),initial_optimizer_steps=[3024],final_optimizer_steps=sorted({int(s['step']) for s in opt.state.values() if 'step' in s}),
      main_plan_sha256=sha(base.OUT/'plans'/f'main_seed{seed}.json'),background_plan_sha256=sha(base.OUT/'plans'/f'background_seed{seed}.json'),mixed_plan_sha256=sha(OUT/'plans'/f'mixed_seed{seed}.json'),strong_fade_draws=sum(e['active'] for e in plan),mixed_ink_exposures=ink_n,mixed_paper_exposures=paper_n,background_draws=864,mixed_draws=864,seconds=time.monotonic()-start,formal_accuracy=False))
    print('Completed',dest.name,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','train']);a.add_argument('--kind',choices=['unet','ragan']);a.add_argument('--seed',type=int,choices=[0,1]);a.add_argument('--arm',choices=ARMS);a.add_argument('--device',default='cuda:0');args=a.parse_args();freeze() if args.action=='freeze' else train(args.kind,args.seed,args.arm,args.device)
