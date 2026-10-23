"""Clear-paper versus dark-nontext BCE on identical source RGB and counts."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json,shutil,time
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import run_paper_location_control as parent
from deterministic_reflection_support import install
from matched_generator_support import hooked as original_hooked,objective,logit_of
from ragan_pilot_support import predict,auxiliaries
from audit_ragan_inference_context import digest_state
from compare_v5_aligned_fade_control import rgb,dump,sha
ROOT=parent.ROOT;OUT=ROOT/'tmp_work/dark_nontext_teacher_control_20261023';OLD=parent.OLD;NEAR=parent.NEAR;base=parent.base
ARMS=['clear','stain'];STEPS=864

def freeze():
    assert not (OUT/'protocol.json').exists();parent.check();cs=json.loads((OUT/'selected_sources.json').read_text())['cases'];assert len(cs)==8
    import evaluate_background_diversity_control as ev
    evalcases=ev.cases();evalfiles={f for c in evalcases for f in c.get('files',[]) if Path(f).suffix=='.jpg'};evalbooks={Path(f).parent.name for f in evalfiles};files=[parent.OUT/'protocol.json',Path(__file__),ROOT/'scripts/benchmark/deterministic_reflection_support.py']
    for name in ['survey.json','candidate_sources.json','selected_sources.json','label_preparation.json']:files.append(OUT/name)
    for name in ['screen_dark_nontext_sources.py','prepare_dark_nontext_candidates.py','prepare_dark_nontext_labels.py']:files.append(ROOT/'scripts/benchmark'/name)
    for c in cs:
        assert c['source'] not in evalfiles and c['book'] not in evalbooks and 'split_dataset_jogai' in c['source'];assert sha(ROOT/c['source'])==c['source_sha256'] and sha(ROOT/c['crop'])==c['crop_sha256'];x,y,w,h=c['roi'];assert np.array_equal(rgb(ROOT/c['source'])[y:y+h,x:x+w],rgb(ROOT/c['crop']))
        files += [ROOT/c['source'],ROOT/c['crop']]
        for arm in ARMS:
            f=OUT/'paper_masks'/arm/f"{c['id']}.png";m=np.asarray(Image.open(f))>0;assert m.sum()==576 and sha(f)==c['mask_sha256'][arm];files.append(f)
    for seed in [0,1]:
        rng=np.random.default_rng(20261025+seed);plan=np.concatenate([rng.permutation(8) for _ in range(108)]).tolist();assert set(Counter(plan).values())=={108};f=OUT/'plans'/f'spot_seed{seed}.json';dump(f,plan);files.append(f)
        files += [base.OUT/'plans'/f'main_seed{seed}.json',base.OUT/'plans'/f'background_seed{seed}.json',OLD/'plans'/f'mixed_seed{seed}.json']
    for c in json.loads((OLD/'selected_sources.json').read_text())['cases']:files.append(NEAR/'near_paper'/f"{c['id']}.png")
    dump(OUT/'source_review.json',dict(original_contact_sheets_reviewed=48,all8_selected_source_and_label_overlays_reviewed=True,new_predictions_seen=False,eval_D04_used_as_teacher=False,evaluation_books_disjoint=True,status='Provisional nontext paper labels, no expertGT. S08 is a weak-colored stain control, not all8 are equally dark.'))
    files.append(OUT/'source_review.json')
    p=dict(utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,kinds=['ragan'],seeds=[0,1],steps=864,lr=3e-5,gradient_clip=1.,saved_steps=[432,864],primary_step=864,threshold=.5,
      changed_factor='On identical8 extra source windows, mean paperBCE on source-reviewed clear-paper box vs isolated nontext-stain box. Both24x24,576pixels perwindow, coefficient1. Same extra forward in botharms.',
      common='Same original14epoch model+Adam3024; original main batch4,16 background crops,8 mixed crops and their old paper+ink+near terms. Same orders/updates/input layouts. Extra source forward batch1 in botharms. Clip accumulated gradient once; one Adam step.',
      loss='main.backward; background central128 paperBCE.backward; mixed oldpaperBCE+oldinkBCE+commonnearBCE.backward; extra clear_or_stain meanpaperBCE.backward; clip1+Adam',
      numeric=dict(strict_deterministic_algorithms=True,cublas_workspace=':4096:8',cudnn_benchmark=False,cudnn_deterministic=True,matmul_tf32=False,cudnn_tf32=False,reflection='Equivalent slice/flip/cat in8 Conv2d; forward unchanged, deterministic backward'),
      source_windows=8,source_pages=len({c['source'] for c in cs}),source_books=len({c['book'] for c in cs}),new_teacher_pixels_each=4608,new_teacher_exposures_each=4608*108,mixed_ink_exposures=1159*108,mixed_paper_exposures=14728*108,mixed_near_exposures=5315*108,
      no_GAN=True,no_new_synthetic_damage=True,formal_accuracy=False,limits='Known development26cases, partial provisional labels,2 initialization seeds. Source stain boxes include only selected interiors/adjacent paper, not complete stain GT. Stain color/texture/location versus clear paper changes together; not a pure intensity ablation. D04/eval books never used for training.',
      frozen_files={str(f.relative_to(ROOT)):sha(f) for f in files})
    dump(OUT/'protocol.json',p);(OUT/'source_snapshot').mkdir(exist_ok=True)
    for f in files:
        if f.suffix=='.py':shutil.copy2(f,OUT/'source_snapshot'/f.name)
    print('Frozen4 deterministic runs,8 windows/6books;4608 negative pixels/arm;D04 excluded',flush=True)
def check():
    parent.check();p=json.loads((OUT/'protocol.json').read_text())
    for n,h in p['frozen_files'].items():assert sha(ROOT/n)==h,n
    return p

def hooked(kind):
    assert kind=='ragan';model,state=original_hooked(kind);assert len(install(model))==8;return model,state
def train(kind,seed,arm,device_name):
    p=check();torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;dest=OUT/'runs'/f'{kind}_seed{seed}_{arm}';assert not dest.exists();dest.mkdir(parents=True)
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
        lab=np.asarray(Image.open(OLD/'labels'/f"{c['id']}.png"));mixed.append((inp(rgb(ROOT/c['crop'])),torch.from_numpy(lab==1).to(device),torch.from_numpy(lab==2).to(device),torch.from_numpy(np.asarray(Image.open(NEAR/'near_paper'/f"{c['id']}.png"))>0).to(device)))
    spotplan=json.loads((OUT/'plans'/f'spot_seed{seed}.json').read_text());spots=[]
    for c in json.loads((OUT/'selected_sources.json').read_text())['cases']:
        spots.append((inp(rgb(ROOT/c['crop'])),torch.from_numpy(np.asarray(Image.open(OUT/'paper_masks'/arm/f"{c['id']}.png"))>0).to(device)))
    weight=1;spot_n=0;history=[];start=time.monotonic();ink_n=paper_n=extra_n=0
    for step in range(STEPS):
        samples=[data.sample(e) for e in plan[step*4:step*4+4]];x,t,v,pos,neg=[torch.stack([a[j] for a in samples]).to(device) for j in range(5)]
        model.train();opt.zero_grad(set_to_none=True);prob=predict(model,kind,x);main,parts=objective(prob,t,v,logit_of(kind,state['raw']),pos,neg);assert torch.isfinite(main);main.backward()
        predict(model,kind,backgrounds[bplan[step]]);z=logit_of(kind,state['raw']);bg_loss=F.softplus(-z[:,:,64:192,64:192]).mean();assert torch.isfinite(bg_loss);bg_loss.backward()
        mx,ip,pp,npaper=mixed[mplan[step]];predict(model,kind,mx);z=logit_of(kind,state['raw'])[0,0]
        paper_loss=F.softplus(-z[pp]).mean();ink_loss=F.softplus(z[ip]).mean();extra_loss=F.softplus(-z[npaper]).mean();loss=paper_loss+ink_loss+weight*extra_loss;assert torch.isfinite(loss);loss.backward()
        sx,smask=spots[spotplan[step]];predict(model,kind,sx);spot_z=logit_of(kind,state['raw'])[0,0];spot_loss=F.softplus(-spot_z[smask]).mean();assert torch.isfinite(spot_loss);spot_loss.backward();spot_n+=int(smask.sum())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip']);assert torch.isfinite(norm);opt.step();ink_n+=int(ip.sum());paper_n+=int(pp.sum());extra_n+=int(npaper.sum())*weight
        history.append(dict(step=step+1,main=float(main.detach()),background=float(bg_loss.detach()),mixed_paper=float(paper_loss.detach()),mixed_ink_unweighted=float(ink_loss.detach()),ink_weight=1,extra_weight=weight,extra_paper_unweighted=float(extra_loss.detach()),spot_paper_loss=float(spot_loss.detach()),gradient_norm=float(norm),active=sum(e['active'] for e in plan[step*4:step*4+4])))
        if (step+1)%216==0:
            tail=history[-216:];progress=dict(step=step+1,means={key:float(np.mean([r[key] for r in tail])) for key in ['main','background','mixed_paper','mixed_ink_unweighted']},seconds=time.monotonic()-start);dump(dest/'progress.json',progress);print(dest.name,progress,flush=True)
        if step+1 in p['saved_steps']:
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),kind=kind,seed=seed,arm=arm,step=3024+step+1,additional_steps=step+1,protocol_sha256=sha(OUT/'protocol.json'),discriminator=None),dest/f'step_{step+1:05d}.pt')
    assert ink_n==p['mixed_ink_exposures'] and paper_n==p['mixed_paper_exposures'] and extra_n==p['mixed_near_exposures']*weight and spot_n==p['new_teacher_exposures_each']
    dump(dest/'loss_components.json',history);dump(dest/'completion.json',dict(kind=kind,seed=seed,arm=arm,initial_state_sha256=initial,initial_checkpoint_sha256=sha(ck),initial_optimizer_steps=[3024],final_optimizer_steps=sorted({int(s['step']) for s in opt.state.values() if 'step' in s}),
      main_plan_sha256=sha(base.OUT/'plans'/f'main_seed{seed}.json'),background_plan_sha256=sha(base.OUT/'plans'/f'background_seed{seed}.json'),mixed_plan_sha256=sha(OLD/'plans'/f'mixed_seed{seed}.json'),strong_fade_draws=sum(e['active'] for e in plan),mixed_ink_exposures=ink_n,extra_paper_exposures=extra_n,spot_paper_exposures=spot_n,spot_draws=864,spot_plan_sha256=sha(OUT/'plans'/f'spot_seed{seed}.json'),mixed_paper_exposures=paper_n,background_draws=864,mixed_draws=864,seconds=time.monotonic()-start,formal_accuracy=False))
    print('Completed',dest.name,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','train']);a.add_argument('--kind',choices=['ragan'],default='ragan');a.add_argument('--seed',type=int,choices=[0,1]);a.add_argument('--arm',choices=ARMS);a.add_argument('--device',default='cuda:0');v=a.parse_args();freeze() if v.action=='freeze' else train(v.kind,v.seed,v.arm,v.device)
