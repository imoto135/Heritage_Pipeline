"""Three paired four-epoch continuations: none, same-book paper, multi-book paper."""
import argparse,json,shutil,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from compare_v5_aligned_fade_control import dump,sha,rgb
import run_matched_supervised_generators as old
from matched_generator_support import hooked,logit_of,objective
from ragan_pilot_support import auxiliaries,predict
from audit_ragan_inference_context import digest_state
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/background_diversity_control_20261022'
ARMS=['continue','samebook','multibook'];STEPS=864
def freeze():
    assert not (OUT/'protocol.json').exists()
    bg=json.loads((OUT/'background_candidates.json').read_text())['candidates']
    assert len(bg)==32 and Counter(r['arm'] for r in bg)=={'samebook':16,'multibook':16}
    train_groups={r['group'] for r in old.full.read_rows('train')}
    val_groups={r['group'] for r in old.full.read_rows('val')}
    real=json.loads((ROOT/'tmp_work/frozen_real_transfer_20261022/protocol.json').read_text())['cases']
    real += [json.loads((ROOT/'tmp_work/frozen_real_transfer_20261022/additional_control/protocol.json').read_text())['case']]
    eval_books={c['book'] for c in real};files=[]
    for r in bg:
        assert sha(ROOT/r['source'])==r['source_sha256'] and sha(ROOT/r['crop'])==r['crop_sha256']
        x,y,w,h=r['roi'];assert np.array_equal(rgb(ROOT/r['source'])[y:y+h,x:x+w],rgb(ROOT/r['crop']))
        if r['arm']=='samebook':
            group=Path(r['source']).stem.rsplit('_',1)[0];assert group in train_groups and group not in val_groups
        else:assert r['book'] not in eval_books and 'split_dataset_jogai' in r['source']
        files += [ROOT/r['source'],ROOT/r['crop']]
    dump(OUT/'source_review.json',dict(utc=datetime.now(timezone.utc).isoformat(),accepted_ids=[r['id'] for r in bg],
      reviewer='Codex original-only visual review of all32 native256 crops; central128 rectangles contain no discernible character strokes',
      supervision='Provisional paper label only in central128x128; all outside pixels excluded from auxiliary loss even if text is visible',
      expert_confirmed=False,predictions_used_for_selection=False,
      caveat='Fine fibers and weak marks are judged paper provisionally. Appearance and surrounding text/context differ across source books; this is a data-source intervention, not isolated book-count causality.'))
    audits=[]
    for seed in [0,1]:
        plan=json.loads((old.OUT/'crops'/f'seed{seed}.json').read_text())[-STEPS*4:]
        assert len(plan)==3456 and sum(e['active'] for e in plan)==692
        dump(OUT/'plans'/f'main_seed{seed}.json',plan)
        rng=np.random.default_rng(20261022+seed);indices=np.concatenate([rng.permutation(16) for _ in range(54)]).tolist()
        assert len(indices)==STEPS and set(Counter(indices).values())=={54}
        dump(OUT/'plans'/f'background_seed{seed}.json',indices)
        active=Counter(old.full.read_rows('train')[e['index']]['patch_id'] for e in plan if e['active'])
        audits.append(dict(seed=seed,main_draws=3456,strong_fade_draws=692,per_patch_active=dict(active),background_draws_per_added_arm=864,
          supervised_paper_pixels_per_added_arm=864*128*128,unique_backgrounds_per_added_arm=16,each_background_draws=54))
    files += [OUT/'background_candidates.json',OUT/'source_review.json',old.OUT/'protocol.json',old.OUT/'source_hashes.json']
    for s in [0,1]:files += [OUT/'plans'/f'main_seed{s}.json',OUT/'plans'/f'background_seed{s}.json']
    for kind in ['unet','ragan']:
      for s in [0,1]:files.append(old.OUT/f'{kind}_seed{s}'/'step_03024.pt')
    for name in ['run_background_diversity_control.py','screen_background_diversity.py','run_matched_supervised_generators.py','matched_generator_support.py','ragan_pilot_support.py','ragan_local_region_support.py']:
        src=ROOT/'scripts/benchmark'/name;files.append(src);dest=OUT/'source_snapshot'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(src,dest)
    dump(OUT/'protocol.json',dict(utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,kinds=['unet','ragan'],seeds=[0,1],steps=STEPS,batch_size=4,
      common_start='Each kind/seed resumes its own frozen14-epoch3024-update checkpoint, identical across3 arms, including Adam state3024',
      main_data='Replay original last4-epoch crop/sampling plans without any RGB/teacher/validity/core/paper change; same3456 draws,692 strong-fade draws per paired condition',
      background_data='one additional native256 paper-context forward per update in added arms; central128 paper-only supervised, outside unknown; same index plan,16 windows x54 draws',
      common_main_loss='Unchanged stable BCE+10L1+inkDice+inkJaccard+existing local ink and paper BCE',
      added_loss='mean softplus(-paper_logit) over central128, coefficient1, in samebook/multibook only; separately forwarded, never concatenated with original batch4',
      control_contrasts='continue vs samebook = added paper-supervision effect; samebook vs multibook = paper source/appearance/context diversity effect with equal auxiliary exposure',
      lr=3e-5,gradient_clip=1.,Adam_restored=True,no_discriminator=True,no_GAN=True,no_new_damage=True,no_new_text_teacher=True,
      primary_step=864,saved_steps=[432,864],no_best_checkpoint_selection=True,threshold=.5,
      old_readout='All19 inherited train/synthetic/known-real cases; retain455 training traces,known49 and isolatedD04 diagnostic',
      recent_readout='All7 reviewed casesR01/R02/R03/R04/R05/R06/R08, R02 secondary; R07 invalid and excluded',
      inference='Fixed512 primary plus prior256 stride128 grids0/16 sensitivity; all conditions and grids reported',
      evaluation_status='Previously inspected development diagnostics, not independent test or expertGT',
      hypothesis='Added diverse paper may reduce false ink, but must preserve thin text and report stain tradeoffs for both seeds',
      adoption_gate='Neither isolated clean-paper success nor ink-center success alone permits adoption; report all failures',
      frozen_files={str(p.relative_to(ROOT)):sha(p) for p in files},audits=audits,formal_accuracy=False))
    print('Frozen12 paired continuations;3arms x2models x2seeds;864updates each',flush=True)
def check():
    p=json.loads((OUT/'protocol.json').read_text())
    for name,h in p['frozen_files'].items():assert sha(ROOT/name)==h,name
    return p
def train(kind,seed,arm,device_name):
    p=check();name=f'{kind}_seed{seed}_{arm}';dest=OUT/'runs'/name;assert not dest.exists();dest.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(20261021+seed);device=torch.device(device_name);torch.cuda.set_device(device)
    checkpoint=old.OUT/f'{kind}_seed{seed}'/'step_03024.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model,state=hooked(kind);model=model.to(device);model.load_state_dict(payload['model']);initial=digest_state(model)
    opt=torch.optim.Adam([q for q in model.parameters() if q.requires_grad],lr=p['lr']);opt.load_state_dict(payload['optimizer']);del payload
    assert {int(s['step']) for s in opt.state.values() if 'step' in s}=={3024}
    for n,h in json.loads((old.OUT/'source_hashes.json').read_text()).items():assert sha(ROOT/n)==h
    data=old.Data(old.full.read_rows('train'));plan=json.loads((OUT/'plans'/f'main_seed{seed}.json').read_text());bplan=json.loads((OUT/'plans'/f'background_seed{seed}.json').read_text())
    bgs=[r for r in json.loads((OUT/'background_candidates.json').read_text())['candidates'] if r['arm']==arm] if arm!='continue' else []
    inputs=[torch.from_numpy(auxiliaries(rgb(ROOT/r['crop'])).copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255 for r in bgs]
    losses=[];start=time.monotonic()
    for step in range(STEPS):
        entries=plan[step*4:step*4+4];samples=[data.sample(e) for e in entries]
        x,t,v,pos,neg=[torch.stack([a[j] for a in samples]).to(device) for j in range(5)]
        assert not (pos&((v==0)|(t>.5))).any() and not (neg&((v==0)|(t<.5))).any()
        model.train();opt.zero_grad(set_to_none=True);prob=predict(model,kind,x)
        main,parts=objective(prob,t,v,logit_of(kind,state['raw']),pos,neg)
        assert torch.isfinite(main);main.backward();paper_loss=main.new_zeros(())
        if inputs:
            q=predict(model,kind,inputs[bplan[step]])
            z=logit_of(kind,state['raw']);paper_loss=F.softplus(-z[:,:,64:192,64:192]).mean()
            assert torch.isfinite(q).all() and torch.isfinite(paper_loss);paper_loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip']);assert torch.isfinite(norm);opt.step()
        losses.append(dict(step=step+1,main=float(main.detach()),added_paper=float(paper_loss.detach()),gradient_norm=float(norm),active=sum(e['active'] for e in entries),**{k:float(q.detach()) for k,q in parts.items()}))
        if (step+1)%216==0:
            tail=losses[-216:];report=dict(step=step+1,mean_main=float(np.mean([r['main'] for r in tail])),mean_added_paper=float(np.mean([r['added_paper'] for r in tail])),seconds=time.monotonic()-start)
            dump(dest/'progress.json',report);print(name,report,flush=True)
        if step+1 in p['saved_steps']:
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),kind=kind,seed=seed,arm=arm,additional_steps=step+1,step=3024+step+1,protocol_sha256=sha(OUT/'protocol.json'),discriminator=None),dest/f'step_{step+1:05d}.pt')
    dump(dest/'loss_components.json',losses)
    dump(dest/'completion.json',dict(kind=kind,seed=seed,arm=arm,initial_state_sha256=initial,initial_checkpoint_sha256=sha(checkpoint),
      initial_optimizer_steps=[3024],final_optimizer_steps=sorted({int(s['step']) for s in opt.state.values() if 'step' in s}),
      main_plan_sha256=sha(OUT/'plans'/f'main_seed{seed}.json'),background_plan_sha256=sha(OUT/'plans'/f'background_seed{seed}.json'),
      strong_fade_draws=sum(e['active'] for e in plan),auxiliary_draws=864 if inputs else 0,paper_pixels=864*16384 if inputs else 0,seconds=time.monotonic()-start,formal_accuracy=False))
    print('Completed',name,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','train']);a.add_argument('--kind',choices=['unet','ragan']);a.add_argument('--seed',type=int,choices=[0,1]);a.add_argument('--arm',choices=ARMS);a.add_argument('--device',default='cuda:0');args=a.parse_args();freeze() if args.action=='freeze' else train(args.kind,args.seed,args.arm,args.device)
