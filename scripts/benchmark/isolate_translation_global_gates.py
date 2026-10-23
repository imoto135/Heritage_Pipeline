"""Pair cached natural translations with baseline-fixed GGC scalars. No training."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
import diagnose_binarization_translation as prev
from compare_v5_aligned_fade_control import dump, sha, save
from ragan_pilot_support import make_model
from deterministic_reflection_support import install
from audit_ragan_inference_context import Reader, digest_state
from apply_ragan_real_regions import panel

ROOT = prev.ROOT
OUT = ROOT/'tmp_work/binarization_global_gate_isolation_20261023'


def freeze():
    parent = prev.check()
    assert not (OUT/'plan.json').exists()
    files = {str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for script in ['audit_ragan_inference_context.py', 'ragan_pilot_support.py', 'deterministic_reflection_support.py']:
        f=ROOT/'scripts/benchmark'/script;files[str(f.relative_to(ROOT))]=sha(f)
    for f in ['plan.json','results.json','verification.json']:
        file=prev.OUT/f;files[str(file.relative_to(ROOT))]=sha(file)
    for row in prev.read(prev.OUT/'results.json')['rows']:
        files[row['probability_file']]=row['probability_sha256']
    dump(OUT/'plan.json',dict(
        status='Adaptive follow-up after natural translations, frozen before forced-gate outputs',
        source_files=files,parent_plan=parent,
        intervention='For each unchanged512 input, cache natural baseline4GGC theta means, then clamp only those means for every joint5channel roll. Other activations and weights unchanged.',
        checks='Every baseline and fixed-gate zero shift must bitmatch previous probability maps. All conditions paired on CPU float32,8threads,deterministic,noncontiguousNCHW as previous.',
        scope='All4models,3inputs,9shifts; no output or threshold selection; exact andr2core and paper black, probability differences.',
        causal_limit='Fixes4globalGGC scalars only. Other local, multiscale, global and boundary paths remain. Does not by itself identify downsampling as sole cause.',
        training_policy='GPU idle is necessary, not sufficient. Train only when a concrete decision cannot be resolved using current checkpoints and a controlled experiment is ready. Never train while another user is using GPU.',
        no_training=True,threshold=.5))
    print('Frozen108 gate interventions; no training',flush=True)


def check():
    p=prev.read(OUT/'plan.json')
    for f,h in p['source_files'].items():assert sha(ROOT/f)==h,f
    for cp in p['parent_plan']['checkpoints']:assert sha(ROOT/cp['path'])==cp['sha256']
    return p


def pad(a):
    if a.ndim==2:a=np.repeat(a[:,:,None],3,axis=2)
    h,w=a.shape[:2];b=np.full((max(h,256),max(w,256),3),255,np.uint8);b[:h,:w]=a;return b


def run():
    plan=check();assert not (OUT/'results.json').exists()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    original=prev.read(prev.OUT/'results.json')['rows'];rows=[];states=[];figures=[];baselines=[]
    for cp in plan['parent_plan']['checkpoints']:
        n=cp['condition'];q=torch.load(ROOT/cp['path'],map_location='cpu',weights_only=False)
        model=make_model('ragan');model.load_state_dict(q['model']);del q;install(model);model.eval()
        before=digest_state(model);reader=Reader(model,torch.device('cpu'))
        for case in plan['parent_plan']['cases']:
            start=time.monotonic();k=case['id'];base=prev.OUT/'inputs'/k;a=np.load(base/'channels.npy')
            labels={s:np.asarray(Image.open(base/f'{s}.png'))>0 for s in ['core','paper']}
            selected=labels['core']|labels['paper']
            natural,theta,_=reader(a,contiguous=False)
            old=next(r for r in original if r['condition']==n and r['case']==k and r['shift_xy']==[0,0])
            assert np.array_equal(natural,np.load(ROOT/old['probability_file']))
            baselines.append(dict(condition=n,case=k,theta=theta,previous_probability_bitexact=True))
            x0,y0,x1,y1=case['figure_box'];sl=np.s_[y0:y1,x0:x1]
            items=[('原画像',pad(a[sl][:,:,:3])),('元出力',pad(np.where(natural[sl]<.5,0,255).astype(np.uint8)))]
            for dx,dy in plan['parent_plan']['shifts_xy']:
                shifted=np.roll(a,(dy,dx),axis=(0,1))
                out,observed,effective=reader(shifted,forced=theta,contiguous=False)
                fixed=np.roll(out,(-dy,-dx),axis=(0,1))
                if dx==dy==0:assert np.array_equal(fixed,natural)
                assert effective==theta
                old=next(r for r in original if r['condition']==n and r['case']==k and r['shift_xy']==[dx,dy])
                reference=np.load(ROOT/old['probability_file']);delta=np.abs(fixed-reference)
                dest=OUT/'predictions'/n/k;dest.mkdir(parents=True,exist_ok=True);stem=f'dx{dx}_dy{dy}'
                pf,mf=dest/f'{stem}.npy',dest/f'{stem}.png';np.save(pf,fixed);save(mf,fixed<.5)
                row=dict(condition=n,case=k,shift_xy=[dx,dy],**prev.parent.score(fixed,labels,.5),
                    natural_core_exact=old['core_exact'],natural_core_r2=old['core_r2'],natural_paper_ink=old['paper_ink'],
                    natural_vs_fixed_selected_disagreement=int((((fixed<.5)!=(reference<.5))&selected).sum()),
                    natural_vs_fixed_selected_probability_mean_abs=float(delta[selected].mean()),
                    natural_vs_fixed_selected_probability_max_abs=float(delta[selected].max()),
                    baseline_theta=theta,theta_observed_under_fixed_upstream=observed,
                    probability_file=str(pf.relative_to(ROOT)),probability_sha256=sha(pf),mask_file=str(mf.relative_to(ROOT)),mask_sha256=sha(mf))
                rows.append(row)
                items.append((f'x{dx} y{dy} 墨{row["core_r2"]} 黒{row["paper_ink"]}',pad(np.where(fixed[sl]<.5,0,255).astype(np.uint8))))
            title=k[:3]+(' 実タイル' if 'actual' in k else ' 中央512')+' / '+n.replace('ragan_','')+' GGC係数固定'
            fig=OUT/'figures'/f'{n}_{k}.png';panel(fig,title,items);figures.append(str(fig.relative_to(OUT)))
            print(n,k,[(r['shift_xy'],r['core_r2'],r['paper_ink'],r['natural_vs_fixed_selected_disagreement']) for r in rows[-9:]],f'{time.monotonic()-start:.1f}s',flush=True)
        after=digest_state(model);assert before==after;states.append(dict(condition=n,state_before=before,state_after=after))
        for h in reader.handles:h.remove()
        del reader,model
    dump(OUT/'results.json',dict(plan_sha256=sha(OUT/'plan.json'),rows=rows,baseline_checks=baselines,unchanged_models=states,figures=figures,no_training=True))


def verify():
    check();r=prev.read(OUT/'results.json');assert r['plan_sha256']==sha(OUT/'plan.json')
    assert len(r['rows'])==108 and len(r['baseline_checks'])==12
    yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    for row in r['rows']:
        a=np.load(ROOT/row['probability_file']);pred=np.asarray(Image.open(ROOT/row['mask_file']))<128
        assert sha(ROOT/row['probability_file'])==row['probability_sha256']
        assert sha(ROOT/row['mask_file'])==row['mask_sha256'] and np.array_equal(pred,a<.5)
        labels={s:np.asarray(Image.open(prev.OUT/'inputs'/row['case']/f'{s}.png'))>0 for s in ['core','paper']}
        cov=ndi.convolve(pred.astype(np.uint8),disk.astype(np.uint8),mode='constant')>0
        assert row['core_exact']==int(pred[labels['core']].sum())
        assert row['core_r2']==int(cov[labels['core']].sum())
        assert row['paper_ink']==int(pred[labels['paper']].sum())
    dump(OUT/'verification.json',dict(plan_sha256=sha(OUT/'plan.json'),results_sha256=sha(OUT/'results.json'),
        independently_rescored_masks=108,exact_natural_baselines=12,exact_fixed_zero_baselines=12,
        figures={f:sha(OUT/f) for f in r['figures']},no_training=True))
    print('Verified108 interventions and24 baseline comparisons',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','verify']);a=p.parse_args()
    {'freeze':freeze,'run':run,'verify':verify}[a.action]()
