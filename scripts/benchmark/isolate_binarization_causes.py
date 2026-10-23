"""Frozen-model development diagnostics: threshold transfer and input interventions.

No new training, annotation, checkpoint selection, or claims of blind validation.
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/binarization_cause_isolation_20261023'
os.environ['MPLCONFIGDIR'] = str(OUT / 'matplotlib_cache')
import argparse, csv, json, time
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.stats import rankdata
from PIL import Image
import evaluate_dark_nontext_teacher_control as ev
import report_background_diversity_control as refs
from compare_v5_aligned_fade_control import sha, dump, rgb, mask, save
from ragan_pilot_support import auxiliaries, make_model, predict
from apply_ragan_real_regions import panel
from audit_ragan_inference_context import digest_state
from deterministic_reflection_support import install

OLD = ev.OUT
CONDITIONS = [f'ragan_seed{s}_{a}' for s in (0, 1) for a in ('clear', 'stain')]
KEYS = ['R03', 'R05', 'R06', 'D04']
GRIDS = ['full512', 'tiled0', 'tiled16']
VARIANTS = ['native256', 'page_local', 'page_page', 'reflect_local', 'reflect_page']


def read(p):
    return json.loads(Path(p).read_text())


def pfile(n, g, k):
    return OLD / 'evaluation' / n / 'probabilities' / g / f'{k}.npy'


def score(p, c, threshold):
    # Widen both operands: float32 scalar casting can erase nextafter boundaries.
    pred = p.astype(np.float64) < np.float64(threshold)
    yy, xx = np.ogrid[-2:3, -2:3]
    covered = ndi.binary_dilation(pred, structure=xx*xx+yy*yy <= 4)
    return dict(core_n=int(c['core'].sum()), core_exact=int((pred & c['core']).sum()),
                core_r2=int((covered & c['core']).sum()), paper_n=int(c['paper'].sum()),
                paper_ink=int((pred & c['paper']).sum()), total_ink=int(pred.sum()))


def choose_threshold(probabilities, labels):
    """Minimum mean of per-region class-balanced exact error, never r2.

    Candidate events come only from calibration scores. Ties: nearest 0.5,
    then smaller threshold. Evaluate every representable score event in float64.
    """
    arrays = [(np.sort(p[c['core']].astype(np.float64)),
               np.sort(p[c['paper']].astype(np.float64))) for p, c in zip(probabilities, labels)]
    assert all(len(f) and len(b) for f, b in arrays)
    events = np.unique(np.r_[0., .5, 1., np.nextafter(np.unique(np.concatenate(
        [a for pair in arrays for a in pair])), np.inf)])
    events = events[(events >= 0) & (events <= 1)]
    error = sum((1-np.searchsorted(f, events, side='left')/len(f)
                 + np.searchsorted(b, events, side='left')/len(b))/2 for f,b in arrays)/len(arrays)
    ii = np.flatnonzero(error <= error.min()+1e-14)
    best = min(ii, key=lambda i: (abs(events[i]-.5), events[i]))
    t = float(events[best])
    # Independent comparison-based objective, not searchsorted again.
    check = np.mean([((f >= t).mean()+(b < t).mean())/2 for f,b in arrays])
    assert abs(check-error[best]) < 1e-12
    return dict(threshold=t, calibration_macro_balanced_error=float(error[best]),
                candidate_count=len(events), calibration_scores_only=True)


def freeze():
    assert not (OUT/'protocol.json').exists()
    ev.check()
    cases = {c['id']:c for c in ev.old.cases()}
    files = {str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for n in CONDITIONS:
        files[str((OLD/'evaluation'/n/'results.json').relative_to(ROOT))] = sha(OLD/'evaluation'/n/'results.json')
        for g in GRIDS:
            for k in KEYS:
                f=pfile(n,g,k);files[str(f.relative_to(ROOT))]=sha(f)
    inputs=[]
    selected={c['id']:c for c in read(OLD/'selected_sources.json')['cases']}
    for k in ['S03','S07']:
        c=selected[k];page=rgb(ROOT/c['source']);x,y,w,h=c['roi']
        left=min(max(x-128,0),page.shape[1]-512);top=min(max(y-128,0),page.shape[0]-512)
        cx,cy=x-left,y-top;sl=np.s_[cy:cy+256,cx:cx+256]
        tile=page[top:top+512,left:left+512];center=page[y:y+256,x:x+256]
        assert tile.shape==(512,512,3) and np.array_equal(tile[sl],center)
        local=auxiliaries(center);page_aux=auxiliaries(tile)[sl][:,:,3:]
        reflected=np.pad(center,((cy,256-cy),(cx,256-cx),(0,0)),mode='reflect')
        tensors={'native256':local}
        for outside,outer in [('page',tile),('reflect',reflected)]:
            for aux,inner in [('local',local[:,:,3:]),('page',page_aux)]:
                a=auxiliaries(outer);a[sl][:,:,3:]=inner
                assert np.array_equal(a[sl][:,:,:3],center)
                tensors[f'{outside}_{aux}']=a
        for aux in ['local','page']:
            assert np.array_equal(tensors['page_'+aux][sl],tensors['reflect_'+aux][sl])
        for outside in ['page','reflect']:
            a,b=tensors[outside+'_local'],tensors[outside+'_page'];changed=np.any(a!=b,axis=2)
            changed[sl]=False;assert not changed.any()
        labels={}
        for arm in ['stain','clear']:
            labels[arm]=np.asarray(Image.open(OLD/'paper_masks'/arm/f'{k}.png'))>0
        labels['core']=np.asarray(Image.open(OLD/'source_guard/S03_ink_guard.png'))>0 if k=='S03' else np.zeros((256,256),bool)
        dest=OUT/'inputs'/k;dest.mkdir(parents=True,exist_ok=True)
        Image.fromarray(center).save(dest/'original.png')
        Image.fromarray(tile).save(dest/'page512.png')
        Image.fromarray(reflected).save(dest/'reflect512.png')
        for name,a in tensors.items():np.save(dest/f'{name}.npy',a)
        for name,a in labels.items():Image.fromarray(a.astype(np.uint8)*255).save(dest/f'{name}.png')
        for f in dest.iterdir():files[str(f.relative_to(ROOT))]=sha(f)
        inputs.append(dict(id=k,source=c['source'],source_sha256=sha(ROOT/c['source']),roi=c['roi'],center_xy_in_512=[cx,cy],
                           core_n=int(labels['core'].sum()),paper_n=int(labels['stain'].sum()),
                           central_aux_changed_pixels=int(np.any(local[:,:,3:]!=page_aux,axis=2).sum())))
    for k in KEYS:
        c=cases[k];dest=OUT/'labels'/k;dest.mkdir(parents=True,exist_ok=True)
        for name in ['core','paper']:
            f=dest/f'{name}.png';Image.fromarray(c[name].astype(np.uint8)*255).save(f);files[str(f.relative_to(ROOT))]=sha(f)
        x,y,w,h=c['roi'];f=dest/'original.png';Image.fromarray(c['page'][y:y+h,x:x+w]).save(f);files[str(f.relative_to(ROOT))]=sha(f)
    dump(OUT/'protocol.json',dict(status='Frozen before these diagnostic readouts; previous development outputs known',
        source_files=files,parent_evaluation_plan_sha256=sha(OLD/'evaluation_plan.json'),conditions=CONDITIONS,
        calibration=dict(folds={'R05_to_R06':dict(calibration=['R03','R05'],transfer=['D04','R06']),
                                'R06_to_R05':dict(calibration=['R03','R06'],transfer=['D04','R05'])},
                         grid='full512',criterion='mean per-region class-balanced exact error; no r2 in selection',
                         candidates='all unique calibration-score events,0,.5,1; ties nearest.5 then smaller',
                         apply='one selected threshold across all3 grids and all4 cases, no re-selection by grid',
                         limits='R03/R05/R06 are known development examples; not independent or blind validation. D04 never used to select threshold.'),
        context=dict(cases=inputs,variants=VARIANTS,backend='CPU float32, MKLDNN,8threads; paired comparisons only on same backend',
                     factors='2x2 at fixed512: outer5channels page vs reflected256, central2aux local256 vs real512. Central RGB unchanged.',
                     native256='secondary anchor: dimensions,boundary and normalization context change together; not a pure factor',
                     limits='Synthetic input interventions on2 training sources; not production preprocessing, heldout transfer or layer identification'),
        no_training=True,no_new_annotations=True,no_GAN=True,formal_accuracy=False))
    print('Frozen48 saved probability arrays,8 threshold selections,40 CPU context forwards',flush=True)


def check():
    p=read(OUT/'protocol.json')
    assert p['parent_evaluation_plan_sha256']==sha(OLD/'evaluation_plan.json')
    for n,h in p['source_files'].items():assert sha(ROOT/n)==h,n
    return p


def calibration():
    plan=check();assert not (OUT/'calibration.json').exists()
    cases={c['id']:c for c in ev.old.cases()};thresholds=[];rows=[];ranks=[];figures=[]
    yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    for n in CONDITIONS:
        prob={(g,k):np.load(pfile(n,g,k)) for g in GRIDS for k in KEYS}
        # All48 baseline maps must reconstruct the unchanged parent masks.
        for (g,k),a in prob.items():assert np.array_equal(a<.5,mask(OLD/'evaluation'/n/'masks'/g/f'{k}.png'))
        for fold,f in plan['calibration']['folds'].items():
            chosen=choose_threshold([prob['full512',k] for k in f['calibration']],[cases[k] for k in f['calibration']])
            thresholds.append(dict(condition=n,fold=fold,**chosen))
            for g in GRIDS:
                for k in KEYS:
                    a=prob[g,k];c=cases[k]
                    for operating,t in [('fixed',.5),('calibrated',chosen['threshold'])]:
                        metrics=score(a,c,t)
                        row=dict(condition=n,fold=fold,grid=g,case=k,role='transfer' if k in f['transfer'] else 'calibration',operating=operating,threshold=t,**metrics)
                        if operating=='calibrated':
                            target=OUT/'calibrated_masks'/n/fold/g/f'{k}.png';save(target,a.astype(np.float64)<t)
                            row.update(mask_file=str(target.relative_to(ROOT)),mask_sha256=sha(target))
                        rows.append(row)
            for k in f['transfer']:
                c=cases[k];raw=rgb(OUT/'labels'/k/'original.png');base=prob['full512',k]<.5
                calibrated=prob['full512',k].astype(np.float64)<chosen['threshold']
                change=raw.copy();change[base&~calibrated]=[255,0,80];change[~base&calibrated]=[0,100,255]
                hp=refs.reference_path(c);annotation=raw.copy();annotation[c['core']]=[0,100,255];annotation[c['paper']]=[0,190,80]
                a=score(prob['full512',k],c,.5);b=score(prob['full512',k],c,chosen['threshold'])
                target=OUT/'figures'/f'{n}_{fold}_{k}.png'
                panel(target,k+' '+n+' / D04を閾値選択に使わない開発診断',[
                    ('原画像',raw),('Howe固定',np.where(mask(hp),0,255).astype(np.uint8)),
                    (f"閾値.5 墨{a['core_r2']} 紙黒{a['paper_ink']}",np.where(base,0,255).astype(np.uint8)),
                    (f"他領域で選択 墨{b['core_r2']} 紙黒{b['paper_ink']}",np.where(calibrated,0,255).astype(np.uint8)),
                    ('差分 赤=黒消失 青=黒追加',change),('部分注釈 青=墨 緑=非文字',annotation)])
                figures.append(str(target.relative_to(OUT)))
        for g in GRIDS:
            bg=prob[g,'D04'][cases['D04']['paper']].astype(np.float64)
            for mode in ['exact','radius2_proxy']:
                fg=np.concatenate([(prob[g,k] if mode=='exact' else ndi.minimum_filter(prob[g,k],footprint=disk,mode='constant',cval=np.inf))[cases[k]['core']] for k in ['R05','R06']]).astype(np.float64)
                allr=rankdata(np.r_[fg,bg],method='average');u=allr[len(fg):].sum()-len(bg)*(len(bg)+1)/2
                # Independent pairwise count from sorted background, including half ties.
                left=np.searchsorted(np.sort(bg),fg,side='left');right=np.searchsorted(np.sort(bg),fg,side='right')
                auc=float(u/(len(fg)*len(bg)));independent=float(np.mean((len(bg)-right+.5*(right-left))/len(bg)))
                assert abs(auc-independent)<1e-12
                ranks.append(dict(condition=n,grid=g,mode=mode,auc_paper_above_ink=auc,
                                  foreground_n=len(fg),background_n=len(bg),foreground_zero=int((fg==0).sum()),
                                  background_zero=int((bg==0).sum()),foreground_quantiles=np.quantile(fg,[0,.25,.5,.75,1]).tolist(),
                                  background_quantiles=np.quantile(bg,[0,.25,.5,.75,1]).tolist()))
    assert len(thresholds)==8 and len(rows)==192 and len(ranks)==24 and len(figures)==16
    dump(OUT/'calibration.json',dict(protocol_sha256=sha(OUT/'protocol.json'),thresholds=thresholds,rows=rows,rank_diagnostics=ranks,figures=figures,formal_accuracy=False))
    print('Completed calibration transfer:192 readouts,96 derived masks,16 figures',flush=True)


def context():
    plan=check();assert not (OUT/'context.json').exists()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=True
    rows=[];states=[];repeat=[];figures=[]
    for n in CONDITIONS:
        checkpoint=OLD/'runs'/n/'step_00864.pt';assert sha(checkpoint)==read(OLD/'evaluation'/n/'results.json')['checkpoint_sha256']
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=make_model('ragan');model.load_state_dict(payload['model']);del payload;install(model);model.eval()
        before=digest_state(model)
        for c in plan['context']['cases']:
            k=c['id'];base=OUT/'inputs'/k;core=np.asarray(Image.open(base/'core.png'))>0;paper=np.asarray(Image.open(base/'stain.png'))>0
            clear=np.asarray(Image.open(base/'clear.png'))>0;labels=dict(core=core,paper=paper);predictions={}
            for variant in VARIANTS:
                start=time.monotonic();a=np.load(base/f'{variant}.npy');tensor=torch.from_numpy(a.copy()).permute(2,0,1)[None].float()/255
                with torch.inference_mode():out=predict(model,'ragan',tensor)[0,0].numpy().copy()
                if variant=='page_local' and k=='S03':
                    with torch.inference_mode():again=predict(model,'ragan',tensor)[0,0].numpy().copy()
                    assert np.array_equal(out,again);repeat.append(dict(condition=n,case=k,variant=variant,bit_exact=True))
                cx,cy=c['center_xy_in_512'];p=out if variant=='native256' else out[cy:cy+256,cx:cx+256]
                assert p.shape==(256,256) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                dest=OUT/'context'/n/k;dest.mkdir(parents=True,exist_ok=True);np.save(dest/f'{variant}.npy',p);save(dest/f'{variant}.png',p<.5)
                metrics=score(p,labels,.5);row=dict(condition=n,case=k,variant=variant,clear_paper_ink=int(((p<.5)&clear).sum()),
                    probability_file=str((dest/f'{variant}.npy').relative_to(ROOT)),probability_sha256=sha(dest/f'{variant}.npy'),
                    mask_file=str((dest/f'{variant}.png').relative_to(ROOT)),mask_sha256=sha(dest/f'{variant}.png'),**metrics)
                if variant=='native256':
                    prior=mask(OLD/'evaluation'/n/'stain_teacher_fit'/f'{k}.png');row['CPU_vs_saved_CUDA_disagreement_pixels']=int(((p<.5)!=prior).sum())
                rows.append(row);predictions[variant]=p<.5
                print(n,k,variant,'ink',metrics['core_r2'],'paperblack',metrics['paper_ink'],f'{time.monotonic()-start:.1f}s',flush=True)
            labelsjp={'native256':'単独256','page_local':'外周原本 補助256','page_page':'外周原本 補助512','reflect_local':'外周反射 補助256','reflect_page':'外周反射 補助512'}
            items=[('中央の原画像・全条件同じ',rgb(base/'original.png'))]
            for variant in VARIANTS:
                r=next(r for r in rows if r['condition']==n and r['case']==k and r['variant']==variant)
                items.append((labelsjp[variant]+f" 墨{r['core_r2']} 黒{r['paper_ink']}",np.where(predictions[variant],0,255).astype(np.uint8)))
            f=OUT/'figures'/f'{n}_{k}_context.png';panel(f,k+' '+n+' / 中央RGB固定・周囲と中央補助を分離',items);figures.append(str(f.relative_to(OUT)))
        after=digest_state(model);assert before==after;states.append(dict(condition=n,checkpoint_sha256=sha(checkpoint),state_before=before,state_after=after))
        del model
    assert len(rows)==40 and len(repeat)==4 and len(figures)==8
    dump(OUT/'context.json',dict(protocol_sha256=sha(OUT/'protocol.json'),rows=rows,unchanged_models=states,repeated_forward=repeat,figures=figures,no_training=True,formal_accuracy=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','calibration','context','check']);args=a.parse_args()
    {'freeze':freeze,'calibration':calibration,'context':context,'check':check}[args.action]()
