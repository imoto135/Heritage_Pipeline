"""Adaptive follow-up: reproduce real512 tiling and intervene on ROI aux only.

Planned after the centered-context readout. Not a preregistered original test.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import isolate_binarization_causes as exp
from compare_v5_aligned_fade_control import dump,sha,rgb,mask,save
from apply_ragan_real_regions import origins,panel
from ragan_pilot_support import auxiliaries,make_model,predict
from deterministic_reflection_support import install
from audit_ragan_inference_context import digest_state

ROOT=exp.ROOT;OUT=exp.OUT


def freeze():
    parent=exp.check();assert not (OUT/'tile_plan.json').exists();files={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    files[str((ROOT/'scripts/benchmark/apply_ragan_real_regions.py').relative_to(ROOT))]=sha(ROOT/'scripts/benchmark/apply_ragan_real_regions.py')
    for c in parent['context']['cases']:
        assert sha(ROOT/c['source'])==c['source_sha256'];files[c['source']]=c['source_sha256']
    files[str((OUT/'context.json').relative_to(ROOT))]=sha(OUT/'context.json')
    dump(OUT/'tile_plan.json',dict(status='Adaptive follow-up frozen after centered-context outputs, before tile-level outputs',
        parent_protocol_sha256=sha(OUT/'protocol.json'),source_files=files,conditions=exp.CONDITIONS,cases=['S03','S07'],
        variants=['native_aux','roi_local_aux'],fixed='Original512 stride384+last origins, originalHann floor.05, threshold.5, CPU8threads,float32',
        intervention='Same RGB and same tiles/weights. Replace only2aux within source256 ROI by precomputed native256 aux; keep outside untouched. Clip at tile/ROI intersection.',
        readouts='Every constituent tile, plus exact same probability-weighted blend; no tile selection or new threshold',
        limits='Training-source diagnosis. Tile location, visible context and edge distance co-vary between constituent tiles; not a pure position-only or layer identification test.',
        no_training=True,formal_accuracy=False))
    print('Frozen adaptive actual-tile decomposition',flush=True)


def check():
    parent=exp.check();p=exp.read(OUT/'tile_plan.json');assert p['parent_protocol_sha256']==sha(OUT/'protocol.json')
    for n,h in p['source_files'].items():assert sha(ROOT/n)==h,n
    return parent,p


def run():
    parent,plan=check();assert not (OUT/'tiles.json').exists();torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    rows=[];blends=[];states=[];figures=[];weight=np.outer(np.maximum(np.hanning(512),.05),np.maximum(np.hanning(512),.05))
    for n in exp.CONDITIONS:
        cp=exp.OLD/'runs'/n/'step_00864.pt';assert sha(cp)==exp.read(exp.OLD/'evaluation'/n/'results.json')['checkpoint_sha256']
        q=torch.load(cp,map_location='cpu',weights_only=False);model=make_model('ragan');model.load_state_dict(q['model']);del q;install(model);model.eval();before=digest_state(model)
        for c in parent['context']['cases']:
            k=c['id'];base=OUT/'inputs'/k;page=rgb(ROOT/c['source']);x0,y0,w,h=c['roi'];local=np.load(base/'native256.npy')
            core=np.asarray(Image.open(base/'core.png'))>0;paper=np.asarray(Image.open(base/'stain.png'))>0;labels=dict(core=core,paper=paper)
            den=np.zeros((h,w),np.float64);sums={v:np.zeros((h,w),np.float64) for v in plan['variants']};items=[('中央の原画像',rgb(base/'original.png'))]
            tile_items=[];start=time.monotonic()
            for y in origins(page.shape[0]):
                for x in origins(page.shape[1]):
                    l,t,r,b=max(x0,x),max(y0,y),min(x0+w,x+512),min(y0+h,y+512)
                    if l>=r or t>=b:continue
                    rs=np.s_[t-y0:b-y0,l-x0:r-x0];ts=np.s_[t-y:b-y,l-x:r-x];tile=page[y:y+512,x:x+512];a=auxiliaries(tile)
                    altered=a.copy();altered[ts][:,:,3:]=local[rs][:,:,3:]
                    changed=np.any(a!=altered,axis=2);check_changed=changed.copy();check_changed[ts]=False;assert not check_changed.any() and np.array_equal(a[:,:,:3],altered[:,:,:3])
                    valid=np.zeros((h,w),bool);valid[rs]=True;tile_label=dict(core=core&valid,paper=paper&valid)
                    ww=weight[ts];den[rs]+=ww
                    for variant,inp in [('native_aux',a),('roi_local_aux',altered)]:
                        tensor=torch.from_numpy(inp.copy()).permute(2,0,1)[None].float()/255
                        with torch.inference_mode():p=predict(model,'ragan',tensor)[0,0].numpy().copy()
                        canvas=np.ones((h,w),np.float32);canvas[rs]=p[ts];sums[variant][rs]+=p[ts]*ww
                        dest=OUT/'tiles'/n/k/variant;dest.mkdir(parents=True,exist_ok=True);name=f'x{x}_y{y}';np.save(dest/f'{name}.npy',canvas);save(dest/f'{name}.png',canvas<.5)
                        metrics=exp.score(canvas,tile_label,.5)
                        # Coverage excludes outside tile; black predictions outside are false by construction.
                        row=dict(condition=n,case=k,variant=variant,tile_xy=[x,y],overlap_roi_xywh=[l-x0,t-y0,r-l,b-t],
                            spot_center_tile_xy=[x0+99-x,y0+129-y] if k=='S07' else None,
                            changed_aux_pixels=int(changed.sum()),mask_file=str((dest/f'{name}.png').relative_to(ROOT)),mask_sha256=sha(dest/f'{name}.png'),
                            probability_file=str((dest/f'{name}.npy').relative_to(ROOT)),probability_sha256=sha(dest/f'{name}.npy'),**metrics)
                        rows.append(row)
                        if variant=='native_aux':
                            shown=np.where(canvas<.5,0,255).astype(np.uint8);shown[~valid]=180
                            tile_items.append((f"原タイル{x},{y} 黒{metrics['paper_ink']}",shown))
            assert (den>0).all()
            for variant in plan['variants']:
                p=(sums[variant]/den).astype(np.float32);dest=OUT/'tiles'/n/k;np.save(dest/f'{variant}_blend.npy',p);save(dest/f'{variant}_blend.png',p<.5)
                metrics=exp.score(p,labels,.5);row=dict(condition=n,case=k,variant=variant,
                    mask_file=str((dest/f'{variant}_blend.png').relative_to(ROOT)),mask_sha256=sha(dest/f'{variant}_blend.png'),
                    probability_file=str((dest/f'{variant}_blend.npy').relative_to(ROOT)),probability_sha256=sha(dest/f'{variant}_blend.npy'),**metrics)
                if variant=='native_aux':
                    old=mask(exp.OLD/'source_context'/n/'masks/full512'/f'{k}.png');row['CPU_vs_saved_CUDA_disagreement_pixels']=int(((p<.5)!=old).sum())
                blends.append(row);items.append((('元の合成' if variant=='native_aux' else '中央補助256で合成')+f" 墨{metrics['core_r2']} 黒{metrics['paper_ink']}",np.where(p<.5,0,255).astype(np.uint8)))
            # Three main panels then every actual tile; gray denotes pixels not seen by that tile.
            f=OUT/'figures'/f'{n}_{k}_actual_tiles.png';panel(f,k+' '+n+' / 実ページ512のタイル分解・灰=範囲外',items+tile_items);figures.append(str(f.relative_to(OUT)))
            print(n,k,'tiles',len(tile_items),'blends',[(r['variant'],r['core_r2'],r['paper_ink']) for r in blends[-2:]],f'{time.monotonic()-start:.1f}s',flush=True)
        after=digest_state(model);assert before==after;states.append(dict(condition=n,state_before=before,state_after=after,checkpoint_sha256=sha(cp)));del model
    dump(OUT/'tiles.json',dict(tile_plan_sha256=sha(OUT/'tile_plan.json'),rows=rows,blends=blends,figures=figures,unchanged_models=states,no_training=True,formal_accuracy=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);args=a.parse_args();freeze() if args.action=='freeze' else run()
