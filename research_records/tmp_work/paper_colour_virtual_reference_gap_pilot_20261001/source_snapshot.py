"""Withhold adjacent qualified paper during colour solve; edit geometry fixed."""
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import probe_observed_absolute_paper_patches as prior
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/paper_colour_virtual_reference_gap_pilot_20261001'
MODES=dict(control0=0,virtual3=3,virtual6=6)
SETTINGS=dict(pages=29,ROIs=90,fixed='current mask, directional hold, original effective paper applied, original fine-L field',
    change='colour only: withhold qualified known paper within3/6px of effective applied, solve in this virtual region too',
    virtual='original effective applied OR (old known-paper AND outside distance<=radius)',
    references='old known-paper minus virtual; no newly qualified pixels',
    numerical='same raw/sigma6 known-only smooth4-neighbour harmonic solver, min area64 and16 adjacent pairs',
    physical_update='virtual solver accepted AND original effective applied only; preserve old baseline elsewhere',
    composition='new raw/smooth colour boundary mix uses original effective applied distance8; baseline float field plus new-minus-old colour',
    control='radius0 re-solves same original paper reference; must reproduce baseline at all90ROIs',
    texture_reference='original old known-paper/fine-L unchanged; distinct from newly withheld colour references',
    virtual_caveat='withheld old-qualified paper is numerical unknown only, never a new physical edit or damage classification',
    unsupported='retain original baseline; do not add applied pixels',
    absolute_patches_or_energy_correction_combined=False,new_DDRM=False,pixel_GT=False,development=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    rows=json.loads((prior.OUT/'diagnostics.json').read_text());deps={str(Path(p).relative_to(ROOT)):sha(Path(p)) for p in [__file__,prior.fill.__file__]}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    manifest=dict(settings=SETTINGS,dependencies=deps,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');results=[];poison=[];current=None
    for row in rows:
        key=row['page'];rid=row['id'];source=prior.OUT/'native'/rid;dest=OUT/'native'/rid;dest.mkdir(parents=True,exist_ok=True)
        if current!=key:
            original=read(ROOT/row['original_source']);full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB);current=key
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        lab=full_lab[cs];crop=read(source/'original.png');base=read(source/'baseline.png');app=read(source/'applied_context.png',True);known=read(source/'reference_context.png',True)
        outside_distance=cv2.distanceTransform((~app).astype(np.uint8),cv2.DIST_L2,5)
        outputs={};stats={};colour_source=None
        if app[s].any():
            field=np.load(source/'baseline_field_Lab.npy',allow_pickle=False)
            colour_source=prior.diagnostic.previous.texture.OUT/'native'/rid/'baseline_color_Lab.npy';oldcolour=np.load(colour_source,allow_pickle=False)
            distance=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5);weight=np.exp(-(np.maximum(distance-1,0)/8.)**2)*app
            np.save(dest/'baseline_field_Lab.npy',field,allow_pickle=False);np.save(dest/'baseline_colour_Lab.npy',oldcolour,allow_pickle=False)
        for mode,radius in MODES.items():
            md=dest/mode;md.mkdir(parents=True,exist_ok=True)
            virtual=app|(known&(outside_distance<=radius));reference=known&~virtual
            assert np.array_equal(virtual&~app,known&~reference) and not np.any(virtual&reference)
            accepted=np.zeros_like(app);actual=np.zeros_like(app);used=np.zeros_like(app);notes=[];update=np.zeros_like(app)
            if app[s].any():
                fields,accepted,used,notes=prior.fill.solve_fields(lab,reference,virtual,16);actual=accepted&app
                colour=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]
                delta=np.zeros_like(field);delta[actual]=colour[actual]-oldcolour[actual];updated=field+delta;update=actual&np.any(delta!=0,axis=2)
                value=prior.fill.frozen.compose(updated,update,base,s)
                assert np.array_equal(value[~actual[s]],base[~actual[s]])
                bad=lab.copy();bad[~reference]=(240,-210,140);repeat=prior.fill.solve_fields(bad,reference,virtual,16)
                assert np.array_equal(accepted,repeat[1])
                for n in fields:assert np.array_equal(fields[n],repeat[0][n])
                poison.append(dict(id=rid,mode=mode))
                np.savez_compressed(md/'harmonic_fields_Lab.npz',**fields)
                for n,a in dict(colour_field_Lab=colour,colour_delta_Lab=delta,result_field_Lab=updated).items():np.save(md/(n+'.npy'),a,allow_pickle=False)
            else:value=base.copy()
            for n,a in dict(output=value,virtual_context=virtual,colour_reference_context=reference,virtual_accepted_context=accepted,
                physical_supported_context=actual,colour_used_context=used,update_context=update,fallback_context=app&~actual).items():write(md/(n+'.png'),a)
            dump(md/'components.json',notes);outputs[mode]=value
            stats[mode]=dict(virtual_extra_context=int((virtual&~app).sum()),removed_colour_reference=int((known&~reference).sum()),
                physical_supported_ROI=int(actual[s].sum()),fallback_ROI=int((app&~actual)[s].sum()),changed_vs_base=int(np.any(value!=base,axis=2).sum()))
            assert not np.any(np.any(value!=base,axis=2)&~app[s])
        assert np.array_equal(outputs['control0'],base)
        for n,a in dict(original=crop,baseline=base,applied_context=app,texture_reference_context=known).items():write(dest/(n+'.png'),a)
        figure([crop,base]+[outputs[m] for m in MODES],['原画像','現行補修・方向保留','紙色再計算・仮領域0','紙色・隣接紙3pxを保留','紙色・隣接紙6pxを保留'],OUT/'comparisons'/(rid+'.png'),row['title']+' · 紙色の直接境界参照を分離',
            '仮領域は計算上だけ。実編集・適用・元細模様は固定。支持不足は元基準を保持。',
            '参照は元の資格紙面だけ。仮領域の原値は補間に使わない。模様量補正は併用なし。')
        results.append({**row,'colour_source':str(colour_source.relative_to(ROOT)) if colour_source else None,'virtual_gap_diagnostics':stats})
        dump(OUT/'progress.json',dict(completed=len(results),total=90,latest=rid));print('Virtual colour gap:',len(results),'/90',rid,[(m,stats[m]['changed_vs_base'],stats[m]['fallback_ROI']) for m in MODES],flush=True)
    dump(OUT/'diagnostics.json',results)
    dump(OUT/'verification.json',dict(status='passed',ROIs=90,pages=29,control0_all_images_exact_baseline=True,
        all_noncolour_reference_poison_invariance=poison,only_old_qualified_paper_withheld=True,
        physical_update_subset_original_effective_applied=True,original_fine_L_and_texture_reference_fixed=True,new_DDRM=False,pixel_GT=False))
    for p,h in deps.items():assert sha(ROOT/p)==h
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete: virtual colour-reference gap90 ROIs',flush=True)


if __name__=='__main__':main()
