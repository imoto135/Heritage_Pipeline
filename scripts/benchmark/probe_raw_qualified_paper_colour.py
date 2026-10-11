"""Match median/raw colour-reference qualification with observed texture fixed."""
import gc
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import diagnose_paper_boundary_reference_bias as diagnostic
import probe_paper_boundary_pair_gate_v2 as fill
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=diagnostic.ROOT
OUT=ROOT/'tmp_work/raw_qualified_paper_colour_pilot_20261001'
MODES=['recomputed_colour','raw_L_colour','raw_Lab_colour']
SETTINGS=dict(pages=29,ROIs=90,baseline='directional abstention77 + new4-books13',
    fixed='candidate edit, directional hold, retained paper-applied geometry, original observed L seam',
    reference_modes=dict(recomputed_colour='same frozen known paper',raw_L_colour='old known AND raw L>=50, Ldrop[-6,4]',
        raw_Lab_colour='raw_L_colour AND abs(raw a-paper a)<=2, abs(raw b-paper b)<=4'),
    qualification_paper='same preprocessor estimate from full-page median Lab; classification only, frozen before colour solve',
    restorer='same raw/sigma6 known-only smooth harmonic fields, boundary mix8, minimum16 adjacent pairs',
    colour_change='baseline float field plus newly computed colour minus saved baseline colour, original fine L retained',
    unsupported='retain saved baseline when new qualification gives insufficient16-pair support; no new applied pixels',
    texture_reference='old qualified paper unchanged, separate from colour-only qualification',
    corrections_combined=False,new_DDRM=False,pixel_GT=False,development=True,
    control='recompute old-known colour using same fixed active mask, to isolate historical-field recomputation')


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(diagnostic.OUT)
    rows=json.loads((diagnostic.OUT/'diagnostics.json').read_text())['regions']
    originals={r['page']['page']:r['page']['original'] for r in diagnostic.previous.texture.history.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    originals.update({r['page']:r['original'] for r in json.loads((diagnostic.fresh.INPUT/'baseline_review.json').read_text())['cases']})
    estimate_fn=diagnostic.previous.diagnostic.references.prior.mask.pale.prior.anchor.prior.mask.prior.MARGIN.first.paper_estimate
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str(Path(fill.__file__).relative_to(ROOT)):sha(Path(fill.__file__))}
    for folder in [diagnostic.OUT,diagnostic.previous.texture.OUT]:
        for n in ['manifest.json','complete.json']:deps[str((folder/n).relative_to(ROOT))]=sha(folder/n)
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=deps,settings=SETTINGS)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');results=[];poison=[];current=None
    for index,row in enumerate(rows,1):
        rid=row['id'];key=row['page']
        if current!=key:
            original=read(ROOT/originals[key]);current=key
            full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            median=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB);paper=estimate_fn(median)
            diff=full_lab-paper
            raw_L=(full_lab[:,:,0]>=50)&(diff[:,:,0]>=-4)&(diff[:,:,0]<=6)
            raw_Lab=raw_L&(np.abs(diff[:,:,1])<=2)&(np.abs(diff[:,:,2])<=4)
            for n,m in [('raw_L_ok',raw_L),('raw_Lab_ok',raw_Lab)]:write(OUT/'pages'/key/(n+'.png'),m)
            del median,paper,diff;gc.collect()
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        lab=full_lab[cs];crop=original[y:y+h,x:x+w];source=ROOT/row['local'];base=read(source/row['output_file'])
        app=read(source/row['applied_file'],True);known=read(source/'reference_context.png',True)
        references=dict(recomputed_colour=known,raw_L_colour=known&raw_L[cs],raw_Lab_colour=known&raw_Lab[cs])
        assert not np.any(references['raw_Lab_colour']&~references['raw_L_colour'])
        outputs={};stats={};dest=OUT/'native'/rid;field_path=None;color_path=None
        if app[s].any():
            parent=diagnostic.previous.texture.OUT/'native'/rid
            field_path=parent/'baseline_field_Lab.npy';color_path=parent/'baseline_color_Lab.npy'
            field=np.load(field_path,allow_pickle=False);basecolor=np.load(color_path,allow_pickle=False)
            assert np.array_equal(fill.frozen.compose(field,app,base,s),base)
            distance=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5)
            weight=np.exp(-(np.maximum(distance-1,0)/8.)**2)*app
            dest.mkdir(parents=True,exist_ok=True)
            np.save(dest/'baseline_field_Lab.npy',field,allow_pickle=False);np.save(dest/'baseline_color_Lab.npy',basecolor,allow_pickle=False)
        for mode,new_known in references.items():
            md=dest/mode;md.mkdir(parents=True,exist_ok=True);actual=np.zeros_like(app);delta=np.zeros_like(lab);notes=[];used=np.zeros_like(app)
            if app[s].any():
                fields,actual,used,notes=fill.solve_fields(lab,new_known,app,16)
                assert not np.any(actual&~app) and not np.any(new_known&app)
                color=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]
                delta[actual]=color[actual]-basecolor[actual]
                updated=field+delta
                update=actual&np.any(delta!=0,axis=2)
                value=fill.frozen.compose(updated,update,base,s)
                bad=lab.copy();bad[~new_known]=(240,-210,140);repeat=fill.solve_fields(bad,new_known,app,16)
                assert np.array_equal(actual,repeat[1])
                for name in fields:assert np.array_equal(fields[name],repeat[0][name])
                poison.append(dict(id=rid,mode=mode))
                np.savez_compressed(md/'harmonic_fields_Lab.npz',**fields)
                for n,a in dict(result_field_Lab=updated,colour_field_Lab=color,colour_delta_Lab=delta).items():np.save(md/(n+'.npy'),a,allow_pickle=False)
                assert np.array_equal(value[~actual[s]],base[~actual[s]])
                assert not np.any(np.any(value!=base,axis=2)&~app[s])
            else:value=base.copy();update=np.zeros_like(app)
            for n,a in dict(output=value,colour_reference_context=new_known,colour_supported_context=actual,
                colour_used_context=used,update_context=update,fallback_context=app&~actual).items():write(md/(n+'.png'),a)
            dump(md/'components.json',notes)
            outputs[mode]=value;stats[mode]=dict(colour_reference=int(new_known.sum()),removed_reference=int((known&~new_known).sum()),
                supported_ROI=int(actual[s].sum()),fallback_ROI=int((app&~actual)[s].sum()),changed_vs_base=int(np.any(value!=base,axis=2).sum()))
        for mode in MODES:
            stats[mode]['changed_vs_recomputed_colour']=int(np.any(outputs[mode]!=outputs['recomputed_colour'],axis=2).sum())
        for n,a in dict(original=crop,baseline=base,applied_context=app,texture_reference_context=known).items():write(dest/(n+'.png'),a)
        figure([crop,base]+[outputs[m] for m in MODES],['原画像','現行補修・方向保留','元参照で紙色を再計算','生Lでも資格ありの紙色参照','生Lab全条件でも資格ありの紙色参照'],OUT/'comparisons'/(rid+'.png'),row['title']+' · 中央値と元画素の紙色参照資格',
            '同じマスク・保留・元の細かいL模様。紙色の参照だけ分ける。16境界組不足は前回を保持。',
            '暖色中央値補正は併用なし。紙色再計算の対照を置き、参照変更の効果と分ける。')
        results.append({**row,'field_source':str(field_path.relative_to(ROOT)) if field_path else None,
            'colour_source':str(color_path.relative_to(ROOT)) if color_path else None,'colour_diagnostics':stats})
        dump(OUT/'progress.json',dict(completed=index,total=90,latest=rid))
        print('Raw-qualified paper colour:',index,'/90',rid,[(stats[m]['changed_vs_base'],stats[m]['changed_vs_recomputed_colour'],stats[m]['fallback_ROI']) for m in MODES],flush=True)
    for n,h in deps.items():assert sha(ROOT/n)==h
    dump(OUT/'diagnostics.json',results)
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,baseline_recomposition=True,
        colour_reference_nested_subsets=True,original_texture_source_fixed=True,all_nonreference_poison=poison,
        outside_colour_supported_exact_baseline=True,no_new_applied_pixels=True,pixel_GT=False,new_DDRM=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:raw-qualified colour-only paper reference trial90 ROIs',flush=True)


if __name__=='__main__':main()
