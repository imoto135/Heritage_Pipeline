"""Training-free red evidence and separate protection/edit decisions.

Development/regression exploration on previously inspected pages, without GT.
No image pixels are restored or deleted. Old experiments are read-only inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/training_free_protection_v2_20260930'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    cv2.setNumThreads(1)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    groups=[dict(result=ROOT/'tmp_work/training_free_color_shape_20260930',
                 pilot=ROOT/'tmp_work/training_free_preprocess_20260930',
                 review=ROOT/'tmp_work/npj_preprocess_review_20260930',name='initial_development'),
            dict(result=ROOT/'tmp_work/training_free_fixed_check_20260930',
                 pilot=ROOT/'tmp_work/training_free_fixed_check_20260930/pilot',
                 review=ROOT/'tmp_work/training_free_fixed_check_20260930/input_review',name='regression_development')]
    template=Path(__file__).with_name('training_free_protection_viewer.html')
    settings=dict(
        red_seed=dict(filtered_image='3x3 median BGR; classification only',lab_a_min=20.,
                      a_over_positive_b=.95,lightness_min=20.,saturation_min=80,
                      hue_max=14,hue_min_wrap=174,minimum_seed_pixels_per_old_component=3),
        red_component=dict(median_lightness_min=25.,growth='whole prior red candidate component; original weak colors preserved'),
        red_uncertain_context=dict(radius=24.,minimum_component_median_lightness=25.,
            rule='unconfirmed old red component within radius of confirmed red is protected with unknown color'),
        thin_extension=dict(radius=2.,support='at least one of prior Sauvola scales; within radius of base ink or confirmed/contextual red'),
        dense_stain='reuse logged prior dense brown-evidence components; not new universal proof',
        edit_guard=dict(radius=3.,rule='stain candidate outside protection guard and hold only'),
        hold='unconfirmed red (may overlap protection), scale disagreement outside protection except prior dense stain, prior dense hold, stain near protection',
        mask_semantics='independent stain_candidate/protect/hold; edit_allowed disjoint from protect and hold',
        pixels='native resolution; no automatic deletion; no fixed border cutoff')
    cases=[];dependencies={};sources={}
    for group in groups:
        review_manifest=json.loads((group['review']/'manifest.json').read_text())
        rois=json.loads((group['review']/'rois.json').read_text())
        completion=json.loads((group['result']/'complete.json').read_text())['outputs']
        pilot_completion=json.loads((group['pilot']/'complete.json').read_text())['outputs']
        for p in [group['result']/'manifest.json',group['review']/'manifest.json',group['review']/'rois.json',group['pilot']/'manifest.json']:
            dependencies[str(p.relative_to(ROOT))]=sha(p)
        for case in review_manifest['cases']:
            stem=case['id']
            for name in ['original.png','red_weak.png','brown_field.png','color_shape/red.png','color_shape/ink.png','color_shape/stain.png']:
                p=group['result']/stem/name
                assert sha(p)==completion[f'{stem}/{name}']
                dependencies[str(p.relative_to(ROOT))]=sha(p)
            for name in ['sauvola_local/foreground.png','scale_votes.png']:
                p=group['pilot']/stem/name
                assert sha(p)==pilot_completion[f'{stem}/{name}']
                dependencies[str(p.relative_to(ROOT))]=sha(p)
            p=group['result']/stem/'components.json';dependencies[str(p.relative_to(ROOT))]=sha(p)
            sources[case['original']]=sha(ROOT/case['original'])
            prior=json.loads((group['result']/'manifest.json').read_text())['input_hashes']
            assert sources[case['original']]==prior[case['original']]
            cases.append(dict(**case,group=group['name'],result=group['result'],pilot=group['pilot'],rois=rois[stem]))
    manifest=dict(purpose='Development/regression improvement without training/manual GT; not an independent test',
        settings=settings,dependencies=dependencies,inputs=sources,
        source_sha256=sha(Path(__file__)),viewer_template_sha256=sha(template),
        versions=dict(python=sys.version,opencv=cv2.__version__,numpy=np.__version__),
        cases=[dict(id=c['id'],original=c['original'],group=c['group']) for c in cases])
    path=OUT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('Existing output has different code/parameters; use a new experiment version')
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(__file__,OUT/'source_snapshot.py');shutil.copyfile(template,OUT/'viewer_snapshot.html')

    def read(path,flag=cv2.IMREAD_COLOR):
        im=cv2.imread(str(path),flag)
        if im is None:raise OSError(path)
        return im

    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        assert cv2.imwrite(str(path),value),path

    colors=dict(protect=(225,100,38),red=(69,52,228),edit_allowed=(23,178,235),
                hold=(176,65,155),ink_color_uncertain=(180,165,0),stain_candidate=(23,178,235))

    def overlay(original,mask,key):
        value=original.copy();value[mask]=np.round(.45*original[mask]+.55*np.array(colors[key])).astype(np.uint8)
        return value

    def decision(original,masks,flat=False):
        value=np.full_like(original,(239,235,231)) if flat else original.copy()
        # Priority is only for display. Saved masks retain their overlaps.
        for key in ['hold','edit_allowed','protect','ink_color_uncertain','red']:
            mask=masks[key]
            value[mask]=colors[key] if flat else np.round(.45*original[mask]+.55*np.array(colors[key])).astype(np.uint8)
        return value

    def figure(images,titles,path,title):
        height=max(1.9,min(6.,4.5*images[0].shape[0]/images[0].shape[1]+.65))
        fig,axes=plt.subplots(2,3,figsize=(13.5,2*height))
        for ax,im,label in zip(axes.flat,images,titles):
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),interpolation='nearest')
            ax.set_title(label,fontproperties=font,fontsize=10);ax.axis('off')
        fig.suptitle(title,fontproperties=font,fontsize=13)
        fig.tight_layout(rect=(0,0,1,.94),h_pad=2.8,w_pad=1.5)
        fig.savefig(path,dpi=145,facecolor='white');plt.close(fig)

    def distance_to(mask):
        if not mask.any():return np.full(mask.shape,np.inf,np.float32)
        return cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)

    metrics=[];gallery=[]
    for case in cases:
        started=time.monotonic();stem=case['id'];old=case['result']/stem;target=OUT/stem
        original=read(old/'original.png');h,w=original.shape[:2]
        foreground=read(case['pilot']/stem/'sauvola_local/foreground.png',0)>0
        votes=read(case['pilot']/stem/'scale_votes.png',0)//85
        old_red=read(old/'color_shape/red.png',0)>0
        old_ink=read(old/'color_shape/ink.png',0)>0
        old_stain=read(old/'color_shape/stain.png',0)>0
        brown=read(old/'brown_field.png',0)>0
        median=cv2.medianBlur(original,3)
        filtered_lab=cv2.cvtColor(median.astype(np.float32)/255.,cv2.COLOR_BGR2LAB)
        filtered_hsv=cv2.cvtColor(median,cv2.COLOR_BGR2HSV)
        L,A,B=filtered_lab[:,:,0],filtered_lab[:,:,1],filtered_lab[:,:,2]
        H,S=filtered_hsv[:,:,0],filtered_hsv[:,:,1]
        robust_seed=(A>=20)&(A>=.95*np.maximum(B,0))&(L>=20)&(S>=80)&((H<=14)|(H>=174))&old_red
        n,labels,stats,_=cv2.connectedComponentsWithStats(old_red.astype(np.uint8),connectivity=8)
        seed_counts=np.bincount(labels.ravel(),weights=robust_seed.ravel(),minlength=n)
        raw_light=cv2.cvtColor(original.astype(np.float32)/255.,cv2.COLOR_BGR2LAB)[:,:,0]
        median_light=np.zeros(n,np.float32)
        component_rows=[]
        for i in range(1,n):
            x,y,cw,ch,area=map(int,stats[i]);sl=(slice(y,y+ch),slice(x,x+cw))
            median_light[i]=np.median(raw_light[sl][labels[sl]==i])
            component_rows.append(dict(id=i,xywh=[x,y,cw,ch],area=area,robust_seed_pixels=int(seed_counts[i]),median_lightness=float(median_light[i])))
        seed_supported_ids=seed_counts>=3;seed_supported_ids[0]=False
        confirmed_ids=seed_supported_ids&(median_light>=25.)
        confirmed_red=confirmed_ids[labels]
        seed_supported_red=seed_supported_ids[labels]
        red_distance=distance_to(confirmed_red)
        minimum_distance=np.full(n,np.inf,np.float32)
        np.minimum.at(minimum_distance,labels.ravel(),red_distance.ravel())
        context_ids=(~confirmed_ids)&(median_light>=25)&(minimum_distance<=24.)
        context_ids[0]=False
        contextual_red=context_ids[labels]
        uncertain_color=old_red&~confirmed_red
        for row in component_rows:
            i=row['id'];row.update(confirmed_red=bool(confirmed_ids[i]),
                protected_by_red_context=bool(context_ids[i]),
                minimum_distance_to_confirmed_red=float(minimum_distance[i]) if np.isfinite(minimum_distance[i]) else None)
        del labels,filtered_lab,filtered_hsv,red_distance

        # Reconstruct the previously logged shape masks without new thresholds.
        n,fg_labels,fg_stats,_=cv2.connectedComponentsWithStats(foreground.astype(np.uint8),connectivity=8)
        dense_stain_ids=np.zeros(n,bool);dense_hold_ids=np.zeros(n,bool)
        for component in json.loads((old/'components.json').read_text()):
            i=component['id'];assert fg_stats[i].tolist()==component['xywh']+[component['area']]
            if component['disposition']=='stain_candidate':dense_stain_ids[i]=True
            else:dense_hold_ids[i]=True
        dense_stain=dense_stain_ids[fg_labels];dense_hold=dense_hold_ids[fg_labels]
        del fg_labels
        base=(foreground&~dense_stain&~dense_hold)|confirmed_red|contextual_red
        thin_extension=(votes>0)&(distance_to(base)<=2.)&~dense_stain&~dense_hold
        protect=base|thin_extension
        stain_candidate=brown|dense_stain
        guard=distance_to(protect)<=3.
        hold=uncertain_color|(((votes>0)&(votes<3))&~protect&~dense_stain)|dense_hold|((stain_candidate&guard)&~protect)
        edit_allowed=stain_candidate&~guard&~hold
        ink_color_uncertain=protect&uncertain_color
        black=protect&~confirmed_red&~ink_color_uncertain
        masks=dict(red=confirmed_red,protect=protect,edit_allowed=edit_allowed,hold=hold,
                   ink_color_uncertain=ink_color_uncertain,stain_candidate=stain_candidate,
                   black=black,robust_seed=robust_seed,seed_supported_red=seed_supported_red,
                   contextual_red=contextual_red,thin_extension=thin_extension,
                   protection_guard=guard,dense_stain=dense_stain,dense_hold=dense_hold,
                   legacy_red=old_red,legacy_ink=old_ink,legacy_stain=old_stain)
        assert not (edit_allowed&protect).any() and not (edit_allowed&hold).any()
        assert np.array_equal(protect,black|confirmed_red|ink_color_uncertain)
        for key,mask in masks.items():write(target/(key+'.png'),mask.astype(np.uint8)*255)
        write(target/'original.png',original)
        write(target/'decision_map.png',decision(original,masks,True))
        write(target/'decision_overlay.png',decision(original,masks))
        for key in colors:write(target/('overlay_'+key+'.png'),overlay(original,masks[key],key))
        write(target/'overlay_legacy_red.png',overlay(original,old_red,'red'))
        (target/'red_components.json').write_text(json.dumps(component_rows,ensure_ascii=False,indent=2)+'\n')
        titles=['原画像','前版：朱候補','改良：色を確認した朱候補','文字保護：色未確定も含む',
                '編集許可の試作：黄','独立出力の重ね合わせ']
        images=[original,overlay(original,old_red,'red'),overlay(original,confirmed_red,'red'),
                overlay(original,protect,'protect'),overlay(original,edit_allowed,'edit_allowed'),decision(original,masks)]
        figure(images,titles,target/'overview.png',stem+'　青＝保護／赤＝朱／青緑＝保護・色未確定／黄＝編集許可／紫＝保留')
        page=dict(id=stem,title=case['title'],group=case['group'],width=w,height=h,rois=[])
        if stem=='200018243_00003_2':page['title']='本文・裏写り候補・欄外'
        for i,roi in enumerate(case['rois']):
            x,y,rw,rh=roi['xywh'];assert 0<=x<x+rw<=w and 0<=y<y+rh<=h
            sl=(slice(y,y+rh),slice(x,x+rw));name=f'roi_{i+1:02d}.png'
            figure([value[sl] for value in images],titles,target/name,roi['title'])
            page['rois'].append(dict(title=roi['title'],xywh=roi['xywh'],image=f'{stem}/{name}'))
        # Repeat previously selected false-red crops for a direct visual comparison.
        audits=case['result']/'audit_diagnostics.json'
        if audits.exists():
            rows=[r for r in json.loads(audits.read_text()) if r['page']==stem]
            for i,c in enumerate(rows[0]['inspected_red_components']):
                x,y,cw,ch=c['xywh'];pad=35;x0,y0,x1,y1=max(0,x-pad),max(0,y-pad),min(w,x+cw+pad),min(h,y+ch+pad)
                sl=(slice(y0,y1),slice(x0,x1));name=f'red_audit_{i+1:02d}.png'
                figure([value[sl] for value in images],titles,target/name,'前版の赤候補を再確認（正解ラベルではない）')
                page['rois'].append(dict(title=f'前版の赤候補{i+1}',xywh=[x0,y0,x1-x0,y1-y0],image=f'{stem}/{name}'))
        # Dense stain and the small red anomaly remain explicit review sites.
        for i,c in enumerate(json.loads((old/'components.json').read_text())):
            x,y,cw,ch=c['xywh'];pad=35;x0,y0,x1,y1=max(0,x-pad),max(0,y-pad),min(w,x+cw+pad),min(h,y+ch+pad)
            sl=(slice(y0,y1),slice(x0,x1));name=f'blob_{i+1:02d}.png'
            figure([value[sl] for value in images],titles,target/name,'前版の濃い塊候補')
            page['rois'].append(dict(title='濃い塊候補',xywh=[x0,y0,x1-x0,y1-y0],image=f'{stem}/{name}'))
        rectangles=[(None,[0,0,w,h])]+[(i+1,r['xywh']) for i,r in enumerate(page['rois'])]
        for roi,rect in rectangles:
            x,y,rw,rh=rect;sl=(slice(y,y+rh),slice(x,x+rw));m={k:v[sl] for k,v in masks.items()}
            metrics.append(dict(page=stem,group=case['group'],roi=roi,xywh=rect,
                **{key+'_pixels':int(value.sum()) for key,value in m.items()},
                previous_red_no_longer_typed=int((m['legacy_red']&~m['red']).sum()),
                previous_red_still_protected=int((m['legacy_red']&m['protect']).sum()),
                previous_ink_no_longer_protected=int((m['legacy_ink']&~m['protect']).sum()),
                new_protected_pixels=int((m['protect']&~m['legacy_ink']).sum()),
                stain_protection_overlap=int((m['stain_candidate']&m['protect']).sum()),
                protection_hold_overlap=int((m['protect']&m['hold']).sum()),
                previous_stain_near_new_protection=int((m['legacy_stain']&m['protection_guard']).sum()),
                previous_stain_edit_allowed=int((m['legacy_stain']&m['edit_allowed']).sum())))
        gallery.append(page)
        print(f'{stem}: red {int(old_red.sum())}->{int(confirmed_red.sum())}, '
              f'previous red protected {int((old_red&protect).sum())}, '
              f'edit {int(edit_allowed.sum())}, {time.monotonic()-started:.1f}s',flush=True)
    (OUT/'diagnostics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    (OUT/'index.html').write_text(template.read_text().replace('__DATA__',json.dumps(dict(cases=gallery),ensure_ascii=False)))
    outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*.png'))}
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',pages=len(cases),
        fixed_rois=sum(len(c['rois']) for c in cases),all_review_crops=sum(len(c['rois']) for c in gallery),
        outputs=outputs),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
