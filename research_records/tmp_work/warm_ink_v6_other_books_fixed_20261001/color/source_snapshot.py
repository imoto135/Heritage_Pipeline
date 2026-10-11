"""Color/shape exploration after the frozen training-free Sauvola pilot.

No training, no manual ground truth, no restoration or automatic deletion.
All thresholds are development heuristics shared by the four diagnostic pages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=ROOT/'tmp_work/training_free_preprocess_20260930')
    parser.add_argument('--review', type=Path, default=ROOT/'tmp_work/npj_preprocess_review_20260930')
    parser.add_argument('--output', type=Path, default=ROOT/'tmp_work/training_free_color_shape_20260930')
    parser.add_argument('--core-radius', type=float, default=8.)
    parser.add_argument('--core-fraction', type=float, default=.25)
    parser.add_argument('--fill', type=float, default=.55)
    parser.add_argument('--min-component-area', type=int, default=200)
    parser.add_argument('--max-aspect', type=float, default=3.)
    args = parser.parse_args()
    out, pilot, review = args.output.resolve(), args.pilot.resolve(), args.review.resolve()
    if not out.is_relative_to(ROOT):
        parser.error('--output must be inside the project')
    if args.core_radius <= 0 or not 0 <= args.core_fraction <= 1 or not 0 <= args.fill <= 1 or args.min_component_area < 1 or args.max_aspect < 1:
        parser.error('Invalid component thresholds')
    out.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out/'plot_cache')
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    cv2.setNumThreads(1)
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')

    def read(path, flag=cv2.IMREAD_COLOR):
        value = cv2.imread(str(path), flag)
        if value is None:
            raise ValueError(f'Cannot read {path}')
        return value

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), value):
            raise OSError(path)

    colors = {'black': (225, 100, 38), 'red': (69, 52, 228),
              'stain': (23, 178, 235), 'hold': (176, 65, 155)}  # BGR
    modes = {'local_baseline': '前回の局所Sauvola：前景のみ',
             'color_only': '原画像の色を追加：形の判定なし',
             'color_shape': '色＋詰まった塊の判定＋保留'}

    def paint(original, masks, selected='all', flat=False):
        if selected == 'ink':
            keys = ('black', 'red')
        else:
            keys = colors if selected == 'all' else (selected,)
        value = np.full_like(original, (239,235,231)) if flat else original.copy()
        for key in keys:
            mask = masks[key]
            if flat:
                value[mask] = colors[key]
            else:
                value[mask] = np.round(.45*original[mask]+.55*np.array(colors[key])).astype(np.uint8)
        return value

    def figure(images, titles, path, title):
        row_height = max(1.9, min(6., 4.5*images[0].shape[0]/images[0].shape[1]+.65))
        fig, axes = plt.subplots(2,3,figsize=(13.5,2*row_height))
        for ax,value,label in zip(axes.flat,images,titles):
            ax.imshow(cv2.cvtColor(value,cv2.COLOR_BGR2RGB),interpolation='nearest')
            ax.set_title(label,fontproperties=font,fontsize=10)
            ax.axis('off')
        fig.suptitle(title,fontproperties=font,fontsize=13)
        fig.tight_layout(rect=(0,0,1,.94),h_pad=2.8,w_pad=1.5)
        fig.savefig(path,dpi=145,facecolor='white')
        plt.close(fig)

    rois = json.loads((review/'rois.json').read_text())
    previous_manifest = json.loads((pilot/'manifest.json').read_text())
    previous_complete = json.loads((pilot/'complete.json').read_text())
    cases = json.loads((review/'manifest.json').read_text())['cases']
    dependencies = {}
    for case in cases:
        for relative in [f"{case['id']}/original.png", f"{case['id']}/sauvola_local/foreground.png",
                         f"{case['id']}/scale_votes.png"]:
            expected = previous_complete['outputs'][relative]
            assert sha(pilot/relative) == expected, relative
            dependencies[relative] = expected
        assert sha(ROOT/case['original']) == previous_manifest['inputs'][case['original']]

    settings = dict(
        color_space='OpenCV float32 BGR [0,1] to CIELab; L [0,100], signed a/b',
        local_color_background='quarter-size Lab, Gaussian sigma16, resized to native',
        red_weak=dict(a_min=6., delta_a_min=3., a_over_positive_b=.45, hue_max=22, hue_min_wrap=168, saturation_min=30),
        red_seed=dict(a_min=14., delta_a_min=6., a_over_positive_b=.8, hue_max=18, hue_min_wrap=172, saturation_min=60),
        red_growth='8-connected weak-color components with at least 2 seed pixels; independent of gray foreground',
        brown_field=dict(a_min=3., b_min=8., delta_a_min=1.3, delta_b_min=-.5,
                         b_over_a=1.3, lightness_below_local=2., minimum_component_area=20),
        dense_component=dict(core_radius=args.core_radius, core_fraction=args.core_fraction,
                             fill=args.fill, min_area=args.min_component_area, max_aspect=args.max_aspect,
                             red_fraction_exclusion=.05),
        component_paper_ring='bbox expanded16 pixels; exclude gray foreground and red candidates; median Lab',
        blob_brown_evidence=dict(delta_a_min=1., delta_b_min=-1., mean_a_min=2., mean_b_min=8.),
        policy='dense brown-evidence blobs -> stain candidate; other dense blobs -> hold; no pixels deleted',
        scale_disagreement='only unprotected pixels with 1 or 2 votes; diagnostic hold, not calibrated uncertainty',
        border='no forced background; all classification at native resolution')
    template = Path(__file__).with_name('training_free_color_shape_viewer.html')
    manifest = dict(purpose='Training-free exploratory color/shape segmentation; no manual GT or deletion',
                    parameters=settings, dependencies=dependencies,
                    parent_manifest_sha256=sha(pilot/'manifest.json'), rois_sha256=sha(review/'rois.json'),
                    source_sha256=sha(Path(__file__)), viewer_template_sha256=sha(template),
                    versions=dict(python=sys.version,opencv=cv2.__version__,numpy=np.__version__),
                    modes=modes, input_hashes=previous_manifest['inputs'])
    manifest_path = out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('Output already belongs to a different experiment; use a new output directory')
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(__file__,out/'source_snapshot.py')
    shutil.copyfile(template,out/'viewer_snapshot.html')
    metrics, gallery = [], []
    for case in cases:
        started = time.monotonic()
        stem = case['id']; target=out/stem; target.mkdir(exist_ok=True)
        original=read(pilot/stem/'original.png'); h,w=original.shape[:2]
        foreground=read(pilot/stem/'sauvola_local/foreground.png',0)>127
        votes=read(pilot/stem/'scale_votes.png',0)//85
        lab=cv2.cvtColor(original.astype(np.float32)/255.,cv2.COLOR_BGR2LAB)
        hsv=cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
        small=cv2.resize(lab,((w+3)//4,(h+3)//4),interpolation=cv2.INTER_AREA)
        local=cv2.resize(cv2.GaussianBlur(small,(0,0),16.,borderType=cv2.BORDER_REFLECT),(w,h),interpolation=cv2.INTER_LINEAR)
        light,a,b=lab[:,:,0],lab[:,:,1],lab[:,:,2]
        da=a-local[:,:,1]; db=b-local[:,:,2]; hue,sat=hsv[:,:,0],hsv[:,:,1]
        weak=(a>=6)&(da>=3)&(a>=.45*np.maximum(b,0))&((hue<=22)|(hue>=168))&(sat>=30)
        seeds=weak&(a>=14)&(da>=6)&(a>=.8*np.maximum(b,0))&((hue<=18)|(hue>=172))&(sat>=60)
        n,labels=cv2.connectedComponents(weak.astype(np.uint8),connectivity=8)
        seed_counts=np.bincount(labels.ravel(),weights=seeds.ravel(),minlength=n)
        accepted=seed_counts>=2; accepted[0]=False
        red=accepted[labels]
        del labels
        brown=(a>=3)&(b>=8)&(da>=1.3)&(db>=-.5)&(b>=1.3*a)&(light<local[:,:,0]-2)&~red
        n,brown_labels,brown_stats,_=cv2.connectedComponentsWithStats(brown.astype(np.uint8),connectivity=8)
        accepted=brown_stats[:,cv2.CC_STAT_AREA]>=20; accepted[0]=False
        brown=accepted[brown_labels]
        del brown_labels
        n,labels,stats,_=cv2.connectedComponentsWithStats(foreground.astype(np.uint8),connectivity=8)
        distance=cv2.distanceTransform(foreground.astype(np.uint8),cv2.DIST_L2,5)
        areas=stats[:,cv2.CC_STAT_AREA].astype(float)
        cores=np.bincount(labels.ravel(),weights=(distance>=args.core_radius).ravel(),minlength=n)/np.maximum(areas,1.)
        red_fraction=np.bincount(labels.ravel(),weights=red.ravel(),minlength=n)/np.maximum(areas,1.)
        means=[np.bincount(labels.ravel(),weights=lab[:,:,i].ravel(),minlength=n)/np.maximum(areas,1.) for i in range(3)]
        fill=areas/np.maximum(stats[:,cv2.CC_STAT_WIDTH]*stats[:,cv2.CC_STAT_HEIGHT],1)
        aspect=np.maximum(stats[:,cv2.CC_STAT_WIDTH]/np.maximum(stats[:,cv2.CC_STAT_HEIGHT],1),
                          stats[:,cv2.CC_STAT_HEIGHT]/np.maximum(stats[:,cv2.CC_STAT_WIDTH],1))
        dense=(areas>=args.min_component_area)&(cores>=args.core_fraction)&(fill>=args.fill)&(aspect<=args.max_aspect)&(red_fraction<.05)
        dense[0]=False
        blob_stain_ids=np.zeros(n,bool); blob_hold_ids=np.zeros(n,bool); components=[]
        for index in np.flatnonzero(dense):
            x,y,cw,ch,area=map(int,stats[index]); pad=16
            x0,y0,x1,y1=max(0,x-pad),max(0,y-pad),min(w,x+cw+pad),min(h,y+ch+pad)
            ring=(~foreground[y0:y1,x0:x1])&(~red[y0:y1,x0:x1])
            ring[max(0,y-y0):y+ch-y0,max(0,x-x0):x+cw-x0]=False
            paper=np.median(lab[y0:y1,x0:x1][ring],axis=0) if ring.sum()>=20 else None
            brown_evidence=bool(paper is not None and means[1][index]-paper[1]>=1.
                                and means[2][index]-paper[2]>=-1.
                                and means[1][index]>=2. and means[2][index]>=8.)
            blob_stain_ids[index]=brown_evidence; blob_hold_ids[index]=not brown_evidence
            components.append(dict(id=int(index),xywh=[x,y,cw,ch],area=area,core_fraction=float(cores[index]),
                                   fill=float(fill[index]),aspect=float(aspect[index]),
                                   mean_lab=[float(channel[index]) for channel in means],
                                   paper_lab=paper.tolist() if paper is not None else None,
                                   disposition='stain_candidate' if brown_evidence else 'hold'))
        blob_stain=blob_stain_ids[labels]&~red
        blob_hold=blob_hold_ids[labels]&~red
        del labels,distance
        masks_by_mode={}
        for mode in modes:
            if mode=='local_baseline':
                masks={key:np.zeros((h,w),bool) for key in colors}; masks['black']=foreground.copy()
            else:
                rejected=(blob_stain|blob_hold) if mode=='color_shape' else np.zeros((h,w),bool)
                black=foreground&~red&~rejected
                stain=(brown&~foreground&~red)|(blob_stain if mode=='color_shape' else False)
                hold=((((votes>0)&(votes<3))| (weak&~red))&~black&~red&~stain)
                if mode=='color_shape': hold|=blob_hold
                masks=dict(black=black,red=red,stain=stain,hold=hold)
            count=sum(mask.astype(np.uint8) for mask in masks.values())
            assert int(count.max())<=1
            masks_by_mode[mode]=masks
            ink=masks['black']|masks['red']
            for key,mask in {**masks,'ink':ink}.items(): write(target/mode/f'{key}.png',mask.astype(np.uint8)*255)
            write(target/mode/'classes.png',paint(original,masks,flat=True))
            for selected in ('all','ink',*colors): write(target/mode/f'overlay_{selected}.png',paint(original,masks,selected))
            metrics.append(dict(page=stem,mode=mode,**{f'{key}_pixels':int(mask.sum()) for key,mask in masks.items()},
                                added_ink_pixels=int((ink&~foreground).sum()),removed_ink_pixels=int((foreground&~ink).sum())))
        write(target/'original.png',original)
        write(target/'red_seeds.png',seeds.astype(np.uint8)*255)
        write(target/'red_weak.png',weak.astype(np.uint8)*255)
        write(target/'brown_field.png',brown.astype(np.uint8)*255)
        write(target/'dense_components.png',(blob_stain|blob_hold).astype(np.uint8)*255)
        (target/'components.json').write_text(json.dumps(components,ensure_ascii=False,indent=2)+'\n')
        proposal=masks_by_mode['color_shape']; ink=proposal['black']|proposal['red']
        delta=np.full_like(original,(239,235,231)); delta[ink&~foreground]=(180,165,0); delta[foreground&~ink]=(25,135,245)
        write(target/'difference_ink.png',delta)
        images=[original,paint(original,masks_by_mode['local_baseline']),
                paint(original,masks_by_mode['color_only'],flat=True),paint(original,proposal,flat=True),
                delta,paint(original,proposal)]
        titles=['原画像','前回の局所Sauvola：前景','色を追加：形の判定なし','色＋形＋保留：分類候補',
                '前景差分：青緑＝追加／橙＝除外','原画像に分類候補を重ねる']
        figure(images,titles,target/'overview.png',stem+'　青＝墨候補／赤＝朱候補／黄＝汚れ候補／紫＝保留')
        page=dict(id=stem,title=case['title'],width=w,height=h,rois=[],components=[])
        if stem=='200018243_00003_2': page['title']='本文・裏写り候補・欄外の混在'
        for i,roi in enumerate(rois[stem]):
            x,y,rw,rh=roi['xywh']; slices=(slice(y,y+rh),slice(x,x+rw)); name=f'roi_{i+1:02d}.png'
            figure([value[slices] for value in images],titles,target/name,roi['title'])
            page['rois'].append(dict(title=roi['title'],xywh=roi['xywh'],image=f'{stem}/{name}'))
            for mode,masks in masks_by_mode.items():
                m_ink=(masks['black']|masks['red'])[slices]; base=foreground[slices]
                metrics.append(dict(page=stem,roi=i+1,mode=mode,
                                    **{f'{key}_pixels':int(mask[slices].sum()) for key,mask in masks.items()},
                                    added_ink_pixels=int((m_ink&~base).sum()),removed_ink_pixels=int((base&~m_ink).sum())))
        for i,component in enumerate(components):
            x,y,cw,ch=component['xywh']; pad=35
            x0,y0,x1,y1=max(0,x-pad),max(0,y-pad),min(w,x+cw+pad),min(h,y+ch+pad)
            slices=(slice(y0,y1),slice(x0,x1)); name=f'component_{i+1:02d}.png'
            figure([value[slices] for value in images],titles,target/name,
                   f"塊候補 {component['id']}　{component['disposition']}（正解ではない）")
            page['components'].append(dict(title=f"塊候補{i+1}：{component['disposition']}",
                                          xywh=[x0,y0,x1-x0,y1-y0],image=f'{stem}/{name}'))
        gallery.append(page)
        print(f'{stem}: red={int(red.sum())}, dense blobs={len(components)}, '
              f'stain blobs={int(blob_stain_ids.sum())}, hold blobs={int(blob_hold_ids.sum())}, '
              f'{time.monotonic()-started:.1f}s',flush=True)
    (out/'diagnostics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    (out/'index.html').write_text(template.read_text().replace('__DATA__',json.dumps(dict(cases=gallery,modes=modes),ensure_ascii=False)))
    outputs={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*.png'))}
    (out/'complete.json').write_text(json.dumps(dict(status='complete',cases=len(gallery),
        rois=sum(len(p['rois']) for p in gallery),outputs=outputs),ensure_ascii=False,indent=2)+'\n')
    print('Color/shape comparison completed; no original pixels deleted.',flush=True)


if __name__=='__main__':
    main()
