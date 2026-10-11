"""Paired fixed local inpainting probe for saved npj and frozen v6 edit masks.

This measures mask consequences with one fixed operator, not DDRM reproduction
or reference-image accuracy. Native pixels, 64px context, Telea radius 3px.
"""
import csv
import json
import os
from pathlib import Path
import textwrap
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/npj_v6_edit_effect_probe_20261001'
os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from build_npj_v6_comparison import FONT_PATH,sha,dump,read,write
from collect_saved_npj_failures import saved_masks

cv2.setNumThreads(1)
SOURCES=[ROOT/'tmp_work/npj_failure_collection_20261001',ROOT/'tmp_work/npj_failure_followup_20261001']
SETTINGS=dict(operator='cv2.INPAINT_TELEA',radius_native_px=3,context_native_px=64,
              mask_dilation=False,resize_before_processing=False,outside_edit='exact original pixels',
              baseline_mask='Saved npj v1 04_degradation.png > 0',v6_mask='Frozen v6 edit.png > 0',
              full_npj_or_DDRM_reproduction=False,pixel_GT=False,restoration_accuracy=False)


def comparison_png(images,labels,path,title,note):
    assert len(images)==3 and len({im.shape for im in images})==1
    h,w=images[0].shape[:2];scale=min(620/w,1200/h);dw,dh=round(w*scale),round(h*scale)
    margin=28;gap=24;width=3*dw+2*margin+2*gap
    canvas=Image.new('RGB',(width,dh+254),(250,248,243));draw=ImageDraw.Draw(canvas)
    large=ImageFont.truetype(FONT_PATH,26);small=ImageFont.truetype(FONT_PATH,21)
    draw.text((margin,16),title,font=large,fill='black')
    draw.text((margin,55),'同じ原画像・同じ処理・同じ領域で、編集マスクだけを変更',font=small,fill=(50,50,50))
    for i,(im,label) in enumerate(zip(images,labels)):
        x=margin+i*(dw+gap);draw.text((x,96),label,font=large,fill='black')
        thumb=cv2.resize(im,(dw,dh),interpolation=cv2.INTER_NEAREST)
        canvas.paste(Image.fromarray(cv2.cvtColor(thumb,cv2.COLOR_BGR2RGB)),(x,136))
    draw.text((margin,dh+154),'固定局所補間（Telea半径3px）。DDRMの復元結果ではありません。',font=small,fill=(50,50,50))
    draw.text((margin,dh+188),note,font=small,fill=(50,50,50))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def stack(paths,labels,path,title):
    images=[Image.open(p).convert('RGB') for p in paths]
    width=max(im.width for im in images);height=90+sum(im.height+82 for im in images)
    canvas=Image.new('RGB',(width,height),(250,248,243));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT_PATH,27);small=ImageFont.truetype(FONT_PATH,23)
    draw.text((28,16),title,font=font,fill='black');draw.text((28,54),'原画像 / npj編集マスクで補間 / v6編集マスクで補間',font=small,fill=(50,50,50))
    y=90
    for im,label in zip(images,labels):
        canvas.paste(im,(0,y));y+=im.height;draw.text((28,y+18),label,font=small,fill='black');y+=82
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def run():
    OUT.mkdir(parents=True,exist_ok=True);dependencies={};rows=[];specs=[]
    for source in SOURCES:
        for name in ['baseline_review.json','comparison_manifest.json','comparison_complete.json']:
            dependencies[str((source/name).relative_to(ROOT))]=sha(source/name)
        complete=json.loads((source/'comparison_complete.json').read_text())['outputs']
        manifest=json.loads((source/'comparison_manifest.json').read_text())
        for rel,digest in manifest['dependencies'].items():assert sha(ROOT/rel)==digest,rel
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            specs.extend(dict(page=page,roi=roi,source=source) for roi in page['rois'])
            original=read(ROOT/page['original']);baseline=saved_masks(page)['edit']
            v6path=source/'v6_masks'/page['page']/'edit.png'
            assert sha(v6path)==complete[str(v6path.relative_to(source))]
            v6edit=read(v6path,True)
            for path in [ROOT/page['original'],ROOT/page['npj_degradation'],v6path]:
                dependencies[str(path.relative_to(ROOT))]=sha(path)
            assert baseline.shape==v6edit.shape==original.shape[:2]
            for roi in page['rois']:
                id=roi['id'];x,y,w,h=roi['xywh'];pad=SETTINGS['context_native_px'];H,W=original.shape[:2]
                x0,y0=max(0,x-pad),max(0,y-pad);x1,y1=min(W,x+w+pad),min(H,y+h+pad)
                context=original[y0:y1,x0:x1].copy();crop_slice=(slice(y-y0,y-y0+h),slice(x-x0,x-x0+w))
                targets={};metrics={};dest=OUT/'native'/id
                write(dest/'original.png',original[y:y+h,x:x+w])
                for method,fullmask in [('npj',baseline),('v6',v6edit)]:
                    mask=fullmask[y0:y1,x0:x1]
                    restored=cv2.inpaint(context,mask.astype(np.uint8)*255,SETTINGS['radius_native_px'],cv2.INPAINT_TELEA) if mask.any() else context.copy()
                    # Guard against any operator changes outside the authorized mask.
                    restored[~mask]=context[~mask]
                    assert np.array_equal(restored[~mask],context[~mask])
                    result=restored[crop_slice];edit=mask[crop_slice];input_crop=context[crop_slice]
                    targets[method]=result
                    changed=np.any(result!=input_crop,axis=2)
                    assert not (changed&~edit).any()
                    delta=np.abs(result.astype(np.int16)-input_crop.astype(np.int16)).mean(axis=2)
                    metrics[method]=dict(edit_pixels=int(edit.sum()),changed_pixels=int(changed.sum()),
                        mean_absolute_change_in_edit=float(delta[edit].mean()) if edit.any() else 0.,
                        nonedit_changed_pixels=0)
                    for filename,value in [(method+'.png',result),(method+'_edit.png',edit),
                                           (method+'_changed.png',changed)]:
                        path=dest/filename;write(path,value);assert np.array_equal(read(path,value.dtype==bool),value)
                title=roi['title']+' · '+page['page']
                note=f'矩形内の変更画素：npj {metrics["npj"]["changed_pixels"]:,} / v6 {metrics["v6"]["changed_pixels"]:,}（精度・除去率ではありません）'
                comparison_png([original[y:y+h,x:x+w],targets['npj'],targets['v6']],['原画像','npjマスクで補間','v6マスクで補間'],
                               OUT/'comparisons'/(id+'.png'),title,note)
                rows.append(dict(id=id,page=page['page'],book=page['book'],xywh=roi['xywh'],
                    baseline_failure=roi['baseline_failure'],failure_type=roi['failure_type'],
                    evaluation_group=page['evaluation_group'],context_xywh=[x0,y0,x1-x0,y1-y0],
                    **{method+'_'+key:value for method,m in metrics.items() for key,value in m.items()}))
                print('Fixed edit-effect probe:',page['page'],id,flush=True)
    dump(OUT/'manifest.json',dict(source_sha256=sha(Path(__file__)),settings=SETTINGS,dependencies=dependencies,
        pages=len({r['page'] for r in rows}),regions=len(rows),selection='All frozen ROIs from both mask reviews, no outcome selection'))
    dump(OUT/'diagnostics.json',rows)
    with (OUT/'diagnostics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    GROUPS={'01_stain_edit_effect.png':['dense_stain','brown_as_red','brown_false_red_new'],
            '02_stroke_edit_effect.png':['thin_black_loop','warm_stroke','faint_left_text'],
            '03_remaining_edit_effect.png':['dense_stain_new','hole_edge_new','brown_near_text','damage_control'],
            '04_control_edit_effect.png':['thin_black_text','control_black','control_text_new','dark_text_control']}
    labels={s['roi']['id']:s['roi']['title'] for s in specs}
    for name,ids in GROUPS.items():stack([OUT/'comparisons'/(id+'.png') for id in ids],
        [labels[id] for id in ids],OUT/name,'編集マスクが見た目に与える影響：固定局所補間')
    archive=OUT/'npj_v6_edit_effect_images.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(OUT.glob('*.png')):z.write(p,'overview/'+p.name)
        for p in sorted((OUT/'comparisons').glob('*.png')):z.write(p,'individual/'+p.name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len(z.namelist())==19
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',pages=11,regions=15,paired_operator_and_context=True,
        native_pixels=True,source_and_mask_hashes_checked=True,nonedit_pixels_unchanged=True,
        written_native_pixels_equal=True,zip_integrity=True,DDRM=False,pixel_GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*.png')},archive_sha256=sha(archive)))
    print('Probe completed: 11 pages / 15 ROIs / 19 comparison PNGs including 4 overview sheets',flush=True)


if __name__=='__main__':run()
