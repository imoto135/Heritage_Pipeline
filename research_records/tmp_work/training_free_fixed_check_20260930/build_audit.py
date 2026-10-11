"""Add inspection artifacts after the frozen algorithm completes; no mask edits."""
from pathlib import Path
import hashlib
import json
import shutil

import cv2
import numpy as np

OUT=Path(__file__).resolve().parent
cv2.setNumThreads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tile(image,width,height):
    h,w=image.shape[:2]
    s=min(1,width/w,height/h)
    small=cv2.resize(image,(max(1,round(w*s)),max(1,round(h*s))),interpolation=cv2.INTER_NEAREST)
    canvas=np.full((height,width,3),245,np.uint8)
    y=(height-small.shape[0])//2;x=(width-small.shape[1])//2
    canvas[y:y+small.shape[0],x:x+small.shape[1]]=small
    return canvas


selection=json.loads((OUT/'input_review/manifest.json').read_text())
rows=[]
for case in selection['cases']:
    folder=OUT/case['id']
    original=cv2.imread(str(folder/'original.png'))
    classes=cv2.imread(str(folder/'color_shape/classes.png'))
    red=cv2.imread(str(folder/'color_shape/red.png'),0)>127
    base=cv2.imread(str(folder/'local_baseline/ink.png'),0)>127
    ink=cv2.imread(str(folder/'color_shape/ink.png'),0)>127
    h,w=red.shape
    x0,y0,x1,y1=round(.15*w),round(.2*h),round(.85*w),round(.85*h)
    central=np.zeros((h,w),bool);central[y0:y1,x0:x1]=True
    n,labels,stats,_=cv2.connectedComponentsWithStats(red.astype(np.uint8),connectivity=8)
    center_counts=np.bincount(labels.ravel(),weights=central.ravel(),minlength=n)
    all_ids=sorted(range(1,n),key=lambda i:int(stats[i,4]),reverse=True)
    central_ids=[i for i in all_ids if center_counts[i]>0]
    # Show largest red component within the central region too, so margin noise
    # cannot hide small internal false candidates in the visual audit.
    chosen=list(dict.fromkeys(all_ids[:4]+central_ids[:4]))
    audit=[]
    if chosen:
        sheet=np.full((len(chosen)*230,1000,3),245,np.uint8)
        for k,i in enumerate(chosen):
            x,y,cw,ch,area=map(int,stats[i]);pad=35
            ax,ay,bx,by=max(0,x-pad),max(0,y-pad),min(w,x+cw+pad),min(h,y+ch+pad)
            slices=(slice(ay,by),slice(ax,bx))
            text=f"red component {i}: {area} px, x={x} y={y} w={cw} h={ch}"
            cv2.putText(sheet,text,(10,k*230+20),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
            sheet[k*230+30:(k+1)*230,:500]=tile(original[slices],500,200)
            sheet[k*230+30:(k+1)*230,500:]=tile(classes[slices],500,200)
            audit.append(dict(id=i,xywh=[x,y,cw,ch],area=area,central_pixels=int(center_counts[i])))
        assert cv2.imwrite(str(folder/'red_audit.png'),sheet)
    # Original/ink overlay central crop: native-size view of fine text/illustration.
    overlay=cv2.imread(str(folder/'color_shape/overlay_ink.png'))
    central_sheet=np.concatenate([tile(original[y0:y1,x0:x1],650,950),
                                  tile(overlay[y0:y1,x0:x1],650,950)],axis=1)
    assert cv2.imwrite(str(folder/'central_ink_audit.png'),central_sheet)
    rows.append(dict(page=case['id'],central_xywh=[x0,y0,x1-x0,y1-y0],
                     red_pixels=int(red.sum()),central_red_pixels=int((red&central).sum()),
                     outside_central_red_pixels=int((red&~central).sum()),
                     added_ink_pixels=int((ink&~base).sum()),removed_ink_pixels=int((base&~ink).sum()),
                     red_components=n-1,inspected_red_components=audit,
                     note='Central rectangle is a location diagnostic, not a paper mask or ground truth.'))
(OUT/'audit_diagnostics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
# Keep frozen renderer output, then change only descriptive text for this run.
raw=OUT/'index_frozen_template.html'
if not raw.exists():
    shutil.copyfile(OUT/'index.html',raw)
replacements={
    '学習なし：色・形と保留の比較':'学習なし：固定条件で別の8冊を確認',
    '既知の4ページでの開発試験です。':'調整に使った4冊とは別の8冊に、同じ条件を固定して適用した確認です。',
    '前回の局所Sauvola':'同じ局所Sauvola',
    '所見と限界':'固定条件の確認結果と限界'}
html=raw.read_text()
for before,after in replacements.items():
    assert before in html,before
    html=html.replace(before,after)
(OUT/'index.html').write_text(html)
(OUT/'viewer_derivation.json').write_text(json.dumps(dict(
    frozen_renderer_index_sha256=sha(raw),display_index_sha256=sha(OUT/'index.html'),
    substitutions=replacements,postprocessor_sha256=sha(Path(__file__))),ensure_ascii=False,indent=2)+'\n')
for row in rows:
    print(row['page'],'red',row['red_pixels'],'central red',row['central_red_pixels'],
          'added',row['added_ink_pixels'],'removed',row['removed_ink_pixels'])
