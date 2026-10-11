"""Audit annotation placement and diagnose Su window estimation without replacing the initial comparison."""
from pathlib import Path
import json, time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from compare import ROOT,HERE,FONT,read,gray,su,binary,save,dump,sha,sheet,LABELS

VALID = {
 'N01':[0,1,2,3,4,5,6,7,8,9,10,11,12,18],
 'T01':[0,1,3,5,6,7,8,10,12,13,14,15,16,17,18,19,20,23,24,25,26,27,29],
 'N05':[0,1,2,4,5,6,7,10,11,13,16,18,19,22,23],
}
FOCUS = {'N01':[30,25,55,100],'T01':[15,285,120,130],'N05':[155,155,100,105]}

def main():
    cv2.setNumThreads(2)
    protocol=json.loads((HERE/'protocol.json').read_text())
    original_ann=json.loads((HERE/'annotations_frozen.json').read_text())
    ann={c['id']:c for c in original_ann['cases']}
    audit=dict(original_annotations_sha256=sha(HERE/'annotations_frozen.json'),
        reason='Original patch audit found manually placed points outside visible ink. Exclude these; do not move them onto a method output.',
        audit_after_first_outputs=True,
        claim_limit='Exploratory, audited selected ink-center retention. Not independent accuracy or pixel precision. No method parameters selected using point scores.',
        original_only_audit_images='originals/*/point_audit.png',
        valid_original_point_indices=VALID,
        excluded_original_point_indices={cid:[i for i in range(len(ann[cid]['ink_points_xy'])) if i not in vv] for cid,vv in VALID.items()})
    dump(HERE/'annotation_audit.json',audit)
    # Register both widths before computing them. These are diagnostic extensions,
    # not retuned versions of the frozen main method or an independent test.
    dump(HERE/'su_width_diagnostic_protocol.json',dict(
        reason='Literal manuscript Algorithm1 reconstruction estimates inter-stroke/column distances (30/47/137 px), too large for thin native strokes.',
        added_after_main_outputs=True,windows=[19,61],same_for_all_pages=True,
        changes='Only local threshold window replaced; same contrast, Canny, threshold formula, postprocessing.',
        status='Exploratory diagnostic; not author Su algorithm parity. Original automatic results preserved.'))
    LABELS['su_fixed19']='Su式 / 窓19固定（診断）';LABELS['su_fixed61']='Su式 / 窓61固定（診断）'
    panels=[];diagnostics=[]
    for c in protocol['cases']:
        cid=c['id'];raw=read(ROOT/c['original']);g=gray(raw)
        x,y,w,h=c['roi_xywh'];sl=np.s_[y:y+h,x:x+w]
        for width in [19,61]:
            k=f'su_fixed{width}';start=time.monotonic();b,_,meta=su(g,width)
            save(HERE/'masks'/cid/(k+'.png'),binary(b));save(HERE/'crops'/cid/(k+'.png'),binary(b[sl]))
            diagnostics.append(dict(id=cid,method=k,seconds=time.monotonic()-start,details=meta))
        keys=['original','sauvola61','su2013_local','su_fixed19','su_fixed61','gatos_initial']
        crops={k:np.asarray(Image.open(HERE/'crops'/cid/(k+'.png')).convert('L'))<127 for k in keys if k!='original'}
        panels.append(sheet(cid,raw[sl],crops,keys,HERE/'figures'/(cid+'_Su幅推定の診断.png'),cid+' Su再実装の幅推定診断'))
    records=[]
    for c in protocol['cases']:
        cid=c['id'];selected=[ann[cid]['ink_points_xy'][i] for i in VALID[cid]]
        x,y,w,h=c['roi_xywh'];negative=np.zeros((h,w),bool)
        for xx,yy,ww,hh in ann[cid]['nonink_regions_xywh']:negative[yy:yy+hh,xx:xx+ww]=True
        for p in sorted((HERE/'crops'/cid).glob('*.png')):
            b=np.asarray(Image.open(p).convert('L'))<127
            hits=[bool(b[yy,xx]) for xx,yy in selected]
            records.append(dict(id=cid,method=p.stem,label=LABELS[p.stem],
                audited_ink_points_kept=sum(hits),audited_ink_points_total=len(hits),
                audited_ink_point_hits=hits,
                nonink_region_false_foreground=int(b[negative].sum()),nonink_region_pixels=int(negative.sum()),
                nonink_region_fraction=float(b[negative].mean()),roi_foreground_pixels=int(b.sum()),crop_sha256=sha(p)))
        # Audit annotation diagram retained alongside original frozen diagram.
        a=read(HERE/'originals'/cid/'original.png');im=Image.fromarray(a).resize((w*3,h*3),Image.Resampling.NEAREST);d=ImageDraw.Draw(im)
        for xx,yy in selected:d.ellipse((xx*3-5,yy*3-5,xx*3+5,yy*3+5),outline='#006bdb',width=2)
        for xx,yy,ww,hh in ann[cid]['nonink_regions_xywh']:d.rectangle((xx*3,yy*3,(xx+ww)*3,(yy+hh)*3),outline='#00a050',width=3)
        im.save(HERE/'originals'/cid/'audited_annotations.png')
        xx,yy,ww,hh=FOCUS[cid];fs=np.s_[yy:yy+hh,xx:xx+ww]
        keys=['original','sauvola61','wolf61','nick61','gatos_initial','gatos_native','su2013_local','su_fixed19','su_fixed61']
        scale=4;pw,ph=ww*scale,hh*scale;gap=20;top=95;row=ph+55
        sheetimg=Image.new('RGB',(3*(pw+gap)+gap,top+3*row+65),'#faf9f5');d=ImageDraw.Draw(sheetimg)
        f=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,17)
        d.text((15,12),cid+' 筆画の拡大：原寸の整数4倍',font=f,fill='#20313b')
        d.text((15,48),'同じ範囲。黒＝候補。原画像の細い線と連続性を比べる。',font=small,fill='#20313b')
        for i,k in enumerate(keys):
            tx=gap+i%3*(pw+gap);ty=top+i//3*row
            d.text((tx,ty),LABELS[k],font=small,fill='#20313b')
            a=read(HERE/'originals'/cid/'original.png') if k=='original' else read(HERE/'crops'/cid/(k+'.png'))
            tile=Image.fromarray(a[fs]).resize((pw,ph),Image.Resampling.NEAREST)
            sheetimg.paste(tile,(tx,ty+30))
        d.text((15,sheetimg.height-52),'Su窓固定は幅推定を切り分ける追加診断。Gatos初期＝後処理前。',font=small,fill='#20313b')
        d.text((15,sheetimg.height-27),'濃淡・朱・修復結果を評価する図ではない。',font=small,fill='#20313b')
        sheetimg.save(HERE/'figures'/(cid+'_細線拡大.png'))
    dump(HERE/'audited_measurements.json',dict(audit=audit,records=records))
    dump(HERE/'su_width_diagnostic_complete.json',dict(results=diagnostics,panels=panels))
    print('\n'.join(f"{r['id']} {r['method']} ink {r['audited_ink_points_kept']}/{r['audited_ink_points_total']} nonink {r['nonink_region_fraction']:.2%}" for r in records if r['method'] in ['sauvola61','wolf61','nick61','gatos_initial','gatos_native','su2013_local','su_fixed19','su_fixed61','nick19']))

if __name__=='__main__':main()
