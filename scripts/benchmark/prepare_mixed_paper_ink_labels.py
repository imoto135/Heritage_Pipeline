"""Source-only review and conservative support filter; never repair missing ink."""
import json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
from compare_v5_aligned_fade_control import rgb,dump,sha
import evaluate_classical_partial_labels as partial
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'tmp_work/mixed_paper_ink_control_20261022'
def labels(case,annotation):
    raw=rgb(ROOT/case['crop']);manual,paths=partial.raster(case,annotation)
    gray=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY).astype(np.float32);paper=ndi.maximum_filter(gray,25)
    # Keep only source-supported traced points. Never join gaps or move RGB pixels.
    supported=(paper-gray)>=12
    result=manual.copy();result[(manual==1)&~supported]=0
    return result,manual
def prepare():
    assert not (OUT/'protocol.json').exists()
    cases=json.loads((OUT/'selected_sources.json').read_text())['cases'];ann={a['id']:a for a in json.loads((OUT/'annotations_source_only.json').read_text())['regions']};counts=[]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    for j in range(2):
        canvas=Image.new('RGB',(4*268,2*290),'#faf9f5');d=ImageDraw.Draw(canvas)
        for i,c in enumerate(cases[j*4:j*4+4]):
            lab,manual=labels(c,ann[c['id']]);raw=rgb(ROOT/c['crop']);over=raw.copy();over[lab==1]=[0,100,255];over[lab==2]=np.rint(.45*raw[lab==2]+.55*np.array([0,200,80])).astype(np.uint8);over[(manual==1)&(lab==0)]=[255,150,0]
            target=OUT/'labels'/f"{c['id']}.png";target.parent.mkdir(exist_ok=True);Image.fromarray(lab).save(target)
            for r,im in enumerate([raw,over]):
                x=i*268;y=r*290;d.text((x,y),c['id']+(' 原本' if r==0 else ' 青=墨 緑=紙 橙=除外'),font=font,fill='black');canvas.paste(Image.fromarray(im),(x,y+28))
            counts.append(dict(id=c['id'],ink_n=int((lab==1).sum()),paper_n=int((lab==2).sum()),unsupported_trace_points=int(((manual==1)&(lab==0)).sum()),label_sha256=sha(target)))
        canvas.save(OUT/f'最終注釈確認_{j:02d}.png')
    dump(OUT/'label_preparation.json',dict(counts=counts,source_only=True,source_confidence='Manual traced foreground, then exclude points with source gray deficit<12 relative to max-filter25 neighborhood. Fixed before any model output; no gap filling or RGB change. Paper polygons eroded1px by inherited rasterizer.',labels={'0':'unknown excluded','1':'source-supported provisional ink trace','2':'selected clear paper interior'},expert_GT=False))
    print('Prepared',sum(r['ink_n'] for r in counts),'ink points;',sum(r['paper_n'] for r in counts),'paper pixels;',sum(r['unsupported_trace_points'] for r in counts),'unsupported trace points excluded',flush=True)
if __name__=='__main__':prepare()
