"""Native source-only thin-text candidates from training books, no predictions."""
import json,hashlib
from pathlib import Path
import cv2,numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/mixed_paper_ink_control_20261022'
PARENT=ROOT/'tmp_work/background_diversity_control_20261022'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def screen():
    assert not (OUT/'source_candidates.json').exists()
    bg=json.loads((PARENT/'background_candidates.json').read_text())['candidates'];books=sorted({r['book'] for r in bg if r['arm']=='multibook'});used={r['source'] for r in bg}
    rng=np.random.default_rng(20261024);rows=[];audit=[]
    for book in books:
        pool=[p for p in sorted((ROOT/'data/split_dataset_jogai'/book).glob('*.jpg')) if str(p.relative_to(ROOT)) not in used]
        indices=sorted(rng.choice(len(pool),2,replace=False))
        for page_index in indices:
            source=pool[int(page_index)];image=np.asarray(Image.open(source).convert('RGB'));gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);h,w=gray.shape;options=[]
            for y in range(round(h*.12),round(h*.84)-255,128):
              for x in range(round(w*.12),round(w*.88)-255,128):
                a=gray[y:y+256,x:x+256];paper=float(np.percentile(a,85));dark=a<paper-35;density=float(dark.mean())
                if density<.025 or density>.13:continue
                distance=ndi.distance_transform_edt(dark);thin=float((distance[dark]<=1.5).mean())
                contrast=paper-float(a[dark].mean());score=thin*40-abs(density-.06)*80+min(contrast,130)*.025
                options.append(dict(xy=[x,y],score=score,dark_density=density,source_thin_fraction=thin,paper85=paper,contrast=contrast))
            options.sort(key=lambda c:(-c['score'],c['xy']));assert options,source
            q=options[0];x,y=q['xy'];key=f'C{len(rows)+1:02d}';path=OUT/'candidates'/f'{key}_original.png';path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(image[y:y+256,x:x+256]).save(path)
            rows.append(dict(id=key,book=book,page=source.stem,source=str(source.relative_to(ROOT)),source_sha256=sha(source),roi=[x,y,256,256],crop=str(path.relative_to(ROOT)),crop_sha256=sha(path),**q))
            audit.append(dict(source=str(source.relative_to(ROOT)),candidate_count=len(options),top5=options[:5]))
    dump(OUT/'source_candidates.json',dict(candidates=rows,seed=20261024,selection='2 randomly selected new training pages/book; rank native windows by source contrast and thin dark structure. No method outputs or GT used. Not all dark structures are text.',source_audit=audit))
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15);canvas=Image.new('RGB',(4*270,4*288),'#faf9f5');d=ImageDraw.Draw(canvas)
    for i,r in enumerate(rows):
        x=i%4*270;y=i//4*288;d.text((x,y),r['id']+' '+r['page'],font=font,fill='black');canvas.paste(Image.open(ROOT/r['crop']).convert('RGB'),(x,y+28))
    canvas.save(OUT/'原画像候補一覧.png');print('Prepared16 source-only native256 candidates',flush=True)
if __name__=='__main__':screen()
