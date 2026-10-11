"""Input-only screening sheets; never rank restoration or mask outputs."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PRIOR=ROOT/'tmp_work/red_positive_collection_20260930'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def main():
    old=json.loads((PRIOR/'selection.json').read_text())
    used={c['book'] for c in old['cases']}
    candidates=[r for r in json.loads((PRIOR/'candidate_pages.json').read_text()) if r['book'] not in used]
    target=OUT/'input_screen'
    target.mkdir(parents=True,exist_ok=True)
    queue=[]
    for i,c in enumerate(candidates):
        p=ROOT/c['original']
        queue.append(dict(queue_index=i,**c,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            previous_history='Input thumbnails were viewed in September collection; no claim of globally unseen data.',
            role='candidate only; not selected, annotated or used for model fitting'))
    (target/'queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2))
    for offset in range(0,len(queue),16):
        sheet=Image.new('RGB',(4*460,4*670),'#faf9f5')
        draw=ImageDraw.Draw(sheet)
        for j,c in enumerate(queue[offset:offset+16]):
            a=cv2.imread(str(ROOT/c['original']),cv2.IMREAD_REDUCED_COLOR_2)
            assert a is not None
            im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB))
            im.thumbnail((440,600),Image.Resampling.LANCZOS)
            xx=(j%4)*460+10; yy=(j//4)*670+10
            draw.text((xx,yy),str(c['queue_index'])+' '+c['id'],font=ImageFont.truetype(FONT,16),fill='#20313b')
            draw.text((xx,yy+24),c['book']+'／原画像のみ',font=ImageFont.truetype(FONT,16),fill='#20313b')
            sheet.paste(im,(xx,yy+52))
        sheet.save(target/f'候補_{offset//16+1:02d}.png')
    # Preserve existing input-only crops with coordinates and hashes as development.
    dev=[]
    for c in old['cases']:
        assert hashlib.sha256((ROOT/c['original']).read_bytes()).hexdigest()==c['original_sha256']
        a=np.asarray(Image.open(ROOT/c['original']).convert('RGB'))
        for r in c['rois']:
            x,y,w,h=r['xywh']
            pp=OUT/'development_inputs'/c['id']/(r['id']+'.png')
            pp.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(a[y:y+h,x:x+w]).save(pp)
            dev.append(dict(page=c['id'],book=c['book'],original=c['original'],source_sha256=c['original_sha256'],
                ROI=r['xywh'],name=r['name'],crop=str(pp.relative_to(OUT)),
                known_prior_method_outputs=True,role='development/regression; not independent validation',pixel_GT=False))
    (OUT/'development_inputs.json').write_text(json.dumps(dev,ensure_ascii=False,indent=2))
    with (OUT/'activity.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),event='input_sheets_prepared',
            candidate_pages=len(queue),candidate_books=len({c['book'] for c in queue}),development_ROIs=len(dev)),ensure_ascii=False)+'\n')
    print('Input-only candidates:',len(queue),'pages;',len({c['book'] for c in queue}),'books. Development crops:',len(dev))

if __name__=='__main__':main()
