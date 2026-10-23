"""Original-only candidate paper windows; manually review central128 before use."""
import json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from compare_v5_aligned_fade_control import dump,sha,rgb
import run_ragan_full_continuation as full
OUT=ROOT/'tmp_work/background_diversity_control_20261022'
BOOKS=['100249376','100249476','100249537','200008316','200014685','200015779','200021802','200021853']
def candidates(path):
    im=rgb(path);h,w=im.shape[:2];g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    result=[]
    # Fixed page-wide32-pixel grid; require bright, low-edge core, never prediction masks.
    for y in range(0,h-255,64):
      for x in range(0,w-255,64):
        a=g[y+64:y+192,x+64:x+192];q5,q95=np.percentile(a,[5,95]);edge=np.hypot(cv2.Sobel(a,cv2.CV_32F,1,0),cv2.Sobel(a,cv2.CV_32F,0,1))
        fraction=float((edge>45).mean());sd=float(a.std())
        if q5<145 or sd>14 or fraction>.035:continue
        # Prefer blank paper, retaining natural coloration and fine texture.
        score=sd+180*fraction+max(0,180-q5)*.2
        result.append(dict(xy=[x,y],score=score,gray_mean=float(a.mean()),gray_std=sd,edge_fraction=fraction,q5=float(q5),q95=float(q95)))
    return sorted(result,key=lambda r:(r['score'],r['xy']))
def screen():
    assert not (OUT/'background_candidates.json').exists();OUT.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261023);rows=full.read_rows('train');groups=sorted({r['group'] for r in rows});selected=[groups[int(i)] for i in sorted(rng.choice(len(groups),8,replace=False))]
    source_sets={}
    source_sets['samebook']=[sorted((ROOT/'data/makura_kiridashi').glob(group+'_*.png')) for group in selected]
    source_sets['multibook']=[]
    for book in BOOKS:
        pool=sorted((ROOT/'data/split_dataset_jogai'/book).glob('*.jpg'));chosen=rng.choice(len(pool),3,replace=False)
        source_sets['multibook'].append([pool[int(k)] for k in chosen])
    chosen=[];survey=[]
    for arm,pools in source_sets.items():
      for j,pool in enumerate(pools):
        options=[]
        for source in pool:
            cc=candidates(source);survey.append(dict(arm=arm,group=j,source=str(source.relative_to(ROOT)),source_sha256=sha(source),candidate_count=len(cc),top_candidates=cc[:5]))
            options.extend(dict(source=source,**c) for c in cc[:24])
        options.sort(key=lambda r:(r['score'],str(r['source']),r['xy']));accepted=[]
        for opt in options:
            if any(opt['source']==q['source'] and np.linalg.norm(np.array(opt['xy'])-q['xy'])<256 for q in accepted):continue
            accepted.append(opt)
            if len(accepted)==2:break
        assert len(accepted)==2,(arm,j)
        for k,c in enumerate(accepted):
            source=c.pop('source');x,y=c.pop('xy');key=f'{arm}_{j:02d}_{k}';im=rgb(source)[y:y+256,x:x+256]
            target=OUT/'backgrounds'/f'{key}.png';target.parent.mkdir(exist_ok=True);Image.fromarray(im).save(target)
            chosen.append(dict(id=key,arm=arm,book='200017197' if arm=='samebook' else source.parent.name,source=str(source.relative_to(ROOT)),source_sha256=sha(source),roi=[x,y,256,256],paper_box_xywh=[64,64,128,128],crop=str(target.relative_to(ROOT)),crop_sha256=sha(target),**c))
    dump(OUT/'source_screening.json',dict(original_only=True,seed=20261023,rows=survey))
    dump(OUT/'background_candidates.json',dict(candidates=chosen,label='Only central128 region provisionally paper; outside unknown. Must visually review before freezing.'))
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14)
    for arm in source_sets:
        subset=[c for c in chosen if c['arm']==arm];canvas=Image.new('RGB',(4*268,4*290),'#faf9f5');d=ImageDraw.Draw(canvas)
        for i,c in enumerate(subset):
            im=Image.open(ROOT/c['crop']).convert('RGB');dd=ImageDraw.Draw(im);dd.rectangle((63,63,192,192),outline=(0,180,80),width=1)
            x=i%4*268;y=i//4*290;d.text((x,y),c['id']+' '+c['book'],font=font,fill='black');canvas.paste(im,(x,y+26))
        canvas.save(OUT/f'{arm}_原画像候補.png')
    print('Prepared32 source-only windows; review central128 rectangles',flush=True)
if __name__=='__main__':screen()
