"""Original-only survey of training books for dark nontext candidates."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from compare_v5_aligned_fade_control import rgb,sha,dump
from screen_background_diversity import BOOKS
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'tmp_work/dark_nontext_teacher_control_20261023'
def main():
    assert not (OUT/'survey.json').exists();OUT.mkdir(exist_ok=True);rng=np.random.default_rng(2026102307);rows=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',13)
    for book in BOOKS:
        paths=sorted((ROOT/'data/split_dataset_jogai'/book).glob('*.jpg'));pick=sorted(rng.choice(len(paths),6,replace=False));canvas=Image.new('RGB',(3*340,2*550),'white');d=ImageDraw.Draw(canvas)
        for j,k in enumerate(pick):
            p=paths[int(k)];raw=rgb(p);small=Image.fromarray(raw);small.thumbnail((330,510));x=j%3*340;y=j//3*550;d.text((x,y),p.stem,font=font,fill='black');canvas.paste(small,(x,y+30));rows.append(dict(book=book,source=str(p.relative_to(ROOT)),source_sha256=sha(p),shape=list(raw.shape)))
        canvas.save(OUT/f'{book}_原画像調査.png')
    dump(OUT/'survey.json',dict(seed=2026102307,selection='6 uniform pages per inherited8 training books, original only; no model/classical outputs',rows=rows))
    print('Surveyed48 training pages,8 source-only contact sheets',flush=True)
if __name__=='__main__':main()
