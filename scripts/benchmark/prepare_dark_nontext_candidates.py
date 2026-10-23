"""Native source windows from original contact-sheet review; no predictions."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from compare_v5_aligned_fade_control import rgb,sha,dump
from screen_dark_nontext_sources import ROOT,OUT
PICKS=[('200015779_00020_1',260,53),('200015779_00165_1',275,484),('200021853_00004_2',197,451),('200021853_00004_2',205,327),('100249537_00076_1',265,40),('100249376_00040_1',181,74),('200008316_00048_2',260,38),('200014685_00017_2',295,45),('100249476_00021_2',239,53),('200008316_00008_1',255,42),('100249537_00013_1',66,47),('200021802_00016_1',64,50)]
def main():
    assert not (OUT/'protocol.json').exists();survey=json.loads((OUT/'survey.json').read_text())['rows'];rows=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',13);canvas=Image.new('RGB',(4*268,3*290),'white');d=ImageDraw.Draw(canvas)
    for i,(name,u,v) in enumerate(PICKS):
        r=next(r for r in survey if Path(r['source']).stem==name);raw=rgb(ROOT/r['source']);h,w=raw.shape[:2];scale=min(330/w,510/h);cx=int(u/scale);cy=int(v/scale);x=max(0,min(w-256,cx-128));y=max(0,min(h-256,cy-128));crop=raw[y:y+256,x:x+256];key=f'S{i+1:02d}';p=OUT/'candidate_inputs'/f'{key}.png';p.parent.mkdir(exist_ok=True);Image.fromarray(crop).save(p);xx=i%4*268;yy=i//4*290;d.text((xx,yy),key+' '+name,font=font,fill='black');canvas.paste(Image.fromarray(crop),(xx,yy+28));rows.append(dict(id=key,book=r['book'],source=r['source'],source_sha256=r['source_sha256'],roi=[x,y,256,256],crop=str(p.relative_to(ROOT)),crop_sha256=sha(p)))
    canvas.save(OUT/'汚れ候補_原寸.png');dump(OUT/'candidate_sources.json',dict(cases=rows,source_only=True))
    print('Prepared12 native256 windows for source review')
if __name__=='__main__':main()
