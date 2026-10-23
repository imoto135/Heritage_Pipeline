"""Additional source-only ink guards, never training or replacement eval labels."""
import json
import cv2,numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont
from compare_v5_aligned_fade_control import rgb,dump,sha
from screen_dark_nontext_sources import ROOT,OUT
PATHS={'D04':[
 ('left_loop',[[70,78],[58,82],[53,90],[52,105],[58,120],[64,126],[80,130],[101,129],[113,121],[117,109],[112,96],[102,82],[92,79],[80,84],[76,95],[72,112],[68,122]]),
 ('center_vertical',[[268,147],[271,157],[270,173],[270,188],[270,202],[269,209]]),
 ('lower_loop',[[244,493],[246,475],[252,459],[266,449],[278,448],[291,454],[297,469],[292,482],[277,490],[260,493],[247,494]]),
 ('right_vertical',[[445,84],[440,98],[439,111],[438,125],[436,138]]),
 ('left_lower_descender',[[80,321],[80,339],[77,355],[69,370],[59,386],[50,392]])],
 'S03':[('upper_curve',[[211,0],[214,11],[212,25],[205,38],[197,49],[188,59],[183,68]]),('lower_hook',[[204,52],[210,51],[214,56],[215,64],[217,70],[227,74],[237,72]])]}
def main():
    assert not (OUT/'text_guard_plan.json').exists();assert not list((OUT/'evaluation').glob('*/results.json'));cs=json.loads((OUT/'selected_sources.json').read_text())['cases'];c=next(c for c in cs if c['id']=='S03');sources={'D04':OUT/'source_guard/D04_original.png','S03':ROOT/c['crop']};rows=[]
    for key,paths in PATHS.items():
        raw=rgb(sources[key]);g=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY).astype(float);deficit=ndi.maximum_filter(g,25)-g;manual=np.zeros(g.shape,np.uint8)
        for name,xy in paths:cv2.polylines(manual,[np.array(xy,np.int32)],False,255,1)
        supported=(manual>0)&(deficit>=12);f=OUT/'source_guard'/f'{key}_ink_guard.png';Image.fromarray(supported.astype(np.uint8)*255).save(f);over=raw.copy();over[supported]=[0,90,255];over[(manual>0)&~supported]=[255,150,0]
        canvas=Image.new('RGB',(raw.shape[1]*2,raw.shape[0]+32),'white');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14);d.text((0,0),key+' 原画像 / 青=文字保持の補助経路 橙=原画像支持不足で除外',font=font,fill='black');canvas.paste(Image.fromarray(raw),(0,32));canvas.paste(Image.fromarray(over),(raw.shape[1],32));canvas.save(OUT/'source_guard'/f'{key}_source_guard_review.png')
        rows.append(dict(id=key,source_file=str(sources[key].relative_to(ROOT)),source_sha256=sha(sources[key]),mask_file=str(f.relative_to(ROOT)),mask_sha256=sha(f),ink_n=int(supported.sum()),unsupported_n=int(((manual>0)&~supported).sum()),paths=[dict(id=n,xy=xy) for n,xy in paths]))
    dump(OUT/'text_guard_annotations.json',dict(rows=rows,status='Additional provisional source-only ink traces for all-white failure guard; not training, not expert boundaryGT, does not change fixed26case labels. Raw source only, no new predictions used. Prior outputs/development context known.',support_filter='maxfilter25-gray>=12; unsupported traced pixels unknown, no gap filling.'))
    print([(r['id'],r['ink_n'],r['unsupported_n']) for r in rows],flush=True)
if __name__=='__main__':main()
