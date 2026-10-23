"""Conservative source-supported near-paper proxy; old annotations stay immutable."""
import json
from pathlib import Path
import cv2,numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont
from compare_v5_aligned_fade_control import rgb,dump,sha
import prepare_mixed_paper_ink_labels as previous
ROOT=previous.ROOT;OLD=previous.OUT;OUT=ROOT/'tmp_work/near_boundary_teacher_control_20261023'
def prepare():
    assert not (OUT/'protocol.json').exists();OUT.mkdir(exist_ok=True)
    cases=json.loads((OLD/'selected_sources.json').read_text())['cases'];anns={a['id']:a for a in json.loads((OLD/'annotations_source_only.json').read_text())['regions']};counts=[]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    for c in cases:
        raw=rgb(ROOT/c['crop']);lab,manual=previous.labels(c,anns[c['id']]);assert np.array_equal(lab,np.asarray(Image.open(OLD/'labels'/f"{c['id']}.png")))
        gray=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY).astype(float);deficit=ndi.maximum_filter(gray,25)-gray
        near=(ndi.distance_transform_edt(lab!=1)<=20)&(deficit<=12)&(ndi.distance_transform_edt(deficit<12)>2)&(ndi.distance_transform_edt(manual!=1)>3)&(lab!=2)
        assert not (near&(manual==1)).any() and not (near&(lab>0)).any() and near.any()
        target=OUT/'near_paper'/f"{c['id']}.png";target.parent.mkdir(exist_ok=True);Image.fromarray(near.astype(np.uint8)*255).save(target)
        over=raw.copy();over[lab==1]=[0,100,255];over[lab==2]=np.rint(.5*raw[lab==2]+.5*np.array([0,200,80])).astype(np.uint8);over[near]=np.rint(.35*raw[near]+.65*np.array([255,0,190])).astype(np.uint8)
        canvas=Image.new('RGB',(768,550),'white');d=ImageDraw.Draw(canvas);d.text((0,0),c['id']+' 原画像 / 青=旧墨芯 緑=旧紙 紫=追加紙（2倍・最近傍）',font=font,fill='black');canvas.paste(Image.fromarray(raw),(0,35));canvas.paste(Image.fromarray(over).resize((512,512),Image.Resampling.NEAREST),(256,35));canvas.save(OUT/f"{c['id']}_原画像と追加教師.png")
        counts.append(dict(id=c['id'],near_paper_n=int(near.sum()),sha256=sha(target)))
    dump(OUT/'label_preparation.json',dict(counts=counts,parameters=dict(local_max_window=25,paper_deficit_max=12,dark_deficit_min=12,dark_exclusion_radius=2,manual_trace_exclusion_radius=3,ink_corridor_radius=20),
      status='Conservative source-contrast proxy for near-stroke paper, not expert boundary GT. Only source RGB and old source annotations used. Prior model results motivated this development experiment; not blinded.',
      unknown='All other pixels keep old status; unsupported manual trace points protected by radius3; no RGB edits, no new ink labels.',old_ink=1159,old_paper=14728))
    print(counts,flush=True)
if __name__=='__main__':prepare()
