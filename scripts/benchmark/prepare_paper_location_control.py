"""Match paper pixel counts and source contrast/luminance; source-only locations."""
import json
import cv2,numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont
import prepare_near_paper_teacher as previous
from compare_v5_aligned_fade_control import rgb,sha,dump
ROOT=previous.ROOT;OLD=previous.OLD;NEAR=previous.OUT;OUT=ROOT/'tmp_work/paper_location_control_20261023'
def prepare():
    assert not (OUT/'protocol.json').exists();OUT.mkdir(exist_ok=True)
    cases=json.loads((OLD/'selected_sources.json').read_text())['cases'];aa={a['id']:a for a in json.loads((OLD/'annotations_source_only.json').read_text())['regions']};rows=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    for i,c in enumerate(cases):
        raw=rgb(ROOT/c['crop']);lab,manual=previous.previous.labels(c,aa[c['id']]);gray=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY).astype(int);deficit=ndi.maximum_filter(gray,25)-gray;dist=ndi.distance_transform_edt(manual!=1)
        near=np.asarray(Image.open(NEAR/'near_paper'/f"{c['id']}.png"))>0
        pool=(dist>=40)&(deficit<=12)&(ndi.distance_transform_edt(deficit<12)>2)&(lab==0)
        strata=deficit*8+gray//32;far=np.zeros(near.shape,bool);rng=np.random.default_rng(2026102300+i);hist=[]
        for s in np.unique(strata[near]):
            n=int((near&(strata==s)).sum());candidates=np.flatnonzero(pool&(strata==s));assert len(candidates)>=n,(c['id'],s,n,len(candidates));far.flat[rng.choice(candidates,n,replace=False)]=True;hist.append(dict(stratum=int(s),n=n))
        assert near.sum()==far.sum() and not (near&far).any() and not (far&(lab>0)).any() and (dist[far]>=40).all()
        for arm,m in [('near',near),('far',far)]:
            dest=OUT/'paper_masks'/arm/f"{c['id']}.png";dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(m.astype(np.uint8)*255).save(dest)
            assert np.array_equal(np.bincount(strata[m],minlength=104),np.bincount(strata[near],minlength=104))
        canvas=Image.new('RGB',(1024,570),'white');d=ImageDraw.Draw(canvas);d.text((0,0),c['id']+' 原画像 / 紫=近傍紙 / 水色=遠方紙（最近傍2倍・画素数と濃度層を一致）',font=font,fill='black')
        for j,(title,m,color) in enumerate([('近傍紙',near,[255,0,190]),('遠方紙',far,[0,175,230])]):
            over=raw.copy();over[m]=np.rint(.3*raw[m]+.7*np.array(color)).astype(np.uint8);over[lab==1]=[0,80,255];d.text((j*512,28),title+f' {int(m.sum())}画素',font=font,fill='black');canvas.paste(Image.fromarray(over).resize((512,512),Image.Resampling.NEAREST),(j*512,50))
        # Native original is supplied separately, untouched.
        Image.fromarray(raw).save(OUT/f"{c['id']}_原画像.png");canvas.save(OUT/f"{c['id']}_近傍と遠方の教師.png")
        rows.append(dict(id=c['id'],near_n=int(near.sum()),far_n=int(far.sum()),near_distance_median=float(np.median(dist[near])),far_distance_min=float(dist[far].min()),far_distance_median=float(np.median(dist[far])),strata=hist,near_sha256=sha(OUT/'paper_masks/near'/f"{c['id']}.png"),far_sha256=sha(OUT/'paper_masks/far'/f"{c['id']}.png")))
    dump(OUT/'label_preparation.json',dict(rows=rows,total_pixels_each=sum(r['near_n'] for r in rows),matching='Within each native256 RGB, exact local gray deficit strata (maxfilter25-gray) and gray brightness bins32, same counts without replacement. width16 strata infeasible for B08, choose width32 globally before training.',
        far_definition='Distance>=40px from ALL manually traced ink paths in this crop, including unsupported trace points; not a claim of distance40 from every unannotated glyph. Same source clear-paper criterion as near: deficit<=12, distance from deficit>=12 pixels>2. Exclude old labels and near pixels.',
        status='Source-reviewed provisional partial paper proxy, not expertGT or complete boundary labels. Far mask scattering/shape differs from near and is part of spatial treatment. Known development experiment. No new RGB or damage.'))
    print('Prepared matched near/far paper masks:',sum(r['near_n'] for r in rows),'pixels each',flush=True)
if __name__=='__main__':prepare()
