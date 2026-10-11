from pathlib import Path
import json, importlib.util, hashlib, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def dump(name,a): (HERE/name).write_text(json.dumps(a,ensure_ascii=False,indent=2))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads((ROOT/p).read_text())
def rgb(p): return np.array(Image.open(p).convert('RGB'))
def text(d,xy,s,size=18): d.text(xy,s,font=ImageFont.truetype(FONT,size),fill='#253440')

def candidates():
    cs=[]
    for c in load('tmp_work/npj_readability_inventory_20261007/selection.json')['candidates']:
        if c['id']=='R01':continue
        cs.append(dict(id='R_'+c['id'],page=c['page'],original=c['source']['original'],seed_xywh=c['page_xywh'],reason=c['category']))
    for c in load('tmp_work/npj_contact_stain_search_20261007/benchmark_candidates.json')['cases']:
        if c['id'] in ['T00','T05']:continue
        cs.append(dict(id='S_'+c['id'],page=c['page'],original=c['source']['original'],seed_xywh=c['page_detail_xywh'],reason=c['title']))
    for c in load('tmp_work/npj_failure_collection_20261001/baseline_review.json')['cases']:
        for r in c['rois']:
            if r['id'] not in ['dense_stain','thin_black_water','thin_black_text','hole_edge_new']:continue
            cs.append(dict(id='D_'+r['id'],page=c['page'],original=c['original'],seed_xywh=r['xywh'],reason=r['title']))
    for c in cs:
        with Image.open(ROOT/c['original']) as im: W,H=im.size
        x,y,w,h=c['seed_xywh'];nw=max(520,min(760,w));nh=max(520,min(760,h))
        x=max(0,min(W-nw,round(x+(w-nw)/2)));y=max(0,min(H-nh,round(y+(h-nh)/2)))
        c['roi_xywh']=[x,y,nw,nh];c['source_sha256']=sha(ROOT/c['original']);c['image_wh']=[W,H]
    dump('candidate_protocol.json',dict(cases=cs,settings=dict(input='Full native original RGB -> OpenCV gray /255, crop after page processing',sauvola=dict(window=61,k=.2,R=.5),wolf=dict(window=61,k=.5),howe_local=dict(c=160,canny='skimage approximation; no automatic tuning')),selection='Known difficult originals from previous inventories, originals inspected first; final hard cases chosen after outputs; exploratory, no held-out accuracy or globally-hardest guarantee.',parameters_frozen_before_new_outputs=True))
    (HERE/'search').mkdir(exist_ok=True)
    for start in range(0,len(cs),6):
        out=Image.new('RGB',(3*370,2*460+65),'#faf9f5');d=ImageDraw.Draw(out)
        text(d,(12,12),'原画像のみの候補探索 / 縮小表示、未補正',24)
        for i,c in enumerate(cs[start:start+6]):
            x,y,w,h=c['roi_xywh'];im=Image.open(ROOT/c['original']).convert('RGB').crop((x,y,x+w,y+h));im.thumbnail((350,360))
            px=10+(i%3)*370;py=65+(i//3)*460
            text(d,(px,py),c['id']+' / '+c['page'],15);text(d,(px,py+25),c['reason'][:22],16)
            out.paste(im,(px,py+65))
        out.save(HERE/'search'/f'original_candidates_{start//6+1:02d}.png')
    return cs

def compute():
    spec=importlib.util.spec_from_file_location('deep',ROOT/'tmp_work/classical_deep_review_20261008/run.py');deep=importlib.util.module_from_spec(spec);spec.loader.exec_module(deep)
    deep.SCRATCH=HERE/'scratch';deep.SCRATCH.mkdir(exist_ok=True)
    cs=json.loads((HERE/'candidate_protocol.json').read_text())['cases'];pages={c['page']:c for c in cs}
    records=json.loads((HERE/'processing.json').read_text()) if (HERE/'processing.json').exists() else []
    for page,c in pages.items():
        if page in {r['page'] for r in records}:continue
        start=time.monotonic();raw=rgb(ROOT/c['original']);g=deep.base.gray(raw)
        masks={};meta={}
        masks['sauvola']=deep.base.sauvola(g,61,.2);meta['sauvola']=dict(window=61,k=.2,R=.5)
        masks['wolf'],meta['wolf']=deep.base.wolf(g,61,.5)
        terms,tmeta=deep.howe_terms(g);masks['howe'],hmeta=deep.howe(terms,160);meta['howe']={**tmeta,**hmeta}
        for key,b in masks.items():
            p=HERE/'masks'/page/(key+'.png');deep.save_mask(p,b)
            assert np.array_equal(deep.read_mask(p),b)
        for row in cs:
            if row['page']==page:render(row,raw,masks,HERE/'candidate_comparisons')
        records.append(dict(page=page,seconds=time.monotonic()-start,meta=meta))
        dump('processing.json',records);print(page,round(records[-1]['seconds'],2),flush=True)

def render(c,raw,masks,folder,title=None):
    x,y,w,h=c['roi_xywh'];scale=c.get('scale',1);pw,ph=w*scale,h*scale
    gap=22;top=108;label=38
    out=Image.new('RGB',(pw*2+gap*3,top+2*(ph+label+gap)+70),'#faf9f5');d=ImageDraw.Draw(out)
    text(d,(18,12),title or c['id']+' / '+c['page'],25)
    text(d,(18,53),'原画像で全ページ処理 → 同じ範囲を切り出し。黒＝前景候補。',18)
    text(d,(18,79),f'座標 {x},{y},{w},{h} / 原画素の整数{scale}倍 / 黒文字を対象',16)
    placements=[]
    for i,(k,s) in enumerate([('original','原画像'),('sauvola','Sauvola / w=61, k=.2'),('wolf','Wolf / w=61, k=.5'),('howe','Howe 基本エネルギー移植 / c=160')]):
        px=gap+(i%2)*(pw+gap);py=top+(i//2)*(ph+label+gap)
        a=raw[y:y+h,x:x+w] if k=='original' else np.repeat(np.where(masks[k][y:y+h,x:x+w],0,255).astype('uint8')[...,None],3,axis=2)
        a=np.repeat(np.repeat(a,scale,0),scale,1)
        text(d,(px,py),s,18);out.paste(Image.fromarray(a),(px,py+label))
        assert np.array_equal(np.array(out)[py+label:py+label+ph,px:px+pw],a)
        placements.append(dict(method=k,xywh=[px,py+label,pw,ph],scale=scale))
    text(d,(18,out.height-53),'HoweはCanny近似・固定パラメータ。著者実装の完全再現ではない。',16)
    text(d,(18,out.height-28),'探索で選んだ難例。画素正解は未作成、精度・総合順位は未確定。',16)
    folder.mkdir(exist_ok=True,parents=True);dest=folder/(c['id']+'.png');out.save(dest)
    return dict(path=str(dest.relative_to(HERE)),panels=placements,sha256=sha(dest))

if __name__=='__main__':
    import sys
    if sys.argv[-1]=='compute':compute()
    else:candidates()
