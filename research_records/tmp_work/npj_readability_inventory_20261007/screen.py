from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
T=ROOT/'tmp_work';OUT=Path(__file__).resolve().parent
S=OUT/'screening';S.mkdir(exist_ok=True)
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fnt=ImageFont.truetype(FONT,22)
frames=[]
for f in json.loads((T/'npj_need_assessment_20261002/manifest.json').read_text())['frames']:
    frames.append(dict(id=f['id'],page=f['page'].replace('_crop',''),original=f['original'],input=f['input'],output=f['saved_npj'],origin_xywh=f['box'],kind='実保存・旧crop' if 'crop' in f['id'] else '実保存・全ページ',version=f['baseline_label']))
c=json.loads((T/'npj_ridge_saved_ranges_20261004/selection.json').read_text())['cases'][0]
frames.append(dict(id='saved_200018243_00002_2',page=c['page'],original=c['original'],input=c['input'],output=c['saved_npj'],origin_xywh=None,kind='実保存・全ページ',version='ローカルv2・外部紙面参照'))
for c in json.loads((T/'npj_full_page_layout_v4_20261004/manifest.json').read_text())['frames']:
    if c['id']=='saved_hnsd':continue
    frames.append(dict(id=c['id'],page=c['page'],original=c['original'],input=c['original_crop'],output=c['saved_npj'],origin_xywh=c['xywh'],kind='凍結コードの新規crop対照',version=c['baseline_kind']))
for c in json.loads((T/'npj_dirt_new_book_20261004/selection.json').read_text())['cases']:
    if 'saved_full' in c:continue
    d=T/'npj_dirt_new_book_20261004/native'/c['id']
    frames.append(dict(id=c['id'],page=c['page'],original=c['original'],input=str((d/'original.png').relative_to(ROOT)),output=str((d/'baseline.png').relative_to(ROOT)),origin_xywh=c['xywh'],kind='凍結コードの新規crop対照',version=c['baseline_kind']))
for page,box,sub,out in [('hnsd001_011',[336,396,528,528],'hnsd001_input','hnsd001_output'),('hnsd007_042',[600,100,768,768],'input','output')]:
    d=T/'npj_manual_stain_oracle_20261004'
    frames.append(dict(id='manual_control_'+page,page=page,original=f'data/split_dataset/train/hnsd00000/{page}.jpg',input=str((d/sub/(page+'.png')).relative_to(ROOT)),output=str((d/out/page/'result'/(page+'.png')).relative_to(ROOT)),origin_xywh=box,kind='凍結コードの新規crop対照',version='通常黒文字マスク・後処理前'))
records=[]
for f in frames:
    o=Image.open(ROOT/f['original']).convert('RGB')
    if f['origin_xywh']:
        x,y,w,h=f['origin_xywh'];o=o.crop((x,y,x+w,y+h))
    inp=Image.open(ROOT/f['input']).convert('RGB');out=Image.open(ROOT/f['output']).convert('RGB')
    assert o.size==inp.size==out.size,f['id']
    f['input_original_exact']=bool(np.array_equal(np.asarray(o),np.asarray(inp)))
    f['shape_wh']=list(o.size)
    f['source_hashes']={k:hashlib.sha256((ROOT/f[k]).read_bytes()).hexdigest() for k in ['original','input','output']}
    cv=Image.new('RGB',(1480,1100),'#faf8f2');d=ImageDraw.Draw(cv)
    d.text((20,10),f"{f['id']} / {f['page']} / {f['kind']}",font=fnt,fill='black')
    for j,(a,l) in enumerate([(o,'原画像'),(inp,'実推論入力'),(out,'NPJ出力')]):
        d.text((20+j*485,55),l,font=fnt,fill='black')
        q=a.copy();q.thumbnail((465,960),Image.Resampling.LANCZOS);cv.paste(q,(20+j*485,100))
    cv.save(S/(f['id']+'_overview.png'))
    # Native-size coverage, fixed spatial grid, independent of mask decisions.
    if f['input_original_exact']:
        pair=[o,out]
    else:pair=[inp,out]
    for yy in range(0,o.height,800):
        for xx in range(0,o.width,800):
            w,h=min(800,o.width-xx),min(800,o.height-yy)
            records.append(dict(frame=f['id'],page=f['page'],xywh=[xx,yy,w,h],pair=[a.crop((xx,yy,xx+w,yy+h)) for a in pair]))
for start in range(0,len(records),4):
    cv=Image.new('RGB',(1520,1060),'#faf8f2');d=ImageDraw.Draw(cv)
    d.text((12,8),'本文候補の走査：左＝実入力、右＝NPJ。縮小。細部は別図で原寸確認。',font=fnt,fill='black')
    for j,r in enumerate(records[start:start+4]):
        x,y=(j%2)*760+12,(j//2)*510+45
        d.text((x,y),f"{r['frame']}  {r['xywh']}",font=fnt,fill='black')
        for k,a in enumerate(r['pair']):
            a=a.copy();a.thumbnail((350,455),Image.Resampling.LANCZOS);cv.paste(a,(x+k*370,y+35))
        r['screen']=f'scan_{start//4+1:02d}.png'
    cv.save(S/f'scan_{start//4+1:02d}.png')
    print(f'scan_{start//4+1:02d}: '+', '.join('{} {}'.format(r['frame'],r['xywh']) for r in records[start:start+4]))
for r in records:r.pop('pair')
(OUT/'frames.json').write_text(json.dumps(frames,ensure_ascii=False,indent=2)+'\n')
(OUT/'screening_grid.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(frames=len(frames),pages=len({f['page'] for f in frames}),tiles=len(records),books=len({f['page'].split('_')[0] for f in frames}))))
