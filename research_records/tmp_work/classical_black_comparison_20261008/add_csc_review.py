"""Add saved CSC v2 and wider, less crowded native comparison sheets."""
from pathlib import Path
import json
import hashlib
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'tmp_work/csc_wolf_review_20261008'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
LABELS={'original':'原画像','wolf61':'Wolf–Jolion / 窓61・k=.5',
    'csc_v2':'CSC v2 / 保存済み黒マスク','sauvola61':'Sauvola / 窓61・k=.2',
    'nick61':'NICK / 窓61・k=-.1','gatos_native':'Gatos再実装 / 原寸 M=1',
    'su2013_local':'Su 2013再実装','csc_v1':'参考：旧CSC v1'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return np.asarray(Image.open(p).convert('RGB')).copy()
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def font(s):return ImageFont.truetype(FONT,s)

def render(c,arrays,keys,name,cols=2):
    scale=2 if c['id']=='N01' else 1
    w,h=c['roi_xywh'][2:];pw,ph=w*scale,h*scale
    gap=24;top=118;heading=44;rows=math.ceil(len(keys)/cols)
    canvas=Image.new('RGB',(cols*(pw+gap)+gap,top+rows*(ph+heading+gap)+100),'#faf9f5')
    d=ImageDraw.Draw(canvas)
    title=name.replace('_','・')
    if name=='原画像_Wolf_CSC_Sauvola' and 'csc_v1' in keys:
        title='原画像・Wolf・CSC v2・Sauvola・CSC v1'
    d.text((18,14),c['id']+' 黒文字：'+title,font=font(25),fill='#20313b')
    caption='同じ座標。黒＝文字候補、白＝背景。'
    if c['id']=='N05':caption+='複数の文字が見える広い範囲。'
    d.text((18,53),caption,font=font(17),fill='#20313b')
    d.text((18,79),'原寸ページの保存マスクを切り出し。朱・DDRMなし。',font=font(17),fill='#20313b')
    panels=[]
    for i,k in enumerate(keys):
        xx=gap+i%cols*(pw+gap);yy=top+i//cols*(ph+heading+gap)
        d.text((xx,yy),LABELS[k],font=font(20),fill='#20313b')
        a=arrays[k]
        tile=Image.fromarray(a).resize((pw,ph),Image.Resampling.NEAREST)
        canvas.paste(tile,(xx,yy+heading))
        assert np.array_equal(np.asarray(canvas)[yy+heading:yy+heading+ph,xx:xx+pw],np.repeat(np.repeat(a,scale,0),scale,1))
        panels.append(dict(method=k,canvas_xywh=[xx,yy+heading,pw,ph],integer_scale=scale))
    d.text((18,canvas.height-81),'CSC v2はACE前処理を含む保存結果。他の古典手法は原画像のグレー入力。',font=font(16),fill='#20313b')
    d.text((18,canvas.height-55),'文字の見やすさ・不要物の残り・筆画の欠落を、原画像に照らして比べる。',font=font(16),fill='#20313b')
    d.text((18,canvas.height-29),'表示の輪郭強調・平滑化なし。N01だけ整数2倍、それ以外は原寸。',font=font(16),fill='#20313b')
    path=OUT/'figures'/(c['id']+'_'+name+'.png');path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    return dict(path=str(path.relative_to(OUT)),sha256=sha(path),panels=panels,roi_xywh=c['roi_xywh'])

def main():
    old=json.loads((HERE/'protocol.json').read_text())
    cases=[];rendering=[];counts=[];sources={};main_orders={}
    for initial in old['cases']:
        c=dict(initial)
        if c['id']=='N05':
            c['prior_roi_xywh']=c['roi_xywh'];c['roi_xywh']=[1019,1617,640,580]
            c['saved_v2']='data/split_dataset_csc_mask_v2/test/200010454/200010454_00021_1/02_text_black.png'
            c['reason']='利用者指定で拡大局所比較を見直し、複数の文字を含む広い範囲へ変更'
        c['saved_v1']=c['saved_v2'].replace('split_dataset_csc_mask_v2','split_dataset_csc_mask')
        x,y,w,h=c['roi_xywh'];sl=np.s_[y:y+h,x:x+w]
        raw=read(ROOT/c['original']);assert sha(ROOT/c['original'])==c['source_sha256']
        assert x+w<=raw.shape[1] and y+h<=raw.shape[0]
        source_paths={'original':ROOT/c['original'],'csc_v2':ROOT/c['saved_v2'],'csc_v1':ROOT/c['saved_v1']}
        for k in ['wolf61','sauvola61','nick61','gatos_native','su2013_local']:
            source_paths[k]=HERE/'masks'/c['id']/(k+'.png')
        full={};arrays={}
        for k,p in source_paths.items():
            a=read(p);assert a.shape==raw.shape,(k,a.shape,raw.shape)
            if k!='original':
                assert set(np.unique(a)).issubset({0,255})
                assert np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,0],a[:,:,2])
            full[k]=a;arrays[k]=a[sl]
            sources[str(p.relative_to(ROOT))]=sha(p)
            dest=OUT/'panels'/c['id']/(k+'.png');dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(arrays[k]).save(dest)
            assert np.array_equal(read(dest),a[sl])
        for k,a in arrays.items():
            if k=='original':continue
            counts.append(dict(id=c['id'],method=k,roi_foreground_pixels=int((a[:,:,0]==0).sum()),
                interpretation='Foreground area only, not accuracy or true ink coverage.'))
        main_keys=['original','wolf61','csc_v2','sauvola61']
        if c['id']=='T01':main_keys.append('csc_v1')
        main_orders[c['id']]=main_keys
        rendering.append(render(c,arrays,main_keys,'原画像_Wolf_CSC_Sauvola'))
        rendering.append(render(c,arrays,['csc_v2','nick61','gatos_native','su2013_local'],'CSCと他の古典手法'))
        rendering.append(render(c,arrays,['original','csc_v2','csc_v1','wolf61'],'CSC版の参考比較'))
        cases.append(c)
    dump(OUT/'protocol.json',dict(cases=cases,
        main_order=['original','wolf61','csc_v2','sauvola61'],
        main_order_per_case=main_orders,
        csc='Saved local CSC v2 black masks, with ACE input path; previous v1 also in T01 main comparison and separate reference sheets.',
        unchanged_parameters=True,new_inference=False,
        note='N05 ROI widened only for viewing; source full-page masks unchanged. No quality ranking inferred from pixel counts.',
        source_sha256=sources))
    dump(OUT/'rendering.json',rendering);dump(OUT/'foreground_counts.json',counts)
    for p,h in sources.items():assert sha(ROOT/p)==h
    dump(OUT/'verification.json',dict(source_hashes_unchanged=True,panels_match_native_source_crops=True,
        integer_display_pixels_exact=True,mask_polarity='0 foreground / 255 background',
        source_script_sha256=sha(__file__),figures=len(rendering)))
    print('Created',len(rendering),'comparison PNGs; N05 ROI',next(c['roi_xywh'] for c in cases if c['id']=='N05'))

if __name__=='__main__':main()
