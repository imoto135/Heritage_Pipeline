"""Fixed mask and genuine repair views; external scripts receive no Japanese DDRM claim."""
import argparse,json,textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,file_hash
from score_independent_points import mask_path
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
LABELS={'csc_v2_red':'ローカルCSC v2','region_spatial_forest':'色・近傍特徴＋RF（開発冊で学習）','texture_paper_texture_sparse':'紙テクスチャ＋少数教師CNN','partial_dense_native':'背景領域も教えたCNN','partial_dense_native_photometry':'撮影色の増強CNN','partial_dense_native_relative_only':'相対色のみのCNN','relative_ridge':'相対色＋線の応答','docres_foreground_chroma':'DocRes前景 ∩ 相対色','sauvola_foreground':'Sauvolaの前景（朱専用ではない）'}
DIRECT={'docres_appearance_author':'DocRes appearance（全体1600）','docres_appearance_native':'DocRes appearance（原寸タイル）','stain':'StainRestorer（通常汚れモデル）','stain_global256':'StainRestorer（全体256・細部縮小）','docres_binarization':'DocRes二値化（色を保持しない）'}
def getmask(name,page,shape):
 if name=='preserve_everything':return np.ones(shape,bool)
 if name=='preserve_nothing':return np.zeros(shape,bool)
 if name=='docres_foreground_chroma':
  a=mask_path('docres_foreground',page);b=mask_path('relative_chroma',page)
  if a.exists() and b.exists():return (np.asarray(Image.open(a).convert('L'))>127)&(np.asarray(Image.open(b).convert('L'))>127)
  return None
 f=mask_path(name,page)
 return np.asarray(Image.open(f).convert('L'))>127 if f.exists() else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',default=['csc_v2_red','region_spatial_forest','texture_paper_texture_sparse','partial_dense_native_photometry','relative_ridge']);p.add_argument('--cases',nargs='*');p.add_argument('--suffix',default='fixed_v2');a=p.parse_args();cases=json.loads((OUT/'independent_cases_v2.json').read_text())['cases'];dest=OUT/'independent_comparisons'/a.suffix;dest.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype(FONT,17);small=ImageFont.truetype(FONT,14);records=[]
 for r in cases:
  cid=r['case_id'];page=r['page']
  if a.cases and cid not in a.cases:continue
  assert file_hash(ROOT/r['original'])==r['source_sha256'];orig=np.asarray(Image.open(ROOT/r['original']).convert('RGB'));x,y,w,h=r['ROI'];sl=np.s_[y:y+h,x:x+w];assert orig[sl].shape[:2]==(h,w)
  title=r.get('name',r.get('description',cid));external=page.startswith('external_');nativebase=OUT/'sauvola_ddrm_base'/page/'complete.json';base=None;baserec=None
  if nativebase.exists() and not external:
   baserec=json.loads(nativebase.read_text());assert file_hash(baserec['result'])==baserec['result_sha256'];base=np.asarray(Image.open(baserec['result']).convert('RGB'));assert base.shape==orig.shape
  methods=a.methods+(['preserve_everything','preserve_nothing'] if base is not None else [])
  avail=[]
  for m in methods:
   mask=getmask(m,page,orig.shape[:2])
   if mask is not None:assert mask.shape==orig.shape[:2];avail.append((m,mask))
  scale=3 if w<=200 else 1;dw,dh=w*scale,h*scale;cw=max(dw,330);cols=3 if base is not None else 2;rowh=dh+45
  if avail:
   sheet=Image.new('RGB',(cols*cw,118+len(avail)*rowh+66),'#faf9f5');draw=ImageDraw.Draw(sheet);draw.text((8,7),cid+'／'+title,font=font,fill='#20313a');draw.text((8,32),str(r['ROI'])+' 原寸ROI・'+('3倍最近傍表示' if scale==3 else '原寸表示')+'・'+r['role'],font=small,fill='#586574');draw.text((8,54),'同じ黒修復に元の赤画素を貼り戻す' if base is not None else '候補マスクの比較。外部文字への日本語DDRM修復は行っていない。',font=small,fill='#586574')
   for j,s in enumerate(['原画像','白＝保存する候補']+(['修復後（同じ黒修復を共有）'] if base is not None else [])):draw.text((j*cw+8,87),s,font=font,fill='#20313a')
   for i,(m,mask) in enumerate(avail):
    yy=118+i*rowh;label=LABELS.get(m,{'preserve_everything':'全部戻す対照＝原画像','preserve_nothing':'赤を戻さない対照'}.get(m,m));draw.text((8,yy+4),label,font=font,fill='#20313a');panels=[orig[sl],np.repeat(mask[sl][:,:,None]*255,3,2).astype(np.uint8)];rec=dict(case_id=cid,method=m,page=page,source_sha256=r['source_sha256'],ROI_original=r['ROI'],scale=scale,selected_ROI_pixels=int(mask[sl].sum()),reading_GT=False)
    if base is not None:
     out=base.copy();out[mask]=orig[mask];assert not (np.any(out!=base,2)&~mask).any();assert np.array_equal(out[mask],orig[mask]);d=dest/page/m;d.mkdir(parents=True,exist_ok=True);full=d/'result.png'
     if full.exists():assert np.array_equal(np.asarray(Image.open(full)),out)
     else:Image.fromarray(out).save(full)
     rec.update(base_sha256=baserec['result_sha256'],output_sha256=file_hash(full),changed_outside_mask=0,genuine_shared_Sauvola_DDRM_base=True);panels.append(out[sl])
    for j,arr in enumerate(panels):drawim=Image.fromarray(arr).resize((dw,dh),Image.Resampling.NEAREST);sheet.paste(drawim,(j*cw,yy+36))
    records.append(rec)
   draw.text((8,sheet.height-55),'未使用は今回の学習過程に対しての区別。実物の全画素・読解・シミの正解はない。',font=small,fill='#586574');draw.text((8,sheet.height-31),'朱の文字と面状の印影は別に評価する。茶色の本文は保存すべき情報である。',font=small,fill='#586574');sheet.save(dest/(cid+'_原画像_マスク'+('_修復' if base is not None else '')+'.png'))
  direct=[]
  for m,label in DIRECT.items():
   f=OUT/'independent_direct'/m/page/'result.png'
   if f.exists():arr=np.asarray(Image.open(f).convert('RGB'));assert arr.shape==orig.shape;direct.append((m,label,arr[sl],f))
  if direct:
   tiles=[('original','原画像',orig[sl],ROOT/r['original'])]+direct;dcw=max(dw,330);sheet=Image.new('RGB',(dcw*3,(dh+45)*((len(tiles)+2)//3)+115),'#faf9f5');draw=ImageDraw.Draw(sheet);draw.text((8,7),cid+'／直接の修復モデルの比較',font=font,fill='#20313a');draw.text((8,31),'同じ原寸ROI。DocResの重みは鏡像配布・作者ベンチマークの完全再現ではない。',font=small,fill='#586574')
   for i,(m,label,arr,f) in enumerate(tiles):
    xx=(i%3)*dcw;yy=58+(i//3)*(dh+45);draw.text((xx+5,yy),label,font=font,fill='#20313a');sheet.paste(Image.fromarray(arr).resize((dw,dh),Image.Resampling.NEAREST),(xx,yy+31));records.append(dict(case_id=cid,method=m,page=page,result_sha256=file_hash(f),ROI_original=r['ROI'],direct_restoration=m!='original',reading_GT=False))
   draw.text((8,sheet.height-32),'二値化は色を捨てる対照。全体256の拡大表示は失われた細部を回復していない。',font=small,fill='#586574');sheet.save(dest/(cid+'_直接修復.png'))
 (dest/'verification.json').write_text(json.dumps(dict(records=records,source_sha256=file_hash(__file__),cases_manifest_sha256=file_hash(OUT/'independent_cases_v2.json'),cases=a.cases,methods=a.methods,external_no_Japanese_DDRM=True,no_real_pixel_or_reading_GT=True),ensure_ascii=False,indent=2));print('Independent rendering records',len(records),flush=True)
if __name__=='__main__':main()
