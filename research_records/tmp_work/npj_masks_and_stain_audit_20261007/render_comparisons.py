from pathlib import Path
import json,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PREV=ROOT/'tmp_work/npj_contact_research_20261007';AUDIT=ROOT/'tmp_work/npj_reproduction_audit_20261007'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';BOXES={'T01':[812,421,450,450],'N01':[428,347,190,220],'N02':[520,890,520,560]}
def read(p):return np.asarray(Image.open(p).convert('RGB')).copy()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,a):p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(a.astype('uint8')).save(p)
def font(n):return ImageFont.truetype(FONT,n)
def textcell(size,lines):
 im=Image.new('RGB',size,'#e8e8e5');d=ImageDraw.Draw(im)
 for j,line in enumerate(lines):d.text((14,28+j*31),line,font=font(18),fill='#53636d')
 return np.asarray(im)
def main():
 assert (OUT/'stain_complete.json').exists();cases=json.loads((AUDIT/'protocol.json').read_text())['cases'];real={c['id']:c for c in json.loads((PREV/'real_models/protocol.json').read_text())['selected']};manifest=[]
 for c in cases:
  id=c['id'];x,y,w,h=BOXES[id];sl=np.s_[y:y+h,x:x+w];frame=real[id]['frame_xywh'];fx,fy,fw,fh=frame;sframe=np.s_[y-fy:y-fy+h,x-fx:x-fx+w];orig=ROOT/c['source']['original'];raw=read(orig);md=(ROOT/c['source']['npj_black']).parent;v2=ROOT/c['v2_dir'];sp=ROOT/c['folder']/'masks/sauvola'/c['page'];page=c['page'];old=ROOT/c['folder']/'output/baseline'/page/'result'/(page+'.png');new=AUDIT/'outputs/v2_masks'/page/'result'/(page+'.png');sau=ROOT/c['folder']/'output/sauvola'/page/'result'/(page+'.png');doc=PREV/'real_models'/id/'docres.png';stain=OUT/'stain'/id/'native_whole_final.png'
  assert np.array_equal(read(PREV/'real_models'/id/'original.png'),raw[fy:fy+fh,fx:fx+fw])
  def bitmap(p):return np.repeat((read(p)[:,:,0]<127)[:,:,None]*255,3,2).astype('uint8')[sl]
  probability=PREV/'real_models'/id/'docres_probability.npy';dp=np.load(probability);dm=np.repeat((dp<.5)[:,:,None]*255,3,2).astype('uint8')[sframe]
  has_red=bool((read(md/'03_text_red.png')[:,:,0][sl]<127).any())
  rowkeys=['black','red','result'] if has_red else ['black','result'];labels=['原画像（参照）','CSC v1＋DDRM','CSC v2＋DDRM','Sauvola黒＋DDRM','DocRes 二値化','StainRestorer 原寸全体'];columns=[]
  for name,black,red,result in [('original',raw[sl],None,raw[sl]),('v1',bitmap(md/'02_text_black.png'),bitmap(md/'03_text_red.png'),read(old)[sl]),('v2',bitmap(v2/'02_text_black.png'),bitmap(v2/'03_text_red.png'),read(new)[sl]),('sauvola',bitmap(sp/'02_text_black.png'),bitmap(sp/'03_text_red.png'),read(sau)[sl]),('docres',dm,None,read(doc)[sframe]),('stain',None,None,read(stain)[sframe])]:
   values=dict(black=black,red=red,result=result)
   if name=='original':values['red']=raw[sl]
   columns.append((name,values))
  scale=2 if id=='N01' else 1;pw,ph=w*scale,h*scale;left=158;header=115;gap=40;W=left+6*(pw+18)+14;H=header+len(rowkeys)*(ph+gap)+144
  canvas=Image.new('RGB',(W,H),'#faf9f5');d=ImageDraw.Draw(canvas);d.text((14,14),id+' マスクから出力まで：同じ位置で確認',font=font(25),fill='#20313b');panels=[]
  for j,label in enumerate(labels):d.text((left+j*(pw+18),66),label,font=font(18),fill='#20313b')
  for i,key in enumerate(rowkeys):
   yy=header+i*(ph+gap);rowlabel={'black':['黒文字マスク','白＝文字候補'],'red':['赤文字マスク','白＝赤候補'],'result':['処理結果','同じ座標']}[key]
   for k,line in enumerate(rowlabel):d.text((14,yy+20+k*30),line,font=font(18),fill='#20313b')
   for j,(name,values) in enumerate(columns):
    a=values[key];placeholder=a is None
    if placeholder:a=textcell((w,h),['明示マスクなし','RGBを直接修復'] if name=='stain' else ['赤の独立分類なし','朱の保存とは別'])
    path=OUT/'panels'/id/name/(key+'.png');write(path,a);im=Image.fromarray(a);im=im.resize((pw,ph),Image.Resampling.NEAREST);xx=left+j*(pw+18);canvas.paste(im,(xx,yy));assert np.array_equal(np.asarray(canvas)[yy:yy+ph,xx:xx+pw],np.repeat(np.repeat(a,scale,0),scale,1));panels.append(dict(method=name,row=key,placeholder=placeholder,path=str(path.relative_to(OUT)),sha256=sha(path),canvas_xywh=[xx,yy,pw,ph],scale=scale))
  foot=['黒・赤は表示を白＝候補に統一。保存CSCの0＝候補を反転表示。未注釈なので正解マスクではない。','DDRMは全ページ処理。DocRes／StainRestorerは原寸768pxの周辺フレーム。DocRes結果は二値候補の一定紙色表示。','StainRestorerは再取得した公式重みで768pxを一度に推論。色補正なし。今回の通常DDRMは黒・赤を使用。']
  if not has_red:foot.append('この切り出しのCSC／Sauvola赤候補は0画素。赤マスク段は省略（個別パネルには保存）。')
  for k,line in enumerate(foot):d.text((14,H-126+k*28),line,font=font(18),fill='#53636d')
  path=OUT/'figures'/(id+'_マスクと処理結果.png');path.parent.mkdir(exist_ok=True);canvas.save(path)
  for name,values in columns:
   if values['red'] is not None and not has_red:write(OUT/'panels'/id/name/'red.png',values['red'])
  sources=[orig,md/'02_text_black.png',md/'03_text_red.png',v2/'02_text_black.png',v2/'03_text_red.png',sp/'02_text_black.png',sp/'03_text_red.png',old,new,sau,doc,probability,stain]
  manifest.append(dict(path=str(path.relative_to(OUT)),ROI=BOXES[id],frame_xywh=frame,panels=panels,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}))
  # Full-page preview keeps mask/output pairs adjacent and uses one display scale.
  sources=[orig,md/'02_text_black.png',old,v2/'02_text_black.png',new];labels=['原画像','v1黒：白＝候補','v1修復後','v2黒：白＝候補','v2修復後'];target_h=740;target_w=round(raw.shape[1]*target_h/raw.shape[0]);sheet=Image.new('RGB',(5*(target_w+18)+20,870),'#faf9f5');draw=ImageDraw.Draw(sheet);draw.text((14,12),id+' 全ページ：マスクと修復後の対応（表示縮小）',font=font(23),fill='#20313b')
  for j,(p,label) in enumerate(zip(sources,labels)):
   a=read(p)
   if j in [1,3]:a=np.repeat((a[:,:,0]<127)[:,:,None]*255,3,2).astype('uint8')
   im=Image.fromarray(a).resize((target_w,target_h),Image.Resampling.NEAREST if j in [1,3] else Image.Resampling.LANCZOS);xx=14+j*(target_w+18);draw.text((xx,55),label,font=font(17),fill='#20313b');sheet.paste(im,(xx,90))
  draw.text((14,837),'各列の元サイズは同じ。細部の確認には同位置の原寸比較PNGを参照。',font=font(18),fill='#53636d');sheet.save(OUT/'figures'/(id+'_全ページ対応.png'))
 # Colour controls: author-size input, fresh model heads, native frame inference.
 for id in ['T01','N01','N02','F41']:
  f=OUT/'stain'/id;paths=[f/'official256_input.png',f/'official256_final.png',f/'official256_coarse.png',f/'native_whole_final.png',f/'native_tiles_final.png'];labels=['公式ローダー入力256','公式設定：最終出力','公式設定：中間出力','原寸768：一括推論','原寸512：タイル推論'];canvas=Image.new('RGB',(5*278+20,425),'#faf9f5');d=ImageDraw.Draw(canvas);d.text((14,12),id+' StainRestorer：読み込み・サイズ・出力段階の確認',font=font(23),fill='#20313b');panels=[]
  for j,(p,label) in enumerate(zip(paths,labels)):
   im=Image.open(p).convert('RGB');im=im.resize((256,256),Image.Resampling.LANCZOS) if im.size!=(256,256) else im;xx=14+j*278;d.text((xx,65),label,font=font(17),fill='#20313b');canvas.paste(im,(xx,104));panels.append(dict(source=str(p.relative_to(OUT)),sha256=sha(p),display_resize=Image.open(p).size!=(256,256),display_size=[256,256],canvas_xywh=[xx,104,256,256]))
  d.text((14,375),'右2列は表示のみ256へ縮小。すべて色補正なし。公式256条件は細線保持の比較ではなく色診断。',font=font(18),fill='#53636d');path=OUT/'figures'/(id+'_StainRestorer色診断.png');canvas.save(path);manifest.append(dict(path=str(path.relative_to(OUT)),panels=panels,kind='colour'))
 # Known synthetic input/GT, kept distinct from author benchmark examples.
 sheet=Image.new('RGB',(4*278+20,3*320+120),'#faf9f5');d=ImageDraw.Draw(sheet);d.text((14,12),'自作の対照文書：紙色による挙動（公式ベンチマークではない）',font=font(23),fill='#20313b')
 for i,name in enumerate(['white','cream','aged']):
  for j,(suffix,label) in enumerate([('clean_input','無追加入力'),('clean_output','無追加：モデル出力'),('stained_input','人工シミ入力'),('stained_output','人工シミ：モデル出力')]):
   im=Image.open(OUT/'controls'/name/(suffix+'.png'));xx=14+j*278;yy=100+i*320;d.text((xx,yy-31),name+' '+label,font=font(17),fill='#20313b');sheet.paste(im,(xx,yy))
 d.text((14,1068),'既知の人工条件の挙動確認。実物のシミ除去・読みやすさ・論文掲載精度の評価ではない。',font=font(17),fill='#53636d');sheet.save(OUT/'figures/自作文書対照.png')
 (OUT/'rendering.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('Comparison sheets and source-linked panels complete',flush=True)
if __name__=='__main__':main()
