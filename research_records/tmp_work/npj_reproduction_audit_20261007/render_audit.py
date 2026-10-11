from pathlib import Path
import sys,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tmp_work/npj_contact_research_20261007'))
from common import read,write,dump,sha,FONT
from render_full import records
BOXES={'T01':[812,421,450,450],'N01':[428,347,190,220],'N02':[520,890,520,560]}
def figure(id,arrays,labels,path,title,footer,scale=1):
 h,w=arrays[0].shape[:2];pw,ph=w*scale,h*scale
 canvas=Image.new('RGB',(len(arrays)*(pw+18)+14,ph+212),'#faf9f5');d=ImageDraw.Draw(canvas)
 d.text((14,14),title,font=ImageFont.truetype(FONT,23),fill='#20313b');panels=[]
 for i,(a,label) in enumerate(zip(arrays,labels)):
  assert a.shape==arrays[0].shape;x=14+i*(pw+18);y=100
  d.text((x,60),label,font=ImageFont.truetype(FONT,18),fill='#20313b');im=Image.fromarray(a)
  if scale>1:im=im.resize((pw,ph),Image.Resampling.NEAREST)
  canvas.paste(im,(x,y));assert np.array_equal(np.asarray(canvas)[y:y+ph,x:x+pw],np.repeat(np.repeat(a,scale,0),scale,1));panels.append([x,y,pw,ph])
 for j,line in enumerate(footer.split('\n')):d.text((14,ph+114+j*26),line,font=ImageFont.truetype(FONT,18),fill='#53636d')
 path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path);return dict(path=str(path.relative_to(OUT)),id=id,ROI=BOXES[id],integer_scale=scale,panels=panels)
def main():
 meta=[]
 for c in records():
  id=c['id']
  if id not in BOXES:continue
  x,y,w,h=BOXES[id];s=np.s_[y:y+h,x:x+w];raw=read(ROOT/c['source']['original']);md=ROOT/c['source']['npj_black'];ace=read(md.parent/'00_input.png')
  b=read(md)[:,:,0]<127;red=read(md.parent/'03_text_red.png')[:,:,0]<127
  v2=Path(str(md.parent).replace('split_dataset_csc_mask/','split_dataset_csc_mask_v2/'));b2=read(v2/'02_text_black.png')[:,:,0]<127;r2=read(v2/'03_text_red.png')[:,:,0]<127
  def paint(base,black,red):
   out=base.copy();out[black]=(20,50,210);out[red]=(220,25,150);return out
  scale=2 if id=='N01' else 1
  arr=[raw[s],ace[s],paint(raw,b,red)[s],paint(raw,b2,r2)[s]]
  meta.append(figure(id,arr,['原画像','分類入力（保存ACE）','保存CSC v1：青=黒、桃=赤','後年CSC v2：同じ色'],OUT/'figures'/(id+'_masks.png'),id+' マスク由来の確認',
   '全文書から生成したマスクの同位置比較。v1は再生成して4クラス全画素一致。\nv2は後年の変更であり、論文の正解マスクではない。本文と汚れの正解注釈なし。',scale))
  baseline=c['folder']/'output/baseline'/c['page']/'result'/(c['page']+'.png');arrays=[raw[s],read(baseline)[s]];labels=['原画像','前回：epoch350＋v1マスク'];sources=[str(ROOT/c['source']['original']),str(baseline)]
  for variant,label in [('local_char_weight','重みだけ model_char に変更'),('local_page_weight','重みだけ model_page に変更'),('v2_masks','マスクだけ後年v2に変更')]:
   op=OUT/'outputs'/variant/c['page']/'result'/(c['page']+'.png')
   if not op.exists():continue
   arrays.append(read(op)[s]);labels.append(label);sources.append(str(op))
  if len(arrays)>2:
   value=figure(id,arrays,labels,OUT/'figures'/(id+'_factor_controls.png'),id+' 原寸全ページ処理：重みとマスクを分けた診断',
    '旧ローカル重みも論文の配布重みとの一致は未確認。これらを正式なNPJ再現とは呼ばない。\n同じ背景・128pxパッチ・重なり32px・seed999。変更条件間の全時刻ノイズ一致は未保証。',scale)
   value['sources']=sources;meta.append(value)
 dump(OUT/'figures/rendering.json',meta);print('audit figures',len(meta),flush=True)
if __name__=='__main__':main()
