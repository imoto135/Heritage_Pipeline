from common import *
from render_full import records
import time
def figure(path,panels,title,footer,scale=1):
 w,h=panels[0]['size'];assert all(p['size']==[w,h] for p in panels);W=len(panels)*(w*scale+24)+24;H=130+h*scale+105
 canvas=Image.new('RGB',(W,H),'#faf9f5');d=ImageDraw.Draw(canvas);f=ImageFont.truetype(FONT,24);sm=ImageFont.truetype(FONT,18);d.text((16,12),title,font=f,fill='#20313b');manifest=[]
 for i,p in enumerate(panels):
  x,y,cw,ch=p['xywh'];a=read(ROOT/p['source'])[y:y+ch,x:x+cw];assert a.shape[:2]==(h,w)
  im=Image.fromarray(a);im=im.resize((w*scale,h*scale),Image.Resampling.NEAREST);px=16+i*(w*scale+24);py=130
  d.text((px,64),p['label'],font=sm,fill='black');canvas.paste(im,(px,py));assert np.array_equal(np.asarray(canvas)[py:py+h*scale,px:px+w*scale],np.asarray(im))
  raw=HERE/'evidence/native'/path.stem/f'{i:02d}.png';write(raw,a);manifest.append(dict(p,source_sha256=sha(ROOT/p['source']),panel_canvas_xywh=[px,py,w*scale,h*scale],integer_scale=scale,native_copy=str(raw.relative_to(HERE))))
 for i,line in enumerate(footer.split('\n')):d.text((16,130+h*scale+12+i*25),line,font=sm,fill='#53636d')
 path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path);return dict(figure=str(path.relative_to(HERE)),panels=manifest)
def panel(source,xywh,label):return dict(source=str(Path(source).relative_to(ROOT)),xywh=xywh,size=xywh[2:],label=label)
def main():
 byid={c['id']:c for c in records()};figures=[]
 for id,roi,scale in [('T01',[812,421,450,450],1),('N01',[428,347,190,220],2),('F39',None,1),('F48',None,1),('F62',None,1)]:
  c=byid[id];folder=c['folder'];page=c['page'];bp=folder/'output/baseline'/page/'result'/(page+'.png');sp=folder/'output/sauvola'/page/'result'/(page+'.png');lp=HERE/'full_local'/id/'sauvola.png'
  if not all(p.exists() for p in [bp,sp,lp]):continue
  if roi is None:
   x,y,w,h=c['detail_page_xywh'];roi=[x+100,y+220,420,360]
   if id=='F62':roi=[x+560,y+300,180,270];scale=2
  panels=[panel(ROOT/c['source']['original'],roi,'原画像'),panel(bp,roi,'NPJ系基準'),panel(sp,roi,'Sauvolaマスク＋DDRM'),panel(lp,roi,'自動範囲＋局所二値化')]
  figures.append(figure(HERE/'evidence'/f'{id}_full_native.png',panels,id+' '+page+' 全ページ処理の同位置比較',
   f'原ページ座標 {roi}。表示は整数{scale}倍最近傍、色・濃さ・輪郭の加工なし。\n文字とシミの分離を確認する図。専門家による読解・筆画の正解判定は未実施。',scale))
 for id,roi,scale in [('N05',[300,165,170,245],2),('F67',[0,0,220,460],1)]:
  folder=HERE/'real_models'/id;paths=[folder/(n+'.png') for n in ['original','sauvola','docres','residual','gain_od_regression']]
  labels=['原画像','Sauvola','DocRes','RGB差分学習','濃度＋明るさ制約']
  figures.append(figure(HERE/'evidence'/f'{id}_thin_native.png',[panel(p,roi,label) for p,label in zip(paths,labels)],id+' 細い線・小さな文字の対照',
   f'原寸フレーム内座標 {roi}、整数{scale}倍最近傍。色調整なし。\n原画像で見える細い部分を照合する図。真の筆画・読解の正解は未注釈。',scale))
 f=HERE/'red_filter_study/complete.json'
 if f.exists():
  protocol=json.loads((f.parent/'protocol.json').read_text())
  for c in protocol['selected']:
   id=c['id'];page=c['page'];roi=c['detail_xywh'];paths=[f.parent/'inputs'/id/(page+'.png')]+[f.parent/'output'/v/page/'result'/(page+'.png') for v in protocol['variants']]
   figures.append(figure(HERE/'evidence'/f'{id}_red_native.png',[panel(p,roi,label) for p,label in zip(paths,['原画像','保存黒＋保存赤','保存黒＋絶対色赤','Sauvola黒＋絶対色赤'])],id+' マスクの役割と色判定の限界',
    '同じ原寸フレーム、コード・重み・乱数初期値。マスクだけを変更。\n厳しい色判定で薄い朱印も候補から外れる。除去への直接採用を意味しない。',1))
 dump(HERE/'evidence/rendering.json',dict(created=time.time(),figures=figures,source_sha256=sha(__file__),all_panels_pixel_equal=True,integer_scaling_only=True));print('native evidence',len(figures),flush=True)
if __name__=='__main__':main()
