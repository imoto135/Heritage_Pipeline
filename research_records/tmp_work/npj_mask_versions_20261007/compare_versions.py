from pathlib import Path
import json,sys,importlib.util,time,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PREV=ROOT/'tmp_work/npj_contact_research_20261007';AUDIT=ROOT/'tmp_work/npj_reproduction_audit_20261007'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def read(p):return np.asarray(Image.open(p).convert('RGB')).copy()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
 cv2.setNumThreads(2);protocol=json.loads((AUDIT/'protocol.json').read_text());modules={};stats=[];code={}
 for version in ['v2','v3']:
  p=ROOT/('modules/ddrm_npj_'+version+'/scripts/csc.py');spec=importlib.util.spec_from_file_location('inspect_'+version,p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);modules[version]=m;code[str(p.relative_to(ROOT))]=sha(p)
 roles={'black':'black_text_mask','red':'red_text_mask','degradation':'degradation_mask','background':'background_mask'}
 names={'black':'02_text_black.png','red':'03_text_red.png','degradation':'04_degradation.png','background':'05_background.png'}
 real={r['id']:r for r in json.loads((PREV/'real_models/protocol.json').read_text())['selected']}
 boxes={'T01':[812,421,450,450],'N01':[428,347,190,220],'N02':[520,890,520,560]};figures=[]
 for c in protocol['cases']:
  id=c['id'];md=(ROOT/c['source']['npj_black']).parent;v2dir=ROOT/c['v2_dir'];inp=cv2.imread(str(md/'00_input.png'));r={}
  for v,m in modules.items():
   start=time.monotonic();a=m.color_space_clustering(inp);r[v]={k:getattr(a,val)>127 for k,val in roles.items()};print(id,v,round(time.monotonic()-start,2),flush=True)
  stored={k:(cv2.imread(str(v2dir/n),0)<127 if k in ['black','red'] else cv2.imread(str(v2dir/n),0)>127) for k,n in names.items()}
  row=dict(id=id,current_v2_vs_stored_v2_changed={k:int(np.count_nonzero(r['v2'][k]!=stored[k])) for k in roles},current_v3_default_vs_current_v2_changed={k:int(np.count_nonzero(r['v3'][k]!=r['v2'][k])) for k in roles})
  stats.append(row);print(row,flush=True)
  x,y,w,h=boxes[id];roi=np.s_[y:y+h,x:x+w];baseline=ROOT/c['folder']/'output/baseline'/c['page']/'result'/(c['page']+'.png');sauvola=ROOT/c['folder']/'output/sauvola'/c['page']/'result'/(c['page']+'.png');v2=AUDIT/'outputs/v2_masks'/c['page']/'result'/(c['page']+'.png')
  sources=[ROOT/c['source']['original'],baseline,v2,sauvola];coords=[boxes[id]]*4;labels=['原画像','保存CSC v1＋DDRM','保存CSC v2＋同じDDRM','Sauvola黒＋同じDDRM']
  frame=real[id]['frame_xywh'];fx,fy,fw,fh=frame;doc=PREV/'real_models'/id/'docres.png';rawframe=read(PREV/'real_models'/id/'original.png');raw=read(sources[0]);assert np.array_equal(rawframe,raw[fy:fy+fh,fx:fx+fw])
  assert x>=fx and y>=fy and x+w<=fx+fw and y+h<=fy+fh,(id,boxes[id],frame)
  sources.append(doc);coords.append([x-fx,y-fy,w,h]);labels.append('DocRes二値化（DDRMなし）')
  arrays=[read(p)[cy:cy+ch,cx:cx+cw] for p,(cx,cy,cw,ch) in zip(sources,coords)];scale=2 if id=='N01' else 1;pw,ph=w*scale,h*scale
  canvas=Image.new('RGB',(len(arrays)*(pw+18)+14,ph+226),'#faf9f5');d=ImageDraw.Draw(canvas);d.text((14,14),id+'：現在の候補を同じ位置で確認',font=ImageFont.truetype(FONT,25),fill='#20313b');panels=[]
  for j,(a,label) in enumerate(zip(arrays,labels)):
   px=14+j*(pw+18);py=98;d.text((px,60),label,font=ImageFont.truetype(FONT,17),fill='#20313b');a=np.repeat(np.repeat(a,scale,0),scale,1);canvas.paste(Image.fromarray(a),(px,py));assert np.array_equal(np.asarray(canvas)[py:py+ph,px:px+pw],a);panels.append(dict(source=str(sources[j].relative_to(ROOT)),sha256=sha(sources[j]),source_ROI=coords[j],canvas_ROI=[px,py,pw,ph]))
  footer=['DDRM3条件は重み・背景・推論設定を固定。v2は黒・赤を変更、Sauvolaは黒のみ変更。','DocResは原寸の周辺フレーム入力→二値化→一定の紙色で表示。色の復元ではない。','これは代表例の目視比較。全資料の順位、文字保持・読解の正解率は未評価。']
  for j,line in enumerate(footer):d.text((14,ph+112+j*28),line,font=ImageFont.truetype(FONT,18),fill='#53636d')
  path=OUT/(id+'_現在の候補比較.png');canvas.save(path);figures.append(dict(path=path.name,integer_scale=scale,panels=panels,panel_pixels_exact=True))
 for p,h in code.items():assert sha(ROOT/p)==h
 dump(OUT/'mask_code_parity.json',dict(code_sha256=code,comparisons=stats,notes='Uses identical saved ACE input for current v2/v3 defaults; stored v2 may have other historical settings. No new DDRM inference.'))
 dump(OUT/'comparison_sources.json',dict(figures=figures,sources_unchanged=True))
if __name__=='__main__':main()
