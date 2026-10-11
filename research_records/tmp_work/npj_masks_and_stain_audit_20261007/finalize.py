from pathlib import Path
import json,hashlib,re,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont,JpegImagePlugin
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;PREV=ROOT/'tmp_work/npj_contact_research_20261007';AUDIT=ROOT/'tmp_work/npj_reproduction_audit_20261007';FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def read(p):return np.asarray(Image.open(p).convert('RGB'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
 checks={};meta=json.loads((OUT/'rendering.json').read_text());protocol=json.loads((AUDIT/'protocol.json').read_text());cases={c['id']:c for c in protocol['cases']}
 for f in meta:
  im=read(OUT/f['path']);ok=True
  for p in f['panels']:
   if f.get('kind')=='colour':
    path=OUT/p['source'];a=Image.open(path).convert('RGB');a=a.resize((256,256),Image.Resampling.LANCZOS) if p['display_resize'] else a;arr=np.asarray(a);xy=p['canvas_xywh'];ok &= sha(path)==p['sha256']
   else:
    path=OUT/p['path'];arr=read(path);arr=np.repeat(np.repeat(arr,p['scale'],0),p['scale'],1);xy=p['canvas_xywh'];ok &= sha(path)==p['sha256']
   x,y,w,h=xy;ok &= np.array_equal(im[y:y+h,x:x+w],arr)
  checks[f['path']+'_panels_exact']=bool(ok)
  if 'source_sha256' in f:checks[f['path']+'_sources_unchanged']=all(sha(ROOT/p)==h for p,h in f['source_sha256'].items())
  if f.get('kind')!='colour':
   id=Path(f['path']).name.split('_')[0];c=cases[id];x,y,w,h=f['ROI'];roi=np.s_[y:y+h,x:x+w];md=(ROOT/c['source']['npj_black']).parent;v2=ROOT/c['v2_dir'];sp=ROOT/c['folder']/'masks/sauvola'/c['page'];all_ok=True
   for method,folder in [('v1',md),('v2',v2),('sauvola',sp)]:
    for key,name in [('black','02_text_black.png'),('red','03_text_red.png')]:
     expected=np.repeat((read(folder/name)[:,:,0]<127)[:,:,None]*255,3,2).astype('uint8')[roi];all_ok &= np.array_equal(read(OUT/'panels'/id/method/(key+'.png')),expected)
   fx,fy,fw,fh=f['frame_xywh'];dp=np.load(PREV/'real_models'/id/'docres_probability.npy');expected=np.repeat((dp<.5)[:,:,None]*255,3,2).astype('uint8')[y-fy:y-fy+h,x-fx:x-fx+w];all_ok &= np.array_equal(read(OUT/'panels'/id/'docres/black.png'),expected)
   checks[id+'_displayed_masks_equal_actual_masks']=bool(all_ok)
 results=json.loads((OUT/'stain_results.json').read_text());complete=json.loads((OUT/'stain_complete.json').read_text());prov=json.loads((OUT/'official_provenance.json').read_text());newprotocol=json.loads((OUT/'stain_protocol.json').read_text())
 checks['fresh_official_weight_identical_previous']=prov['fresh_weight_matches_previous'] and sha(OUT/'official/StainDoc_fresh.pth')==prov['weight_sha256']
 checks['release_model_and_used_model_byte_equal']=all(v['release_vs_current_equal'] and v['old_used_vs_current_equal'] for v in prov['files'].values())
 checks['four_cases_previous_tiled_PNGs_pixel_exact']=len(results)==4 and all(r['fresh_tiles_vs_previous_max_255']==0 for r in results)
 checks['actual_loader_inputs_verified_all_cases']=all(r['official_loader_manual_tensor_equal'] for r in results)
 checks['adapter_matches_direct_within_numerical_rounding']=max(r['adapter_vs_direct_float_max'] for r in results)<2e-6 and max(r['adapter_vs_direct_float_mean'] for r in results)<2e-7
 checks['CPU_GPU_direct_match_within_1e_minus_4']=next(r for r in results if r['id']=='N01')['CPU_GPU_max']<1e-4
 checks['torchvision_save_matches_manual_within_one_level']=all(r['save_image_vs_manual_quantization_max_255']<=1 for r in results)
 checks['new_stain_code_weights_unchanged']=all(sha(ROOT/p)==h for p,h in newprotocol['dependencies'].items())
 checks['saved_model_outputs_unchanged']=all(sha(OUT/p)==h for p,h in complete['outputs'].items())
 report=OUT/'マスク比較とStainRestorer確認報告.md';checks['report_links_exist']=all((OUT/p).exists() for p in re.findall(r'\]\(([^)]+)\)',report.read_text()) if not p.startswith('https://'))
 assert all(checks.values()),[k for k,v in checks.items() if not v]
 dump(OUT/'verification.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),no_readability_GT=True,official_paired_benchmark=False))
 summary=Image.new('RGB',(1800,1200),'#faf9f5');d=ImageDraw.Draw(summary);lines=[('マスクと修復結果／StainRestorer実装確認',38),('同じ位置で黒・赤マスクと処理結果を上下に確認できるPNGを作成。',28),('CSC v1／v2、Sauvola、DocRes、StainRestorerを並べて表示。',28),('白＝候補に統一。StainRestorerは明示的なマスクを使わない。',28),('公開StainDoc重みを再取得：以前のファイルと完全一致。',28),('公開時・現行・以前使用したモデル本体も一致。',28),('公式DataLoaderValで4例を再推論：公式256px入力でも色変化が残る。',28),('RGB/BGR・正規化・最終出力選択・保存の取り違えは見つからなかった。',28),('独自アダプターとの差は数値丸めの範囲。CPUとGPUも整合。',28),('新しい重みで以前のタイル出力を再計算：4例すべてPNG全画素一致。',28),('原寸768pxの一括推論でも変色が残る。タイルだけでは説明できない。',28),('これは今回の公開重み・入力条件の結果。モデル全体を一律に却下しない。',28),('色補正を加えた成功画像は作っていない。内部原因は未確定。',28),('原寸PNG、個別マスク、出典・座標・ハッシュは同梱報告とJSONを参照。',25),('2026年10月7日／読解の正解注釈なし／公式paired benchmarkの再現ではない',24)]
 y=35
 for t,s in lines:d.text((28,y),t,font=ImageFont.truetype(FONT,s),fill='#20313b');y+=73
 summary.save(OUT/'figures/確認要点.png');pages=[summary]
 files=[OUT/'figures'/(id+'_マスクと処理結果.png') for id in ['N01','T01','N02']]+[OUT/'figures'/(id+'_全ページ対応.png') for id in ['N01','T01','N02']]+[OUT/'figures'/(id+'_StainRestorer色診断.png') for id in ['N01','T01','N02','F41']]+[OUT/'figures/自作文書対照.png']
 for p in files:
  im=Image.open(p).convert('RGB');page=Image.new('RGB',(1800,1200),'#faf9f5');im.thumbnail((1760,1100),Image.Resampling.LANCZOS);page.paste(im,((1800-im.width)//2,(1150-im.height)//2));ImageDraw.Draw(page).text((24,1160),'原寸で確認する場合は、対応するPNGを開く。色補正なし。',font=ImageFont.truetype(FONT,23),fill='#53636d');pages.append(page)
 pages[0].save(OUT/'マスクと修復結果_実装確認.pdf',save_all=True,append_images=pages[1:],resolution=150)
 bundle=[p for folder in ['figures','panels'] for p in (OUT/folder).rglob('*.png')]+[p for p in OUT.glob('*') if p.suffix in ['.md','.json','.py','.pdf']]+[p for p in (OUT/'stain').glob('*/*.png')]+[p for p in (OUT/'controls').rglob('*.png')]
 dump(OUT/'資料ハッシュ.json',{str(p.relative_to(OUT)):sha(p) for p in bundle});bundle.append(OUT/'資料ハッシュ.json')
 with zipfile.ZipFile(OUT/'マスクと修復結果_資料.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in bundle:z.write(p,p.relative_to(OUT))
 with zipfile.ZipFile(OUT/'マスクと修復結果_資料.zip') as z:assert z.testzip() is None
 print(json.dumps(dict(checks_passed=sum(checks.values()),checks_total=len(checks),pdf_pages=len(pages),zip_files=len(bundle),zip_bytes=(OUT/'マスクと修復結果_資料.zip').stat().st_size),ensure_ascii=False))
if __name__=='__main__':main()
