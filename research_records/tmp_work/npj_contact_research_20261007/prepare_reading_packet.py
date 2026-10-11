from common import *
from render_full import records
from apply_real_models import selection
import csv,time
IDS=['T01','T03','N01','F39','F48','F62','F59','F65','F67']
def main():
 out=HERE/'reading_packet';byid={c['id']:c for c in records()};frames={c['id']:c for c in selection()};rng=np.random.default_rng(104707);ids=list(IDS);rng.shuffle(ids)
 methods=['original','npj_baseline','sauvola_mask_ddrm','docres','automatic_local'];rng.shuffle(methods);key=[];source_panels=[]
 for j,id in enumerate(ids):
  c=byid[id];page=c['page'];x,y,w,h=c['detail_page_xywh']
  if id.startswith('F'):x+=100;y+=200;w=420;h=400
  if id=='F62':
   xx,yy,_,_=c['detail_page_xywh'];x,y,w,h=xx+560,yy+300,180,270
  elif id=='N01':x,y,w,h=428,347,190,220
  bp=c['folder']/'output/baseline'/page/'result'/(page+'.png');sp=c['folder']/'output/sauvola'/page/'result'/(page+'.png')
  fp=frames[id];fx,fy,fw,fh=fp['frame_xywh']
  sources={'original':(ROOT/c['source']['original'],[x,y,w,h]),'npj_baseline':(bp,[x,y,w,h]),'sauvola_mask_ddrm':(sp,[x,y,w,h]),'automatic_local':(HERE/'full_local'/id/'sauvola.png',[x,y,w,h]),'docres':(HERE/'real_models'/id/'docres.png',[x-fx,y-fy,w,h])}
  for group in range(5):
   method=methods[(j+group)%5];path,roi=sources[method];xx,yy,ww,hh=roi;array=read(path)[yy:yy+hh,xx:xx+ww];assert array.shape[:2]==(h,w)
   q=f'Q{j+1:02d}';dest=out/f'set_{chr(65+group)}'/(q+'.png');write(dest,array)
   key.append(dict(set=chr(65+group),question=q,id=id,page=page,method=method,image=str(dest.relative_to(out)),page_xywh=[x,y,w,h],source=str(path.relative_to(ROOT)),source_roi=roi,source_sha256=sha(path),role='contact_candidate' if id not in ['F59','F65','F67'] else 'control'))
 for group in range(5):
  folder=out/f'set_{chr(65+group)}';fields=['question','image','transcription','seconds','confidence_0_to_4','suspected_missing_strokes','suspected_added_marks','comment']
  with (folder/'回答票.csv').open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
   for j in range(len(ids)):writer.writerow(dict(question=f'Q{j+1:02d}',image=f'Q{j+1:02d}.png'))
 dump(out/'管理者用_割付と出典.json',dict(created=time.time(),key=key,source_sha256=sha(__file__),GT_transcription=False,reading_evaluation_completed=False,counterbalance='cyclic across5sets with shuffled case/method order; each set sees each location only once',perfect_method_blinding=False,all_panels_native=True))
 (out/'実施前に読む.md').write_text('''# 読解評価の準備用セット（評価は未実施）

接触候補6箇所と正常・近接対照3箇所。各画像は同じ原ページ座標の原寸PNG。条件は原画像、NPJ系基準、Sauvolaマスク＋DDRM、DocRes、自動局所修正の5つ。

回答者ごとにset_A〜Eの一つだけを渡す。同じ回答者に他セットや比較図を先に見せない。同一箇所を一種類だけ見せ、条件を回答者間で入れ替える割付である。画像様式から処理が推測できるため、完全な手法盲検ではない。

DDRM条件は原寸全ページを処理した出力、DocRes条件は768pxフレームを原寸タイルで処理した出力であり、入力文脈は同一ではない。今回の適用条件の比較として記録し、各手法の公式ベンチマーク再現や文脈を揃えた優劣とは扱わない。

表示倍率・端末・時間計測の開始終了を統一し、画像ごとに翻刻、時間、確信度、欠落・追加の疑いを回答票に記入する。読めない箇所は推測で埋めず保留を記録する。専門家が原資料情報を照合して別途正解を作成する。正解なしの自己申告文字数を正答率として扱わない。

管理者用JSONには条件名・座標・元ファイルのハッシュがある。回答者には先に渡さない。採点と方法間比較は未実施。本セットだけで一般化や読解改善を主張しない。
''')
 print('reading preparation',len(ids),'locations',len(key),'native images, no responses',flush=True)
if __name__=='__main__':
 while not (HERE/'full_render_complete.json').exists():time.sleep(5)
 main()
