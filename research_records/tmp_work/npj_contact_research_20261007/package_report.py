from common import *
from PIL import JpegImagePlugin
import time,zipfile,os
BG='#faf9f5';INK='#20313b';MUTED='#53636d'
def font(size):return ImageFont.truetype(FONT,size)
def wrap(draw,text,width,size=23):
 lines=[];line=''
 for char in text:
  if char=='\n':lines.append(line);line='';continue
  if draw.textlength(line+char,font=font(size))>width:
   if char in '。、，．！？：；）】」』' and line:
    lines.append(line[:-1]);line=line[-1]+char
   else:lines.append(line);line=char
  else:line+=char
 if line:lines.append(line)
 return lines
def block(draw,x,y,text,width,size=23,color=INK,leading=None):
 lines=wrap(draw,text,width,size)
 for i,line in enumerate(lines):draw.text((x,y+i*(leading or size+11)),line,font=font(size),fill=color)
 return y+len(lines)*(leading or size+11)
def overview():
 canvas=Image.new('RGB',(1840,1450),BG);d=ImageDraw.Draw(canvas)
 d.text((44,25),'本文に重なるシミ：改善と未解決の点',font=font(38),fill=INK)
 block(d,44,89,'別資料でも固定条件の改善例が見つかった。単純二値化・既存モデルとの差と、弱い情報の保持はまだ課題。',1740,24)
 evidence=Image.open(HERE/'evidence/N01_full_native.png').convert('RGB');evidence.thumbnail((1750,680),Image.Resampling.LANCZOS);canvas.paste(evidence,((1840-evidence.width)//2,180))
 y=900
 cards=[('探索の進展','6,022ページを機械探索、94ページを原画像で確認。接触候補は計12ページ・5資料群。成功数・正解率ではない。'),('分かったこと','NPJ系の黒い塊は、共通SauvolaやDocResで改善する例が多い。自動局所化は縁の残り・範囲不足がある。'),('次の主題','シミ除去と薄い筆記の保持、必要箇所の判断を一つの問題にする。実際の読解改善を先に確認する。')]
 for i,(title,body) in enumerate(cards):
  x=44+i*598;d.rounded_rectangle((x,y,x+572,y+288),radius=18,fill=['#e8eff0','#f1ebe1','#eee9f2'][i]);block(d,x+22,y+19,title,528,28);block(d,x+22,y+74,body,528,23)
 block(d,44,1252,'実施：原寸全ページ25ページ×2条件、原画像からの手法比較、資料群を分けた小規模学習。\n未達：実物での一般化、専門家による読解・筆画の正解評価、新手法としての既存法への優位。',1740,23)
 canvas.save(HERE/'figures/00_progress_overview.png')
def method_table():
 canvas=Image.new('RGB',(1840,1440),BG);d=ImageDraw.Draw(canvas);d.text((44,25),'成功した条件と、採用しない理由',font=font(38),fill=INK)
 rows=[('Sauvola／DocRes','接触シミを落とし、曲線・小文字を抽出する複数例。','色・濃淡の保存ではない。薄い人工線や灰色像を除く場合がある。'),
 ('自動範囲＋局所置換','マスク外はNPJ系出力と画素一致。','縁が残る、小さなシミを見落とす、不要な修正。'),
 ('RGB差分学習','学習型の人工汚れで誤差が減る。','色・形の変化に弱い。実シミが残り、正常部分も変わる。'),
 ('明るさ制約＋L1','画素位置を再描画しない設計を試せた。','3試行でゼロ補正へ崩壊。今回の最適化失敗。'),
 ('濃度の教師を追加','合成汚れ濃度で学ぶと崩壊を回避。','実物で白い斑。線が残っても濃さは変わる。'),
 ('汚れ種類・薄線を追加','合成の色・形への転用誤差が低下。','実物の斑は残る。微小斑点では入力より悪い。'),
 ('赤判定を厳格化','茶色を貼り戻す赤マスクを除ける。','薄い朱印も外す。単純な除外は採用しない。'),
 ('周波数・広い文脈','広い色むらや白い斑が滑らかになる。','シミの縁・文字周囲の明るい縁、汚れの残留。')]
 widths=[360,632,744];xs=[44,424,1076];headers=['方法','改善できたこと','失敗・限界'];y=112
 for x,w,h in zip(xs,widths,headers):d.rectangle((x,y,x+w,y+54),fill='#dfe7e9');block(d,x+12,y+8,h,w-24,25)
 y+=65
 for i,row in enumerate(rows):
  for x,w,txt in zip(xs,widths,row):
   d.rectangle((x,y,x+w,y+126),fill='#eeede7' if i%2==0 else BG);end=block(d,x+12,y+13,txt,w-24,22);assert end<=y+123
  y+=137
 block(d,44,1288,'「改善」は画像上の観察または人工汚れの誤差低下。専門家の読解改善、実物の一般化、筆画保持の保証とは区別する。',1740,24)
 canvas.save(HERE/'figures/05_methods_outcome.png')
def next_steps():
 canvas=Image.new('RGB',(1840,1260),BG);d=ImageDraw.Draw(canvas);d.text((44,25),'論文化に向けた次の判断',font=font(38),fill=INK)
 blocks=[('1　読解改善が必要な問題か','原画像、NPJ系基準、Sauvola、DocResを同じ文字位置で評価する。正しく読める量、時間、確信度を比較。画像がきれいなだけで成功にしない。'),
 ('2　保存すべき弱い情報があるか','薄い本文・注記・色のある筆記を原資料情報と専門家で確定する。灰色像を裏写りか表の筆画か決めずに成功・失敗へ数えない。'),
 ('3　必要な箇所だけを直せるか','条件を固定して未使用資料へ適用する。接触シミだけでなく、近接シミと正常細字も含め、除去効果と有害な変化を別に評価する。')]
 for i,(title,body) in enumerate(blocks):
  y=124+i*250;d.rounded_rectangle((44,y,1796,y+224),radius=18,fill=['#e8eff0','#f1ebe1','#eee9f2'][i]);block(d,72,y+22,title,1696,30);block(d,72,y+84,body,1696,26)
 block(d,44,952,'主張候補：原画像の観測情報から、基準処理の読みやすさ低下を必要箇所だけで修正し、弱い筆記を保つ。',1740,29)
 block(d,44,1080,'現状は、この両立・実物での一般化・読解改善・既存法への優位をまだ示せていない。\n一般化と無関係な別課題の数を増やすより、一つの問題で必要性と差を示す。',1740,24)
 canvas.save(HERE/'figures/06_paper_direction.png')
def make_pdf():
 order=['figures/00_progress_overview.png','evidence/T01_full_native.png','evidence/N01_full_native.png','evidence/F39_full_native.png','evidence/F48_full_native.png','evidence/F62_full_native.png',
  'evidence/N02_red_native.png','evidence/F41_red_native.png','real_models/N03/comparison.png','real_augmented/N01/comparison.png',
  'figures/01_synthetic_domains.png','figures/02_preservation_tradeoff.png','figures/03_controlled_strokes.png','figures/04_stress_domains.png',
  'frequency/N01/comparison.png','figures/05_methods_outcome.png','figures/06_paper_direction.png']
 pages=[]
 for i,rel in enumerate(order):
  im=Image.open(HERE/rel).convert('RGB');im.thumbnail((1770,1190),Image.Resampling.LANCZOS);page=Image.new('RGB',(1840,1320),BG);page.paste(im,((1840-im.width)//2,24));d=ImageDraw.Draw(page)
  d.text((36,1250),f'{i+1:02d}/{len(order):02d}　'+rel,font=font(18),fill=MUTED)
  d.text((36,1280),'PDFは閲覧用に縮小。細部は原寸PNGを確認。専門家の読解評価は未実施。',font=font(18),fill=MUTED);pages.append(page)
 pages[0].save(HERE/'接触シミ_2時間検証.pdf',save_all=True,append_images=pages[1:],resolution=150,quality=92,title='接触シミと読みやすさ：2時間検証',author='Heritage Pipeline research')
 dump(HERE/'pdf_rendering.json',dict(pages=order,count=len(order),resolution=150,presentation_resampling='LANCZOS for PDF only; evidence PNGs remain integer-scaled native pixels'))
def bundle():
 selected=set()
 for fn in ['2時間作業報告.md','研究方針.md','読解評価の実施案.md','verification.json','session_complete.json','pdf_rendering.json','接触シミ_2時間検証.pdf','full_comparison_stats.json','full_render_complete.json','contact_candidates.json','contact_candidates.csv','visual_output_review.json']:
  if (HERE/fn).exists():selected.add(HERE/fn)
 selected.update(HERE.glob('*.py'));selected.update((HERE/'figures').glob('*.png'));selected.update((HERE/'full_comparisons').glob('*.png'));selected.update((HERE/'evidence').rglob('*'))
 selected.add(HERE/'進捗報告_短縮版.md')
 selected.update((HERE/'reading_packet').rglob('*'))
 for folder in ['full_page','full_followup/B','full_followup/C','red_filter_study']:
  selected.update((HERE/folder).glob('*.json'))
 for folder in ['other_book_screen/interim','other_book_screen/final']:
  selected.update((HERE/folder).glob('*.json'))
 for folder in ['synthetic_eval','stress_eval']:
  selected.update((HERE/folder).glob('*.csv'));selected.update((HERE/folder).glob('*.json'))
  for path in (HERE/folder).glob('*/protocol.json'):selected.add(path)
  for path in (HERE/folder).glob('*/summary.json'):selected.add(path)
 for folder in ['learned','controlled_strokes','controlled_strokes_augmented','context_scale','red_audit','real_models','real_augmented']:
  selected.update((HERE/folder).glob('*.json'))
  if folder=='learned':
   for path in (HERE/folder).glob('*/*.json'):selected.add(path)
 for id in ['T00','T01','T02','T03','T04','T06','N01','N02','F39','F48','F60','F62','N03','N05','F35','F41','F65','F67']:
  selected.add(HERE/'real_models'/id/'comparison.png')
 for id in ['N01','N02','F62','N05']:selected.add(HERE/'real_augmented'/id/'comparison.png')
 selected.add(HERE/'frequency/N01/comparison.png');selected.add(HERE/'red_audit/F41/whole_page.png')
 selected.add(HERE/'context_scale/T01/gain_od_regression_comparison.png')
 readme=HERE/'資料の読み方.txt';readme.write_text('最初に2時間作業報告.mdとfigures/00_progress_overview.pngを参照してください。\nPDFは17ページの代表比較、細部はevidence/とfull_comparisons/のPNGを確認してください。\nreading_packet/は未実施の読解評価の準備です。管理者用の割付は回答者へ先に渡さないでください。\n学習済み重み、元の全ページ画像、巨大な中間配列はZIPに含みません。再実行には同じHeritage_Pipelineワークスペースと依存資源が必要です。\n以前の報告への相対リンクはワークスペース内で参照するためのものです。\n');selected.add(readme)
 selected={p for p in selected if p.is_file()};dest=HERE/'接触シミ_2時間検証_資料.zip';tmp=HERE/'bundle.tmp.zip'
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
  for p in sorted(selected):z.write(p,str(p.relative_to(HERE)))
 os.replace(tmp,dest)
 with zipfile.ZipFile(dest) as z:assert z.testzip() is None
 dump(HERE/'bundle_verification.json',dict(files=len(selected),zip_bytes=dest.stat().st_size,zip_sha256=sha(dest),entries={str(p.relative_to(HERE)):sha(p) for p in sorted(selected)},integrity=True))
 print('bundle',len(selected),'files',round(dest.stat().st_size/1e6,1),'MB',flush=True)
def main():
 assert (HERE/'full_render_complete.json').exists();overview();method_table();next_steps();make_pdf();bundle()
if __name__=='__main__':main()
