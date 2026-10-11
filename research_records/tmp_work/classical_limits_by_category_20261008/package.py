import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import build as b

H=b.HERE;R=b.ROOT
cs=json.loads((H/'candidate_protocol.json').read_text())['cases'];byid={c['id']:c for c in cs}
definitions=[
 ('01','R_L01','極細・薄い筆画','中央列の細い垂れ線と輪郭',[1320,540,230,370],
  'Sauvolaは細線を断片として残す。Wolfは中央の薄い線を大きく失う。Howeは上の垂れ線を保持するが、下の薄い輪郭には切れ目が残る。','3法で細線保持に差。Howeでも全ての薄い輪郭はつながらない。','visible_stroke_loss'),
 ('02','D_thin_black_water','広い水染みと低コントラスト文字','中央から左の薄い本文と、右の濃い本文',[580,1800,300,350],
  '濃い右列は3法とも比較的残る。中央・左の薄い文字は3法とも複数の部分が欠ける。Wolfの欠けが目立ち、Howeでも断片化が残る。','局所の数画素の切れ目より広い、文字全体の保持の問題。','visible_stroke_loss'),
 ('03','R_S01','本文に接触する濃い染み','上の罫線から中央の字へ重なる染み',[970,280,250,360],
  'Sauvola・Wolfは染みの粒状像を本文の脇に残す。Howeは粒状像を減らす一方、染み下の中央筆画を途中で切り、上の字の一部を太くまとめる。','背景の除去と、接触部の細い筆画を残すことが両立していない。','stain_ink_contact'),
 ('04','D_dense_stain','墨と似た濃い斑点','本文の間にある楕円状の暗い斑点',[400,540,260,270],
  '原画像の濃い茶色の楕円は、Sauvola・Wolfでは黒い塊として残るが、Howeでは除去されている。前回の「3法とも残る」という記述は目視の誤りであり訂正する。斑点は本文の間にあるが、筆画そのものとは離れている。','Howeが斑点を除去できる成功対照。接触汚れや読解改善とは別に評価する。','howe_stain_removal_success'),
 ('05','R_N02','密な粒状像と暗い帯','右の暗い帯と、本文・罫線に重なる粒状像',[1540,2350,320,320],
  '3法とも本文近傍の粒状像と右側の暗い帯を残す。Howeで小粒が減る部分はあるが、密な範囲は黒い像のまま。入力自体が既に強い白黒像に見える。','濃淡の手掛かりが乏しい入力。粒状像の材質と、帯が撮影影かは未確定。','material_uncertain'),
 ('06','R_L05b','紙面の欠損が文字と重なる','右下の白い欠損らしい範囲と筆画',[1200,1510,350,310],
  '3法とも右下の文字を連続した形として残せない。Wolfは欠損外の薄い部分も削る。Sauvola・Howeでは欠損境界の像も残る。','原画像で筆画情報を確認できない部分と、処理で追加された欠けを区別する必要がある。','missing_source_information'),
 ('07','G_200015843','裏写り候補と薄い表文字の混在','左側の薄い本文・細い添え書きと灰色像',[800,1400,300,340],
  '3法とも広い灰色像を減らす一方、薄い筆画の切れ目が残る。Howeは左側の細線を比較的つなぐ。灰色像の全てを不要と確定していない。','材質・筆画の正解未確定。裏写り除去の成功／失敗は判定保留。','material_uncertain'),
]
selected=[];figures=[];originals={}
for ident,seed,title,focus,detail,obs,limit,kind in definitions:
    c=dict(byid[seed],id=ident+'_'+title,seed_id=seed,category=title,focus=focus,detail_xywh=detail,observation=obs,limit=limit,judgment=kind)
    raw=b.rgb(R/c['original']);W,Hh=c['image_wh'];x,y,w,h=c['roi_xywh']
    originals[c['original']]=c['source_sha256'];assert b.sha(R/c['original'])==c['source_sha256']
    masks={k:np.array(Image.open(H/'masks'/c['page']/(k+'.png')).convert('L'))==0 for k in ['sauvola','wolf','howe']}
    for m in masks.values(): assert m.shape==raw.shape[:2]
    native=H/'native'/c['id'];native.mkdir(parents=True,exist_ok=True)
    Image.fromarray(raw[y:y+h,x:x+w]).save(native/'original.png')
    for key,m in masks.items():Image.fromarray(np.where(m[y:y+h,x:x+w],0,255).astype('uint8')).save(native/(key+'.png'))
    c['comparison']=b.render(c,raw,masks,H/'comparisons',ident+' '+title+' / '+c['page']);figures.append(c['comparison'])
    dc=dict(c,id=c['id']+'_詳細2倍',roi_xywh=detail,scale=2)
    c['detail']=b.render(dc,raw,masks,H/'details',ident+' '+title+'：注目部分の2倍表示');figures.append(c['detail'])
    # Position markings belong only to the auxiliary context, never native crops.
    context=Image.fromarray(raw);context.thumbnail((650,900));sx=context.width/W;sy=context.height/Hh
    d=ImageDraw.Draw(context);d.rectangle((x*sx,y*sy,(x+w)*sx,(y+h)*sy),outline='#d32b30',width=4)
    xx,yy,ww,hh=detail;d.rectangle((xx*sx,yy*sy,(xx+ww)*sx,(yy+hh)*sy),outline='#ed9b00',width=3)
    canvas=Image.new('RGB',(max(680,context.width+30),context.height+100),'#faf9f5');d=ImageDraw.Draw(canvas)
    b.text(d,(15,10),ident+' '+title,22);b.text(d,(15,44),c['page']+' / 赤：広域比較、橙：詳細2倍',16);canvas.paste(context,(15,80))
    (H/'contexts').mkdir(exist_ok=True);canvas.save(H/'contexts'/(c['id']+'.png'))
    selected.append(c)

# Read every exported comparison again and check the PNG's actual panel pixels.
checked=0
for c in selected:
    raw=b.rgb(R/c['original'])
    for fig,roi in [(c['comparison'],c['roi_xywh']),(c['detail'],c['detail_xywh'])]:
        saved=np.array(Image.open(H/fig['path']).convert('RGB'));x,y,w,h=roi
        for panel in fig['panels']:
            key=panel['method'];a=raw[y:y+h,x:x+w] if key=='original' else np.repeat(np.array(Image.open(H/'masks'/c['page']/(key+'.png')).convert('L'))[y:y+h,x:x+w,None],3,axis=2)
            scale=panel['scale'];a=np.repeat(np.repeat(a,scale,0),scale,1);px,py,pw,ph=panel['xywh']
            assert np.array_equal(saved[py:py+ph,px:px+pw],a);checked+=1
    x,y,w,h=c['roi_xywh'];assert np.array_equal(b.rgb(H/'native'/c['id']/'original.png'),raw[y:y+h,x:x+w])

# Compact side-by-side overview. All detailed comparisons above preserve pixels.
cw,rh=300,390;overview=Image.new('RGB',(4*cw+40,100+rh*len(selected)),'#faf9f5');d=ImageDraw.Draw(overview)
b.text(d,(15,10),'黒文字の課題別難例：原画像 / Sauvola / Wolf / Howe移植',25)
b.text(d,(15,47),'一覧は縮小。原寸比較・2倍詳細で確認。⑦の材質・除去正解は判定保留。',17)
for i,c in enumerate(selected):
    py=100+i*rh;b.text(d,(15,py),c['id']+' / '+c['page'],19)
    for j,key in enumerate(['original','sauvola','wolf','howe']):
        tile=Image.open(H/'native'/c['id']/(key+'.png')).convert('RGB');tile.thumbnail((285,330),Image.Resampling.NEAREST)
        b.text(d,(15+j*cw,py+28),key,16);overview.paste(tile,(15+j*cw,py+53))
overview.save(H/'課題別_比較一覧.png')
b.dump('selection.json',dict(cases=selected,classification='Seven challenge categories including a corrected Howe stain-removal success and material-uncertain cases; not seven common failures',selection_after_outputs=True,methods_tuned_for_these_cases=False,globally_hardest_verified=False))

controls=[]
for cid,seed,title,obs in [
 ('C01','S_T01','広い染みを比較的分離できる対照','染みは広いが3法とも本文との分離が比較的良い。広さだけでは最難例にならない。'),
 ('C02','R_N01','灰色像を減らせる対照','灰色像はかなり減り、主要な濃い本文は残る。材質が未確定なので灰色像の除去精度は断定しない。'),
 ('C03','D_hole_edge_new','本文外の穴・裂けらしい像','3法とも損傷の縁を黒い前景として残す。ただし本文外であり、文字の読解改善とは別の外観評価。'),
]:
    c=dict(byid[seed],id=cid+'_'+title)
    if cid=='C01':c['roi_xywh']=[760,360,600,600]
    raw=b.rgb(R/c['original']);masks={k:np.array(Image.open(H/'masks'/c['page']/(k+'.png')).convert('L'))==0 for k in ['sauvola','wolf','howe']}
    fig=b.render(c,raw,masks,H/'controls',cid+' '+title)
    controls.append(dict(id=cid,title=title,page=c['page'],observation=obs,figure=fig,roi_xywh=c['roi_xywh']))
b.dump('controls.json',controls)

lines=['# 黒文字の課題別難例と、古典的手法の限界', '',
 '2026-10-08。21ページ・24領域をWolf、Sauvola、Howe基本エネルギーのローカル移植で比較し、7種類の課題候補を選んだ。④の濃い斑点はHoweの除去成功例であり、共通の失敗例という前回の記述を訂正した。提案手法の追加実装・調整は行っていない。', '',
 '**薄い筆画の欠損と染みとの接触部には問題が残る。一方、④の孤立した濃い斑点はHoweが除去できている。原画像の紙面欠損と、表文字か裏写りかの不確定性は、それらと別に扱う。**', '',
 '利用者の所感と今回の訂正を踏まえ、次の主比較はSauvolaとHoweに絞る。[2手法の比較と訂正の確認](sauvola_howe/比較報告.md)を参照。', '',
 '[課題別の比較一覧PNG](課題別_比較一覧.png)。一覧は縮小表示。判定には以下の原寸比較と整数2倍の詳細を使う。各PNGは左上が原画像、右上Sauvola、左下Wolf、右下Howe移植。', '',
 '## 課題別の候補', '',
 '| ID | 課題 | ページ | 注目する問題 | 原寸比較 | 詳細2倍 |', '| --- | --- | --- | --- | --- | --- |']
for c in selected:
    lines.append(f'| {c["id"][:2]} | {c["category"]} | {c["page"]} | {c["limit"]} | [PNG]({c["comparison"]["path"]}) | [PNG]({c["detail"]["path"]}) |')
for c in selected:
    lines += ['',f'### {c["id"]}', '',f'注目箇所：{c["focus"]}。', '',c['observation'], '',f'位置は[原ページの赤枠](contexts/{c["id"]}.png)で確認できる。橙枠が詳細2倍。原ページ座標 x,y,width,height は `{c["roi_xywh"]}`、詳細は `{c["detail_xywh"]}`。']
lines += ['', '## 難例にしなかった対照', '',
 '以前の処理で難しかった箇所を、今回の3法でも必ず難しいとは扱わなかった。T01の広い染みなどを比較対照として残した。', '']
for c in controls:lines.append(f'- [{c["id"]} {c["title"]}]({c["figure"]["path"]})：{c["page"]}。{c["observation"]}')
lines += ['', '## 探索と判断の範囲', '',
 '- 以前の6,072ページの原画像スクリーニングに基づく汚れ・筆画候補一覧から、今回21領域・18ページを原画像だけで見直した。裏写り候補は過去の近傍丁照合による11ページを原画像の一覧で追加確認し、そのうち3ページを固定条件の比較へ追加した。今回6,072ページを全て再処理したわけではない。',
 '- 合計24領域・21ページを3条件で全ページ処理し、7領域を選択した。全候補は[candidate_protocol.json](candidate_protocol.json)、原画像の探索一覧は[search/](search/)、全候補の比較は[candidate_comparisons/](candidate_comparisons/)にある。',
 '- 最終選択は出力を見た後の探索的選択。「確認した範囲で厳しい代表候補」であり、資料全体で最も難しいことや独立テストの精度を保証しない。03と04・02は同じhnsd資料群で、別資料への一般化例とは数えない。分類は排他的ではなく、同じ領域に複数の課題がある。',
 '- 影・折れだけに由来する独立した最難例は今回まだ確定していない。02は水染みを伴う背景むら、05は暗い帯を含む像。05の帯を撮影影と決めつけない。',
 '- ⑥の欠損境界は画像上の所見。物理的な紙欠損の確定や、欠ける前の字形の正解は得ていない。原画像で見えない筆画を、二値化出力から正しく戻せたとは判定できない。',
 '- ⑦の灰色像は裏写り候補。近傍丁の照合は候補探しに使い、今回ROIの画素材質GTにはしていない。表の薄い文字を消しているかの専門的確認を残す。⑤の粒状像も必要な線を含みうるため全てを汚れGTにはしない。',
 '- 朱文字は評価対象外。原画像に含まれる赤い像も分離の成功・失敗には数えない。濃い孤立斑点の除去、文字の保持、読解改善は別々の問題。', '',
 '## 比較条件と検証', '',
 '無補正の原画像を全ページ・元解像度で読み、OpenCV RGB2GRAYを255で割った同じ入力を3法に渡した。ACE、CSC、修復モデルは通していない。全ページの処理後にROIを切り出した。窓サイズ61は元画像の画素単位で、解像度や線幅に合わせた自動調整はしていない。', '',
 '- Sauvola：w=61、k=0.2、R=0.5。Wolf：w=61、k=0.5、全ページの最小輝度・最大局所標準偏差を使用。',
 '- Howe：前回の基本エネルギー移植、c=160。Cannyはskimageでの近似、著者実装のCannyと自動パラメータ選択は再現していない。Howe原法そのものの限界とは断定しない。',
 '- 3法の設定は新しい出力を見る前に固定し、この候補に合わせた調整は行っていない。今回の比較から「古典的手法は全て頭打ち」とは結論しない。',
 '- 原寸PNGは画素を維持、詳細は原画素の整数2倍反復。原寸・詳細の56パネルを保存PNGから再読込し、原画像・全ページマスクの該当画素と完全一致を確認した。選択7領域の原画像切り出しも全画素一致。21原画像のSHA-256が処理前後で一致し、全ページマスク63枚を保存した。',
 '- 密な画素正解、専門家の翻刻、読解試験は未実施。今回の所見は目視での限界可視化であり、精度の数値や総合順位ではない。', '',
 '再現用：[build.py](build.py)、[package.py](package.py)、[候補一覧CSV](候補一覧.csv)、[selection.json](selection.json)、[検証記録](verification.json)。前回コードの数式とグラフカットの検証記録は[前回の報告](../classical_deep_review_20261008/検討報告.md)を参照。', '',
 '次に提案手法を検討するときは、この難例と比較的分離できる対照を固定して使える。③は原画像で見える筆画と染み、①②は必要な薄い筆画、⑤⑦は材質確認、⑥は原画像で確認可能な情報と欠損を区別するための確認対象になる。']
(H/'課題別難例_報告.md').write_text('\n'.join(lines)+'\n')

with (H/'候補一覧.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f);writer.writerow(['ID','分類','ページ','原画像','xywh','詳細xywh','注目箇所','観察','判断の限界','比較','詳細'])
    for c in selected:writer.writerow([c['id'],c['category'],c['page'],c['original'],c['roi_xywh'],c['detail_xywh'],c['focus'],c['observation'],c['limit'],c['comparison']['path'],c['detail']['path']])
sources={str(p.relative_to(R)):b.sha(p) for p in [H/'build.py',H/'package.py',R/'tmp_work/classical_black_comparison_20261008/compare.py',R/'tmp_work/classical_deep_review_20261008/run.py',R/'tmp_work/classical_deep_review_20261008/graphcut_cli']}
for c in cs:assert b.sha(R/c['original'])==c['source_sha256']
all_pages={c['page'] for c in cs};proc=json.loads((H/'processing.json').read_text());assert {p['page'] for p in proc}==all_pages
b.dump('verification.json',dict(full_page_input_sources_unchanged=len(all_pages),new_full_page_masks=len(all_pages)*3,selected_regions=len(selected),main_and_detail_pngs=len(figures),exported_panels_exact_pixel_check=checked,native_original_crops_exact=True,input_sources={c['original']:c['source_sha256'] for c in cs},code_sha256=sources,figures=figures))
print('Selected',len(selected),'categories;',len(all_pages),'pages;',len(cs),'candidate regions;',checked,'verified panels')
