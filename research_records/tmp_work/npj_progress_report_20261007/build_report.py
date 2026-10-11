"""Compose evidence figures from archived pixels; no restoration/inference is run."""
from pathlib import Path
import json, hashlib, zipfile, csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin

Image.init()

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
T = ROOT / 'tmp_work'
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
NATIVE = OUT / 'native_crops'
NATIVE.mkdir(exist_ok=True)
W, H = 1920, 1080
BG = '#f5f3ed'
INK = '#192f3b'
MUTED = '#53666c'
GREEN = '#147865'
RED = '#b4423a'
GOLD = '#a36918'
BLUE = '#245a91'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
pages, provenance, panel_checks = [], [], []
metrics = {}

def font(size, bold=False): return ImageFont.truetype(BOLD if bold else FONT, size, index=0)
def txt(im, xy, text, size=28, color=INK, bold=False):
    ImageDraw.Draw(im).text(xy, text, font=font(size,bold), fill=color, spacing=12)
def wrap(s, width, size):
    f=font(size); out=[]
    for paragraph in s.split('\n'):
        line=''
        for ch in paragraph:
            if f.getlength(line+ch)>width and line:
                out.append(line); line=ch
            else: line+=ch
        out.append(line)
    return '\n'.join(out)
def para(im, x,y,s,width,size=28,color=INK):
    lines=wrap(s,width,size)
    txt(im,(x,y),lines,size,color)
    return y+(size+12)*len(lines.splitlines())
def base(num,title,subtitle,tag='',color=BLUE):
    im=Image.new('RGB',(W,H),BG)
    d=ImageDraw.Draw(im)
    d.rectangle((0,0,W,13),fill=color)
    txt(im,(60,35),'HERITAGE PIPELINE  /  NPJからの進捗',24,MUTED)
    txt(im,(60,82),title,49,INK,True)
    txt(im,(60,159),subtitle,26,MUTED)
    d.line((60,1015,1860,1015),fill='#c9ceca',width=2)
    txt(im,(60,1030),'作成 2026-10-07  |  実験記録は2026-10-04まで  |  実験出力を使用・色補正なし',21,MUTED)
    txt(im,(1770,1030),f'{num:02d} / 10',21,MUTED)
    if tag: txt(im,(1450,35),tag,24,color,True)
    return im
def box(im, xy, title, body, color=BLUE, size=28):
    x,y,w,h=xy
    d=ImageDraw.Draw(im); d.rounded_rectangle((x,y,x+w,y+h),radius=16,fill='white')
    d.rectangle((x,y+16,x+6,y+h-16),fill=color)
    txt(im,(x+25,y+17),title,32,color,True)
    para(im,x+25,y+76,body,w-50,size)
def source(p): return T/p
def read(p): return Image.open(p).convert('RGB')
def mask(p): return np.asarray(Image.open(p).convert('L'))>127
def change(p,q): return np.any(np.asarray(read(p)) != np.asarray(read(q)),axis=2)
def panel(im, path, xywh, x,y,scale,label, pid):
    src=read(path)
    if xywh is None: xywh=(0,0,*src.size)
    a,b,w,h=xywh
    assert min(a,b)>=0 and a+w<=src.width and b+h<=src.height,(path,xywh,src.size)
    crop=src.crop((a,b,a+w,b+h))
    native=NATIVE/(pid+'.png'); crop.save(native)
    rendered=crop.resize((w*scale,h*scale),Image.Resampling.NEAREST)
    txt(im,(x,y-45),label,29,INK,True)
    im.paste(rendered,(x,y))
    ImageDraw.Draw(im).rectangle((x-1,y-1,x+rendered.width,y+rendered.height),outline='#aab5b7',width=1)
    assert np.array_equal(np.asarray(im.crop((x,y,x+w*scale,y+h*scale))),np.asarray(rendered))
    provenance.append(dict(panel=pid,source=str(path.relative_to(ROOT)),crop_xywh=list(xywh),paste_xy=[x,y],scale=scale,native=str(native.relative_to(OUT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    panel_checks.append(dict(panel=pid,pixels_match=True))
    return rendered.size
def save(im,name):
    p=FIG/name; im.save(p); pages.append(p)
def note(im,s,y=940,color=MUTED): para(im,60,y,s,1800,25,color)

M=source('npj_manual_stain_oracle_20261004')
orig1=M/'hnsd001_input/hnsd001_011.png'
base1=M/'hnsd001_output/hnsd001_011/result/hnsd001_011.png'
out1=M/'hnsd001_harmonic_postprocess_v3.png'
m1=mask(M/'hnsd001_manual_mask_v3.png')
c1=change(base1,out1)
metrics['isolated_v3']=dict(mask_pixels=int(m1.sum()),changed_pixels=int(c1.sum()),changed_outside=int((c1&~m1).sum()))
assert metrics['isolated_v3']==dict(mask_pixels=7259,changed_pixels=7227,changed_outside=0)
orig7=M/'input/hnsd007_042.png'
base7=M/'output/hnsd007_042/result/hnsd007_042.png'
pre7=M/'precleaned_output/hnsd007_042/result/hnsd007_042.png'
out7=M/'reclassified_output/hnsd007_042/result/hnsd007_042.png'
m7=mask(M/'manual_stain_mask.png'); c7=change(base7,out7)
black7=~mask(M/'masks/hnsd007_042/02_text_black.png')
metrics['contact']=dict(mask_pixels=int(m7.sum()),preclean_changed=int(change(base7,pre7).sum()),override_changed=int(c7.sum()),outside=int((c7&~m7).sum()),changed_on_black_mask=int((c7&black7).sum()),mask_overlap_black=int((m7&black7).sum()))
assert metrics['contact']==dict(mask_pixels=1330,preclean_changed=0,override_changed=19462,outside=18132,changed_on_black_mask=10635,mask_overlap_black=1330)

im=base(1,'局所的な改善と、汚れ除去を妨げる条件を特定','現在の到達点：実装・比較基盤・局所効果は前進。NPJへの総合優位と自動化は未確定。','進捗の要約')
box(im,(60,245,570,310),'進展 1  比較を正しくした','原画像・実推論入力・保存出力を照合。通常NPJと、編集マスクだけを替えた診断実験を区別した。',BLUE)
box(im,(675,245,570,310),'進展 2  局所効果を確認','紙面だけを補間参照にすると灰色のにじみが減る。孤立斑点は手指定の後処理で除去できた。',GREEN)
box(im,(1290,245,570,310),'進展 3  原因を具体化','茶色いシミが黒文字に分類される例を確認。保存される劣化マスクは、監査した推論経路では読まれていない。',BLUE)
box(im,(60,595,870,280),'現在の壁  汚れと筆画の分離','除去を強めると筆画が消え、保護を広げると汚れも残る。本文に接触した汚れは、黒文字マスクから外すだけでは穴や周辺変化を生む。',RED)
box(im,(975,595,885,280),'未達  別冊での有効性と自然な紙面','別冊への転用で誤編集が発生。別4ページ・6範囲では目立つ追加除去なし。補修面の平坦さ・境界も残る。',GOLD)
note(im,'次：孤立汚れ／接触汚れを分けた固定評価セットで、汚れ改善・筆画保持・紙面の自然さを別々に判定する。',920)
save(im,'01_progress_summary.png')

im=base(2,'局所成功：孤立した汚れは、後処理で取り除ける','hnsd001_011  |  同じ位置を3倍・最近傍で表示  |  凍結コードによる新規crop対照','限定的な成功',GREEN)
roi=(80,145,180,160)
for i,(p,l) in enumerate([(orig1,'原画像：茶色い斑点'),(base1,'NPJ対照：黒い塊が残る'),(out1,'手指定の紙面補修 v3')]):
    panel(im,p,roi,60+i*610,275,3,l,f'02_{i}')
box(im,(60,790,865,135),'変更範囲を局所化','7,259画素の指定内で7,227画素変更。指定外の変更は0。',GREEN,25)
box(im,(975,790,885,135),'限界','出力を見てマスクを調整した既知1例。補修面の平坦さは残る。',GOLD,25)
note(im,'原ページcrop [336,396,528,528] 内の拡大。成功はこの孤立斑点の局所除去に限定し、自動検出や一般化とは扱わない。',950)
save(im,'02_isolated_stain_success.png')

im=base(3,'失敗：接触汚れは、マスクから外すだけでは直らない','hnsd007_042  |  同じ位置を2倍・最近傍で表示  |  凍結コードによる新規crop対照','接触汚れが未解決',RED)
roi7=(353,175,190,160)
for i,(p,l) in enumerate([(orig7,'原画像：筆画・枠線と接触'),(base7,'NPJ対照：黒く残る'),(pre7,'入力画像を事前補修'),(out7,'黒文字マスクを解除')]):
    panel(im,p,roi7,60+i*460,280,2,l,f'03_{i}')
box(im,(60,650,865,230),'入力の事前補修：効果0','マスク等を固定すると、NPJ出力は全画素一致。この設定では、入力画像の見た目を直すだけでは出力に反映されない。',GOLD,27)
box(im,(975,650,885,230),'マスク解除：不自然な穴と周辺変化','指定1,330画素に対し、出力19,462画素が変化。うち指定外18,132画素。局所的な汚れ除去として制御できていない。',RED,27)
note(im,'黒文字マスクとの重なり10,635画素は「実筆画の損傷数」ではない。マスクは汚れも含み、画素ごとの材質正解は未作成。',930)
save(im,'03_contact_stain_failure.png')

R=source('npj_colour_independent_ridge_20261004/native/new_new_warm_lower_loss')
roi=(615,411,80,79)
old=change(R/'baseline.png',R/'before.png')
new=change(R/'baseline.png',R/'all_colour_ridge/combined.png')
x,y,w,h=roi
metrics['stroke_example']=dict(previous_roi_changes=int(old[y:y+h,x:x+w].sum()),new_roi_changes=int(new[y:y+h,x:x+w].sum()))
assert metrics['stroke_example']==dict(previous_roi_changes=37,new_roi_changes=0)
im=base(4,'誤削除の改善：茶色い筆画も、形状で保持する','暖色冊の既知反例  |  80×79画素を5倍表示  |  新規crop対照に対する追加処理の比較','保持の局所改善',GREEN)
for i,(p,l) in enumerate([(R/'original.png','原画像：茶色い筆画'),(R/'baseline.png','NPJ対照'),(R/'before.png','従来の線保持：37画素変更'),(R/'all_colour_ridge/combined.png','色を問わない保持：変更0')]):
    panel(im,p,roi,60+i*460,280,5,l,f'04_{i}')
    xx=60+i*460+(650-615)*5; yy=280+(446-411)*5
    ImageDraw.Draw(im).rectangle((xx-5,yy-5,xx+55,yy+50),outline=RED,width=3)
box(im,(60,725,865,180),'何が進んだか','この37画素に加え、別の22・52画素を含む計111画素の誤編集を既知例で防止。黒残留の局所除去も維持した。',GREEN,26)
box(im,(975,725,885,180),'何が残るか','形が線状の汚れも守ってしまう。既知の反例を見て設計した条件なので、別冊への一般性能は未確定。',GOLD,26)
note(im,'赤枠は比較用の位置表示。基準NPJが残していた筆画を追加処理で壊さない改善であり、NPJ自体の文字復元改善とは分ける。',945)
save(im,'04_stroke_preservation.png')

C=source('npj_dirt_clean_reference_20261002/native/current_control')
im=base(5,'局所成功：補間に使う「紙面」を選ぶと、灰色のにじみが減る','hnsd001_011 の下余白  |  同じ編集候補・同じ位置を表示  |  新規対照は過去保存出力と全画素一致','参照の選別が有効',GREEN)
roi=(240,310,350,250)
for i,(p,l) in enumerate([(C/'original.png','原画像'),(C/'baseline.png','NPJ対照'),(C/'all_boundary.png','全境界を参照：灰色化'),(C/'clean_reference.png','紙面だけ参照：軽減')]):
    panel(im,p,roi,60+i*460,300,1,l,f'05_{i}')
box(im,(60,620,865,255),'失敗の原因を切り分けた','黒い枠線や断片まで参照にすると、その暗さが補間領域へにじむ。明るさ・局所差による紙面選別と境界平滑化を併用して改善した。',GREEN,27)
box(im,(975,620,885,255),'汚れ除去の完成ではない','本文側の黒い塊、範囲外の小片、平坦な補修面は残る。枠外判定を別ページへ広げると本文を消した反例もある。',GOLD,27)
note(im,'既知1ページの局所結果。暗い画素数の減少は見た目の診断値であり、汚れ除去精度や正解率ではない。',930)
save(im,'05_paper_reference_success.png')

L=source('npj_literature_trial_review_20261002/native/07_red_seal_control')
im=base(6,'失敗と手掛かり：外部モデルの出力を、そのまま採用できない','朱印の同一範囲  |  原寸表示  |  StainRestorerは色復元、DocResは文字候補の二値化','役割の違いを明示',RED)
for i,(p,l) in enumerate([(L/'original_image.png','原画像'),(L/'saved_npj_image.png','実保存NPJ：朱を保持'),(L/'stain_image.png','StainRestorer：朱が退色'),(L/'docres_image.png','DocRes：白黒の文字候補')]):
    panel(im,p,None,60+i*460,270,1,l,f'06_{i}')
note(im,'StainRestorer：紙色変化と朱の退色を256px条件でも確認。DocRes：曲線・汚れ分離の手掛かりはあるが、小筆点や朱を落とす。',850)
note(im,'この資料・重み・適用条件での判断。DocResの白黒出力を「自然な色復元」と数えず、補助分類としてのみ検討する。',930)
save(im,'06_external_model_limits.png')

G=source('npj_ridge_saved_ranges_20261004/native/known_upper_stain')
im=base(7,'追加処理の限界：NPJで既に除去できている例もある','200010454_00002_2  |  実保存の全ページ出力から同じ範囲を切り出し、3倍表示','追加効果なし',GOLD)
roi=(230,130,180,140)
for i,(p,l) in enumerate([(G/'original.png','原画像：茶色いシミ'),(G/'baseline.png','実保存NPJ：既に軽減'),(G/'all_colour_ridge/combined.png','追加処理：目立つ利点なし')]):
    panel(im,p,roi,60+i*610,280,3,l,f'07_{i}')
box(im,(60,745,865,170),'固定条件を別範囲へ適用','4ページ・6範囲で、目立つ追加汚れ除去の肯定例なし。新候補の変更107画素も材質未確定。',GOLD,27)
box(im,(975,745,885,170),'評価上の前進','原画像にシミがあるだけで追加処理は正当化できない。NPJ出力に残る問題と、処理が必要な場所を先に確認する。',BLUE,27)
note(im,'固定v6の紙色変更を含むため、右列を「全画素不変」とは呼ばない。同冊・既知ページを含み、独立冊の成功率ではない。',945)
save(im,'07_no_additional_benefit.png')

im=base(8,'原因の特定：シミが黒文字に入り、劣化マスクは推論に渡らない','監査したローカル pipeline_ddrm.py の経路  |  既知2例の分類監査・対照実験','設計上の課題',BLUE)
box(im,(60,240,410,210),'CSCの4分類','黒文字／朱文字／劣化／背景として保存する。',BLUE,27)
box(im,(525,240,625,210),'黒文字・朱文字の経路','黒文字 → DDRMの復元対象\n朱文字 → 推論後に原画像から合成',BLUE,27)
box(im,(1205,240,655,210),'劣化・背景の経路','04_degradation / 05_background は、この推論スクリプトでは読み込まれない。',RED,27)
txt(im,(479,310),'→',38,BLUE,True)
txt(im,(60,490),'手指定したシミ範囲が、どの分類に入っていたか',31,INK,True)
counts=[('孤立斑点  hnsd001',7259,[3474,0,1294,2491]),('接触シミ  hnsd007',1330,[1330,0,0,0])]
colors=['#243d50','#b4423a','#d3932c','#c9c7bc']
for j,(label,n,vals) in enumerate(counts):
    yy=565+j*120; txt(im,(60,yy),label,29,INK,True)
    xx=480
    for v,col in zip(vals,colors):
        ww=1200*v/n
        if v:
            ImageDraw.Draw(im).rectangle((round(xx),yy,round(xx+ww),yy+52),fill=col)
            if ww>120: txt(im,(xx+15,yy+8),f'{v/n*100:.1f}%',24,'white' if col==colors[0] else INK,True)
        xx+=ww
    txt(im,(1710,yy+7),f'{n:,} px',24,MUTED)
for j,(label,col) in enumerate(zip(['黒文字','朱文字','劣化','背景'],colors)):
    x=480+j*240; ImageDraw.Draw(im).rectangle((x,805,x+26,831),fill=col); txt(im,(x+40,796),label,26)
note(im,'接触シミの1,330画素は全て黒文字。劣化マスクを追加するだけでは解けず、分類と推論への渡し方の両方を直す必要がある。',875)
note(im,'実行時used_maskは未保存。保存入力・CSCマスク・コードと同じ変換から再構成し、入力cropと黒マスクの全画素一致を確認。',950)
save(im,'08_mask_routing_diagnosis.png')

im=base(9,'手法の成否：局所成功、条件付き、失敗を分けて報告する','成功は確認した範囲に限定。出力が変わらないことや変更量の多さを、除去成功とは数えない。','主要手法の整理')
rows=[
('紙面参照を選ぶ調和補間','局所成功','灰色のにじみを軽減。接触汚れ・平坦さは残る。',GREEN),
('手指定マスクによる後処理','局所成功','孤立斑点を除去。出力を見た後の手調整で、自動化は未達。',GREEN),
('色に依存しない線保持','条件付き','既知の筆画誤編集111画素を防止。線状汚れも保護する。',GOLD),
('DocResによる補助分類','条件付き','細曲線を拾う一方、小筆点・朱・薄い主線を落とす。',GOLD),
('低域／高域分離・紙模様転写','不十分','細模様量は改善するが、色面・縁・自然さは未解決。',GOLD),
('黒文字マスクからシミを解除','失敗','穴と周辺変化。接触汚れを安全に分離できない。',RED),
('枠外判定の単純な拡張','失敗','別ページの本文を消す。全ページ文脈でも適用範囲が狭い。',RED),
('StainRestorerの色出力','不採用','今回の条件では紙色変化・朱の退色・色むら。',RED),
('紙／筆記／汚れの層分解','不採用','汚れを分けると薄い朱や暖色筆記も失う。',RED),
]
y=245
for j,(method,status,reason,col) in enumerate(rows):
    ImageDraw.Draw(im).rectangle((60,y,1860,y+74),fill='white' if j%2==0 else '#eaece7')
    txt(im,(78,y+16),method,28,INK,True); txt(im,(610,y+16),status,28,col,True); txt(im,(800,y+16),reason,27)
    y+=78
note(im,'初期のDDNM比較、CSC修正、自己参照背景、裏写り検出などの経緯と根拠は、同梱の進捗報告.mdに記載。',965)
save(im,'09_method_outcomes.png')

im=base(10,'次の検証：汚れ検出の問題と、補修処理の問題を分ける','現在は閾値の拡張を止め、成功条件と失敗条件を固定して評価する段階。','次段階の提案')
box(im,(60,245,565,400),'1  小さな固定評価セット','孤立汚れ、文字接触汚れ、虫食いの穴・縁を分ける。\n\n隣接筆画・薄い筆画・朱印・枠線・正常紙面を保持対照にする。',BLUE,29)
box(im,(677,245,565,400),'2  同じ補修器で比較','NPJそのまま／手指定の汚れマスク／自動マスクを比較する。\n\n手指定でも失敗するなら補修側、自動だけ失敗するなら検出側を見直す。',BLUE,29)
box(im,(1294,245,566,400),'3  条件固定後に別冊へ','汚れ改善、筆画の追加損傷、紙色・模様・境界を別々に判定する。\n\n開発で見た冊と分離し、処理が動かなかった例も記録する。',BLUE,29)
box(im,(60,700,1800,215),'進捗報告で伝える結論','NPJ後の局所的な汚れ除去と、筆画を壊す原因の切り分けまで進んだ。現在の中心課題は「何を消してよいか」の判定であり、文字に接触した汚れの安全な分離と、別冊での有効性の確認が残っている。',GREEN,31)
note(im,'周波数・多尺度の処理は紙面の馴染みに役立つが、それだけでは同じ色・形・スケールにある筆画と汚れを識別できない。',950)
save(im,'10_next_steps.png')

(OUT/'verified_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
(OUT/'image_sources.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
(OUT/'verification.json').write_text(json.dumps(dict(figures=len(pages),native_panels=len(provenance),panel_pixel_checks=panel_checks,inference_rerun=False,notes=['Images are assembled from saved outputs.','All comparison crops use integer nearest-neighbor enlargement.','Only figure 04 has explicit red locator boxes; unannotated crops are supplied.','hnsd001 v3 report text had v2 counts; final v3 pixels were recounted.']),ensure_ascii=False,indent=2)+'\n')
imgs=[Image.open(p).convert('RGB') for p in pages]
imgs[0].save(OUT/'NPJ進捗報告_20261007.pdf',save_all=True,append_images=imgs[1:],resolution=144.0,title='NPJからの進捗報告',author='Heritage Pipeline',quality=96,subsampling=0)
# One compact overview for quick review; exported PNG pages remain the primary evidence.
sheet=Image.new('RGB',(W,H),BG)
for idx,page in enumerate([pages[1],pages[2],pages[3],pages[5]]):
    p=Image.open(page); p.thumbnail((960,540),Image.Resampling.LANCZOS)
    sheet.paste(p,((idx%2)*960,(idx//2)*540))
sheet.save(OUT/'比較画像一覧.png')
print(json.dumps(dict(figures=len(pages),panels=len(provenance),metrics=metrics),ensure_ascii=False))
