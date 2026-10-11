"""Package the contact-stain search and frozen NPJ baseline evidence."""
from pathlib import Path
import csv
import hashlib
import json
import re
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
BG, INK, MUTED = '#faf9f5', '#202d38', '#53616c'
manifest = json.loads((HERE/'inference_selection.json').read_text())
inferred = {c['test_id']: c for c in manifest['cases']}
screen = json.loads((HERE/'screening.json').read_text())


def c(id, roi, priority, group, title, finding, role):
    return dict(id=id, detail_xywh=roi, priority=priority, group=group,
                title=title, finding=finding, proposed_role=role)


cases = [
    c('T00',[300,150,320,400],'高','既知参照',
      '枠線から本文までシミが黒い塊になる',
      '原画像で見えるシミ下の筆画が、出力では大きな黒い塊に覆われる。枠線にも接続する。',
      '元の提案例。比較基準として再利用する。'),
    c('T01',[130,150,450,450],'高','追加・本文重なり',
      '複数の筆画を広い黒い塊が覆う',
      '原画像では茶色いシミを横切る細い線・曲線が見えるが、出力では広い黒い塊に埋まり、隣の列とも接続する。',
      '追加例の第一候補。汚れと筆画を分ける処理の開発対象にする。'),
    c('T02',[120,200,440,440],'高','追加・本文重なり',
      '2つのシミがそれぞれ本文を覆う',
      '原画像の2つの茶色い範囲に見える筆画が、出力では大小2つの黒い塊に埋まる。正常な隣の列は概ね追える。',
      '大きさの違うシミと、変更しない隣接本文を同時に確認する主要例。'),
    c('T03',[250,180,350,480],'高','追加・本文重なり',
      '本文の横の小さい文字が埋もれる',
      'シミの中の本文と、その右の小さい文字に見える線が出力では塊・断片に変わる。広い単一の塊だけでない失敗。',
      '小さい添え書きを守れるか確認する主要例。'),
    c('T04',[250,190,330,470],'中','追加・筆画接触',
      'シミが本文の細い線と一体になる',
      '原画像のシミに重なる細い曲線の近くが、出力では黒い塊になり、本文の線と接続する。',
      '筆画と汚れの接触部を調べる例。薄い線の復元根拠は原画像で確認する。'),
    c('T05',[245,185,330,420],'低','近接対照',
      '小さいシミは黒く残るが本文とは離れる',
      '本文の線の下に小さい黒い塊が残るが、拡大では主要な線との間に隙間が見える。本文を覆う主要例には数えない。',
      '黒い塊があるだけで処理を必要と判断しない対照。'),
    c('T06',[280,190,300,440],'中','追加・筆画接触',
      '小さい茶色い範囲が筆画に接続する',
      '原画像の曲線に近接・重複する小さい茶色い範囲が、出力では筆画に接続した黒い塊として残る。影響は局所的。',
      '小さな接触範囲だけを直して正常な曲線を保護できるか確認する。'),
    c('T07',[210,170,450,470],'保留','別資料・材質確認',
      '別資料の茶色い跡が本文の一画につながる',
      '濃い茶色い跡の下端が本文の横線に接し、出力では黒い塊として一体になる。ただし跡には白い中心もあり、シミか損傷かは未確定。',
      '別資料への展開候補。物理的な穴や欠損が含まれるか先に確認し、純粋なシミの主評価へはまだ入れない。'),
    c('C01',[230,160,440,490],'保留','材質・重なり不確定対照',
      '粒状像は残るが本文との重なりは不確定',
      '原画像にある粒状の濃い像が出力でも残る。主要な本文は左右に見え、像の下の灰色線は表の筆画か裏写りか未確定。',
      '本文を覆うシミの確定例にはしない。材質と必要な筆画の確認用に記録する。'),
]

# Existing reference: same frozen-code crop inference, reused rather than rerun.
reference = dict(test_id='T00',page='hnsd007_042',book='hnsd00000',
    original='data/split_dataset/train/hnsd00000/hnsd007_042.jpg',
    npj_black='data/split_dataset_csc_mask/train/hnsd00000/hnsd007_042/02_text_black.png',
    npj_red='data/split_dataset_csc_mask/train/hnsd00000/hnsd007_042/03_text_red.png',
    actual_input='tmp_work/npj_manual_stain_oracle_20261004/input/hnsd007_042.png',
    output='tmp_work/npj_manual_stain_oracle_20261004/output/hnsd007_042/result/hnsd007_042.png',
    crop_xywh=[600,100,768,768],source_kind='2026-10-04の既存crop対照・今回再利用')

for name in ('comparisons','contexts','mask_diagnostics','native','figures'):
    (HERE/name).mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def font(n,bold=False):
    return ImageFont.truetype(BOLD if bold else FONT,n)


def para(d,text,x,y,width,n=28,bold=False,fill=INK):
    line='';f=font(n,bold)
    for ch in text:
        if ch=='\n' or d.textlength(line+ch,font=f)>width:
            d.text((x,y),line,font=f,fill=fill);y+=n+12;line='' if ch=='\n' else ch
        else:line+=ch
    if line:d.text((x,y),line,font=f,fill=fill);y+=n+12
    return y


images={};panels=[]
for item in cases:
    r=dict(reference if item['id']=='T00' else inferred[item['id']])
    if item['id']!='T00':r['source_kind']='2026-10-07の新規crop対照'
    original=Image.open(ROOT/r['original']).convert('RGB')
    x,y,w,h=r['crop_xywh'];fullbox=(x,y,x+w,y+h)
    input_im=Image.open(ROOT/r['actual_input']).convert('RGB')
    output=Image.open(ROOT/r['output']).convert('RGB')
    assert input_im.size==output.size==(w,h)
    assert np.array_equal(np.asarray(original.crop(fullbox)),np.asarray(input_im)),item['id']
    mask=Image.open(ROOT/r['npj_black']).convert('L').crop(fullbox).convert('RGB')
    assert set(np.unique(np.asarray(mask))).issubset({0,255})
    if item['id']!='T00':
        cropped=Image.open(HERE/'masks'/r['page']/'02_text_black.png').convert('RGB')
        assert np.array_equal(np.asarray(mask),np.asarray(cropped))
    images[item['id']]=(input_im,output,mask,r,original)
    dx,dy,dw,dh=item['detail_xywh'];box=(dx,dy,dx+dw,dy+dh)
    assert dx>=0 and dy>=0 and dx+dw<=w and dy+dh<=h
    item.update(page=r['page'],book=r['book'],crop_xywh=r['crop_xywh'],
                page_detail_xywh=[x+dx,y+dy,dw,dh],source_kind=r['source_kind'],
                source={k:r[k] for k in ('original','npj_black','npj_red','actual_input','output')})
    item['source_sha256']={k:sha(ROOT/v) for k,v in item['source'].items()}
    native=HERE/'native'/item['id'];native.mkdir(exist_ok=True)
    for label,im in [('original',input_im),('npj',output),('black_mask',mask)]:
        im.save(native/f'{label}_context.png');im.crop(box).save(native/f'{label}_detail.png')
    # Main comparison: integer nearest-neighbour repetition, without annotations in evidence panels.
    scale=2;pw,ph=dw*scale,dh*scale
    cw=max(1800,2*pw+140);ch=ph+390
    canvas=Image.new('RGB',(cw,ch),BG);d=ImageDraw.Draw(canvas)
    d.rectangle((0,0,cw,10),fill='#a04428' if item['priority']=='高' else '#2c627d')
    d.text((35,28),f'{item["id"]}  {item["title"]}',font=font(40,True),fill=INK)
    d.text((35,91),f'{item["page"]}  ／  優先度：{item["priority"]}  ／  {r["source_kind"]}',font=font(25),fill=MUTED)
    py=185;positions=[35,pw+95]
    for label,im,px in [('original',input_im,positions[0]),('npj',output,positions[1])]:
        d.text((px,138),'原画像' if label=='original' else 'NPJ系・通常黒文字マスクの処理後',font=font(28,True),fill=INK)
        canvas.paste(im.crop(box).resize((pw,ph),Image.Resampling.NEAREST),(px,py))
        panels.append(dict(file=f'comparisons/{item["id"]}.png',rect=[px,py,pw,ph],scale=scale,
                           native=f'native/{item["id"]}/{label}_detail.png'))
    yy=para(d,item['finding'],35,py+ph+18,cw-70,28)
    para(d,f'原ページROI (x,y,w,h) = {item["page_detail_xywh"]}。同じ座標・無補正・原画素の2倍表示。',35,yy+8,cw-70,24,fill=MUTED)
    para(d,'目視候補。汚れの材質・残す筆画の正解・読解効果は未注釈。全ページ工程後の出力ではない。',35,yy+54,cw-70,24,fill=MUTED)
    rel=f'comparisons/{item["id"]}.png';canvas.save(HERE/rel);item['comparison']=rel
    # Context: true input + true output, same 1x pixels.
    cv=Image.new('RGB',(1640,1100),BG);dd=ImageDraw.Draw(cv)
    dd.text((30,16),f'{item["id"]}  {item["page"]}  {item["group"]}',font=font(32,True),fill=INK)
    dd.text((30,65),'原画像（実入力と全画素一致）',font=font(26,True),fill=INK)
    dd.text((840,65),'NPJ系のcrop対照',font=font(26,True),fill=INK)
    cv.paste(input_im,(30,115));cv.paste(output,(840,115))
    para(dd,item['finding'],30,910,1570,27)
    para(dd,f'切り出し：{r["crop_xywh"]} ／ {r["source_kind"]}。原画素1倍。',30,1030,1570,24,fill=MUTED)
    rel=f'contexts/{item["id"]}.png';cv.save(HERE/rel);item['context']=rel
    for label,px in [('original',30),('npj',840)]:
        panels.append(dict(file=rel,rect=[px,115,w,h],scale=1,native=f'native/{item["id"]}/{label}_context.png'))
    # Diagnostic shows why a dark stain is routed into black-text regeneration.
    ww=max(1700,3*dw+120);hh=dh+370
    cv=Image.new('RGB',(ww,hh),BG);dd=ImageDraw.Draw(cv)
    dd.text((25,20),f'{item["id"]}  {item["page"]}：原画像・保存黒文字マスク・出力',font=font(29,True),fill=INK)
    for j,(label,im,title) in enumerate([('original',input_im,'原画像'),('black_mask',mask,'既存CSC v1黒文字マスク'),('npj',output,'NPJ系の出力')]):
        px=25+j*(dw+30);dd.text((px,78),title,font=font(22,True),fill=INK);cv.paste(im.crop(box),(px,125))
        panels.append(dict(file=f'mask_diagnostics/{item["id"]}.png',rect=[px,125,dw,dh],scale=1,
                           native=f'native/{item["id"]}/{label}_detail.png'))
    yy=para(dd,'中央列の黒は黒文字候補。汚れ候補の正解マスクではない。広い茶色い像が黒文字マスクに含まれ、出力でも黒い塊になる対応を確認する。',25,dh+150,ww-50,26)
    para(dd,'02_text_blackを使用。04_degradationを欠損として渡した以前の診断とは異なる。',25,yy+8,ww-50,23,fill=MUTED)
    rel=f'mask_diagnostics/{item["id"]}.png';cv.save(HERE/rel);item['mask_diagnostic']=rel


def atlas(rel,title,subtitle,ids):
    W,H=2220,220+((len(ids)+1)//2)*970
    cv=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(cv)
    d.text((35,22),title,font=font(48,True),fill=INK)
    para(d,subtitle,35,98,W-70,28,fill=MUTED)
    for j,id in enumerate(ids):
        item=next(c for c in cases if c['id']==id);a,b,_,r,_=images[id]
        x,y,w,h=item['detail_xywh'];box=(x,y,x+w,y+h)
        px=35+(j%2)*1100;py=220+(j//2)*970
        d.line((px,py,px+1050,py),fill='#ccd0d4',width=2)
        d.text((px,py+20),f'{id}  {item["page"]}',font=font(35,True),fill=INK)
        para(d,item['title'],px,py+80,1050,31,bold=True)
        d.text((px,py+143),'原画像',font=font(27,True),fill=INK)
        d.text((px+530,py+143),'NPJ系の処理後',font=font(27,True),fill=INK)
        for im,xx in [(a,px),(b,px+530)]:
            t=im.crop(box);factor=min(500/t.width,620/t.height)
            t=t.resize((round(t.width*factor),round(t.height*factor)),Image.Resampling.LANCZOS)
            cv.paste(t,(xx,py+195))
        para(d,item['finding'],px,py+835,1050,27)
    cv.save(HERE/rel)


atlas('figures/01_本文を覆うシミ.png','本文を覆うシミ：元の例と、追加の主要3例',
      '原画像では筆画が見えるが、NPJ系の出力ではシミと筆画が黒く一体になる。',['T00','T01','T02','T03'])
atlas('figures/02_接触と保留例.png','接触の小さい例と、別資料への展開候補',
      '小さな接触部も記録する。別資料のT07はシミ・損傷の材質確認が残る。',['T04','T06','T07','T05'])

# Record all 88 original/mask screening decisions, including exclusions.
art={1,2,8,10,13,14,15,16,25,26,27,29,30,31,32,37,41,44,45,46,54,57,78,79,80,88}
cover={4,5,7,11,72,77}
seal={3,6,20,22,33,87}
binding={21,23,34,38,40,52,55,56,58,59,60,61,62,63,64,65,66,67,69,71,73,74,75,86}
selected={r['id']:r['test_id'] for r in manifest['cases']}
reviews=[]
for row in screen['screened_candidates']:
    n=int(row['id'][1:]);id=row['id']
    if id in selected:
        item=next(c for c in cases if c['id']==selected[id])
        decision=item['group'];finding=item['finding'];test=item['id']
    elif n==28:decision='既知参照';finding='元のhnsd007_042。旧crop対照を再利用。';test='T00'
    elif n in art:decision='挿絵・色図を主評価から除外';finding='選ばれた濃い茶色領域は絵の線・面、または色刷りの領域。本文を覆うシミとは確定しない。';test=''
    elif n in cover:decision='表紙・ラベルを除外';finding='茶色い候補は表紙・ラベルの色や面。本文への接触汚れの主評価から除外。';test=''
    elif n in seal:decision='印・装飾を除外';finding='濃い候補は印影・装飾。シミと認定して除去しない。';test=''
    elif n in binding:decision='綴じ・ページ束の影を除外';finding='候補の多くは枠外の綴じ側やページ束の像。本文を覆うシミとは分ける。';test=''
    else:decision='同種の接触シミを未特定';finding='本文の筆画・薄い灰色像・紙面の濃淡が主。選んだ範囲で大きなシミの黒塊化を主評価へ追加する根拠は弱い。';test=''
    reviews.append(dict(screen_id=id,page=row['page'],decision=decision,test_id=test,
        crop_xywh=json.dumps(row['crop_xywh']),finding=finding,original=row['original'],black_mask=row['npj_black']))


def csvfile(name,rows):
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


csvfile('探索88ページの所見.csv',reviews)
csvfile('接触シミの候補一覧.csv',[{k:(json.dumps(c[k],ensure_ascii=False) if isinstance(c[k],list) else c[k])
    for k in ('id','page','group','priority','title','finding','page_detail_xywh','proposed_role','source_kind','comparison')}
    for c in cases])
benchmark=dict(objective='NPJ系の出力後、本文に重なる汚れを局所的に除去し、原画像で確認できる筆画を保つ',
    inputs_required=['original RGB image','NPJ-derived baseline output'],
    cases=cases,main_new_candidates=['T01','T02','T03','T04','T06'],reference=['T00'],
    separate_material_confirmation=['T07','C01'],nearby_not_overlapping_control=['T05'],
    initial_development_priority=['T01','T02','T03'],
    source_grouping_note='T00,T01,T02,T03,T04,T05,T06 are hnsd007 pages in the same collection. Nearby pages may share stain origins. Not independent documents.',
    unknown_document_generalization_demonstrated=False,reading_improvement_demonstrated=False,
    automatic_stain_detection_demonstrated=False,material_GT=False,
    selection_before_inference=True,parameters_tuned_to_outputs=False,
    coordinate_note='page_detail_xywh is a rectangular review ROI, not an edit mask, stain segmentation or pixel ground truth',
    next_evaluation=['Annotate visible necessary strokes and confirmed stain regions on original',
        'Compare original, frozen NPJ baseline, and proposed additional processing at same coordinates',
        'Check adjacent dark writing, faint strokes, small annotations, and rules remain readable',
        'Evaluate human reading/line tracking after processing rather than cosmetic black-pixel reduction'])
(HERE/'benchmark_candidates.json').write_text(json.dumps(benchmark,ensure_ascii=False,indent=2))

table='\n'.join(f'| {c["id"]} | {c["page"]} | {c["priority"]} | {c["title"]} | [拡大]({c["comparison"]})・[文脈]({c["context"]}) |' for c in cases)
text='''# 本文に重なるシミが黒塊化する例の追加探索

2026-10-07。前回の `hnsd007_042` と同種の問題を、原画像・旧文字マスク・実際のNPJ系出力で探した。

**追加の主候補は5ページ。特に `hnsd007_023`、`hnsd007_025`、`hnsd007_022` は、茶色いシミの下に見える本文・小さい文字が、処理後の黒い塊に覆われる差が明瞭。この課題を研究の中心に置いて局所的な分離・修復を検討するための具体例は増えた。**

ただし5ページは全て元の例と同じhnsd007資料内。近いページのシミには共通する形・位置もあり、同じ汚れの由来や近い撮影条件の可能性がある。5種類の独立した資料で再現したとは扱わない。別資料の `200021869_00014_2` にも茶色い跡と本文の一画の接触を確認したが、白い中心があり、シミか損傷かの確認を残して補助候補とした。

## 比較画像

- [主要例4つの一覧PNG：既知1＋追加3](figures/01_本文を覆うシミ.png)
- [小さい接触部・別資料・近接対照の一覧PNG](figures/02_接触と保留例.png)
- [全比較PDF](本文に重なるシミ_追加探索.pdf)
- [画像・座標・出自・実行記録のZIP](本文に重なるシミ_追加探索一式.zip)

| ID | ページ | 優先度 | 所見の要点 | 原画像／NPJ系出力 |
| --- | --- | --- | --- | --- |
{TABLE}

T00は2026-10-04の既存出力を再利用した参照。T01〜T07・C01は今回新しく推論した8crop。既知例を新しい発見として数えていない。T05は黒い塊が本文の近くにあるが、拡大すると隙間が見えるため、本文を覆う主要例から外した。C01は残留像の材質と、下に見える灰色線が表の筆画か裏写りかが不確定で、同種の確定例にはしない。

## どのように探したか

過去に原画像6,072ページを走査したランキングを再利用し、茶色い連結領域の上位350件から資料ごとの上限3ページで54ページを選び、元の例の資料群や既存の近接シミ候補を追加した。さらにhnsd007内を重点探索し、重複を除いた**88ページの選定cropとページ縮小像、保存黒文字マスクを目視**した。今回6,072ページ全ての復元出力を調べたわけではない。探索対象は原画像・マスクから選んだ問題の候補であり、発生頻度や全資料での失敗率は算出しない。

濃い茶色領域には、文字そのもの、印影、挿絵の黒い面、表紙のラベル、綴じ側の影も多数あった。これらを同じシミとして数えず、本文を覆う茶色い範囲を優先した。[全88ページの所見CSV](探索88ページの所見.csv)に採否を残した。

8cropの範囲・理由は、出力を見る前に[選定記録](inference_selection.json)へ固定した。原画像は変更せず、既存CSC v1の黒・朱マスクを同座標で切り出した。文字／汚れの手動マスク変更や、新処理の閾値探索はしていない。

## ベースラインの条件

元の例と同じ2026-10-02凍結ローカルddrm_npj_v2コード、同じ文字モデル重み、同じ紙面参照を使用。各cropを別プロセスで初期化し、seed 999、20段、128pxパッチ、重なり32px、masking=True、mask_timesteps=50、shared_noise=False。保存の `02_text_black` と `03_text_red` を使用した。04_degradationを欠損として渡した以前の診断出力は使っていない。

これは**NPJ系の通常黒文字マスクによるcrop対照**で、論文時点の完成した全ページ処理の忠実再現ではない。条件を変えて有利な見た目を選ぶことなく、同じ固定条件の結果を記録した。今回の推論は追加処理の成功確認ではなく、問題が残るベースラインを確かめるためのもの。

原画像と実入力の全画素一致、マスク切り出しの一致、出力寸法、Otsuへのフォールバックや推論結果欠落がないことを確認した。[推論検証](inference_verified.json)に実行コマンド・出力ハッシュを保存した。原画像・既存マスク・重み・凍結コードは変更していない。

## この問題を研究の中心に置く場合

対象を「NPJ系の出力後、本文に重なる汚れを局所的に除去し、原画像で確認できる筆画を残す」と設定する。最初の開発例はT01・T02・T03。T01では広い塊に覆われた線、T02では大きさの違う2つのシミ、T03では小さい添え書きの保持を確認できる。T04・T06は小さい接触部の確認に使う。T00は元の参照として保持する。

比較画像の中央の診断列では、茶色い像の一部・広い範囲が既に黒文字候補に入っていることを確認できる。[T01の原画像／保存黒文字マスク／出力](mask_diagnostics/T01.png)。この対応は誤分類候補を示すが、どこまでが汚れでどこからが必要な筆画かを自動判定できたわけではない。

**二段階目には原画像も渡す。** NPJ系出力の黒い塊だけから下の文字を取り戻そうとせず、原画像に残る線・輪郭を確認して汚れと分離する。太い字だけでなく、小さい文字、薄い線、枠線を保護する。原画像で判別できない部分を、正解の字形が分かったかのように補完した評価にはしない。

現在ある比較ROIは調査用の矩形であり、汚れの除去マスクや筆画の正解ではない。次はこの少数例について、原画像で確認できる汚れ／必要な筆画／不確定部分を注釈し、原画像・NPJ系ベースライン・NPJ系＋追加処理を同位置で比較する。シミが薄くなったか、筆画を消していないか、人が文字を追いやすくなったかを別に評価する。

T07は別資料への展開候補として材質確認を先に行う。今回は同資料内の関連例が中心なので、未知資料への一般化を実証したとは扱わない。T05や孤立シミ、正常な本文を併せて、余計な変更をしない対照にする。

[次の評価に渡す候補JSON](benchmark_candidates.json)と[候補CSV](接触シミの候補一覧.csv)に、優先順位・出自・原ページ座標・役割を記載した。現段階では自動検出、筆画分離、追加除去、読解改善の成功は未評価。今回確認できたのは、元の1例に限らず、同種の読みづらさを作る接触シミが実際の固定条件出力に残ること。

## 表示・検証

原画像とNPJ系出力の比較PNGは同じ座標、色調補正なし。拡大比較は原画素を整数2倍に反復、文脈比較は1倍。マスク診断は同位置の既存黒文字マスクを中央に表示した。2つの一覧PNGとPDFだけは閲覧用縮小・リサイズを含む。[表示パネルの座標](rendering.json)、[検証記録](verification.json)。

汚れの材質を専門家が認定した記録、筆画の画素正解、翻刻・読解試験はまだない。目視で筆画が見える／覆われることと、どの文字が読めるかの確定は分けている。
'''.replace('{TABLE}',table)
(HERE/'追加探索報告.md').write_text(text)
(HERE/'rendering.json').write_text(json.dumps(panels,ensure_ascii=False,indent=2))

# PDF collects the overview and all nine detailed evidence comparisons.
pdf_order=['figures/01_本文を覆うシミ.png','figures/02_接触と保留例.png']+[c['comparison'] for c in cases]
pdf=[]
for rel in pdf_order:
    im=Image.open(HERE/rel).convert('RGB');im.thumbnail((2220,2280),Image.Resampling.LANCZOS)
    page=Image.new('RGB',(2220,2340),BG);page.paste(im,((2220-im.width)//2,20));pdf.append(page)
pdf[0].save(HERE/'本文に重なるシミ_追加探索.pdf',save_all=True,append_images=pdf[1:],resolution=170,
            title='本文に重なるシミの追加探索・2026-10-07')

for p in panels:
    cv=Image.open(HERE/p['file']).convert('RGB');x,y,w,h=p['rect']
    got=np.asarray(cv.crop((x,y,x+w,y+h)))
    source=np.asarray(Image.open(HERE/p['native']).convert('RGB'))
    expected=np.repeat(np.repeat(source,p['scale'],axis=0),p['scale'],axis=1)
    assert np.array_equal(got,expected),p
for item in cases:
    for key,h in item['source_sha256'].items():assert sha(ROOT/item['source'][key])==h
for p,h in manifest['source_code_hashes'].items():assert sha(ROOT/p)==h
assert sha(ROOT/manifest['checkpoint'])==manifest['checkpoint_sha256']
assert sha(ROOT/manifest['background'])==manifest['background_sha256']
assert len({r['page'] for r in reviews})==88
verify=dict(original_mask_screen_pages=88,inference_crops=8,new_main_contact_candidates=5,
    main_new_pages=[c['page'] for c in cases if c['group'].startswith('追加')],
    all_new_main_candidates_same_hnsd007_source=True,reference_reused=1,
    separate_source_material_pending=1,near_contact_not_counted=1,uncertain_material_control=1,
    exact_input_original_crop=True,exact_mask_crop=True,all_inference_outputs_verified=True,
    evidence_panels_verified=len(panels),source_files_unchanged=True,pdf_pages=len(pdf_order),
    reading_GT=False,automatic_restoration_success_claimed=False)
(HERE/'verification.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2))
bundle=['追加探索報告.md','接触シミの候補一覧.csv','探索88ページの所見.csv','benchmark_candidates.json',
        'screening.json','inference_selection.json','inference_verified.json','verification.json',
        'rendering.json','本文に重なるシミ_追加探索.pdf','screen.py','extend.py','infer.py','package.py']
for folder in ('comparisons','contexts','mask_diagnostics','native','figures'):
    bundle += [str(p.relative_to(HERE)) for p in sorted((HERE/folder).rglob('*.png'))]
bundle += [str(p.relative_to(HERE)) for p in sorted((HERE/'logs').glob('*_verified.json'))]
with zipfile.ZipFile(HERE/'本文に重なるシミ_追加探索一式.zip','w',zipfile.ZIP_DEFLATED) as z:
    for rel in bundle:z.write(HERE/rel,rel)
for rel in re.findall(r'\]\(([^)]+)\)',text):
    assert (HERE/rel).exists(),rel
print(json.dumps(verify,ensure_ascii=False))
