"""Reframe the saved, unchanged outputs around Sauvola versus Howe."""
from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'sauvola_howe';OUT.mkdir(exist_ok=True)
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def text(d,xy,s,n=18):d.text(xy,s,font=ImageFont.truetype(FONT,n),fill='#233640')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rgb(p):return np.array(Image.open(p).convert('RGB'))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))

cases=json.loads((HERE/'selection.json').read_text())['cases']
notes={
 '01':'Howeは上の極細線をつなぐ。下の淡い輪郭は両方で欠ける。',
 '02':'Howeでも広い筆画欠損が残る。文字保持を確認する共通の難例。',
 '03':'Howeは染みを減らすが筆画を切る。Sauvolaはつながりと汚れを残す。',
 '04':'Howeが斑点を除去。前回の「両方で残る」という判断を訂正。',
 '05':'密な粒状像・暗い帯は両方で残る。不要画素の範囲は未確定。',
 '06':'原画像の欠損と処理による欠けを分けて確認する。',
 '07':'Howeは細線を比較的つなぐ。灰色像の除去正解は判定保留。',
}
records=[];dependencies={};checked=0
for c in cases:
    source=ROOT/c['original'];assert sha(source)==c['source_sha256']
    dependencies[c['original']]=sha(source)
    arrays={'original':rgb(source)}
    for key in ['sauvola','howe']:
        p=HERE/'masks'/c['page']/(key+'.png');arrays[key]=rgb(p);dependencies[str(p.relative_to(ROOT))]=sha(p)
    for detail in [False,True]:
        x,y,w,h=c['detail_xywh'] if detail else c['roi_xywh'];scale=2 if detail else 1
        pw,ph=w*scale,h*scale;gap=22;top=145
        out=Image.new('RGB',(3*pw+gap*4,top+ph+70),'#faf9f5');d=ImageDraw.Draw(out)
        text(d,(20,12),c['id']+' / '+('詳細2倍' if detail else '原寸比較'),24)
        text(d,(20,49),notes[c['id'][:2]],18)
        text(d,(20,79),c['page']+f' / xywh={x},{y},{w},{h} / 同じ全ページ処理結果',16)
        panels=[]
        for i,(key,label) in enumerate([('original','原画像'),('sauvola','Sauvola / w=61, k=.2'),('howe','Howe 基本エネルギー移植 / c=160')]):
            px=gap+i*(pw+gap);text(d,(px,111),label,18)
            a=arrays[key][y:y+h,x:x+w];a=np.repeat(np.repeat(a,scale,0),scale,1)
            out.paste(Image.fromarray(a),(px,top));panels.append(dict(method=key,xywh=[px,top,pw,ph]))
        text(d,(20,top+ph+18),'原画素を維持。HoweはCanny近似・固定設定。原法の自動調整は未再現。',16)
        text(d,(20,top+ph+43),'探索例の比較。正解マスクによる精度・総合順位ではない。',16)
        folder=OUT/('details' if detail else 'comparisons');folder.mkdir(exist_ok=True)
        p=folder/(c['id']+'.png');out.save(p);saved=rgb(p)
        for panel in panels:
            a=arrays[panel['method']][y:y+h,x:x+w];a=np.repeat(np.repeat(a,scale,0),scale,1)
            px,py,ww,hh=panel['xywh'];assert np.array_equal(saved[py:py+hh,px:px+ww],a);checked+=1
        records.append(dict(case=c['id'],detail=detail,roi_xywh=[x,y,w,h],scale=scale,panels=panels,path=str(p.relative_to(OUT)),sha256=sha(p)))

# The rectangle isolates the dark spot from neighboring handwriting; it is not
# dense GT and the count is not a document-wide precision measure.
spot=[460,590,90,60];x,y,w,h=spot
counts={key:int((rgb(HERE/'masks/hnsd001_011'/f'{key}.png')[y:y+h,x:x+w,0]==0).sum()) for key in ['sauvola','wolf','howe']}
assert counts['howe']==0
c=next(c for c in cases if c['id'].startswith('04'))
native=Image.open(ROOT/c['original']).convert('RGB');native=native.crop((340,400,860,920))
d=ImageDraw.Draw(native);d.rectangle((x-340,y-400,x+w-341,y+h-401),outline='#e00030',width=2)
native.save(OUT/'斑点_確認矩形.png')
audit=dict(page=c['page'],rectangle_xywh=spot,rectangle_area=w*h,foreground_black_pixels=counts,scope='Post-output manual rectangle isolating the spot, not dense GT or overall accuracy.',finding='Howe removes this isolated dark spot; previous common-failure description was a visual-review error.',algorithm_changed=False)
dump(OUT/'斑点_再確認.json',audit)

overview=Image.new('RGB',(1240,100+380*len(cases)),'#faf9f5');d=ImageDraw.Draw(overview)
text(d,(15,12),'Sauvola と Howe：課題別比較',26)
text(d,(15,51),'左：原画像　中央：Sauvola　右：Howe移植。縮小一覧、詳細PNGは原画素維持。',18)
for i,c in enumerate(cases):
    py=100+i*380;text(d,(15,py),c['id'],20);text(d,(15,py+29),notes[c['id'][:2]],16)
    x,y,w,h=c['roi_xywh']
    for j,key in enumerate(['original','sauvola','howe']):
        im=Image.open(HERE/'native'/c['id']/(key+'.png')).convert('RGB');im.thumbnail((390,300),Image.Resampling.NEAREST)
        overview.paste(im,(15+j*410,py+65))
overview.save(OUT/'Sauvola_Howe_比較一覧.png')

lines=['# SauvolaとHoweの比較・斑点所見の訂正','',
 '主比較をSauvolaとHoweに絞った。既存の原画像と保存マスクをそのまま使い、7領域の原寸比較と詳細2倍のPNGを作成した。Wolfは以前の3法比較に残している。','',
 '**04の濃い孤立斑点はHoweで除去できている。前回の「3法とも黒い塊を残す」は目視の誤りだった。** 元の報告・JSON・CSV・再生成コードも訂正した。アルゴリズムや出力マスクを変更した結果ではない。','',
 '[比較一覧PNG](Sauvola_Howe_比較一覧.png)','',
 '| 課題 | 所見 | 原寸 | 詳細2倍 |','| --- | --- | --- | --- |']
for c in cases:
    lines.append(f'| {c["category"]} | {notes[c["id"][:2]]} | [PNG](comparisons/{c["id"]}.png) | [PNG](details/{c["id"]}.png) |')
lines += ['', '## 斑点の訂正を確認した範囲','',
 '対象は hnsd001_011 の原ページ座標 x=460, y=590, width=90, height=60。[原画像上の赤い確認矩形](斑点_確認矩形.png)で位置を確認できる。', '',
 f'この矩形内の黒画素はSauvola {counts["sauvola"]}、Wolf {counts["wolf"]}、Howe {counts["howe"]}。Howeは当該矩形内で全て背景となり、比較画像でも孤立斑点が消えている。矩形は隣の文字を避けて選んだ確認用領域で、画素正解や一般的な汚れ除去精度ではない。[確認記録](斑点_再確認.json)。', '',
 '## 2手法の争点','',
 '現在の例では、Howeは孤立斑点の除去と極細線の連続性で有力。一方、hnsd007_042の接触染みでは、Howeが染みを減らす際に原画像で見える中央筆画を大きく切っている。Sauvolaは汚れが残るものの、その筆画のつながりを比較的保持する。したがって「背景が白い方が良い」だけで優劣を決めない。', '',
 '今後の比較で中心に置くのは、03の接触染みで必要な筆画を保てるか、01・02の薄い文字を残せるか、04で得られた斑点除去を維持できるか。これらは提案手法を判断する条件として固定できる。', '',
 '05は材質不明の粒状像、06は原画像の欠損、07は表文字と裏写りの正解未確定を含むため、全てを同じ二値化の失敗率に合算しない。', '',
 '## 条件と検証','',
 '入力・窓・パラメータは前回と同一。Sauvola w=61, k=.2, R=.5、Howe基本エネルギー移植 c=160。Howeはskimage Canny近似で、自動調整は未再現。この固定条件での探索的比較であり、手法一般の順位ではない。', '',
 f'原寸7枚・詳細7枚、計{checked}パネルを保存後に再読込し、元の原画像／全ページマスクの該当画素と一致することを確認。詳細は整数2倍反復。元画像と使用マスクのSHA-256を[verification.json](verification.json)に保存した。']
(OUT/'比較報告.md').write_text('\n'.join(lines)+'\n')
for p,h in dependencies.items():assert sha(ROOT/p)==h
dump(OUT/'verification.json',dict(exact_panels=checked,figures=records,inputs_sha256=dependencies,spot_audit=audit,method_parameters_changed=False,script_sha256=sha(Path(__file__))))
print(json.dumps(dict(figures=len(records),exact_panels=checked,spot_black_pixels=counts),ensure_ascii=False))
