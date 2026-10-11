"""Compose evidence from saved images; no inference or source-image modification."""
from pathlib import Path
import csv
import hashlib
import json
import math
import zipfile
from PIL import Image, ImageDraw, ImageFont
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
frames = {f['id']: f for f in json.loads((HERE / 'frames.json').read_text())}
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
BG, INK, MUTED = '#faf9f5', '#202d38', '#52616d'
for name in ('comparisons', 'native', 'figures'):
    (HERE / name).mkdir(exist_ok=True)


def case(id, frame, roi, title, priority, category, observation, reading, action, caveat):
    return dict(id=id, frame=frame, roi_xywh=roi, title=title, priority=priority,
                category=category, observation=observation, reading=reading,
                action=action, caveat=caveat)


cases = [
    case('S01', 'manual_control_hnsd007_042', [300,150,320,400],
         '本文に重なるシミが黒い塊になる', '高', '接触汚れ',
         '原画像では茶色い範囲の下に筆画が見える。出力ではその範囲が黒く埋まり、上の枠線と本文につながる。',
         '字の内部や隣接する筆画の区切りが見えにくくなる。今回、汚れ処理の目的に最も直接対応する候補。',
         '原画像も参照し、シミと筆画を分離して局所修復する。正常な列と枠線を保護する。',
         '2026-10-04の通常黒文字マスクによるcrop対照。完成した改善例ではない。NPJ出力だけの塊除去では、下の文字まで消す危険がある。'),
    case('S02', 'crop_hnsd', [0,270,540,330],
         '下部の粒状像が本文・枠線と接触する', '中', '接触汚れ候補',
         '下部の濃淡が、出力では粒状の黒い塊になり、枠線と本文下端に接触する。',
         '枠線に近い字の輪郭が背景側の黒い像と区別しづらい。ただし大半は枠外で、S01より本文への影響が局所的。',
         '本文に接触する範囲だけを候補にする。枠外の汚れ除去と本文の改善を別々に評価する。',
         '比較には2026-10-02の原画像そのままのcrop対照を使う。旧保存入力は濃淡が異なるが、今回表示する対照出力と旧出力は全画素一致。濃淡の材質は未確定。'),
    case('L01', 'known_200010454_v2', [1230,500,280,650],
         '薄い長い曲線が点線状になる', '高', '筆画欠損',
         '薄い本文列の長い曲線と輪郭が、出力では点線状・断片状になる。隣の濃い列は比較的よく残る。',
         '字形や連続した筆運びを追う手掛かりが減る。孤立シミよりも読解に関わる問題候補。',
         '原画像に見える筆画の保持・復元を評価する。汚れを消すだけでは回復しない。',
         '実保存のCSC v2全ページ版。同ページの別保存版を独立した文書とは数えない。筆画の正解と読解効果は未注釈。'),
    case('L02', 'page_200019865', [1750,2490,280,550],
         '下部の細い線に切れ目ができる', '中', '局所的な筆画欠損',
         '左の細い縦方向の線の下端付近に、原画像との比較で切れ目が見える。右の濃い列は概ね明瞭。',
         '線のつながりを追いにくい候補。ただし広範囲の文字消失ではなく、読解への影響は小さい可能性がある。',
         '小さな切れ目の接続補完の対照にする。大きな筆画欠損とは別の難易度で評価する。',
         'ddrm_visual_checkの全ページ保存出力。実入力と原画像は一致。特定の文字の誤読は確認していない。'),
    case('L03', 'saved_200018243_00002_2', [350,800,350,700],
         '本文の大きな輪郭が途中で欠ける', '高', '筆画欠損',
         '左側の本文に見える大きな曲線が途中で失われ、その下の小さな字形も黒い断片へ変わる。右側の濃い本文は残る。',
         '字形の輪郭と文字列の連続性が崩れ、原画像で追えていた形を追いにくくなる。',
         '正常な濃い本文を保護し、薄い表の筆画と背景の灰色像を原画像上で分けて評価する。',
         'ローカルv2・外部紙面参照の全ページ保存出力。灰色像をすべて表の筆画として復元することは正解とは限らない。'),
    case('L04', 'new_warm_faint_writing', [480,0,270,650],
         '薄い本文と小さな添え書きが断片化する', '高', '筆画欠損',
         '原画像の縦の本文と小さな添え書きに見える線が、出力では部分的に途切れる。大きな字形の内部にも欠けがある。',
         '本文の形だけでなく、隣の小さい文字を追う手掛かりも減る。',
         '本文と小さい文字の保持を一緒に確認する。原画像から確かめられる線を対象にする。',
         '2026-10-04に凍結ローカルコードで実行したcrop対照。歴史的保存版・全ページ段階後の出力ではない。'),
    case('L05a', 'new_warm_upper_loss', [390,290,360,450],
         '複数の本文文字が広く欠ける・上部', '高', '広範囲の筆画欠損',
         '原画像で見える複数の本文の輪郭・線が、出力では大きく欠け、点や断片に変わる。',
         '1か所の切れ目ではなく、複数の字形を追う情報が失われる。今回の中でも目視の差が大きい候補。',
         '薄い本文の保持・復元の主要評価例にする。近傍の残った字形も変更しすぎない。',
         '2026-10-04のcrop対照。L05bと同じページであり、独立した2ページではない。このROIは紙欠損の穴埋めを評価していない。'),
    case('L05b', 'new_warm_lower_loss', [0,120,480,450],
         '複数の本文文字が広く欠ける・下部', '高', '広範囲の筆画欠損',
         '左側の本文の輪郭と小さい文字が出力で断片化する。原画像の紙面に残っている線を比較している。',
         '字の内部と小さい文字を追いにくくなる。上部とは異なる場所でも同種の損失が見える。',
         '同ページ内の追加ROIとして、筆画を保持する処理の局所的な偏りを確認する。',
         'L05aと同じページの別crop。紙が物理的に失われている部分を、汚れ除去で復元できる例には数えない。'),
    case('N01', 'old_full_page', [760,1870,220,480],
         '主線の周囲に黒い点・断片が残る', '中', '残留像・材質未確定',
         '主線の近くにあった灰色像は大きく減るが、出力には黒い点・断片が残り、一部は主線に近接する。',
         '点・短い線を表の文字の一部と取り違える可能性がある。実際の読解妨害の程度は未確認。',
         '表の筆画、裏写り、紙の像を確認してから局所除去する。自動で全ての小点を消さない。',
         'v1trueという名前の全ページ保存出力。残留像の材質は目視だけでは確定できない。'),
    case('N02', 'unseen_lower_binding_mark', [360,0,370,400],
         '本文の周囲の黒い粒状像が残る', '中', '残留像・材質未確定',
         '原画像に既にある黒い粒状像が本文の周囲と輪郭近くにあり、出力でも残っている。',
         '筆画と周辺の細かい像の区別がしづらい候補。新たな悪化ではなく、読みづらさが残る例。',
         '本文近傍の粒状像の分離を試す候補。太い本文を保護し、判別できる粒状像だけを対象にする。',
         '2026-10-04のcrop対照。粒状像が汚れ・印刷のかすれ・別の線のどれかは未確定。綴じ側の黒い帯だけの除去は主目的にしない。'),
    case('R01', 'seal_control', [200,50,330,400],
         '薄い朱印がほぼ見えなくなる', '補足', '朱印の保持',
         '原画像では矩形の朱印と印面の線が見えるが、このcrop対照の出力ではほぼ見えない。',
         '印面・来歴情報の確認には関係するが、本文の汚れ除去とは評価対象が異なる。',
         '朱の保持を確認する補助例として記録する。本文の読解改善の主評価には混ぜない。',
         '2026-10-04のcrop対照。このページの薄い朱印についての所見であり、旧報告の別ページの赤黒重なり対照とは異なる。'),
]

# S02: use an exact-original-input control, rather than the darker historical input.
s02 = dict(frames['crop_hnsd'])
s02['input'] = 'tmp_work/npj_dirt_control_20261002/input/hnsd_raw_control.png'
s02['output'] = 'tmp_work/npj_dirt_control_20261002/output/hnsd_raw_control/result/hnsd_raw_control.png'
s02['input_original_exact'] = True
s02['kind'] = '2026-10-02・原画像入力crop対照'
s02['version'] = '凍結ローカルv2・既存CSC v1マスク'
historical = frames['crop_hnsd']

controls = [
    case('C01', 'manual_control_hnsd001_011', [40,100,310,310],
         '孤立シミ：除去の読解価値は低優先', '低', '対照',
         'シミは黒く残るが、近くの本文の筆画と離れている。',
         '孤立シミを消して見た目が整っても、文字が読みやすくなるかは別に確認する必要がある。',
         '外観改善の対照にする。接触汚れと区別する。', '手動マスクの成功例と同じページの処理前出力。'),
    case('C02', 'new_gray_binding_mark', [310,0,440,730],
         '綴じ側の帯：本文の外側', '低', '対照',
         '綴じ側の黒い縦帯は残るが、本文の主要部分から枠線で隔てられている。',
         '本文の読解改善の根拠としては弱い。', '位置が本文と接触しない対照として保持する。',
         '2026-10-04のcrop対照。帯の材質は未確定。'),
    case('C03', 'fresh_printed_text', [190,80,460,580],
         '濃い本文：形が概ね残る', '対照', '対照',
         '出力でも濃い本文の主要な形を追える。',
         '局所改善を追加するときに、読める部分を崩さないことを確認する対照。',
         '新処理でこのような本文を変更しすぎないことを評価する。', '細部の完全一致や専門家による読解確認を意味しない。'),
    case('C04', 'known_200010454_v2', [660,80,370,380],
         '版の違い：全ページ版で既に減ったシミ', '低', '対照',
         '旧切り出し版では残っていた茶色い斑点が、CSC v2全ページ保存版では大きく減る。',
         'この全ページ出力への追加除去の必要性は低い。',
         '版を固定し、既に除去できたものを改善対象として数えない。',
         '旧cropと全ページでは推論範囲・参照背景等も異なる。差を特定の1要因の効果とはしない。'),
]


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def wrap(text, draw, f, width):
    lines, line = [], ''
    for ch in text:
        if ch == '\n' or draw.textlength(line+ch, font=f) > width:
            lines.append(line)
            line = '' if ch == '\n' else ch
        else:
            line += ch
    if line:
        lines.append(line)
    return lines


def para(draw, text, xy, width, size=25, fill=INK, bold=False):
    x, y = xy
    for line in wrap(text, draw, font(size, bold), width):
        draw.text((x,y), line, font=font(size,bold), fill=fill)
        y += size+10
    return y


cache = {}
def images(c):
    f = s02 if c['id'] == 'S02' else frames[c['frame']]
    if c['id'] not in cache:
        orig = Image.open(ROOT / f['original']).convert('RGB')
        if f['origin_xywh']:
            x,y,w,h = f['origin_xywh']
            orig = orig.crop((x,y,x+w,y+h))
        inp = Image.open(ROOT / f['input']).convert('RGB')
        out = Image.open(ROOT / f['output']).convert('RGB')
        assert orig.size == inp.size == out.size
        assert np.array_equal(np.asarray(orig), np.asarray(inp)), c['id']
        cache[c['id']] = (orig, out, f)
    return cache[c['id']]


panels = []
all_cases = cases + controls
for c in all_cases:
    a,b,f = images(c)
    x,y,w,h = c['roi_xywh']
    assert 0 <= x < a.width and 0 <= y < a.height
    assert x+w <= a.width and y+h <= a.height
    box = (x,y,x+w,y+h)
    c['page'] = f['page']
    offset = f['origin_xywh'] or [0,0,0,0]
    c['page_xywh'] = [x+offset[0], y+offset[1], w,h]
    c['source'] = {k: f[k] for k in ('original','input','output','origin_xywh','kind','version')}
    c['source']['sha256'] = {k: hashlib.sha256((ROOT/f[k]).read_bytes()).hexdigest() for k in ('original','input','output')}
    c['native'] = {}
    dest = HERE/'native'/c['id']
    dest.mkdir(exist_ok=True)
    for label,im in (('original',a),('npj',b)):
        im.crop(box).save(dest/f'{label}.png')
        c['native'][label] = str((dest/f'{label}.png').relative_to(HERE))
    s = max(1,min(3,740//w,1400//h))
    pw,ph = w*s,h*s
    height = max(1200, ph+365)
    canvas = Image.new('RGB',(2000,height),BG)
    d = ImageDraw.Draw(canvas)
    accent = '#a04228' if c['priority']=='高' else '#2c627d'
    d.rectangle((0,0,2000,12),fill=accent)
    d.text((40,28),f'{c["id"]}  {c["title"]}',font=font(44,True),fill=INK)
    d.text((40,92),f'{f["page"]}  |  優先度：{c["priority"]}  |  {c["category"]}',font=font(26),fill=MUTED)
    mode = '保存ページ出力' if '全ページ' in f['kind'] else ('原画像入力のcrop対照・2026-10-02' if c['id']=='S02' else '凍結コードのcrop対照・2026-10-04')
    d.text((40,136),mode+'  ／  同じ位置・色調補正なし',font=font(26),fill=MUTED)
    x1,x2 = 40,800
    d.text((x1,194),'原画像（実入力と全画素一致）',font=font(28,True),fill=INK)
    d.text((x2,194),'NPJ系の処理後',font=font(28,True),fill=INK)
    for label,im,px in (('original',a,x1),('npj',b,x2)):
        scaled = im.crop(box).resize((pw,ph),Image.Resampling.NEAREST)
        canvas.paste(scaled,(px,248))
        panels.append(dict(file=f'comparisons/{c["id"]}.png',id=c['id'],label=label,
                           rect=[px,248,pw,ph],scale=s,native=c['native'][label]))
    side_x = 1580
    yy = para(d,'確認したこと',(side_x,192),375,28,bold=True)
    yy = para(d,c['observation'],(side_x,yy+10),375,25)
    yy = para(d,'読解との関係',(side_x,yy+20),375,28,bold=True)
    yy = para(d,c['reading'],(side_x,yy+10),375,25)
    yy = para(d,'原画像内の位置',(side_x,yy+20),375,25,bold=True)
    # Only this context thumbnail receives a locator box. Evidence panels are untouched.
    thumb = a.copy()
    thumb.thumbnail((340,max(100,min(440,height-yy-210))),Image.Resampling.LANCZOS)
    td = ImageDraw.Draw(thumb)
    fx,fy = thumb.width/a.width, thumb.height/a.height
    td.rectangle((x*fx,y*fy,(x+w)*fx,(y+h)*fy),outline='#ce302a',width=3)
    canvas.paste(thumb,(side_x,yy+10))
    foot_y = 248+ph+20
    d.text((40,foot_y),f'表示：原画素を{s}倍に整数反復。原ページROI (x,y,w,h) = {c["page_xywh"]}',font=font(24),fill=MUTED)
    para(d,'判断の範囲：'+c['caveat'],(40,foot_y+43),1460,24,fill=MUTED)
    c['comparison'] = f'comparisons/{c["id"]}.png'
    canvas.save(HERE/c['comparison'])

assert np.array_equal(np.asarray(Image.open(ROOT/s02['output']).convert('RGB')),
                      np.asarray(Image.open(ROOT/historical['output']).convert('RGB')))


def overview(name, title, subtitle, ids):
    canvas = Image.new('RGB',(2000,200+len(ids)*660),BG)
    d = ImageDraw.Draw(canvas)
    d.text((40,25),title,font=font(47,True),fill=INK)
    para(d,subtitle,(40,96),1900,28,fill=MUTED)
    for i,id in enumerate(ids):
        c=next(c for c in all_cases if c['id']==id)
        a,b,f=images(c)
        x,y,w,h=c['roi_xywh']; box=(x,y,x+w,y+h)
        yy=200+i*660
        d.line((40,yy,1960,yy),fill='#ccd1d5',width=2)
        d.text((40,yy+20),f'{id}  {c["title"]}',font=font(33,True),fill=INK)
        d.text((40,yy+76),f'{c["page"]}  /  優先度：{c["priority"]}',font=font(24),fill=MUTED)
        py=yy+154
        d.text((40,yy+116),'原画像',font=font(25,True),fill=INK)
        d.text((590,yy+116),'NPJ系の処理後',font=font(25,True),fill=INK)
        for im,px in ((a,40),(b,590)):
            t=im.crop(box); t.thumbnail((510,470),Image.Resampling.LANCZOS)
            canvas.paste(t,(px,py))
        sy=para(d,c['observation'],(1160,yy+116),760,31)
        sy=para(d,'追加処理：'+c['action'],(1160,sy+25),760,28)
        para(d,'出力：'+('実保存・全ページ' if '全ページ' in f['kind'] else '原画像入力のcrop対照'),(1160,sy+25),760,25,fill=MUTED)
    rel=f'figures/{name}.png'; canvas.save(HERE/rel)
    return rel


figures=[
    overview('01_汚れ関連の読みにくさ','汚れ除去の候補：本文に接触する黒い像',
             'S01が第一候補。材質・必要な筆画を確認し、本文に影響する範囲を優先する。',['S01','S02','N02']),
    overview('02_筆画欠損の読みにくさ','筆画の保持・復元の候補：字形が断片化する',
             '薄い線だけでなく、複数の本文の形が失われる例もある。汚れ除去とは別に評価する。',['L03','L04','L05a']),
    overview('03_低優先と正常対照','追加処理の優先度が低い例と、保護する本文',
             '汚れが残っていることだけで、読解改善の必要性が示されたとは扱わない。',['C01','C02','C03','C04']),
]

disposition = {
    'page_200019865': ('候補','L02：下部の細線。影響は局所的。'),
    'page_200017197': ('自然劣化の根拠から除外','原画像と実入力が異なり、実入力に原画像にない斑点がある。'),
    'known_200010454_v2': ('候補＋対照','L01：薄い本文の分断。C04：茶色い斑点は既に減る。'),
    'crop_hnsd': ('候補・原画像入力対照で照合','S02。旧入力の濃淡は異なるため原画像入力の対照で比較。出力は一致。'),
    'old_upper_crop': ('補助・同ページ','L01/C04と同じページ。小さな切れ目・旧版だけの斑点を独立ページにしない。'),
    'old_full_page': ('候補','N01：主線周辺の断片。材質未確定。'),
    'saved_200018243_00002_2': ('候補','L03：本文の大きな輪郭の欠損。'),
    'fresh_lower_paper': ('低優先・対照','濃い本文は残る。左下の薄い像の材質・読解影響は未確定。'),
    'fresh_binding_mark': ('低優先','綴じ側の像は本文の外側が中心。'),
    'wormhole': ('低優先','見た範囲では孤立した穴が主で、本文との重なりは明確でない。'),
    'seal_control': ('補足','R01：薄い朱印の保持。本文の汚れ除去とは分ける。'),
    'fresh_printed_text': ('保護対照','C03：濃い本文の主要な形が残る。'),
    'unseen_lower_frame_mark': ('低優先','粒状像の多くは枠外。本文への明確な接触を優先候補にはできなかった。'),
    'unseen_lower_binding_mark': ('候補','N02：本文近傍の粒状像が残る。材質未確定。'),
    'new_warm_upper_loss': ('候補','L05a：広範囲の筆画欠損。'),
    'new_warm_lower_loss': ('候補・同ページ','L05b：L05aと同ページの別位置。物理的紙欠損を除去課題とはしない。'),
    'new_warm_faint_writing': ('候補','L04：薄い本文・小さい文字の断片化。'),
    'new_gray_binding_mark': ('低優先','C02：綴じ帯は本文の外側が中心。'),
    'new_gray_writing': ('保護対照','濃い本文の主要な形は概ね残る。'),
    'manual_control_hnsd001_011': ('低優先','C01：孤立シミ。文字との接触は明確でない。'),
    'manual_control_hnsd007_042': ('第一候補','S01：シミの黒塊化で本文が隠れる。'),
}
assert set(disposition)==set(frames)
review = [dict(frame=f['id'], page=f['page'], disposition=disposition[f['id']][0],
               finding=disposition[f['id']][1], kind=f['kind'], version=f['version'],
               input_original_exact=f['input_original_exact'], original=f['original'],
               input=f['input'], output=f['output']) for f in frames.values()]


def csvfile(name, rows):
    with (HERE/name).open('w',newline='',encoding='utf-8-sig') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


csvfile('全フレームの確認結果.csv',review)
csvfile('読みにくさ候補一覧.csv',[
    {k:(json.dumps(c[k],ensure_ascii=False) if isinstance(c[k],list) else c[k])
     for k in ('id','page','priority','category','title','page_xywh','observation','reading','action','caveat','comparison')}
    for c in cases])
(HERE/'selection.json').write_text(json.dumps(dict(candidates=cases,controls=controls,
    figures=figures,scanned_frames=len(frames),unique_pages=len({f['page'] for f in frames.values()}),
    roi_coordinate_convention='top-left x,y,width,height; page coordinates refer to original file pixels',
    note='Visual candidates, not reading/transcription ground truth. No new inference in this task.'),ensure_ascii=False,indent=2))
(HERE/'rendering.json').write_text(json.dumps(panels,ensure_ascii=False,indent=2))

grid=json.loads((HERE/'screening_grid.json').read_text())
main=[c for c in cases if c['category']!='朱印の保持']
assert len(main)==10 and len({c['page'] for c in main})==9
assert len(frames)==21 and len({f['page'] for f in frames.values()})==17
report = '''# NPJ系の処理後にも読みにくさが残る箇所の洗い出し

2026-10-07。原画像と保存出力の同じ位置を比較した。今回、新しい修復推論や閾値調整は行っていない。

**「NPJ系でまず修復し、残った問題のある箇所だけを直す」という提案を検討できる例はある。汚れ除去の第一候補は、本文に重なる茶色いシミが黒い塊になる `hnsd007_042`。一方、薄い筆画や複数の字形が欠ける例もあり、こちらは筆画の保持・復元を別に評価する必要がある。**

今回確認したのは既存の21フレーム・17ページ、{tiles}タイル。本文に関わる目視候補を9ページ・10領域、朱印を補足1領域として整理した。同ページの上部・下部や版違いは独立ページと数えない。過去に問題を調べた開発例を多く含み、文書全体の失敗率を示す調査ではない。

## まず見る比較

- [汚れ関連：本文に接触する黒い像](figures/01_汚れ関連の読みにくさ.png)
- [筆画欠損：字形が断片化する](figures/02_筆画欠損の読みにくさ.png)
- [低優先の汚れと、保護すべき本文](figures/03_低優先と正常対照.png)
- [比較資料PDF](NPJ処理後の読みにくさ_比較資料.pdf)、[画像・一覧・出自のZIP](NPJ処理後の読みにくさ_資料一式.zip)

## 候補一覧

優先度は目視での調査順序であり、読解試験の成績ではない。「高」は差が明瞭で読解の手掛かりに関わりそうな領域、「中」は局所的な切れ目や材質の確認が残る像。

| ID | ページ | 優先度 | 見えている問題 | 次に試すこと | 比較 |
| --- | --- | --- | --- | --- | --- |
{table}

## 二段階の提案として何を扱うか

### 1. 汚れが残って本文を妨げる箇所

**S01を最初の評価例にする。** 原画像では茶色いシミの下に筆画が見えるが、NPJ系の出力ではシミが黒く埋まり、文字と枠線に接続している。ここで筆画を残して汚れを分離できれば、「紙がきれいになる」だけでなく「文字の形を追いやすくする」という目的に結びつく。

ただし、黒い塊を丸ごと背景で埋めると文字まで消える。この例を直す二段階目は、NPJ出力に加えて原画像を参照できる設計が必要になる。出力内で失われた筆画を原画像から確かめ、正常な列と枠線を保護する。汚れ・筆画の少数の確認注釈から始める。

N02は「新しく悪化した」例ではなく、原画像で既に本文の周囲にある粒状像が残る例。S02では本文と接触する部分を優先し、枠外の大半を消すだけで本文の改善が示されたとは扱わない。N01/N02の粒状像が裏写り・汚れ・必要な線のどれかは未確定で、除去前に確認する。

### 2. 処理で筆画が欠ける箇所

L01、L03、L04、L05a/bは原画像に見える形が出力で途切れたり断片化したりする候補。L05a/bでは複数の本文の形が広く欠け、数画素の切れ目をつなぐ問題とは規模が違う。L02は比較的小さな切れ目の対照として使う。

この群は、汚れを除去するだけでは直らない。研究の中心を汚れ除去に置く場合も、これらを「消してはいけない薄い線」の保護対照にする価値がある。字形が復元されたかを評価するときは、原画像で確認できる必要な筆画をまず確認する。灰色像を全て裏写りとして消すことも、全て本文として戻すことも正解と決めない。

### 3. 低優先と正常対照

C01の孤立シミは本文から離れており、除去できることと読解が改善することは別。C02の綴じ側の帯も本文の外側が中心で、まずは本文に接触する像を優先する。C03のように主要な本文の形が残る例を、追加処理で崩さない対照にする。

C04の茶色い斑点は旧切り出し版では残っていたが、今回対象のCSC v2全ページ保存版では既に大きく減る。[以前の旧crop／全ページの比較](../npj_need_assessment_20261002/26_stain_crop_vs_page.png)も参照する。これを共通の未解決問題には数えない。R01の薄い朱印の消失は補足で、本文の読解改善とは分ける。

## 出力の出自と判断の範囲

- 全ページの歴史的保存出力と、2026-10-02/04に凍結ローカルコードで実行済みのcrop対照を明示して比較した。保存名だけから論文時点の最終出力の忠実再現とは認定していない。crop対照は全ページ工程後の結果ではない。
- S02の旧保存入力は原画像と濃淡が異なる。今回は原画像をそのまま入力した既存対照を表示し、その出力が旧保存出力と全画素一致することを再確認した。2実行を独立した2成功／失敗とは数えない。
- `200017197_00008_1`は実入力に原画像にない斑点があり、自然劣化を直す根拠から除外した。
- 04_degradationを欠損として渡した以前の診断出力は、今回の候補に使っていない。通常黒文字マスクによる修復出力との混同を避ける。
- 比較は原画像・実入力の全画素一致と寸法を確認して作成した。領域ごとのPNGは色調補正なし、原画素の整数反復で拡大。位置を示す赤枠は補助縮小画像にだけ描いた。全体一覧とPDFは閲覧用に縮小する。
- 読解試験、専門家の翻刻、筆画の正解注釈は未実施。ここでの「読みにくさ」は次の評価に使う目視候補であり、特定の文字の誤読や手法の改善効果を確定していない。

## 次の評価に渡せるもの

[領域ごとの座標・所見CSV](読みにくさ候補一覧.csv)、[21フレームの確認結果CSV](全フレームの確認結果.csv)、[出自・座標JSON](selection.json)、[検証記録](verification.json)。`native/<ID>/original.png` と `npj.png` は同一位置の無補正切り出し。座標は原ページの左上を基準とした `x,y,width,height`。

最初の小規模評価はS01に読解を妨げる汚れ／残す筆画の注釈を付け、C01・C03を対照にして行える。比較条件は「原画像」「NPJ系のベースライン」「NPJ系＋追加処理」を固定し、汚れの見た目、必要な筆画の保持、人が文字を追えるかを別々に記録する。現在の手動閾値・手動マスクの成功を、この候補群での自動的な改善や一般化の達成とは扱わない。

進捗報告には、次のように記載できる。

> NPJ系の出力後にも、本文に接触する汚れの黒塊化と、薄い筆画の分断・欠損が残る候補を確認した。追加処理は本文に影響する局所領域を対象とし、まず原画像を参照した汚れと筆画の分離を検討する。現段階では読解改善の実証と未知文書への一般化は未評価である。
'''
table='\n'.join(f'| {c["id"]} | {c["page"]} | {c["priority"]} | {c["title"]} | '+
                ('局所汚れ分離・筆画保護' if c['id']=='S01' else
                 '筆画の保持・復元' if c['id'].startswith('L') else
                 '朱の保持を別途確認' if c['id']=='R01' else '材質と本文への影響を確認')+
                f' | [PNG]({c["comparison"]}) |' for c in cases)
(HERE/'調査報告.md').write_text(report.replace('{tiles}',str(len(grid))).replace('{table}',table))

# Report PDF: overview pages, then unannotated evidence comparisons, then controls.
pdf_order=figures[:2]+[c['comparison'] for c in cases]+[figures[2]]
pdf_pages=[]
for rel in pdf_order:
    im=Image.open(HERE/rel).convert('RGB')
    im.thumbnail((2000,1800),Image.Resampling.LANCZOS)
    page=Image.new('RGB',(2000,1900),BG)
    page.paste(im,((2000-im.width)//2,20))
    pdf_pages.append(page)
pdf_pages[0].save(HERE/'NPJ処理後の読みにくさ_比較資料.pdf',save_all=True,
                  append_images=pdf_pages[1:],resolution=160.0,title='NPJ処理後の読みにくさ候補・2026-10-07')

# Verify exported native pixels and evidence panel pixel repetition.
verified=[]
for p in panels:
    im=Image.open(HERE/p['file']).convert('RGB')
    x,y,w,h=p['rect']
    got=np.asarray(im.crop((x,y,x+w,y+h)))
    native=np.asarray(Image.open(HERE/p['native']).convert('RGB'))
    expected=np.repeat(np.repeat(native,p['scale'],axis=0),p['scale'],axis=1)
    assert np.array_equal(got,expected),p
    verified.append(dict(id=p['id'],label=p['label'],panel_exact=True))
for c in all_cases:
    for key,digest in c['source']['sha256'].items():
        assert hashlib.sha256((ROOT/c['source'][key]).read_bytes()).hexdigest()==digest

# Fixed-grid coverage audit. Rectangles do not overlap and cover each saved frame.
for id,f in frames.items():
    area=sum(t['xywh'][2]*t['xywh'][3] for t in grid if t['frame']==id)
    assert area==f['shape_wh'][0]*f['shape_wh'][1],id
    tiles=[t['xywh'] for t in grid if t['frame']==id]
    for i,(x,y,w,h) in enumerate(tiles):
        for xx,yy,ww,hh in tiles[i+1:]:
            assert min(x+w,xx+ww)<=max(x,xx) or min(y+h,yy+hh)<=max(y,yy)
verification=dict(scanned_frames=len(frames),unique_pages=len({f['page'] for f in frames.values()}),
    grid_tiles=len(grid),grid_full_coverage=True,main_candidate_pages=9,main_candidate_regions=10,
    supplementary_regions=1,evidence_panels=verified,source_files_unchanged=True,
    selected_inputs_exact_to_original=True,hnsd_control_equal_historical=True,
    new_inference_executed=False,pdf_pages=len(pdf_order))
(HERE/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
bundle_files=['調査報告.md','読みにくさ候補一覧.csv','全フレームの確認結果.csv',
              'selection.json','frames.json','screening_grid.json','rendering.json','verification.json',
              'NPJ処理後の読みにくさ_比較資料.pdf','build_inventory.py','screen.py']
bundle_files += [str(p.relative_to(HERE)) for folder in ('figures','comparisons','native') for p in sorted((HERE/folder).rglob('*.png'))]
with zipfile.ZipFile(HERE/'NPJ処理後の読みにくさ_資料一式.zip','w',zipfile.ZIP_DEFLATED) as z:
    for rel in bundle_files:
        z.write(HERE/rel,rel)
print(json.dumps({k:v for k,v in verification.items() if k!='evidence_panels'},ensure_ascii=False))
