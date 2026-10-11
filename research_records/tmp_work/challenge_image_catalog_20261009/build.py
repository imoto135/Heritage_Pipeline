"""Build a source-checked catalog of existing binarization challenge images.

Uses existing frozen selections and PNGs. It does not rerun any method.
"""
from pathlib import Path
import csv
import hashlib
import json

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / 'tmp_work/classical_limits_by_category_20261008'
TWO = PRIOR / 'sauvola_howe'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

ROLES = {
    '01': ('主対象', '薄い筆画の保持', '原画像で追える細線が処理で途切れる。下側の淡い輪郭を重点確認。',
           '線の連続・文字形・余分な黒点を別々に記録する。'),
    '02': ('主対象', '水染み中の薄い文字', '染みの中の複数文字が欠ける。濃い右列は内部対照。',
           '読める字形と曖昧な筆画を分けて注記する。画素正解は未確定。'),
    '03': ('主対象', '染みと筆画の接触', '染みを消すと原画像で見える中央筆画も途切れる。',
           '筆画保持と周辺の汚れ残存を対で確認する。'),
    '04': ('成功対照', '孤立斑点の除去', '本文から離れた濃い斑点をHoweが除去できている。',
           '新しい条件でも斑点を再出現させない。'),
    '05': ('判定保留', '色のある紙面の粒状像', '黄色みのある紙面で濃い粒状像が目立つ。粒状像の材質は未確定。',
           '粒状像と近くの筆画を分けて確認し、材質を確認するまでは除去を改善と数えない。'),
    '06': ('別課題', '虫食い状の紙面損傷', '左上に枝分かれ状の跡があり、後続ページにも近い位置の跡が続く。原因は未確定。',
           '原本の調査記録・現物で穴や表面損傷の有無を確認する。本文外なので文字二値化の失敗件数には含めない。'),
    '07': ('追加対象', '強い裏移りと表文字', '青灰色の像は直前の図柄ページと反転・位置合わせで対応する。表文字との画素境界は未確定。',
           '裏移りの除去と、重なる表文字の保持を別々に目視確認する。'),
}


def source_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def selection():
    raw = json.loads((PRIOR / 'selection.json').read_text())
    rows = []
    for c in raw['cases']:
        number = c['id'][:2]
        assert number in ROLES
        if number in ('05', '06', '07'):
            continue
        role, challenge, criterion, next_check = ROLES[number]
        source = ROOT / c['original']
        assert source.exists() and source_hash(source) == c['source_sha256']
        with Image.open(source) as image:
            width, height = image.size
        x, y, w, h = c['roi_xywh']
        assert 0 <= x < width and 0 <= y < height and x+w <= width and y+h <= height
        comparison = PRIOR / 'comparisons' / (c['id'] + '.png')
        detail = PRIOR / 'details' / (c['id'] + '_詳細2倍.png')
        two_comparison = TWO / 'comparisons' / (c['id'] + '.png')
        context = PRIOR / 'contexts' / (c['id'] + '.png')
        for path in (comparison, detail, two_comparison, context):
            assert path.exists(), path
        title=challenge if number=='06' else c['id'].split('_',1)[1]
        rows.append(dict(id=number, title=title, page=c['page'],
                         original=c['original'], source_sha256=c['source_sha256'],
                         roi_xywh=c['roi_xywh'], detail_xywh=c['detail_xywh'],
                         role=role, challenge=challenge, criterion=criterion,
                         next_check=next_check, judgment=c['judgment'],
                         original_context=str(context.relative_to(ROOT)),
                         comparison_sauvola_howe=str(two_comparison.relative_to(ROOT)),
                         comparison_three_methods=str(comparison.relative_to(ROOT)),
                         detail_three_methods=str(detail.relative_to(ROOT)),
                         primary_comparison=str(two_comparison.relative_to(ROOT)),
                         detail_comparison=str(detail.relative_to(ROOT))))
    source=ROOT/'data/split_dataset/val/200021851/200021851_00007_2.jpg'
    rows.append(dict(id='05',title='色のある紙面の粒状像',page='200021851_00007_2',
                     original=str(source.relative_to(ROOT)),source_sha256=source_hash(source),
                     roi_xywh=[1635,2175,520,520],detail_xywh=[1680,2250,340,330],
                     role=ROLES['05'][0],challenge=ROLES['05'][1],criterion=ROLES['05'][2],
                     next_check=ROLES['05'][3],judgment='colored_granular_mark_material_uncertain',
                     original_context='tmp_work/challenge_image_catalog_20261009/05_原ページ位置.png',
                     comparison_sauvola_howe='',comparison_three_methods='',detail_three_methods='',
                     primary_comparison='tmp_work/challenge_image_catalog_20261009/05_原画像_CSC_Sauvola_Howe.png',
                     detail_comparison='tmp_work/challenge_image_catalog_20261009/05_原画像_CSC_Sauvola_Howe_詳細2倍.png'))
    source=ROOT/'data/test/200003803/200003803_00003_1.jpg'
    rows.append(dict(id='06',title='左上の虫食い状の紙面損傷',page='200003803_00003_1',
                     original=str(source.relative_to(ROOT)),source_sha256=source_hash(source),
                     roi_xywh=[80,95,620,530],detail_xywh=[175,150,250,220],
                     role=ROLES['06'][0],challenge=ROLES['06'][1],criterion=ROLES['06'][2],
                     next_check=ROLES['06'][3],judgment='insect_like_damage_cause_unconfirmed',
                     original_context='tmp_work/challenge_image_catalog_20261009/06_原ページ位置.png',
                     comparison_sauvola_howe='',comparison_three_methods='',detail_three_methods='',
                     primary_comparison='tmp_work/challenge_image_catalog_20261009/06_連続ページ同領域.png',
                     detail_comparison='tmp_work/challenge_image_catalog_20261009/06_損傷詳細2倍.png'))
    source=ROOT/'data/split_dataset/val/200021660/200021660_00028_1.jpg'
    rows.append(dict(id='07',title='強い裏移りと表文字の重なり',page='200021660_00028_1',
                     original=str(source.relative_to(ROOT)),source_sha256=source_hash(source),
                     roi_xywh=[1093,1723,768,768],detail_xywh=[1240,1880,360,360],
                     role=ROLES['07'][0],challenge=ROLES['07'][1],criterion=ROLES['07'][2],
                     next_check=ROLES['07'][3],judgment='showthrough_supported_by_adjacent_page',
                     original_context='tmp_work/challenge_image_catalog_20261009/07_原ページ位置.png',
                     reverse_candidate_page='200021660_00027_2',
                     reverse_candidate_comparison='tmp_work/challenge_image_catalog_20261009/07_裏面候補対照.png',
                     reverse_alignment_evidence='tmp_work/challenge_image_catalog_20261009/07_裏移り_前後ページ照合.json',
                     comparison_sauvola_howe='',comparison_three_methods='',detail_three_methods='',
                     primary_comparison='tmp_work/challenge_image_catalog_20261009/07_原画像_CSC保存マスク.png',
                     detail_comparison='tmp_work/challenge_image_catalog_20261009/07_原画像_CSC保存マスク_詳細2倍.png'))
    assert len(rows)==7 and len({r['id'] for r in rows})==7
    return sorted(rows,key=lambda r:r['id'])


def new_05_assets():
    source=ROOT/'data/split_dataset/val/200021851/200021851_00007_2.jpg'
    maskdir=ROOT/'data/split_dataset_csc_mask_v2/val/200021851/200021851_00007_2'
    classical=PRIOR/'masks/200021851_00007_2'
    paths=[source,maskdir/'02_text_black.png',classical/'sauvola.png',classical/'howe.png']
    labels=['原画像','CSC v2 黒文字','Sauvola','Howe 移植']
    ims=[]
    for path in paths:
        with Image.open(path) as image: ims.append(image.convert('RGB'))
    assert all(im.size==ims[0].size for im in ims)
    font=ImageFont.truetype(FONT,22);small=ImageFont.truetype(FONT,16)
    import numpy as np
    for box,scale,name in [((1635,2175,2155,2695),1,'05_原画像_CSC_Sauvola_Howe.png'),
                           ((1680,2250,2020,2580),2,'05_原画像_CSC_Sauvola_Howe_詳細2倍.png')]:
        crops=[im.crop(box) for im in ims]
        if scale==2:
            crops=[im.resize((im.width*2,im.height*2),Image.Resampling.NEAREST) for im in crops]
        w,h=crops[0].size;gap=20;top=105
        canvas=Image.new('RGB',(w*4+gap*5,h+top+72),'#f8f7f3');d=ImageDraw.Draw(canvas)
        d.text((20,12),'05 色のある紙面の粒状像：4法比較',font=font,fill='#1b2c38')
        d.text((20,49),f'200021851_00007_2 / xywh={box[0]},{box[1]},{box[2]-box[0]},{box[3]-box[1]} / {scale}倍',font=small,fill='#405364')
        for i,(label,crop) in enumerate(zip(labels,crops)):
            px=gap+i*(w+gap)
            d.text((px,top-35),label,font=small,fill='#1b2c38')
            canvas.paste(crop,(px,top))
        d.text((20,canvas.height-49),'粒状像の材質は未確定。黒画素の多少だけで良否を決めない。',font=small,fill='#405364')
        pixels=np.asarray(canvas)
        for i,crop in enumerate(crops):
            px=gap+i*(w+gap)
            assert np.array_equal(pixels[top:top+h,px:px+w],np.asarray(crop))
        canvas.save(HERE/name)
    x,y,w,h=1635,2175,520,520
    page=ims[0].copy();d=ImageDraw.Draw(page);d.rectangle((x,y,x+w,y+h),outline='red',width=12)
    ImageOps.contain(page,(900,1350),Image.Resampling.LANCZOS).save(HERE/'05_原ページ位置.png')


def new_06_assets():
    first=ROOT/'data/test/200003803/200003803_00003_1.jpg'
    later=ROOT/'data/test/200003803/200003803_00005_1.jpg'
    x,y,w,h=80,95,620,530
    font=ImageFont.truetype(FONT,22);small=ImageFont.truetype(FONT,16)
    crops=[]
    for path in (first,later):
        with Image.open(path) as image:
            raw=image.convert('RGB')
        crops.append(raw.crop((x,y,x+w,y+h)))
    gap=24;top=100
    canvas=Image.new('RGB',(w*2+gap*3,h+top+72),'#f8f7f3')
    d=ImageDraw.Draw(canvas)
    d.text((24,12),'06 同資料の連続ページ：左上の枝分かれ状の跡',font=font,fill='#1b2c38')
    for i,(path,crop) in enumerate(zip((first,later),crops)):
        px=gap+i*(w+gap)
        d.text((px,62),path.stem,font=small,fill='#405364')
        canvas.paste(crop,(px,top))
    d.text((24,canvas.height-51),'形の連続は見えるが、虫害による穴かどうかは画像だけでは未確定。',font=small,fill='#405364')
    import numpy as np
    pixels=np.asarray(canvas)
    for i,crop in enumerate(crops):
        px=gap+i*(w+gap)
        assert np.array_equal(pixels[top:top+h,px:px+w],np.asarray(crop))
    canvas.save(HERE/'06_連続ページ同領域.png')
    detail=crops[0].crop((175-x,150-y,175-x+250,150-y+220))
    detail=detail.resize((detail.width*2,detail.height*2),Image.Resampling.NEAREST)
    detail.save(HERE/'06_損傷詳細2倍.png')
    with Image.open(first) as image: page=image.convert('RGB')
    d=ImageDraw.Draw(page);d.rectangle((x,y,x+w,y+h),outline='red',width=14)
    ImageOps.contain(page,(900,1300),Image.Resampling.LANCZOS).save(HERE/'06_原ページ位置.png')
    names=[f'200003803_{page:05d}_{side}.jpg' for page in range(2,8) for side in (1,2)]
    tw,th=450,470
    sheet=Image.new('RGB',(tw*4,th*3),'white');draw=ImageDraw.Draw(sheet)
    for i,name in enumerate(names):
        with Image.open(first.parent/name) as image:
            crop=image.convert('RGB').crop((60,70,550,580))
        crop.thumbnail((tw,th-20),Image.Resampling.LANCZOS)
        px=i%4*tw;py=i//4*th
        sheet.paste(crop,(px,py+20))
        draw.text((px+5,py+2),name,font=small,fill='#1b2c38')
    sheet.save(HERE/'06_新候補_連続ページ左上.png')


def new_07_assets():
    source=ROOT/'data/split_dataset/val/200021660/200021660_00028_1.jpg'
    maskfile=ROOT/'data/split_dataset_csc_mask_v2/val/200021660/200021660_00028_1/02_text_black.png'
    with Image.open(source) as image:raw=image.convert('RGB')
    with Image.open(maskfile) as image:mask=image.convert('L')
    assert raw.size==mask.size
    # The previously saved original crop is an independent pixel-identity check.
    x,y,w,h=1093,1723,768,768
    old=ROOT/'tmp_work/npj_contact_research_20261007/other_book_screen/final/native/final_05'
    import numpy as np
    assert np.array_equal(np.asarray(raw.crop((x,y,x+w,y+h))),
                          np.asarray(Image.open(old/'original.png').convert('RGB')))
    f=ImageFont.truetype(FONT,24);small=ImageFont.truetype(FONT,17)
    for title,box,scale,name in [
        ('07 強い裏移り：原画像とCSC保存マスク',(x,y,x+w,y+h),1,'07_原画像_CSC保存マスク.png'),
        ('07 強い裏移り：詳細2倍',(1240,1880,1600,2240),2,'07_原画像_CSC保存マスク_詳細2倍.png')]:
        crop=raw.crop(box);black=mask.crop(box).convert('RGB')
        if scale>1:
            crop=crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.NEAREST)
            black=black.resize((black.width*scale,black.height*scale),Image.Resampling.NEAREST)
        gap=24;top=112;canvas=Image.new('RGB',(crop.width*2+gap*3,crop.height+top+78),'#f8f7f3')
        d=ImageDraw.Draw(canvas);d.text((20,12),title,font=f,fill='#1b2c38')
        d.text((24,52),f'200021660_00028_1 / xywh={box[0]},{box[1]},{box[2]-box[0]},{box[3]-box[1]}',font=small,fill='#405364')
        d.text((gap,top-33),'原画像',font=f,fill='#1b2c38')
        d.text((crop.width+gap*2,top-33),'保存済みCSC v2黒文字マスク',font=f,fill='#1b2c38')
        canvas.paste(crop,(gap,top));canvas.paste(black,(crop.width+gap*2,top))
        d.text((24,canvas.height-52),'灰色像は直前ページの図柄と対応。表文字との境界は未確定。',font=small,fill='#405364')
        pixels=np.asarray(canvas)
        assert np.array_equal(pixels[top:top+crop.height,gap:gap+crop.width],np.asarray(crop))
        assert np.array_equal(pixels[top:top+black.height,crop.width+gap*2:crop.width+gap*2+black.width],np.asarray(black))
        canvas.save(HERE/name)
    page=raw.copy();d=ImageDraw.Draw(page);d.rectangle((x,y,x+w,y+h),outline='red',width=14)
    shown=ImageOps.contain(page,(900,1300),Image.Resampling.LANCZOS)
    shown.save(HERE/'07_原ページ位置.png')
    candidate=ROOT/'data/split_dataset/val/200021660/200021660_00027_2.jpg'
    with Image.open(candidate) as image:
        reverse=ImageOps.mirror(image.convert('RGB'))
    reverse=reverse.resize(raw.size,Image.Resampling.LANCZOS)
    front=raw.crop((x,y,x+w,y+h))
    back=reverse.crop((x,y,x+w,y+h))
    gap=24;top=110
    panel=Image.new('RGB',(w*2+gap*3,h+top+78),'#f8f7f3')
    d=ImageDraw.Draw(panel)
    d.text((24,12),'07 表画像と反対面候補の仮配置',font=f,fill='#1b2c38')
    d.text((gap,64),'表：200021660_00028_1',font=small,fill='#405364')
    d.text((w+gap*2,64),'候補：00027_2 を左右反転',font=small,fill='#405364')
    panel.paste(front,(gap,top));panel.paste(back,(w+gap*2,top))
    d.text((24,panel.height-52),'全体寸法だけを合わせた仮配置。図柄と灰色像の厳密な位置一致は未検証。',font=small,fill='#405364')
    pixels=np.asarray(panel)
    assert np.array_equal(pixels[top:top+h,gap:gap+w],np.asarray(front))
    assert np.array_equal(pixels[top:top+h,w+gap*2:w+gap*2+w],np.asarray(back))
    panel.save(HERE/'07_裏面候補対照.png')


def montage(rows):
    tile_w,tile_h=680,690
    canvas=Image.new('RGB',(tile_w*2,95+tile_h*4),'#f5f3ed')
    draw=ImageDraw.Draw(canvas)
    titlefont=ImageFont.truetype(FONT,32)
    font=ImageFont.truetype(FONT,22)
    small=ImageFont.truetype(FONT,17)
    draw.text((22,15),'課題対象の原画像：7領域',font=titlefont,fill='#162a39')
    draw.text((24,60),'索引用の縮小画像。判定は原寸・詳細比較PNGで行う。',font=small,fill='#405364')
    colors={'主対象':'#146082','成功対照':'#217a4b','判定保留':'#a8611e','別課題':'#684c91','追加対象':'#276e75'}
    for index,row in enumerate(rows):
        col=index%2;line=index//2;left=col*tile_w;top=95+line*tile_h
        draw.rectangle((left+12,top+8,left+tile_w-12,top+tile_h-8),fill='white',outline='#c8cdd0',width=2)
        draw.text((left+28,top+19),row['id']+' '+row['title'],font=font,fill='#162a39')
        draw.text((left+28,top+54),row['role']+'｜'+row['page'],font=small,fill=colors[row['role']])
        with Image.open(ROOT/row['original']) as image:
            x,y,w,h=row['roi_xywh'];crop=image.convert('RGB').crop((x,y,x+w,y+h))
        shown=ImageOps.contain(crop,(620,555),Image.Resampling.LANCZOS)
        px=left+(tile_w-shown.width)//2;py=top+90+(555-shown.height)//2
        canvas.paste(shown,(px,py))
        draw.rectangle((px,py,px+shown.width-1,py+shown.height-1),outline='#778b98',width=1)
        draw.text((left+28,top+652),f"原座標 {x},{y},{w},{h}",font=small,fill='#405364')
    output=HERE/'原画像一覧.png';canvas.save(output)
    return output


def markdown(rows):
    lines=[
        '# 課題対象画像の台帳', '',
        '2026-10-09。既存の二値化比較と原画像候補から7領域を選び、次の実験で担う役割ごとに整理した。新たな推論やパラメータ変更はしていない。', '',
        '**主対象は01〜03、維持すべき成功対照は04。05は材質未確定、06は紙面損傷、07は裏移りの追加対象。** 7領域は既存の出力を見た後に整理した開発用事例であり、独立テストの精度には使わない。02と04は同じページ、02〜04は同じ hnsd 資料群なので、独立した3資料とは数えない。', '',
        '元データの分割は **train 3件（02・03・04）、val 2件（05・07）、test 2件（01・06）**。テストだけから抽出した7件ではない。testに属する2件も選定後に出力を見ているため、今後のパラメータ調整に対する未使用テストとは呼べない。', '',
        'DDRMへ接続する際の開発例と最終評価の分け方は[評価分割方針](評価分割方針.md)に記録した。', '',
        '[原画像の索引PNG](原画像一覧.png)は縮小表示。[7課題のCSC・Sauvola・Howe比較PNG](csc_sauvola_howe/7課題_CSC_Sauvola_Howe_比較.png)では、原画像を含む4列を同じ座標で原寸表示する。CSCは保存済みv2黒文字マスク、SauvolaとHoweは生の原画像を入力とする。', '',
        '| ID・役割 | 課題と現状 | ページ・原画像座標 | 比較 |',
        '| --- | --- | --- | --- |',
    ]
    for row in rows:
        c=row
        # All related paths are under tmp_work, so from this catalog use ../.
        def link(target): return '../'+target.removeprefix('tmp_work/')
        x,y,w,h=c['roi_xywh']
        comparison_label={'05':'原画像／CSC／Sauvola／Howe','06':'連続ページの同領域','07':'原画像／CSC保存マスク'}.get(c['id'],'原画像／Sauvola／Howe')
        detail_label='詳細2倍' if c['id'] in ('05','06','07') else '3法の詳細'
        lines.append(f"| **{c['id']} {c['role']}** {c['challenge']} | {c['criterion']} | `{c['page']}` `{x},{y},{w},{h}` | [原画像の位置]({link(c['original_context'])}) · [{comparison_label}]({link(c['primary_comparison'])}) · [{detail_label}]({link(c['detail_comparison'])}) |")
    lines += ['', '## 次の確認', '']
    for row in rows:
        lines.append(f"- **{row['id']} {row['challenge']}：** {row['next_check']}")
    lines += ['',
        '01〜04の Howe 診断は[自動調整と失敗原因](../howe_diagnosis_20261008/診断報告.md)、原実装照合と疎な評価点は[再現照合と評価固定](../howe_diagnosis_20261008/evaluation/再現照合と評価固定.md)を参照。03では自動調整が接触筆画をさらに削り、04では孤立斑点を消せる。この2例を対にして判断する。', '',
        '05は白黒に近い旧候補 `200003967_00043_1` から、黄色みのある紙面と濃い粒状像がある `200021851_00007_2` に差し替えた。材質は未確定であり、粒状像の除去を画素正解として扱わない。旧05は元の比較報告に残す。', '',
        '07は[前後ページの反転・位置合わせ照合](07_裏移り_前後ページ照合.png)で直前の `200021660_00027_2` の樹木の図柄が灰色像と最もよく対応した。6近傍候補での局所相関はこのページが0.463、他候補の最大が0.191であり、裏移りとして扱う根拠になる。ただし画素単位の正解や表文字との境界を確定した値ではない。[照合条件と全候補の数値](07_裏移り_前後ページ照合.json)。', '',
        '06は旧候補 `200021925_00007_1` の文字に重なる欠損から、ユーザー提示の `200003803_00003_1` 左上の枝分かれ状の跡へ差し替えた。[連続ページの索引](06_新候補_連続ページ左上.png)でも近い形の跡が続く。本文外にあるため、文字二値化の失敗例ではなく紙面損傷の別課題として扱う。虫害・穴・表面剥離・染みの判別には原本の調査記録または現物確認が必要。旧06の古典3法比較は元の報告に残す。', '',
        '対照画像は[C01：広い染みを比較的分離できる例](../classical_limits_by_category_20261008/controls/C01_広い染みを比較的分離できる対照.png)、[C02：灰色像を減らせる例](../classical_limits_by_category_20261008/controls/C02_灰色像を減らせる対照.png)、[C03：本文外の穴・裂けらしい像](../classical_limits_by_category_20261008/controls/C03_本文外の穴・裂けらしい像.png)。主対象と同じ失敗件数に合算しない。', '',
        '原画像パス、SHA-256、座標、比較PNGの機械可読な一覧は[catalog.json](catalog.json)と[catalog.csv](catalog.csv)。原画像と比較画像の存在、原画像ハッシュ、座標範囲を生成時に検証した。', '',
        '現行7件の原画像・座標・役割は[選定ロック](selection_lock.json)に固定した。再生成後は `python tmp_work/challenge_image_catalog_20261009/freeze_selection.py` で照合する。最初の承認時点の選定は[旧ロック](selection_lock_v1_20261009.json)に残す。01〜03を探索的な筆画保持の主対象、04を汚れ除去の対照、07を裏移りの追加対象とする。05の材質と06の損傷原因は未確定。7件はいずれも独立テストの精度に加えない。',
    ]
    (HERE/'対象画像.md').write_text('\n'.join(lines)+'\n')


def main():
    HERE.mkdir(exist_ok=True)
    new_05_assets()
    new_06_assets()
    new_07_assets()
    rows=selection()
    (HERE/'catalog.json').write_text(json.dumps(dict(scope='Exploratory development images selected after previous outputs',cases=rows),ensure_ascii=False,indent=2)+'\n')
    with (HERE/'catalog.csv').open('w',newline='') as file:
        fields=['id','role','challenge','page','original','source_sha256','roi_xywh','detail_xywh','judgment','next_check','original_context','primary_comparison','detail_comparison','reverse_candidate_page','reverse_candidate_comparison','comparison_sauvola_howe','comparison_three_methods','detail_three_methods']
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in rows:writer.writerow({**row,'roi_xywh':json.dumps(row['roi_xywh']),'detail_xywh':json.dumps(row['detail_xywh'])})
    montage(rows)
    markdown(rows)
    print(HERE/'対象画像.md')


if __name__=='__main__':main()
