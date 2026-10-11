"""Render fixed-red comparisons without changing inference pixels or colors."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BOXES = {'N01': [428,347,190,220], 'T01': [812,421,450,450], 'N02': [520,890,520,560]}
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def font(size):
    return ImageFont.truetype(FONT, size)

def read(p):
    return np.asarray(Image.open(p).convert('RGB')).copy()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, a):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a.astype('uint8')).save(p)

def text(draw, xy, value, size=20):
    draw.text(xy, value, font=font(size), fill='#20313b')

def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    complete = json.loads((OUT / 'complete.json').read_text())
    cases = {c['id']: c for c in protocol['cases']}
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    manifest, measurements = [], []
    overview = Image.new('RGB', (1640, 1830), '#faf9f5')
    od = ImageDraw.Draw(overview)
    text(od, (18,14), '赤マスクをv2に固定：黒マスクだけを比較', 28)
    for j, label in enumerate(['原画像', 'v2黒＋v2赤＋DDRM', 'Sauvola黒＋v2赤＋DDRM']):
        text(od, (155+j*490, 68), label, 23)
    overview_panels = []
    for i, cid in enumerate(['N01','T01','N02']):
        c = cases[cid]
        assert sha(ROOT / c['fixed_sauvola_result']) == complete['output_sha256'][cid]
        x,y,w,h = BOXES[cid]
        sl = np.s_[y:y+h, x:x+w]
        raw = read(ROOT / c['original'])
        v2 = read(ROOT / c['v2_result'])
        fixed = read(ROOT / c['fixed_sauvola_result'])
        old = read(ROOT / c['old_sauvola_result'])
        vmask = read(ROOT / c['v2_black'])[:,:,0] < 127
        smask = read(ROOT / c['sauvola_black'])[:,:,0] < 127
        red = read(ROOT / c['v2_red'])[:,:,0] < 127
        old_red = read(ROOT / c['old_red'])[:,:,0] < 127
        copied_red = OUT / 'masks/sauvola_v2_red' / c['page'] / '03_text_red.png'
        assert sha(copied_red) == sha(ROOT / c['v2_red'])
        assert sha(OUT / 'masks/sauvola_v2_red' / c['page'] / '02_text_black.png') == sha(ROOT / c['sauvola_black'])
        assert raw.shape == v2.shape == fixed.shape
        measurements.append(dict(id=cid, roi_xywh=BOXES[cid],
            v2_black_pixels=int(vmask[sl].sum()), sauvola_black_pixels=int(smask[sl].sum()),
            sauvola_only_pixels=int((smask & ~vmask)[sl].sum()),
            v2_only_pixels=int((vmask & ~smask)[sl].sum()),
            v2_red_pixels_in_roi=int(red[sl].sum()), old_red_pixels_in_roi=int(old_red[sl].sum()),
            v2_mask_pixels_output_brighter_than_128=int((vmask & (v2.mean(2)>128))[sl].sum()),
            sauvola_mask_pixels_output_brighter_than_128=int((smask & (fixed.mean(2)>128))[sl].sum()),
            note='Foreground counts and brightness checks are descriptive, not accuracy or reading scores.'))
        cols = [dict(name='original', label='原画像（参照）', black=raw[sl], red=raw[sl], result=raw[sl]),
            dict(name='v2', label='v2黒＋v2赤＋DDRM', black=np.repeat(vmask[sl][:,:,None],3,2)*255,
                red=np.repeat(red[sl][:,:,None],3,2)*255, result=v2[sl]),
            dict(name='sauvola_v2_red', label='Sauvola黒＋v2赤＋DDRM', black=np.repeat(smask[sl][:,:,None],3,2)*255,
                red=np.repeat(red[sl][:,:,None],3,2)*255, result=fixed[sl])]
        if cid == 'N02':
            cols.append(dict(name='sauvola_old_red', label='参考：Sauvola黒＋旧赤',
                black=np.repeat(smask[sl][:,:,None],3,2)*255,
                red=np.repeat(old_red[sl][:,:,None],3,2)*255, result=old[sl]))
        rows = ['black','red','result'] if cid == 'N02' else ['black','result']
        scale = 2 if cid == 'N01' else 1
        pw, ph = w*scale, h*scale
        left, top, gap = 150, 112, 30
        width = left + len(cols)*(pw+20)+20
        height = top + len(rows)*(ph+gap)+130
        canvas = Image.new('RGB', (width,height), '#faf9f5')
        d = ImageDraw.Draw(canvas)
        text(d, (14,14), cid+' 同じ座標・原寸比較：赤を固定、黒だけ変更', 25)
        for j, col in enumerate(cols):
            text(d, (left+j*(pw+20), 67), col['label'], 18)
        panels = []
        for r, key in enumerate(rows):
            yy = top+r*(ph+gap)
            text(d,(14,yy+20), {'black':'黒マスク','red':'赤マスク','result':'修復結果'}[key],20)
            if key != 'result': text(d,(14,yy+50),'白＝候補',18)
            for j,col in enumerate(cols):
                a = col[key].astype('uint8')
                pp = OUT / 'panels' / cid / col['name'] / (key+'.png')
                save(pp,a)
                im = Image.fromarray(a).resize((pw,ph),Image.Resampling.NEAREST)
                xx = left+j*(pw+20)
                canvas.paste(im,(xx,yy))
                assert np.array_equal(np.asarray(canvas)[yy:yy+ph,xx:xx+pw],np.repeat(np.repeat(a,scale,0),scale,1))
                panels.append(dict(method=col['name'],row=key,path=str(pp.relative_to(OUT)),
                    sha256=sha(pp), canvas_xywh=[xx,yy,pw,ph], integer_display_scale=scale))
        foot = ['DDRM：同じ重み・入力・背景・seed999・20段・128px／重なり32。Sauvola：窓61、k=0.2。',
            '推論は全ページ原寸。色調整なし。N01だけ整数2倍表示。マスク候補は正解注釈ではない。',
            '赤は中央2条件で全画素同一。初期seedは共通だが、黒条件間の各段の乱数一致は未保証。']
        if cid == 'N02': foot.append('右端は旧赤の参考条件。中央2列が今回の比較対象。')
        else: foot.append('全ページの旧赤＝v2赤を確認済み。既存の推論結果を再利用。対象範囲の赤候補は0画素。')
        for k,line in enumerate(foot): text(d,(14,height-120+k*27),line,17)
        path = figures / (cid+'_赤固定比較.png')
        canvas.save(path)
        manifest.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path),roi_xywh=BOXES[cid],panels=panels))
        yy = 112+i*530
        text(od,(18,yy+20),cid,26)
        for j, a in enumerate([raw[sl],v2[sl],fixed[sl]]):
            # This overview is resized for navigation; detail sheets keep native pixels.
            im = Image.fromarray(a)
            if cid == 'N01':
                im = im.resize((2*w,2*h),Image.Resampling.NEAREST)
            im.thumbnail((470,480),Image.Resampling.LANCZOS)
            xx = 155+j*490
            overview.paste(im,(xx,yy))
            overview_panels.append(dict(id=cid,method=['original','v2','sauvola_v2_red'][j],
                canvas_xywh=[xx,yy,*im.size], display_resized=True))
    for k,line in enumerate(['上：N01（整数2倍）、中央：T01、下：N02（表示縮小）。細部は各例の原寸比較PNGを参照。',
        '黒マスクだけを変更。赤はv2で共通。汚れと筆画の正解・専門家の読解評価は未取得。',
        'N02は今回新規に全ページ推論。N01／T01は赤マスクが同一だったため確認済み出力を再利用。']):
        text(od,(18,1735+k*28),line,20)
    overview.save(figures / '3例_赤固定比較.png')
    manifest.append(dict(path='figures/3例_赤固定比較.png',panels=overview_panels))
    # Page overview to expose differences outside the selected detail.
    for cid, c in cases.items():
        keys = ['original','v2_result','fixed_sauvola_result']
        labels = ['原画像','v2黒＋v2赤','Sauvola黒＋v2赤']
        if cid == 'N02':
            keys.append('old_sauvola_result')
            labels.append('参考：Sauvola黒＋旧赤')
        height = 900
        images = [Image.open(ROOT / c[k]).convert('RGB') for k in keys]
        width = round(images[0].width*height/images[0].height)
        sheet = Image.new('RGB',(len(keys)*(width+20)+20,1020),'#faf9f5')
        sd = ImageDraw.Draw(sheet)
        text(sd,(14,12),cid+' 全ページ：赤固定の比較（表示のみ縮小）',25)
        for j,(im,label) in enumerate(zip(images,labels)):
            xx = 14+j*(width+20)
            text(sd,(xx,57),label,19)
            sheet.paste(im.resize((width,height),Image.Resampling.LANCZOS),(xx,95))
        sheet.save(figures / (cid+'_全ページ比較.png'))
    (OUT / 'rendering.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (OUT / 'measurements.json').write_text(json.dumps(measurements,ensure_ascii=False,indent=2))
    # Source/output verification, without interpreting these as fidelity tests.
    for p,expected in protocol['source_sha256'].items(): assert sha(ROOT/p) == expected,p
    (OUT / 'verification.json').write_text(json.dumps(dict(source_hashes_unchanged=True,
        fixed_red_hashes_identical=True, native_panels_exact=True, measurements_are_not_accuracy=True),indent=2))
    print(json.dumps(measurements,ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
