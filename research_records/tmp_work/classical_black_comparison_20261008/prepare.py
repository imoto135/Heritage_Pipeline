from pathlib import Path
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
prior = json.loads((ROOT/'tmp_work/npj_sauvola_v2_red_20261007/protocol.json').read_text())
old = {c['id']: c for c in prior['cases']}
cases = [
    dict(id='N01', original=old['N01']['original'], roi_xywh=[428,347,190,220],
         saved_v2=old['N01']['v2_black'], saved_sauvola=old['N01']['sauvola_black'],
         reason='細い曲線と円形シミ：線の連続性とシミの誤検出'),
    dict(id='T01', original=old['T01']['original'], roi_xywh=[812,421,450,450],
         saved_v2=old['T01']['v2_black'], saved_sauvola=old['T01']['sauvola_black'],
         reason='広いシミと本文：筆画の欠落・太り・紙面ノイズ'),
    dict(id='N05', original='data/split_dataset/test/200010454/200010454_00021_1.jpg',
         roi_xywh=[1329,1757,280,380], reason='細い墨線の対照：細い線の保持と紙面の誤検出'),
]
for c in cases:
    p = ROOT/c['original']
    c['source_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    im = Image.open(p).convert('RGB')
    c['page_size'] = list(im.size)
    x,y,w,h = c['roi_xywh']
    crop = im.crop((x,y,x+w,y+h))
    dest = HERE/'originals'/c['id']; dest.mkdir(parents=True,exist_ok=True)
    crop.save(dest/'original.png')
    grid = crop.resize((w*3,h*3),Image.Resampling.NEAREST)
    d = ImageDraw.Draw(grid)
    for xx in range(0,w,20):
        d.line((xx*3,0,xx*3,h*3),fill='#96bdd0',width=1)
        d.text((xx*3+1,1),str(xx),font=ImageFont.truetype(FONT,15),fill='#003366')
    for yy in range(0,h,20):
        d.line((0,yy*3,w*3,yy*3),fill='#96bdd0',width=1)
        d.text((1,yy*3+1),str(yy),font=ImageFont.truetype(FONT,15),fill='#003366')
    grid.save(dest/'original_grid.png')
protocol = dict(cases=cases, selection='Original-image inspection before new classical outputs; purposive development comparison.',
    input='Full native page, common OpenCV RGB2GRAY; crop only after processing; no ACE, CLAHE, red overlay or DDRM.',
    gt='No dense pixel GT. Any point labels are tentative original-only annotations, not expert transcription.',
    frozen_settings={
        'sauvola61':dict(window=61,k=.2,R=.5),
        'wolf61':dict(window=61,k=.5,global_statistics='whole native page'),
        'nick61':dict(window=61,k=-.1,formula='m + k sqrt(mean(I^2) - m^2 / window_area)'),
        'nick19':dict(window=19,k=-.1),
        'gatos_native':dict(initial_window=61,k=.2,background_window=181,q=.6,p1=.5,p2=.8,
            wiener_window=3,upsampling=1,postprocessing='paper shrink/swell with dominant component height >=10; min odd window3'),
        'su2013_local':dict(gamma=1,contrast_window=3,threshold='edge mean + edge std / 2',
            window='nearest odd >= 2*estimated horizontal edge width',
            canny='skimage sigma sqrt(2), automatic gradient 70th percentile high and low=.4*high',
            postprocessing='paper Algorithm2 raster neighbor-pair relabel; remove isolated foreground pixels',
            limitation='Local reconstruction, not author MATLAB p-code parity; reference [4] boundary logical operators not reproduced in full.'),
        'controls':['global Otsu','Niblack61 k=-.2','saved CSC v2 where available'],
        'sensitivity':['Sauvola31 k=.2','Sauvola91 k=.2','Sauvola61 k=.1','Wolf61 k=.2'],
    },
    implementation_limits='Gatos native M=1 variant and Su local reconstruction must be named as such. No paper-level superiority claims.')
(HERE/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
print(json.dumps(cases,ensure_ascii=False,indent=2))
