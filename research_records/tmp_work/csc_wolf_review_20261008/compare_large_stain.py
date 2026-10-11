"""Compare saved CSC v2, Sauvola and Wolf over T01's complete large stain."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'large_stain_v2_sauvola_wolf'
OUT.mkdir(exist_ok=True)
prior = json.loads((HERE/'protocol.json').read_text())
case = next(c for c in prior['cases'] if c['id']=='T01')
roi = [760,360,600,600]
x,y,w,h = roi
sources = {
    'original': ROOT/case['original'],
    'csc_v2': ROOT/case['saved_v2'],
    'sauvola61': ROOT/'tmp_work/classical_black_comparison_20261008/masks/T01/sauvola61.png',
    'wolf61': ROOT/'tmp_work/classical_black_comparison_20261008/masks/T01/wolf61.png',
}
labels = {'original':'原画像', 'csc_v2':'CSC v2（保存済み・ACE入力）',
          'sauvola61':'Sauvola（窓61・k=0.2）', 'wolf61':'Wolf（窓61・k=0.5）'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {k:sha(p) for k,p in sources.items()}
arrays = {}
for key,path in sources.items():
    assert hashes[key] == prior['source_sha256'][str(path.relative_to(ROOT))]
    a = np.asarray(Image.open(path).convert('RGB'))
    assert a.shape == (2843,1860,3)
    if key != 'original': assert set(np.unique(a)).issubset({0,255})
    arrays[key] = a[y:y+h,x:x+w].copy()
    Image.fromarray(arrays[key]).save(OUT/(key+'_crop.png'))

font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(n): return ImageFont.truetype(font_path,n)
gap,top,heading = 24,118,44
canvas = Image.new('RGB',(2*w+3*gap,top+2*(h+heading+gap)+100),'#faf9f5')
draw = ImageDraw.Draw(canvas)
draw.text((18,14),'T01：大きな染みの黒文字比較',font=font(25),fill='#20313b')
draw.text((18,53),'染み全体と周囲の文字を含む同じ範囲。黒＝文字候補、白＝背景。',font=font(17),fill='#20313b')
draw.text((18,79),'原画像・CSC v2・Sauvola・Wolf／各600×600画素を原寸表示。',font=font(17),fill='#20313b')
panels = []
for i,(key,a) in enumerate(arrays.items()):
    px = gap + (i%2)*(w+gap)
    py = top + (i//2)*(h+heading+gap)
    draw.text((px,py),labels[key],font=font(20),fill='#20313b')
    canvas.paste(Image.fromarray(a),(px,py+heading))
    assert np.array_equal(np.asarray(canvas)[py+heading:py+heading+h,px:px+w],a)
    assert np.array_equal(np.asarray(Image.open(OUT/(key+'_crop.png'))),a)
    panels.append(dict(method=key,canvas_xywh=[px,py+heading,w,h]))
draw.text((18,canvas.height-81),'CSC v2はACE前処理を含む保存結果。Sauvola・Wolfは原画像のグレー入力。',font=font(16),fill='#20313b')
draw.text((18,canvas.height-55),'前回の設定と全ページマスクを使用。表示の平滑化・輪郭強調なし。',font=font(16),fill='#20313b')
draw.text((18,canvas.height-29),'同じ筆画の連続性、染みの残り、不要な黒点を原画像に照らして比較。',font=font(16),fill='#20313b')
image_path = OUT/'T01_大きな染み_原画像_CSCv2_Sauvola_Wolf.png'
canvas.save(image_path)
for key,path in sources.items(): assert sha(path)==hashes[key]
metadata = dict(case='T01',roi_xywh=roi,order=list(arrays),
    rationale='Entire large stain and surrounding text, expanded from previous 450x450 ROI.',
    new_inference=False,parameters_unchanged=True,display_scale=1,
    csc_input='Saved CSC v2 mask with ACE preprocessing.',
    classical_input='Saved full-page masks from raw RGB-to-gray input.',
    source_paths={k:str(p.relative_to(ROOT)) for k,p in sources.items()},source_sha256=hashes,
    verification=dict(source_hashes_unchanged=True,source_hashes_match_previous_protocol=True,
        mask_polarity='0 foreground / 255 background',display_pixels_match_crops=True,
        saved_crops_match_sources=True),panels=panels,figure_sha256=sha(image_path),
    source_script_sha256=sha(Path(__file__)))
(OUT/'protocol.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
print(image_path)
