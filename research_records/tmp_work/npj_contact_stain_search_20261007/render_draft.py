from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',25)
(HERE/'draft').mkdir(exist_ok=True)
for r in json.loads((HERE/'inference_selection.json').read_text())['cases']:
 if not (ROOT/r['output']).exists():continue
 a=Image.open(ROOT/r['actual_input']).convert('RGB');b=Image.open(ROOT/r['output']).convert('RGB')
 c=Image.new('RGB',(1640,890),'#faf9f5');d=ImageDraw.Draw(c)
 d.text((30,15),f'{r["test_id"]} {r["page"]} crop={r["crop_xywh"]}',font=font,fill='black')
 d.text((30,62),'原画像',font=font,fill='black');d.text((840,62),'NPJ系・通常黒文字マスクのcrop対照',font=font,fill='black')
 c.paste(a,(30,110));c.paste(b,(840,110));c.save(HERE/'draft'/f'{r["test_id"]}.png')
print('Rendered existing outputs')
