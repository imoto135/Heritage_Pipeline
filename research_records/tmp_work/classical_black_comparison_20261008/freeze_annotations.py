"""Freeze sparse original-only ink centers and small non-ink regions before inference."""
from pathlib import Path
import json, hashlib
import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
POINTS = {
 'N01': [(65,40),(64,52),(61,64),(56,79),(52,89),(47,101),(40,117),
         (41,133),(42,144),(43,156),(61,145),(76,135),(91,125),(102,124),
         (111,130),(115,144),(111,161),(104,175),(87,186)],
 'T01': [(22,99),(33,101),(72,115),(82,141),(32,168),(28,184),(34,204),
         (67,216),(51,229),(32,241),(36,255),(55,307),(34,328),(70,355),
         (42,376),(54,394),(113,59),(130,71),(149,54),(147,80),
         (138,113),(138,139),(240,29),(236,68),(223,129),(243,154),
         (240,216),(249,273),(235,277),(227,327),(253,330)],
 'N05': [(82,20),(82,32),(63,46),(76,55),(80,63),(70,73),(95,95),
         (102,111),(89,127),(73,140),(70,154),(69,173),(71,188),(85,192),
         (59,206),(63,226),(76,236),(213,81),(211,103),(209,119),
         (205,144),(247,114),(239,138),(237,162),(224,177),(205,192),
         (190,205),(178,217)]
}
NEGATIVE = {
 'N01': [[80,57,25,28],[90,91,35,20],[75,111,25,10]],
 'T01': [[160,100,30,45],[280,130,50,45],[160,330,40,50]],
 'N05': [[120,140,50,35],[110,205,35,40],[110,30,65,40]],
}
records = []
for cid, points in POINTS.items():
    src = HERE/'originals'/cid/'original.png'
    a = np.asarray(Image.open(src).convert('RGB'))
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    selected=[]
    for x,y in points:
        # Only the raw source is consulted; snap to the ink center within 1 native pixel.
        local=g[y-1:y+2,x-1:x+2]; yy,xx=np.unravel_index(local.argmin(),local.shape)
        selected.append([int(x-1+xx),int(y-1+yy)])
    annotated=Image.fromarray(a).resize((a.shape[1]*3,a.shape[0]*3),Image.Resampling.NEAREST)
    d=ImageDraw.Draw(annotated)
    for x,y in selected: d.ellipse((x*3-5,y*3-5,x*3+5,y*3+5),outline='#006bdb',width=2)
    for x,y,w,h in NEGATIVE[cid]:d.rectangle((x*3,y*3,(x+w)*3,(y+h)*3),outline='#00a050',width=3)
    annotated.save(HERE/'originals'/cid/'tentative_annotations.png')
    records.append(dict(id=cid,ink_points_xy=selected,nonink_regions_xywh=NEGATIVE[cid],
        original_crop_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
(HERE/'annotations_frozen.json').write_text(json.dumps(dict(
    origin='Manual inspection of original-only native crops and grid; ink centers snapped <=1px to darkest original pixel.',
    frozen_before_new_classical_outputs=True,
    caveat='Purposive tentative annotations by Codex. Not dense pixel GT, expert reading GT, or unbiased page accuracy. Region fractions measure these selected non-ink patches only.',
    cases=records),ensure_ascii=False,indent=2))
print('Frozen',sum(len(r['ink_points_xy']) for r in records),'ink points and',sum(len(r['nonink_regions_xywh']) for r in records),'non-ink patches')
