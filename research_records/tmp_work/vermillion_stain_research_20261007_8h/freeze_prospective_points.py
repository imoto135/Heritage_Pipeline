"""Tentative original-only visual points, fixed before any prospective predictions."""
import json,datetime
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import ROOT,OUT,file_hash
P=OUT/'prospective_cases';manifest=P/'cases_frozen.json';data=json.loads(manifest.read_text());cases={r['case_id']:r for r in data['cases']}
# Coordinates are hand selected from original coordinate sheets, not from colour thresholds or model output.
coords={
'prospective_external_00':([(26,37),(42,34),(53,36),(60,34),(68,37),(75,36),(81,35)],[(10,15),(45,10),(75,15),(40,55),(65,60)]),
'prospective_external_01':([(27,33),(27,43),(43,36),(61,36),(81,37),(101,31)],[(12,15),(60,12),(112,52),(128,15)]),
'prospective_external_02':([(16,37),(28,36),(39,39),(46,37),(55,38),(65,39),(23,181),(36,179),(49,179)],[(60,100),(25,78),(15,137),(73,206)]),
'prospective_external_04':([(26,45),(39,44),(50,44),(58,45),(65,44),(74,44),(26,125),(39,126),(25,185),(38,186),(49,184),(25,235),(40,235),(49,234),(56,235)],[(20,80),(55,95),(80,158),(25,270),(78,289)]),
'prospective_external_05':([(36,43),(33,94),(41,94),(27,149),(37,149),(44,149),(25,204),(32,204),(40,204),(47,204),(30,252),(27,307),(35,307),(25,360),(32,360),(40,360),(25,413),(34,413),(42,413),(49,413),(29,468),(38,468),(45,468),(53,468),(34,523)],[(10,75),(12,185),(65,230),(62,330),(63,440),(65,510)]),
'prospective_external_06':([(26,64),(38,64),(50,65),(57,64),(66,64),(77,64),(87,64),(29,93),(43,90),(30,122),(49,122),(58,120),(30,151),(49,150),(58,151),(28,180),(43,180),(58,180),(29,222),(45,222),(56,222),(67,222),(30,253),(43,253),(57,253),(33,283),(47,283),(58,283),(70,283),(30,360),(48,360),(60,360),(28,419),(47,419),(59,419),(74,419),(87,419),(31,479),(48,479),(61,479),(71,479),(78,479),(34,511)],[(12,37),(75,42),(50,199),(77,324),(66,388),(79,540)]),
'prospective_jp_00':([(112,216),(164,216),(231,216),(274,227),(276,300),(274,400),(228,452),(176,444),(132,409),(117,342),(114,273)],[(62,155),(83,330),(194,144),(224,529),(253,364),(65,474)]),
'prospective_jp_01':([(155,95),(205,87),(262,78),(319,80),(344,140),(342,282),(328,417),(230,428),(157,364),(170,145),(275,215),(215,294),(270,351),(194,530),(250,521),(325,524),(264,625),(237,742),(273,818)],[(94,126),(409,217),(423,433),(100,544),(107,776),(415,664)]),
'prospective_jp_02':([(115,114),(143,116),(166,139),(177,171),(138,210),(105,196),(83,169),(92,142),(118,163)],[(35,105),(222,161),(204,245),(70,290),(150,320)])}
points=[]
for cid,(red,paper) in coords.items():
 r=cases[cid];x0,y0,w,h=r['ROI']
 for label,xylist in [('observed_red_mark',red),('paper_or_brown_nonmark',paper)]:
  for x,y in xylist:
   assert 0<=x<w and 0<=y<h
   points.append(dict(case_id=cid,page=r['page'],book=r['book'],role=r['role'],xy_roi=[x,y],xy_original=[x+x0,y+y0],label=label))
path=P/'points_frozen.json';assert not path.exists(),'Do not replace frozen labels'
rec=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),points=points,cases_manifest_sha256=file_hash(manifest),source_sha256=file_hash(__file__),labels='Tentative Codex visual observations on original images only; no pigment chemistry, expert reading, unbiased pixel GT, or confirmed clean background',no_model_outputs_inspected=True,not_for_training=True,annotation_selection='Hand-picked original coordinate sheets; correlated stroke centres and selected paper points',red_points=sum(p['label']=='observed_red_mark' for p in points),nonmark_points=sum(p['label']!='observed_red_mark' for p in points))
path.write_text(json.dumps(rec,ensure_ascii=False,indent=2))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
for cid in coords:
 r=cases[cid];rgb=Image.open(ROOT/r['original']).convert('RGB');x,y,w,h=r['ROI'];im=rgb.crop((x,y,x+w,y+h));scale=3 if w<200 else 1;im=im.resize((w*scale,h*scale),Image.Resampling.NEAREST);d=ImageDraw.Draw(im)
 for p in [q for q in points if q['case_id']==cid]:
  xx,yy=p['xy_roi'];xx*=scale;yy*=scale;color=(0,90,255) if p['label']=='observed_red_mark' else (0,220,90);d.ellipse((xx-3,yy-3,xx+3,yy+3),outline=color,width=1)
 im.save(P/(cid+'_確認点.png'))
print('Frozen prospective original points',rec['red_points'],rec['nonmark_points'])
