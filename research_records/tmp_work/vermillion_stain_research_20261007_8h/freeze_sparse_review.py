"""Tentative original-only sparse marks; never model-derived ground truth.

These are diagnostic points, not expert pigment identification, full masks,
readability labels, or an unbiased estimate of page-wide accuracy.
"""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
POINTS={
 'dev_10':{
  'observed_red_mark':[(317,111),(317,180),(314,260),(312,340),(311,425),(351,229),(291,272),(265,304),(170,502),(145,539),(128,570),(143,600),(111,649),(230,510),(263,532),(300,561),(275,639),(281,748),(273,802),(264,852),(333,822),(296,892),(256,942),(210,997),(166,1042),(121,1087),(75,1125),(64,1110),(160,790),(145,864),(130,919),(88,944)],
  'paper_or_brown_nonmark':[(30,200),(50,500),(480,700),(500,1000),(410,1100),(400,600),(470,430),(100,430),(70,750),(225,1170)]},
 'dev_08':{
  'observed_red_mark':[(120,160),(133,280),(130,450),(134,620),(145,900),(215,950),(390,954),(501,300),(490,550),(495,760),(275,276),(335,343),(434,437),(419,669),(320,690),(345,803),(465,931),(222,635)],
  'paper_or_brown_nonmark':[(65,320),(560,200),(540,700),(90,1000),(330,1040),(370,250),(300,440),(460,520)]},
 'dev_06':{
  'observed_red_mark':[(65,160),(64,240),(78,330),(132,366),(233,375),(330,368),(338,232),(318,112),(181,112),(106,185),(133,219),(183,180),(216,168),(261,294),(277,332),(210,315)],
  'paper_or_brown_nonmark':[(30,170),(400,180),(410,335),(160,445),(280,450),(182,60),(30,400),(275,78)]},
 'validation_00':{
  'observed_red_mark':[(70,83),(89,64),(140,62),(185,65),(218,138),(217,221),(207,292),(136,292),(76,259),(70,177),(152,179),(165,216),(128,258),(104,124),(181,114),(181,195),(265,168),(300,228),(384,190),(376,375),(438,282),(250,139)],
  'paper_or_brown_nonmark':[(35,60),(29,180),(41,260),(234,210),(226,310),(188,320),(151,336),(458,120),(470,399),(249,281)]},
 'supplemental_known_book_06':{
  'observed_red_mark':[(71,67),(73,89),(74,111),(79,137),(85,160)],
  'paper_or_brown_nonmark':[(111,55),(128,119),(117,210),(229,70),(186,140),(230,230)]},
 'supplemental_known_book_08':{
  'observed_red_mark':[(137,74),(135,79),(150,187),(163,185)],
  'paper_or_brown_nonmark':[(192,70),(183,155),(190,215),(58,161),(40,50)]},
}

def main():
    dest=OUT/'sparse_original_review';dest.mkdir(exist_ok=True)
    pp=dest/'points_frozen.json';assert not pp.exists(),'Never silently change points after predictions'
    rows={r['case_id']:r for r in json.loads((OUT/'cases_frozen.json').read_text())['cases']}
    points=[]
    palette={'observed_red_mark':'#d4288e','paper_or_brown_nonmark':'#1295be'}
    for case,classes in POINTS.items():
        r=rows[case];im=Image.open(OUT/'case_inputs'/(case+'.png')).convert('RGB')
        overlay=im.copy();d=ImageDraw.Draw(overlay)
        for name,xy in classes.items():
            for idx,(x,y) in enumerate(xy):
                assert 0<=x<im.width and 0<=y<im.height
                d.ellipse((x-4,y-4,x+4,y+4),outline=palette[name],width=2)
                points.append(dict(case=case,page=r['page'],book=r['book'],role=r['role'],
                    label=name,xy_local=[x,y],xy_original=[x+r['ROI'][0],y+r['ROI'][1]],
                    reviewer='Codex original-only tentative visual review',confidence='tentative',
                    chemical_or_reading_GT=False))
        overlay.save(dest/(case+'_暫定点.png'))
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),points=points,
        cases_source_sha256=hashlib.sha256((OUT/'cases_frozen.json').read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='Manual coordinates from original-only PNG/grid, before inspecting new learned outputs',
        interpretation='Observed red marking versus paper/brown nonmark at selected points; not definitive stain GT',
        metric='Strict center hit and optional 2px neighbourhood support; report as selected-point proportions only',
        limitations=['Not expert reviewed','Not unbiased page sampling','No full pixel GT','No character transcription GT',
            'Validation annotated case only one seal book; not handwriting generalization'],
        use='Diagnostic evaluation; never learning labels')
    pp.write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    print('Frozen original-only tentative points:',len(points),'; cases:',len(POINTS))

if __name__=='__main__':main()
