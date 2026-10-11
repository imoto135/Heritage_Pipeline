"""Tentative colored-ink/paper points on external originals only."""
import json
from datetime import datetime,timezone
from PIL import Image,ImageDraw
from synthetic_layers import OUT,file_hash

SPECS={
 'external_00':dict(red=[(34,86),(92,182),(93,193),(31,254),(257,106)],paper=[(10,43),(365,95),(300,240),(230,165)]),
 'external_04':dict(red=[(48,45),(73,52),(105,46),(130,46),(150,45)],paper=[(15,15),(90,20),(178,40),(160,75)]),
 'external_05':dict(red=[(24,37),(38,37),(54,37),(67,38),(76,37)],paper=[(10,15),(70,10),(100,65),(20,70)]),
 'external_11':dict(red=[(28,37),(80,28),(104,33),(148,36),(197,35),(263,40),(282,37),(322,37),(43,87),(65,85),(150,84),(235,84),(283,89),(60,130),(148,131),(220,130),(280,133)],paper=[(360,32),(340,65),(390,45),(350,110),(15,10)]),
}

def main():
    dest=OUT/'external_originals/frozen_points';dest.mkdir(exist_ok=True)
    cases={r['case']:r for r in json.loads((OUT/'external_originals/frozen_cases/cases_frozen.json').read_text())['cases']};points=[]
    for cid,spec in SPECS.items():
        r=cases[cid];x0,y0,w,h=r['xywh_original'];im=Image.open(OUT/'external_originals/frozen_cases'/(cid+'.png')).convert('RGB');d=ImageDraw.Draw(im)
        for key,coords in spec.items():
            label='observed_red_mark' if key=='red' else 'paper_or_brown_nonmark'
            for x,y in coords:
                assert 0<=x<w and 0<=y<h
                points.append(dict(case=cid,page=r['page'],book=r['book'],role='external_qualitative_only',label=label,xy_local=[x,y],xy_original=[x+x0,y+y0],
                    reviewer='Codex original-only tentative visual review',confidence='tentative',chemical_or_reading_GT=False))
                color='#009fe3' if key=='red' else '#08c149';d.ellipse((x-3,y-3,x+3,y+3),outline=color,width=1)
        im.save(dest/(cid+'_暫定点.png'))
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),points=points,
        cases_sha256=file_hash(OUT/'external_originals/frozen_cases/cases_frozen.json'),script_sha256=file_hash(__file__),
        no_selector_outputs_inspected=True,no_training_or_threshold_selection=True,
        limitation='Selected tentative visible-ink and paper points, not full-pixel ground truth or representative population accuracy',
        uncertain_brown_ink_points_not_forced_into_stain_labels=True)
    (dest/'points_frozen.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));print('Frozen tentative external points',len(points),flush=True)

if __name__=='__main__':main()
