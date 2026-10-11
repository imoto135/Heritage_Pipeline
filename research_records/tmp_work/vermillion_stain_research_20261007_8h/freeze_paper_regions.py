"""Tentative original-only paper negatives, in development books only."""
from pathlib import Path
import json
from datetime import datetime,timezone
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,file_hash

REGIONS={
 'dev_10':[[400,420,120,170],[410,1000,110,180],[25,405,70,80],[445,800,75,140]],
 'dev_08':[[520,100,110,140],[525,550,105,150],[520,775,100,175],[50,1010,80,70],[150,990,320,80]],
 'dev_06':[[370,130,95,100],[380,270,85,80],[150,430,120,65],[10,80,40,200]],
}

def main():
    dest=OUT/'original_paper_negative_regions';dest.mkdir(exist_ok=True)
    pp=dest/'regions_frozen.json';assert not pp.exists(),'Use another version if revising review'
    cases={r['case_id']:r for r in json.loads((OUT/'cases_frozen.json').read_text())['cases']}
    records=[]
    for case,boxes in REGIONS.items():
        r=cases[case];assert r['role']=='development'
        im=Image.open(OUT/'case_inputs'/(case+'.png')).convert('RGB');shown=im.copy();d=ImageDraw.Draw(shown)
        for idx,(x,y,w,h) in enumerate(boxes):
            assert x>=0 and y>=0 and x+w<=im.width and y+h<=im.height
            record=dict(case=case,book=r['book'],page=r['page'],xywh_local=[x,y,w,h],
                xywh_original=[x+r['ROI'][0],y+r['ROI'][1],w,h],original=r['original'],
                original_sha256=r['source_sha256'],label='No visible red mark in tentative original review',
                definite_stain_or_clean_GT=False,reviewer='Codex tentative visual review')
            records.append(record)
            im.crop((x,y,x+w,y+h)).save(dest/(case+f'_paper_{idx:02d}.png'))
            d.rectangle((x,y,x+w,y+h),outline='#1595b8',width=2)
        shown.save(dest/(case+'_紙領域.png'))
    pp.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),regions=records,
        selection='Original review only, development books; noticed early CNN grain failure but no real validation outputs inspected',
        use='Exploratory additional negative supervision for mark head only; never impose zero stain labels',
        annotation_status='Not expert checked, not definitive full pixel GT',
        source_sha256=file_hash(__file__)),ensure_ascii=False,indent=2))
    print('Frozen tentative paper negative regions:',len(records),'in development only')

if __name__=='__main__':main()
