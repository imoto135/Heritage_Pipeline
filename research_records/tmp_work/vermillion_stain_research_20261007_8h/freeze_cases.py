"""Freeze original-only cases and book separation before new predictions."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def main():
    old=json.loads((OUT/'development_inputs.json').read_text())
    sources={r['id']:r for r in json.loads((OUT/'native_candidate_inspection/sources.json').read_text())}
    specs=[
        ('200021637_00003_2',[1160,2320,500,460],'淡い印影と濃い朱印・茶色の折り目','validation','pale_seal'),
        ('100249416_00002_2',[1570,2480,390,380],'朱印と墨の交差','validation','seal_black_contact'),
        ('100249416_00002_2',[1580,1575,390,470],'茶色の場と墨印','validation','brown_negative'),
        ('200017458_00003_2',[1830,130,280,380],'赤褐色の朱印と紙の斑点','validation','seal'),
        ('200017458_00003_2',[1590,180,240,300],'朱印横の茶色の紙・斑点','validation','brown_negative'),
        ('200015779_00089_2',[1730,2700,300,400],'細い朱印と墨の近接','validation','seal'),
        ('200010454_00012_1',[1060,500,250,260],'細い朱の傍線','supplemental_known_book','red_line'),
        ('200010454_00012_1',[1490,680,220,260],'赤褐色の句点と墨','supplemental_known_book','red_punctuation'),
        ('200010454_00006_2',[1440,620,250,270],'薄い朱の句点・小線','supplemental_known_book','red_punctuation'),
        ('200010454_00006_2',[1660,370,100,500],'朱近傍の茶色の折り目','supplemental_known_book','brown_negative'),
        ('200010454_00002_2',[1655,1810,75,195],'茶色の折り目・過去の誤保護診断','supplemental_known_book','brown_negative'),
    ]
    records=[]
    for i,r in enumerate(old):
        records.append(dict(case_id=f'dev_{i:02d}',page=r['page'],book=r['book'],original=r['original'],
            source_sha256=r['source_sha256'],ROI=r['ROI'],name=r['name'],role='development',
            category='writing' if '筆記' in r['name'] or '囲み' in r['name'] else 'seal_or_colored_content',
            prior_history='Known prior outputs; development only',pixel_GT=False))
    for j,(page,box,name,role,category) in enumerate(specs):
        r=sources[page]
        x,y,w,h=box; assert x>=0 and y>=0 and x+w<=r['width'] and y+h<=r['height']
        records.append(dict(case_id=f'{role}_{j:02d}',page=page,book=r['book'],original=r['original'],
            source_sha256=r['source_sha256'],ROI=box,name=name,role=role,category=category,
            prior_history='Original screened only in this session; project-wide prior usage not excluded',pixel_GT=False))
    train_books=sorted({r['book'] for r in records if r['role']=='development'})
    validation_books=sorted({r['book'] for r in records if r['role']=='validation'})
    assert not set(train_books)&set(validation_books)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),cases=records,
        development_books=train_books,validation_books=validation_books,
        scope='RGB mark/stain separation, not pigment identification or reconstructed text',
        validation='Local fixed-configuration book-separated diagnostic; not globally unseen external test',
        selection='Only original images; no new method masks or restoration outputs yet',
        rule='Do not use validation books for background synthesis, fitting or parameter selection; changes after inspection are exploratory',
        main_generalization_limit='Validation positives are seals, not independent red handwriting books',
        no_full_pixel_GT=True)
    pp=OUT/'cases_frozen.json'
    assert not pp.exists(),'Do not silently reselect cases'
    pp.write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    dest=OUT/'case_inputs'; dest.mkdir(exist_ok=True)
    for r in records:
        p=ROOT/r['original']; assert hashlib.sha256(p.read_bytes()).hexdigest()==r['source_sha256']
        x,y,w,h=r['ROI']; im=Image.open(p).convert('RGB').crop((x,y,x+w,y+h))
        im.save(dest/(r['case_id']+'.png'))
        shown=im.resize((w*2,h*2),Image.Resampling.NEAREST) if max(w,h)<=600 else im.copy()
        d=ImageDraw.Draw(shown); scale=shown.width/w
        for xx in range(0,w,50):
            d.line((int(xx*scale),0,int(xx*scale),shown.height),fill='#009db9',width=1)
            d.text((int(xx*scale)+1,2),str(xx),font=ImageFont.truetype(FONT,14),fill='#17455a')
        for yy in range(0,h,50):
            d.line((0,int(yy*scale),shown.width,int(yy*scale)),fill='#009db9',width=1)
            d.text((1,int(yy*scale)+1),str(yy),font=ImageFont.truetype(FONT,14),fill='#17455a')
        shown.save(dest/(r['case_id']+'_coordinate.png'))
    print('Frozen:',len(records),'ROIs;',len(train_books),'development books;',len(validation_books),'validation books')

if __name__=='__main__':main()
