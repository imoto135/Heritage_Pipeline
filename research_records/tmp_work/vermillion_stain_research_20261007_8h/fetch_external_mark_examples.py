"""Small public author-hosted corpus for original-only mark/stain screening.

Text-line GT is NOT red-mark or stain GT. No fitting on external originals.
"""
from pathlib import Path
import requests
import json
import hashlib
import zipfile
import io
from datetime import datetime,timezone
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
URL='https://ai4ch.uniud.it/udiadstl/U-DIADS-TL.zip'

def main():
    dest=OUT/'external_originals/udiads';dest.mkdir(parents=True,exist_ok=True)
    r=requests.get(URL,timeout=60);r.raise_for_status();assert r.content[:2]==b'PK'
    archive=dest/'U-DIADS-TL.zip';archive.write_bytes(r.content)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        files=[n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg')) and not n.startswith('__MACOSX')]
        records=[]
        for n in files:
            relative=Path(n);assert '..' not in relative.parts and not relative.is_absolute()
            f=dest/'originals'/relative;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(z.read(n))
            im=Image.open(f);records.append(dict(index=len(records),archive_entry=n,path=str(f.relative_to(OUT)),
                source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),width=im.width,height=im.height,
                role='input-only external candidate, no training',red_or_stain_GT=False))
    protocol=dict(downloaded_utc=datetime.now(timezone.utc).isoformat(),author_download=URL,
        author_site='https://sites.google.com/view/avml-lab-udiadstl/home',
        paper='https://link.springer.com/article/10.1007/s10032-026-00585-7',
        image_source='Bibliothèque nationale de France / Gallica as described by authors',
        task_difference='Original dataset is text-line segmentation; its GT classes do not label red ink or stains',
        extracted_originals=len(records),archive_sha256=hashlib.sha256(r.content).hexdigest(),archive_bytes=len(r.content),
        fitting='None; screening before selecting additional fixed diagnostic cases')
    (dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    (dest/'candidates.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    font=ImageFont.truetype(FONT,16)
    for offset in range(0,len(records),16):
        sheet=Image.new('RGB',(4*390,4*620),'#faf9f5');d=ImageDraw.Draw(sheet)
        for j,record in enumerate(records[offset:offset+16]):
            im=Image.open(OUT/record['path']).convert('RGB');im.thumbnail((370,540),Image.Resampling.LANCZOS)
            xx=j%4*390+8;yy=j//4*620+10
            d.text((xx,yy),str(record['index'])+' '+Path(record['archive_entry']).name,font=font,fill='#20313a')
            d.text((xx,yy+25),'外部原画像のみ／朱・シミGTなし',font=font,fill='#20313a')
            sheet.paste(im,(xx,yy+58))
        sheet.save(dest/(f'原画像候補_{offset//16+1:02d}.png'))
    print('External original-only screening:',len(records),'pages; archive',len(r.content),'bytes',flush=True)

if __name__=='__main__':main()
