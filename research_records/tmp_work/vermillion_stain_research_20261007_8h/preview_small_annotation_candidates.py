"""Input-only native crops of small-note search candidates, including partial scans."""
import json
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,file_hash

def main():
    folder=OUT/'small_annotation_screen';cache=folder/'per_page_checkpoint.jsonl';rows=[]
    for line in cache.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r['status']=='scored' and r['candidate_locations'] and r['score']>=2:rows.append(r)
    splits=json.loads((OUT/'synthetic_audit/protocol.json').read_text())['splits']
    prior={b for books in splits.values() for b in books}
    candidates=[]
    for book in sorted({r['book'] for r in rows}):
        if book in prior:continue
        candidates+=sorted([r for r in rows if r['book']==book],key=lambda r:-r['score'])[:2]
    candidates=sorted(candidates,key=lambda r:-r['score'])[:48]
    dest=folder/'native_original_preview';dest.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14)
    records=[]
    for offset in range(0,len(candidates),12):
        sheet=Image.new('RGB',(4*360,3*620),'#faf9f5');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(candidates[offset:offset+12]):
            original=Image.open(ROOT/r['original']).convert('RGB');w,h=original.size;x=j%4*360+5;y=j//4*620+5
            c=r['candidate_locations'][0];xx=max(0,min(w-330,c['x']-165));yy=max(0,min(h-430,c['y']-215))
            draw.text((x,y),str(offset+j)+' '+r['id'],font=font,fill='#20313a');draw.text((x,y+23),'原画像／赤候補は未判定・'+str([xx,yy,330,430]),font=font,fill='#536470')
            thumb=original.copy();thumb.thumbnail((150,135));sheet.paste(thumb,(x,y+52));sheet.paste(original.crop((xx,yy,xx+330,yy+430)),(x,y+190))
            records.append(dict(**r,preview_index=offset+j,preview_ROI=[xx,yy,330,430],excluded_all_synthetic_bank_books=True,source_sha256=file_hash(ROOT/r['original'])))
        sheet.save(dest/f'原画像_小朱候補_{offset//12+1:02d}.png')
    (dest/'queue.json').write_text(json.dumps(dict(records=records,partial_input_scan=not (folder/'complete.json').exists(),source_sha256=file_hash(__file__),
        purpose='Native original-only discovery; no learned/baseline masks viewed for these candidates; no semantic ground truth',
        excluded_synthetic_bank_books=sorted(prior)),ensure_ascii=False,indent=2))
    print('Original native previews',len(records),flush=True)

if __name__=='__main__':main()
