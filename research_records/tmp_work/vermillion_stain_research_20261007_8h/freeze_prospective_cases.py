"""Freeze later ORIGINAL-only Japanese controls and actual thin red numerals."""
import json
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,file_hash

JP=[
 ('200014685_00002_2',[1720,2600,300,580],'非常に薄い縦長の印影','faint_seal'),
 ('200021660_00003_2',[2180,2760,500,900],'本文の外側の薄い印影','faint_seal'),
 ('200021071_00003_2',[1600,1480,320,390],'小さい淡朱の円形印','small_faint_seal'),
 ('200021071_00047_2',[620,1030,400,550],'墨本文に重なる褐色斑点／本文は保存','tentative_brown_nonmark_control'),
 ('200021071_00087_2',[50,1270,400,500],'褐色斑点と本文／本文は保存','tentative_brown_nonmark_control'),
]
EX=[
 (29,[190,380,115,75],'左余白の細い赤い数記号','thin_red_note'),
 (29,[705,125,150,65],'上部の細い赤い見出し','thin_red_heading'),
 (29,[720,735,100,215],'本文間の小さい赤い数記号','thin_red_note'),
 (29,[100,950,105,400],'文字のない褐色余白','tentative_paper_negative'),
 (38,[530,215,105,310],'本文間の赤い番号列','thin_red_note'),
 (38,[25,535,90,530],'折れのある外側余白の細い赤番号','thin_red_note'),
 (50,[530,705,125,560],'本文間の淡い赤番号列','thin_red_note'),
 (50,[530,90,170,70],'淡い赤見出し','thin_red_heading'),
 (50,[1140,1640,125,265],'下側の褐色余白','tentative_paper_negative'),
]

def main():
    dest=OUT/'prospective_cases';dest.mkdir(exist_ok=True);manifest=dest/'cases_frozen.json';assert not manifest.exists(),'Frozen inputs must not be replaced'
    small=json.loads((OUT/'bright_annotation_screen/queue.json').read_text())['records'];lookup={r['id']:r for r in small};external=json.loads((OUT/'external_originals/udiads/candidates.json').read_text());rows=[]
    splits=json.loads((OUT/'synthetic_audit/protocol.json').read_text())['splits'];excluded={b for books in splits.values() for b in books}
    earlier=json.loads((OUT/'cases_frozen.json').read_text());excluded|=set(earlier['development_books']+earlier['validation_books']+['200010454'])
    for i,(page,box,name,kind) in enumerate(JP):
        r=lookup[page];assert r['book'] not in excluded
        rows.append(dict(case_id=f'prospective_jp_{i:02d}',page=page,book=r['book'],original=r['original'],source_sha256=r['source_sha256'],ROI=box,name=name,kind=kind,
            role='prospective_japanese',origin='Original-only secondary bright-core screen, no method outputs inspected',all_synthetic_bank_books_excluded=True))
    for i,(idx,box,name,kind) in enumerate(EX):
        r=external[idx];book=Path(r['archive_entry']).parts[2];page=book+'_'+Path(r['archive_entry']).stem
        rows.append(dict(case_id=f'prospective_external_{i:02d}',page='external_'+page,book=book,original=str((OUT/r['path']).relative_to(ROOT)),source_sha256=r['source_sha256'],ROI=box,name=name,kind=kind,
            role='prospective_external',origin='Original-only full-page inspection; no method outputs inspected',archive_entry=r['archive_entry'],candidate_index=idx,
            warning='More cases from the existing Latin2 manuscript, not extra independent manuscripts'))
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    for r in rows:
        assert file_hash(ROOT/r['original'])==r['source_sha256'];im=Image.open(ROOT/r['original']).convert('RGB');x,y,w,h=r['ROI'];assert 0<=x<x+w<=im.width and 0<=y<y+h<=im.height
        crop=im.crop((x,y,x+w,y+h));crop.save(dest/(r['case_id']+'.png'))
        scale=3 if w<=170 else 1;sheet=Image.new('RGB',(max(500,w*scale),h*scale+75),'#faf9f5');sheet.paste(crop.resize((w*scale,h*scale),Image.Resampling.NEAREST),(0,75));d=ImageDraw.Draw(sheet)
        d.text((5,5),r['case_id']+'／原画像だけ・予測前固定',font=font,fill='#20313a');d.text((5,29),r['name'],font=font,fill='#20313a');d.text((5,51),'座標はROI内・拡大 '+str(scale)+'倍',font=font,fill='#536470')
        spacing=10 if scale==3 else 50
        for xx in range(0,w,spacing):d.line((xx*scale,75,xx*scale,75+h*scale),fill='#53a5ae');d.text((xx*scale+1,77),str(xx),font=font,fill='#17303d')
        for yy in range(0,h,spacing):d.line((0,75+yy*scale,w*scale,75+yy*scale),fill='#53a5ae');d.text((1,77+yy*scale),str(yy),font=font,fill='#17303d')
        sheet.save(dest/(r['case_id']+'_座標.png'))
    manifest.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),cases=rows,source_sha256=file_hash(__file__),
        no_method_outputs_seen=True,no_training_or_model_selection=True,purposive_original_only_sampling=True,
        annotation_status='Codex tentative visual case interpretation; no expert reading, chemical stain or dense foreground GT',
        external_warning='Latin2 was already in the fixed external corpus; count once at manuscript level. No Japanese-character DDRM claim.',
        japanese_books=sorted({r['book'] for r in rows if r['role']=='prospective_japanese'})),ensure_ascii=False,indent=2));print('Frozen prospective cases',len(rows),flush=True)

if __name__=='__main__':main()
