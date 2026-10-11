"""Freeze external original-only diagnostic examples before selector outputs."""
import json
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,file_hash

SPECS=[
 (7,[90,530,380,280],'Small red initials among black Latin writing','observed_red'),
 (7,[545,120,450,260],'Faint red initials next to dark script','observed_red'),
 (27,[350,810,430,180],'Red and blue heading; blue must also remain','multiple_ink_control'),
 (27,[80,1250,240,350],'Warm paper and folds without visible colored script','tentative_paper_negative'),
 (33,[500,100,210,100],'Thin faded red heading on warm paper','observed_red'),
 (33,[15,950,120,90],'Small pale red marginal numeral','observed_red'),
 (33,[520,930,120,100],'Small red marginal numeral near brown body ink','observed_red'),
 (33,[120,650,430,200],'Brown body writing is retained content, not unwanted stain','brown_ink_preservation_control'),
 (64,[210,170,340,140],'Faded reddish/brown Syriac heading; pigment identity uncertain','ambiguous_ink'),
 (64,[580,900,310,230],'Brown Syriac body writing is retained content','brown_ink_preservation_control'),
 (64,[20,1250,160,420],'Discolored outer parchment/fold, tentative no annotation','tentative_paper_negative'),
 (75,[780,140,420,170],'Red Syriac heading on brown parchment','observed_red'),
 (75,[1180,700,130,450],'Brown parchment margin, tentative no annotation','tentative_paper_negative'),
]

def main():
    records=json.loads((OUT/'external_originals/udiads/candidates.json').read_text())
    dest=OUT/'external_originals/frozen_cases';dest.mkdir(exist_ok=True)
    rows=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
    for i,(idx,box,description,kind) in enumerate(SPECS):
        r=records[idx];src=OUT/r['path'];assert file_hash(src)==r['source_sha256'];im=Image.open(src).convert('RGB')
        x,y,w,h=box;assert x+w<=im.width and y+h<=im.height
        crop=im.crop((x,y,x+w,y+h));cid=f'external_{i:02d}';crop.save(dest/(cid+'.png'))
        sheet=Image.new('RGB',(w,h+65),'#faf9f5');sheet.paste(crop,(0,65));d=ImageDraw.Draw(sheet)
        d.text((4,4),cid+' 原画像のみ・予測前固定',font=font,fill='#20313a')
        d.text((4,30),kind,font=font,fill='#20313a')
        for xx in range(0,w,50):d.line((xx,65,xx,65+h),fill='#38aebd');d.text((xx+1,68),str(xx),font=font,fill='#003b47')
        for yy in range(0,h,50):d.line((0,65+yy,w,65+yy),fill='#38aebd');d.text((1,68+yy),str(yy),font=font,fill='#003b47')
        sheet.save(dest/(cid+'_座標.png'))
        book=Path(r['archive_entry']).parts[2];page=book+'_'+Path(r['archive_entry']).stem
        rows.append(dict(case=cid,page=page,book=book,original=r['path'],source_sha256=r['source_sha256'],
            xywh_original=box,description=description,kind=kind,role='external_qualitative_only',candidate_index=idx,
            no_method_predictions_seen=True,no_training_or_model_selection=True,no_expert_red_stain_GT=True,
            original_line_segmentation_split=Path(r['archive_entry']).parent.name,
            independent_of_local_Japanese_training=True))
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),cases=rows,
        selection='Original-only visual sampling of colored writing and negative controls; not random population estimate',
        task='Preserve observed historic colored writing; pigment chemistry and readability not expert annotated',
        baseline_warning='Do not apply Japanese-character DDRM as validated Latin/Syriac repair',
        external_text_line_GT_unused=True,selector_evaluation_only=True,
        archive_protocol_sha256=file_hash(OUT/'external_originals/udiads/protocol.json'),script_sha256=file_hash(__file__))
    (dest/'cases_frozen.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('Frozen',len(rows),'external ROIs from',len({r['page'] for r in rows}),'pages in',sorted({r['book'] for r in rows}),flush=True)

if __name__=='__main__':main()
