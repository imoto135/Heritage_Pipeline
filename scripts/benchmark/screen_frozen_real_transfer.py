"""Audit page exposure and render originals only; never load prediction masks."""
import json, re, hashlib
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/frozen_real_transfer_20261022'
PAT=re.compile(r'(?:\d{9}|[a-z]{4}\d{3})_\d{3,5}(?:_[012])?')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def audit():
    assert not (OUT/'exposure_audit.json').exists()
    seen=defaultdict(set);counts=defaultdict(int)
    for base in [ROOT/'tmp_work',ROOT/'scripts']:
        for p in base.rglob('*'):
            if OUT in p.parents:continue
            for page in PAT.findall(str(p.relative_to(ROOT))):seen[page].add('path:'+str(p.relative_to(ROOT)))
            if p.is_file() and p.suffix.lower() in ('.json','.jsonl','.csv','.md','.txt','.py') and p.stat().st_size<8_000_000:
                counts[p.suffix]+=1
                s=p.read_text(errors='replace')
                for page in set(PAT.findall(s)):seen[page].add(str(p.relative_to(ROOT)))
    inventory=[]
    for split in ['val','test']:
        for p in sorted((ROOT/'data'/split).rglob('*')):
            if p.suffix.lower() not in ('.png','.jpg','.jpeg','.tif','.tiff'):continue
            page=p.stem;prior=sorted(seen.get(page,[]))
            inventory.append(dict(page=page,book=p.parent.name,split=split,source=str(p.relative_to(ROOT)),prior_mentions=len(prior),examples=prior[:3]))
    unused=[r for r in inventory if r['prior_mentions']==0 and r['book']!='200017197']
    dump(OUT/'exposure_audit.json',dict(scope='All tmp_work and scripts paths; text JSON/JSONL/CSV/MD/TXT/PY below8MB. A mention is conservatively treated as exposure. Not a complete external usage history.',text_files=dict(counts),inventory=inventory,unmentioned=unused))
    print('pages',len(inventory),'unmentioned',len(unused),'books',len({r['book'] for r in unused}),flush=True)
def render():
    import numpy as np
    rows=json.loads((OUT/'refined_exposure_audit.json').read_text())['eligible']
    rng=np.random.default_rng(20261022);chosen=[]
    for book in sorted({r['book'] for r in rows}):
        pool=[r for r in rows if r['book']==book]
        chosen.extend(pool[int(i)] for i in sorted(rng.choice(len(pool),min(4,len(pool)),replace=False)))
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    for k in range(0,len(chosen),16):
        batch=chosen[k:k+16];canvas=Image.new('RGB',(4*310,4*465),'#faf9f5');d=ImageDraw.Draw(canvas)
        for i,r in enumerate(batch):
            a=Image.open(ROOT/r['source']).convert('RGB');r['shape_wh']=list(a.size)
            a.thumbnail((300,425));x=(i%4)*310;y=(i//4)*465
            d.text((x+3,y+3),r['page'],font=font,fill='black');canvas.paste(a,(x+5,y+31))
        canvas.save(OUT/f'source_survey_{k//16:02d}.jpg')
    dump(OUT/'source_survey.json',dict(seed=20261022,selection='Up to4 uniformly sampled pages per previously unmentioned val/test book; source-only survey',pages=chosen))
    print('survey',len(chosen),'sheets',(len(chosen)+15)//16,flush=True)
def refine():
    import subprocess, csv
    safe={
      'tmp_work/npj_contact_research_20261007/other_book_screen/ranked.json',
      'tmp_work/npj_contact_research_20261007/other_book_screen/rows.jsonl',
      'tmp_work/red_positive_collection_20260930/screening_scores.json',
      'tmp_work/npj_failure_collection_20261001/screening_scores.csv',
      'tmp_work/vermillion_stain_research_20261007_8h/relative_red_screen/scores.json',
      'tmp_work/test_only_challenge_search_20261009/original_features.csv',
      'tmp_work/unseen_dark_margin_screening_20261001/scores.json',
      'tmp_work/unseen_directional_margin_screening_20261001/scores.json',
      'tmp_work/unseen_margin_validation_20261001/screening_scores.json',
      'tmp_work/unseen_thin_margin_validation_20261001/screening_scores.json',
      'tmp_work/vermillion_stain_research_20261007_8h/small_annotation_screen/scores.json',
      'tmp_work/vermillion_stain_research_20261007_8h/small_annotation_screen/per_page_checkpoint.jsonl',
      'tmp_work/vermillion_stain_research_20261007_8h/bright_annotation_screen/per_page_checkpoint.jsonl'}
    inventories={'tmp_work/npj_failure_collection_20261001/page_inventory.csv','tmp_work/npj_failure_followup_20261001/page_inventory.csv'}
    seen=defaultdict(set)
    paths=subprocess.check_output(['rg','--files','-uuu','tmp_work','scripts'],cwd=ROOT,text=True).splitlines()
    for name in paths:
        if name.startswith(str(OUT.relative_to(ROOT))+'/') or name in safe:continue
        p=ROOT/name
        for page in set(PAT.findall(name)):seen[page].add('path:'+name)
        if name in inventories:
            for r in csv.DictReader(p.open()):
                if r.get('review_status') not in ('unreviewed','',None) or r.get('observation') or r.get('roi_xywh'):
                    seen[r['page']].add(name+':reviewed')
            continue
        if p.suffix.lower() in ('.json','.jsonl','.csv','.md','.txt','.py') and p.stat().st_size<8_000_000:
            for page in set(PAT.findall(p.read_text(errors='replace'))):seen[page].add(name)
    rows=json.loads((OUT/'exposure_audit.json').read_text())['inventory']
    for r in rows:r['other_mentions']=sorted(seen.get(r['page'],[]))
    eligible=[r for r in rows if not r['other_mentions'] and r['book']!='200017197']
    revision='refined_exposure_audit_v2.json'
    dump(OUT/revision,dict(excluded_source_registry_files=sorted(safe),inventories_unreviewed_rows_only=sorted(inventories),inventory=rows,eligible=eligible,
       limitation='All pages previously automatically screened and CSC-preprocessed; eligible means no other local text/file-path mention in audited scope. Not proof of complete historical or external blindness.'))
    print('eligible',len(eligible),'books',len({r['book'] for r in eligible}),flush=True)
if __name__=='__main__':
    import sys
    {'audit':audit,'refine':refine,'render':render}[sys.argv[1]]()
