"""Stratified selected-point diagnostics; no mask-based score for color restorers."""
import json,collections
from pathlib import Path
import numpy as np
from PIL import Image
from synthetic_layers import OUT,file_hash

def mask_path(method,page):
 d=OUT/'independent_predictions'
 if method=='csc_v2_red':return d/'csc_v2'/page/'red_mask.png'
 if method in ['hsv_loose','relative_chroma','relative_ridge']:return d/'source_heuristics'/page/(method+'.png')
 if method=='docres_foreground':return OUT/'independent_direct/docres_binarization'/page/'ink_mask.png'
 if method=='sauvola_foreground':return d/'source_heuristics'/page/'sauvola_black.png'
 return d/method/page/'mark_mask.png'

def main():
 protocol_file=OUT/('independent_evaluation_protocol_v3.json' if (OUT/'independent_evaluation_protocol_v3.json').exists() else 'independent_evaluation_protocol_v2.json');proto=json.loads(protocol_file.read_text());points=[];sources=[]
 f=OUT/'sparse_original_review/points_frozen.json';sources.append(f)
 for p in json.loads(f.read_text())['points']:
  if p['role'] in proto['local_roles']:points.append({**p,'case_id':p['case']})
 f=OUT/'external_originals/frozen_points/points_frozen.json';sources.append(f)
 for p in json.loads(f.read_text())['points']:points.append({**p,'case_id':p['case'],'page':'external_'+p['page']})
 f=OUT/'prospective_cases/points_frozen.json';sources.append(f);points+=json.loads(f.read_text())['points']
 methods=list(proto['model_resources'])+['csc_v2_red','hsv_loose','relative_chroma','relative_ridge','docres_foreground','sauvola_foreground','docres_foreground_chroma','preserve_everything','preserve_nothing']
 rows=[];details=[];missing=[]
 for method in methods:
  groups=collections.defaultdict(lambda:dict(points=0,selected_center=0,selected_within_2px=0));cache={}
  for p in points:
   page=p['page'];x,y=p['xy_original'];kind=method
   if method in ['preserve_everything','preserve_nothing']:center=support=method=='preserve_everything'
   else:
    if page not in cache:
     paths=[mask_path(method,page)] if method!='docres_foreground_chroma' else [mask_path('docres_foreground',page),mask_path('relative_chroma',page)]
     if not all(f.exists() for f in paths):missing.append(dict(method=method,page=page,case_id=p['case_id']));continue
     masks=[np.asarray(Image.open(f).convert('L'))>127 for f in paths];m=masks[0]
     for mm in masks[1:]:assert m.shape==mm.shape;m=m&mm
     cache[page]=(m,{str(f.relative_to(OUT)):file_hash(f) for f in paths})
    m,hashes=cache[page];center=bool(m[y,x]);support=bool(m[max(0,y-2):y+3,max(0,x-2):x+3].any())
   k=(p['case_id'],p['book'],p['role'],p['label']);g=groups[k];g['points']+=1;g['selected_center']+=int(center);g['selected_within_2px']+=int(support);details.append(dict(method=method,point=p,selected_center=center,selected_within_2px=support))
  for k,g in groups.items():rows.append(dict(method=method,case_id=k[0],book=k[1],role=k[2],label=k[3],**g,center_fraction=g['selected_center']/g['points']))
 d=OUT/'independent_point_scores';d.mkdir(exist_ok=True);record=dict(rows=rows,point_sources={str(f.relative_to(OUT)):file_hash(f) for f in sources},protocol_sha256=file_hash(protocol_file),source_sha256=file_hash(__file__),missing=sorted({json.dumps(r,sort_keys=True) for r in missing}),interpretation='Selected tentative original points only; stratify annotation/seal and manuscript; not independent pixels, chemical stain GT, or reading accuracy; color restorers evaluated visually separately')
 (d/'scores.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));(d/'point_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2));print('Independent scores rows',len(rows),'missing method/cases',len(record['missing']))
if __name__=='__main__':main()
