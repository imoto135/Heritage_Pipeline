"""Proportions at tentative selected points, explicitly not pixel precision/recall."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image
from synthetic_layers import OUT,file_hash
from render_selector_diagnostics import mask_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',default=['saved_csc_v1_red','saved_csc_v2_red','csc_v2_red','hsv_loose','relative_ridge','relative_pointwise','relative_overlap'])
    p.add_argument('--roles',nargs='+',default=['development']);p.add_argument('--suffix',default='initial_development')
    a=p.parse_args();points=json.loads((OUT/'sparse_original_review/points_frozen.json').read_text())['points']
    dest=OUT/'sparse_scores'/a.suffix;dest.mkdir(parents=True,exist_ok=True)
    records=[];details=[];cache={}
    for method in a.methods+['preserve_everything','preserve_nothing']:
        groups={}
        for point in points:
            if point['role'] not in a.roles:continue
            key=(method,point['page'])
            if method not in ['preserve_everything','preserve_nothing']:
                path=mask_path(*key)
                if not path.exists():continue
                if key not in cache:cache[key]=np.asarray(Image.open(path).convert('L'))>127
                mask=cache[key];x,y=point['xy_original'];strict=bool(mask[y,x])
                support=bool(mask[max(0,y-2):y+3,max(0,x-2):x+3].any())
            else:strict=support=method=='preserve_everything'
            group_key=(point['case'],point['label'])
            group=groups.setdefault(group_key,dict(method=method,case=point['case'],book=point['book'],role=point['role'],label=point['label'],points=0,
                selected_center=0,selected_within_2px=0))
            group['points']+=1;group['selected_center']+=int(strict);group['selected_within_2px']+=int(support)
            details.append(dict(method=method,point=point,selected_center=strict,selected_within_2px=support))
        for group in groups.values():
            group['center_fraction']=group['selected_center']/group['points']
            group['within_2px_fraction']=group['selected_within_2px']/group['points'];records.append(group)
    (dest/'scores.json').write_text(json.dumps(dict(records=records,
        interpretation='Selected original-only tentative points, not unbiased full-page accuracy or definite stain GT',
        points_sha256=file_hash(OUT/'sparse_original_review/points_frozen.json'),script_sha256=file_hash(__file__),
        roles=a.roles,methods=a.methods),ensure_ascii=False,indent=2))
    (dest/'point_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
    for r in records:
        print(r['method'],r['case'],r['label'],f"{r['selected_center']}/{r['points']}",flush=True)

if __name__=='__main__':main()
