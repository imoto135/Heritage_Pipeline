"""Selected no-visible-mark regions; supervised dev diagnostics, not pixel GT."""
import argparse,json
import numpy as np
from PIL import Image
from synthetic_layers import OUT,file_hash
from render_selector_diagnostics import mask_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',required=True);p.add_argument('--suffix',required=True);a=p.parse_args()
    regions=json.loads((OUT/'original_paper_negative_regions/regions_frozen.json').read_text())['regions'];rows=[];cache={}
    for method in a.methods:
        for idx,r in enumerate(regions):
            path=mask_path(method,r['page'])
            if not path.exists():continue
            key=(method,r['page'])
            if key not in cache:cache[key]=np.asarray(Image.open(path).convert('L'))>127
            x,y,w,h=r['xywh_original'];selected=int(cache[key][y:y+h,x:x+w].sum())
            excluded=method.split('_exclude')[-1] if '_exclude' in method else None
            training_region=('region_' in method or method.startswith('texture_paper_texture') or method.startswith('partial_')) and excluded!=r['book']
            rows.append(dict(method=method,region=idx,book=r['book'],page=r['page'],xywh_original=r['xywh_original'],selected=selected,pixels=w*h,selected_fraction=selected/(w*h),
                region_supervision_used_by_method=training_region,no_visible_red_annotation='Tentative original-only Codex review',definite_stain_GT=False,
                evaluation_scope='Held-out-book diagnostic' if excluded==r['book'] else 'Development diagnostic, may include supervised regions'))
    dest=OUT/'paper_region_scores'/a.suffix;dest.mkdir(parents=True,exist_ok=True)
    (dest/'scores.json').write_text(json.dumps(dict(rows=rows,script_sha256=file_hash(__file__),regions_sha256=file_hash(OUT/'original_paper_negative_regions/regions_frozen.json'),
        interpretation='Proportion selected in tentative paper-negative rectangles; not representative full-page precision or definitive real stain removal'),ensure_ascii=False,indent=2))
    for method in a.methods:
        current=[r for r in rows if r['method']==method];print(method,sum(r['selected'] for r in current),'/',sum(r['pixels'] for r in current),flush=True)

if __name__=='__main__':main()
