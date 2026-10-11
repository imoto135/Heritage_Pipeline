"""Score saved full-page masks on the exploratory frozen visual probes."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
ROOT = HERE.parents[2]
METHODS = ('sauvola', 'fixed', 'auto', 'thi025_c160')


def score_points(mask, points, radius, corridor_radius):
    h, w = mask.shape
    hits = []
    exact_hits = []
    centers = []
    for x, y in points:
        assert 0 <= x < w and 0 <= y < h
        yy, xx = np.ogrid[max(0,y-radius):min(h,y+radius+1),
                          max(0,x-radius):min(w,x+radius+1)]
        circle = (xx-x)**2 + (yy-y)**2 <= radius**2
        patch = mask[max(0,y-radius):min(h,y+radius+1),
                     max(0,x-radius):min(w,x+radius+1)]
        hits.append(bool((patch & circle).any()))
        exact_hits.append(bool(mask[y,x]))
        centers.append((x,y))
    line_pixels = []
    for (x0,y0),(x1,y1) in zip(centers,centers[1:]):
        n=max(abs(x1-x0),abs(y1-y0))+1
        for x,y in zip(np.rint(np.linspace(x0,x1,n)).astype(int),
                       np.rint(np.linspace(y0,y1,n)).astype(int)):
            yy,xx=np.ogrid[max(0,y-corridor_radius):min(h,y+corridor_radius+1),
                           max(0,x-corridor_radius):min(w,x+corridor_radius+1)]
            near=(xx-x)**2+(yy-y)**2<=corridor_radius**2
            line_pixels.append(bool((mask[max(0,y-corridor_radius):min(h,y+corridor_radius+1),
                                          max(0,x-corridor_radius):min(w,x+corridor_radius+1)] & near).any()))
    return dict(exact_points_retained=sum(exact_hits),near_points_retained=sum(hits),points_total=len(hits),
                path_near_foreground_fraction=float(np.mean(line_pixels)),
                note='Proximity to black pixels only; this does not establish connected or correct lettering.')


def main():
    frozen=json.loads((HERE/'frozen_probes.json').read_text())
    for source in frozen['source_files'].values():
        assert hashlib.sha256((ROOT/source['original']).read_bytes()).hexdigest()==source['sha256']
    rows=[]
    mask_hashes={}
    for probe in frozen['probes']:
        page=frozen['source_files'][probe['case']]['page']
        for method in METHODS:
            path=SOURCE/'masks'/page/(method+'.png')
            mask_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            mask=np.asarray(Image.open(path).convert('L'))==0
            if 'points_xy' in probe:
                measurement=score_points(mask,probe['points_xy'],probe['radius'],probe['corridor_radius'])
            else:
                x,y,w,h=probe['box_xywh'];assert x>=0 and y>=0 and x+w<=mask.shape[1] and y+h<=mask.shape[0]
                measurement=dict(black_pixels=int(mask[y:y+h,x:x+w].sum()),area=w*h,
                                 note='Black area of one selected nontext patch; no dense stain outline.')
            rows.append(dict(probe=probe['id'],page=page,method=method,**measurement))
    out=dict(frozen_sha256=hashlib.sha256((HERE/'frozen_probes.json').read_bytes()).hexdigest(),
             input='Previously saved full-page masks; no new inference or parameter search',
             mask_sha256=mask_hashes,rows=rows)
    (HERE/'scores.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for row in rows:print(row)


if __name__=='__main__':main()
