from pathlib import Path
import cv2,numpy as np,json,time,argparse,csv
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def main(all_pages=False):
 cv2.setNumThreads(1);im=cv2.imread(str(OUT/'paper/Fig10.png'))
 boxes=[[4,5,480,397],[4,739,480,394],[4,1470,480,394]]
 templates=[]
 for i,(x,y,w,h) in enumerate(boxes):
  a=im[y:y+h,x:x+w];cv2.imwrite(str(OUT/'paper'/f'Fig10_original_row{i+1}_match_crop.png'),a)
  kp,de=cv2.SIFT_create(nfeatures=1800).detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None);templates.append((kp,de))
 files=[ROOT/r['original'] for r in csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv').open())] if all_pages else [p for s in ['train','val','test'] for p in (ROOT/'data/split_dataset'/s/'200010454').glob('*.jpg')]
 assert files,'Image inventory is empty; pathlib.rglob does not descend split symlinks'
 def process(p):
  a=cv2.imread(str(p));scale=min(1,(1000 if all_pages else 1400)/max(a.shape[:2]));a=cv2.resize(a,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
  kp,de=cv2.SIFT_create(nfeatures=5000).detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None);rr=[]
  if de is None:return rr
  for i,(kt,dt) in enumerate(templates):
   matches=cv2.BFMatcher().knnMatch(dt,de,k=2);good=[m for m,n in matches if m.distance<.72*n.distance];n=0;H=None
   if len(good)>=6:
    src=np.float32([kt[m.queryIdx].pt for m in good]);dst=np.float32([kp[m.trainIdx].pt for m in good]);H,mask=cv2.findHomography(src,dst,cv2.RANSAC,4)
    if mask is not None:n=int(mask.sum())
   rr.append(dict(row=i+1,source=str(p.relative_to(ROOT)),ratio_matches=len(good),homography_inliers=n,scale=scale,template_to_scaled_original=H.tolist() if H is not None else None))
  return rr
 results=[]
 with ThreadPoolExecutor(max_workers=8 if all_pages else 4) as pool:
  for j,rr in enumerate(pool.map(process,files),1):
   results.extend(rr)
   if j%250==0:print('paper-source search',j,'/',len(files),'best inliers',max(r['homography_inliers'] for r in results),flush=True)
 ranked={str(i):sorted([r for r in results if r['row']==i],key=lambda r:r['homography_inliers'],reverse=True)[:5] for i in [1,2,3]}
 (OUT/'paper'/('page_matches_all.json' if all_pages else 'page_matches.json')).write_text(json.dumps(dict(created=time.time(),search_scope=f'{len(files)} local JPG pages; low inlier counts do not establish a source match',template_boxes=boxes,ranked=ranked),indent=2))
 for i,rr in ranked.items():print(i,[(r['source'],r['homography_inliers']) for r in rr[:3]],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--all-pages',action='store_true');a=p.parse_args();main(a.all_pages)
