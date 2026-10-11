from common import *
import csv,time,concurrent.futures
def score(r):
 cv2.setNumThreads(1)
 im=Image.open(ROOT/r['original']).convert('RGB');W,H=im.size
 im.thumbnail((1400,1400),Image.Resampling.LANCZOS);a=np.asarray(im);h,w=a.shape[:2]
 lab=cv2.cvtColor(a,cv2.COLOR_RGB2LAB).astype('float32');g=gray(a)
 # Local chromatic changes relative to broad paper estimate; original-only screening.
 ls=cv2.GaussianBlur(lab,(0,0),2);lp=cv2.GaussianBlur(lab,(0,0),55)
 warm=(ls[:,:,2]-lp[:,:,2]>2.5)&(ls[:,:,1]-lp[:,:,1]>-.5)&(g<.88)&(g>.2)
 warm[:int(h*.06)]=False;warm[int(h*.94):]=False;warm[:,:int(w*.06)]=False;warm[:,int(w*.94):]=False
 warm=cv2.morphologyEx(warm.astype('uint8'),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
 n,labels,stats,_=cv2.connectedComponentsWithStats(warm,8)
 close=cv2.morphologyEx(g,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
 ridge=(close-g)/np.maximum(close,.1)>.12
 best=[]
 for i in range(1,n):
  x,y,bw,bh,area=map(int,stats[i]);region=labels==i
  if area<160 or area>h*w*.13 or bw<15 or bh<15:continue
  inside=region&ridge;contact=int(inside.sum());density=contact/area
  # Prefer broad colored regions containing some thin dark detail, not solid dark shapes.
  if density<.008 or density>.35:continue
  val=float(round(np.sqrt(area)*np.sqrt(contact)*min(bw,bh)/max(bw,bh),3))
  best.append(dict(score=val,area=area,stroke_hint_pixels=contact,stroke_hint_density=density,
   thumb_xywh=[x,y,bw,bh],page_xywh=[round(x*W/w),round(y*H/h),round(bw*W/w),round(bh*H/h)]))
 best.sort(key=lambda v:-v['score'])
 return dict(page=r['page'],book=r['book'],split=r['split'],original=r['original'],
   original_size=[W,H],screen_size=[w,h],candidates=best[:3],score=best[0]['score'] if best else 0)
def main():
 cv2.setNumThreads(1);out=HERE/'other_book_screen';out.mkdir(exist_ok=True)
 rows=list(csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv').open()))
 rows=[r for r in rows if not r['page'].startswith('hnsd007_')]
 dump(out/'manifest.json',dict(source_sha256=sha(__file__),input_pages=len(rows),
  protocol='original-only local warm region + dark local-detail intersection; not material or writing GT',
  previous_selection_sha256=sha(SOURCE/'benchmark_candidates.json'),created=time.time()))
 results=[json.loads(s) for s in (out/'rows.jsonl').read_text().splitlines()] if (out/'rows.jsonl').exists() else []
 done={r['page'] for r in results};remaining=[r for r in rows if r['page'] not in done];start=time.monotonic()
 with (out/'rows.jsonl').open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for i,v in enumerate(pool.map(score,remaining),start=len(results)):
   results.append(v);f.write(json.dumps(v,ensure_ascii=False)+'\n');f.flush()
   if (i+1)%100==0:
    stat=dict(done=i+1,total=len(rows),seconds=round(time.monotonic()-start,1),candidates=sum(r['score']>0 for r in results))
    dump(out/'progress.json',stat);print(json.dumps(stat),flush=True)
 results.sort(key=lambda r:-r['score']);dump(out/'ranked.json',results)
 dump(out/'complete.json',dict(pages=len(results),seconds=time.monotonic()-start,positive=sum(r['score']>0 for r in results)))
 print('other book scan complete',len(results),flush=True)
if __name__=='__main__':main()
