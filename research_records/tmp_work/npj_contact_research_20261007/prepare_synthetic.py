from common import *
import csv,collections,time
def main():
 cv2.setNumThreads(1);rng=np.random.default_rng(20261007)
 rows=list(csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv').open()))
 scores={r['page']:r for r in csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/screening_scores.csv').open())}
 groups=collections.defaultdict(list)
 for r in rows:
  if not r['page'].startswith('hnsd'):groups[r['book']].append(r)
 eligible=[b for b,rs in groups.items() if len(rs)>=15]
 eligible.sort(key=lambda b:hashlib.sha256(('contact-pilot-v1-'+b).encode()).hexdigest())
 chosen=eligible[:28];splits={b:('train' if i<16 else 'dev' if i<20 else 'test') for i,b in enumerate(chosen)}
 out=HERE/'synthetic';out.mkdir(exist_ok=True);patches=[];manifest=[]
 dump(out/'protocol.json',dict(created=time.time(),source_sha256=sha(__file__),books=splits,
  native_patch_size=256,resize=False,known_added_stain_GT_only=True,clean_material_GT=False,
  purpose='recovery of digitally added contamination, not proof of real stain removal',
  source_selection='hash order of eligible non-hnsd books; low brown score within central 70% of book; text-like local contrast filtering',
  contamination='multiplicative RGB optical-density ellipses, with independent color/shape OOD stress tests',
  train_originals_exclude_hnsd007=True,book_disjoint=True))
 for b in chosen:
  rs=sorted(groups[b],key=lambda r:r['page']);middle=rs[max(1,len(rs)//8):max(3,7*len(rs)//8)]
  middle.sort(key=lambda r:(int(scores[r['page']]['brown_blob_score']),hashlib.sha256(r['page'].encode()).hexdigest()))
  for r in middle[:3]:
   a=read(ROOT/r['original']);h,w=a.shape[:2];candidates=[]
   for attempt in range(80):
    x=int(rng.integers(max(0,int(w*.12)),max(int(w*.12)+1,w-256-int(w*.12))))
    y=int(rng.integers(max(0,int(h*.12)),max(int(h*.12)+1,h-256-int(h*.12))))
    p=a[y:y+256,x:x+256]
    if p.shape!=(256,256,3):continue
    d,_=morph_contrast(p,9);density=float((d>.14).mean())
    if .025<density<.26:candidates.append((p,[x,y,256,256]))
    if len(candidates)>=12:break
   for p,xywh in candidates:
    patches.append(p);manifest.append(dict(index=len(patches)-1,split=splits[b],book=b,page=r['page'],original=r['original'],xywh=xywh,
       source_sha256=sha(ROOT/r['original'])))
  print('patch prep',b,splits[b],len(patches),flush=True)
 np.save(out/'patches.npy',np.stack(patches));dump(out/'patches.json',manifest)
 # Contact-sheet samples are only source review, before learned outputs.
 for split in ['train','dev','test']:
  inds=[m['index'] for m in manifest if m['split']==split]
  canvas=Image.new('RGB',(8*220,((len(inds[::12])+7)//8)*250),'white');d=ImageDraw.Draw(canvas);f=ImageFont.truetype(FONT,14)
  for j,i in enumerate(inds[::12]):
   p=Image.fromarray(patches[i]);p=p.resize((220,220),Image.Resampling.LANCZOS);x=(j%8)*220;y=(j//8)*250
   canvas.paste(p,(x,y));d.text((x,y+222),manifest[i]['page'],font=f,fill='black')
  canvas.save(out/(split+'_sources.png'))
 dump(out/'prepared.json',dict(patches=len(patches),books=len(chosen),split_counts=dict(collections.Counter(m['split'] for m in manifest)),
  patch_array_sha256=sha(out/'patches.npy'),source_manifest_sha256=sha(out/'patches.json')))
if __name__=='__main__':main()
