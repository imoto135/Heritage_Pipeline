"""Screen original JPEGs for visual collection of red-ink examples.

Input-only color screening, not segmentation evaluation or annotation.
The heuristic selection is biased and can miss faint red; record that limitation.
"""
from pathlib import Path
import hashlib
import json
import shutil
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/red_positive_collection_20260930'
PRIOR=ROOT/'tmp_work/training_free_protection_v2_20260930/manifest.json'
cv2.setNumThreads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prior=json.loads(PRIOR.read_text())
    used_books=set(Path(p).parent.name for p in prior['inputs'])
    # The source PNG for makura has a page folder, so also record its book ID.
    used_books.add('200017197')
    settings=dict(original_roots=['data/train','data/val','data/test'],
                  decode='OpenCV IMREAD_REDUCED_COLOR_4; triage only',
                  loose_red='HSV H<=22 or >=168, S>=35, V>=20, R>=1.12*G and R>=1.12*B',
                  saturated_red='loose AND S>=75 AND R>=1.25*G AND R>=1.25*B',
                  dark_red='loose AND V<140',ranking_region='inner 5%..95%, not a paper mask',
                  per_book='up to 2 saturated and 2 loose/dark candidates, separating sorted page indices by max(2,N//20)',
                  caveat='Color-based purposive screening, not a representative or unbiased benchmark. Faint thin red may be missed.')
    manifest=dict(purpose='Collect original-image candidates; no method outputs or manual pixel labels used',
                  settings=settings,source_sha256=sha(Path(__file__)),prior_manifest_sha256=sha(PRIOR),
                  excluded_development_book_ids=sorted(used_books),opencv=cv2.__version__)
    mp=OUT/'screening_manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:
        raise RuntimeError('Existing scan has different parameters/source')
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(__file__,OUT/'screening_source_snapshot.py')
    rows=[];candidates=[];book_log=[]
    for split in ['train','val','test']:
        for book in sorted((ROOT/'data'/split).iterdir()):
            if not book.is_dir() or book.name in used_books:continue
            paths=sorted(book.glob('*.jpg'));book_rows=[];start=time.monotonic()
            for index,path in enumerate(paths):
                im=cv2.imread(str(path),cv2.IMREAD_REDUCED_COLOR_4)
                if im is None:raise OSError(path)
                hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
                H,S,V=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
                B,G,R=[im[:,:,i].astype(np.float32) for i in range(3)]
                loose=((H<=22)|(H>=168))&(S>=35)&(V>=20)&(R>=1.12*G)&(R>=1.12*B)
                strong=loose&(S>=75)&(R>=1.25*G)&(R>=1.25*B)
                dark=loose&(V<140)
                h,w=loose.shape;sl=(slice(round(.05*h),round(.95*h)),slice(round(.05*w),round(.95*w)))
                row=dict(id=path.stem,original=str(path.relative_to(ROOT)),book=book.name,split=split,index=index,
                         reduced_width=w,reduced_height=h,saturated_pixels=int(strong[sl].sum()),
                         loose_pixels=int(loose[sl].sum()),dark_pixels=int(dark[sl].sum()),
                         full_loose_pixels=int(loose.sum()))
                book_rows.append(row);rows.append(row)
            chosen=[];spacing=max(2,len(paths)//20)
            for key in ['saturated_pixels','dark_pixels']:
                count=0
                for row in sorted(book_rows,key=lambda r:(-r[key],r['id'])):
                    if row[key]<10:break
                    if any(abs(row['index']-r['index'])<spacing for r in chosen):continue
                    chosen.append(dict(row,ranking=key));count+=1
                    if count==2:break
            candidates.extend(chosen)
            book_log.append(dict(book=book.name,split=split,pages=len(paths),candidates=len(chosen)))
            print(f'{split}/{book.name}: {len(paths)} pages, {len(chosen)} candidates, {time.monotonic()-start:.1f}s',flush=True)
    (OUT/'screening_scores.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    (OUT/'candidate_pages.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2)+'\n')
    (OUT/'book_coverage.json').write_text(json.dumps(book_log,ensure_ascii=False,indent=2)+'\n')
    # Input-only thumbnail sheets, with separate strong and dark examples per book.
    for offset in range(0,len(candidates),16):
        subset=candidates[offset:offset+16];sheet=np.full((4*650,4*460,3),245,np.uint8)
        for j,row in enumerate(subset):
            im=cv2.imread(str(ROOT/row['original']),cv2.IMREAD_REDUCED_COLOR_2)
            h,w=im.shape[:2];scale=min(440/w,590/h)
            small=cv2.resize(im,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
            x=(j%4)*460;y=(j//4)*650
            cv2.putText(sheet,str(offset+j)+' '+row['id'],(x+6,y+20),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1)
            cv2.putText(sheet,row['ranking']+' '+str(row[row['ranking']]),(x+6,y+40),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1)
            sheet[y+50:y+50+small.shape[0],x+10:x+10+small.shape[1]]=small
        assert cv2.imwrite(str(OUT/f'candidates_{offset//16+1:02d}.jpg'),sheet)
    (OUT/'screening_complete.json').write_text(json.dumps(dict(pages_screened=len(rows),books_screened=len(book_log),
        candidate_pages=len(candidates),sheet_count=(len(candidates)+15)//16),indent=2)+'\n')
    print(f'Completed input screening: {len(rows)} pages, {len(candidates)} candidates.',flush=True)


if __name__=='__main__':
    main()
