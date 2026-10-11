"""Recreate the exact region-RF book-holdout folds and save native masks."""
import json,pickle,time
import cv2,numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from synthetic_layers import OUT,ROOT,file_hash
from fit_sparse_color_context import feature_maps

def main():
    cv2.setNumThreads(2)
    records=json.loads((OUT/'region_supported_trials/training_samples.json').read_text())
    cases={r['page']:r for r in json.loads((OUT/'cases_frozen.json').read_text())['cases'] if r['role']=='development'}
    xs=[];ys=[];books=[]
    for page in sorted({r['page'] for r in records}):
        c=cases[page];assert file_hash(ROOT/c['original'])==c['source_sha256'];rgb=np.asarray(Image.open(ROOT/c['original']).convert('RGB'));maps,names=feature_maps(rgb,True)
        for r in [q for q in records if q['page']==page]:
            x,y=r['xy_original'];xs.append([m[y,x] for m in maps]);ys.append(r['label']=='observed_red_mark');books.append(r['book'])
    x=np.asarray(xs,np.float32);y=np.asarray(ys,np.int64);b=np.asarray(books);dest=OUT/'region_supported_holdout';dest.mkdir(exist_ok=True)
    old=next(r for r in json.loads((OUT/'region_supported_trials/models.json').read_text()) if r['name']=='region_spatial_forest')
    for book in sorted(set(books)):
        model=RandomForestClassifier(n_estimators=256,max_depth=5,min_samples_leaf=2,class_weight='balanced',random_state=524,n_jobs=4)
        train=b!=book;test=~train;model.fit(x[train],y[train]);selected=model.predict_proba(x[test])[:,1]>=.5;truth=y[test].astype(bool)
        prior=next(r for r in old['development_book_holdout'] if r['book']==book)
        assert int((selected&truth).sum())==prior['mark_selected'];assert int((selected&~truth).sum())==prior['nonmark_selected']
        path=dest/f'region_spatial_forest_exclude{book}.pkl';path.write_bytes(pickle.dumps(model,protocol=4));method=path.stem
        for page,c in cases.items():
            if c['book']!=book or page not in {r['page'] for r in records}:continue
            folder=OUT/'native_predictions'/method/page;folder.mkdir(parents=True,exist_ok=True);done=folder/'complete.json'
            if done.exists():
                rec=json.loads(done.read_text());assert rec['model_sha256']==file_hash(path);assert rec['mask_sha256']==file_hash(folder/'mark_mask.png');continue
            start=time.monotonic();rgb=np.asarray(Image.open(ROOT/c['original']).convert('RGB'));h,w=rgb.shape[:2];maps,features=feature_maps(rgb,True);assert features==names;prob=np.zeros((h,w),np.float32)
            for yy in range(0,h,96):
                z=np.stack([m[yy:yy+96].reshape(-1) for m in maps],1);prob[yy:yy+96]=model.predict_proba(z)[:,1].reshape(min(96,h-yy),w)
            Image.fromarray((prob>=.5).astype(np.uint8)*255).save(folder/'mark_mask.png')
            Image.fromarray(np.rint(prob*255).astype(np.uint8)).save(folder/'mark_probability_preview.png')
            done.write_text(json.dumps(dict(name=method,page=page,excluded_book=book,actual_training_books=sorted(set(b[train])),threshold=.5,model_sha256=file_hash(path),
                source_sha256=c['source_sha256'],script_sha256=file_hash(__file__),training_samples_sha256=file_hash(OUT/'region_supported_trials/training_samples.json'),
                mask_sha256=file_hash(folder/'mark_mask.png'),seconds=time.monotonic()-start,fold_counts_exact_original_manifest=True,expert_GT=False),indent=2))
            print('Region held-book',book,page,round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
