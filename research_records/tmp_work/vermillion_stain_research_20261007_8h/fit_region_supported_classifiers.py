"""Exploratory sparse mark points plus tentative paper-region negatives.

Region sampling changes supervision; it is not a training-free improvement.
"""
from pathlib import Path
import json
import pickle
import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from synthetic_layers import OUT,ROOT,file_hash
from fit_sparse_color_context import feature_maps

def main():
    cv2.setNumThreads(2);dest=OUT/'region_supported_trials';dest.mkdir(exist_ok=True)
    points=[p for p in json.loads((OUT/'sparse_original_review/points_frozen.json').read_text())['points'] if p['role']=='development']
    regions=json.loads((OUT/'original_paper_negative_regions/regions_frozen.json').read_text())['regions']
    cases={r['page']:r for r in json.loads((OUT/'cases_frozen.json').read_text())['cases']}
    rng=np.random.default_rng(761);negatives=[]
    for idx,r in enumerate(regions):
        x,y,w,h=r['xywh_original']
        for _ in range(64):
            negatives.append(dict(page=r['page'],book=r['book'],xy_original=[int(rng.integers(x,x+w)),int(rng.integers(y,y+h))],
                label='paper_or_brown_nonmark',role='development',region_index=idx))
    records=points+negatives;books=sorted({r['book'] for r in records})
    heldout=set(json.loads((OUT/'cases_frozen.json').read_text())['validation_books']);assert not set(books)&heldout
    protocol=dict(created_from_development_only=True,source_sha256=file_hash(__file__),training_books=books,
        positive_points=sum(p['label']=='observed_red_mark' for p in points),original_points=len(points),
        paper_negative_regions=len(regions),paper_negative_samples=len(negatives),seed=761,
        correlated_samples='832 sampled negatives are only 13 small regions in 3 books; not 832 independent annotations',
        annotation_status='Tentative Codex original visual labels; not expert reviewed',
        never_use_validation_books=sorted(heldout),threshold=.5,
        fixed_hyperparameters='Same as sparse point classifiers: logistic C1/balanced; forest256/depth5/leaf2/balanced',
        book_holdout='Known development books only; any improvement is exploratory')
    (dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    (dest/'training_samples.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    outputs=[]
    for spatial in [False,True]:
        xs=[];ys=[];bs=[]
        for page in sorted({r['page'] for r in records}):
            case=cases[page];rgb=np.asarray(Image.open(ROOT/case['original']).convert('RGB'));maps,names=feature_maps(rgb,spatial)
            for r in [rr for rr in records if rr['page']==page]:
                x,y=r['xy_original'];xs.append([m[y,x] for m in maps]);ys.append(r['label']=='observed_red_mark');bs.append(r['book'])
        x=np.asarray(xs,np.float32);y=np.asarray(ys,np.int64);b=np.asarray(bs)
        for classifier in ['logistic','forest']:
            def create():
                if classifier=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=1000,random_state=524))
                return RandomForestClassifier(n_estimators=256,max_depth=5,min_samples_leaf=2,class_weight='balanced',random_state=524,n_jobs=4)
            name='region_'+('spatial_' if spatial else 'color_')+classifier;folds=[]
            for book in books:
                model=create();train=b!=book;test=~train;model.fit(x[train],y[train]);pred=model.predict_proba(x[test])[:,1]>=.5
                t=y[test].astype(bool);folds.append(dict(book=book,mark_selected=int((pred&t).sum()),mark_points=int(t.sum()),
                    nonmark_selected=int((pred&~t).sum()),nonmark_samples=int((~t).sum()),
                    label_status='Tentative and correlated selected points/regions'))
            model=create();model.fit(x,y);path=dest/(name+'.pkl');path.write_bytes(pickle.dumps(model,protocol=4))
            record=dict(name=name,spatial=spatial,classifier=classifier,features=names,training_points=len(y),
                development_book_holdout=folds,model_sha256=file_hash(path),model_path=str(path),training_on_tentative_labels=True)
            outputs.append(record);print(name,folds,flush=True)
    (dest/'models.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2))
    print('Region-supported exploratory models:',len(outputs),flush=True)

if __name__=='__main__':main()
