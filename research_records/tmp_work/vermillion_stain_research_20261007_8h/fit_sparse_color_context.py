"""Exploratory small-data classifiers, with and without spatial chromatic features.

Fits original-only tentative DEVELOPMENT points, never validation points.
No claim that sparse supervision or random forests are novel.
"""
from pathlib import Path
import json
import time
import pickle
import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from synthetic_layers import OUT,ROOT,local_paper,file_hash
from prepare_fullpage_baselines import ridge

def feature_maps(rgb,spatial=True):
    a=cv2.medianBlur(rgb,3).astype(np.float32)
    bg=local_paper(a.astype(np.uint8));od=np.log((bg+1)/(a+1))
    lab=cv2.cvtColor(a/255,cv2.COLOR_RGB2LAB);hsv=cv2.cvtColor(a.astype(np.uint8),cv2.COLOR_RGB2HSV)
    dg=od[:,:,1]-od[:,:,0];db=od[:,:,2]-od[:,:,0]
    maps=[a[:,:,0]/255,a[:,:,1]/255,a[:,:,2]/255,
        od[:,:,0],od[:,:,1],od[:,:,2],dg,db,lab[:,:,1]/40,lab[:,:,2]/40,
        hsv[:,:,1].astype(np.float32)/255,lab[:,:,0]/100]
    names=['R','G','B','OD_R','OD_G','OD_B','relative_red_G','relative_red_B','Lab_a','Lab_b','saturation','Lab_L']
    if spatial:
        chroma=np.minimum(dg,db)
        for sigma in [1.,3.,8.,20.]:
            low=cv2.GaussianBlur(chroma,(0,0),sigma)
            maps.extend([low,chroma-low]);names.extend([f'chroma_mean_sigma{sigma}',f'chroma_band_sigma{sigma}'])
        maps.append(ridge(chroma));names.append('multiscale_chroma_ridge')
        gradient=cv2.magnitude(cv2.Sobel(chroma,cv2.CV_32F,1,0),cv2.Sobel(chroma,cv2.CV_32F,0,1))
        maps.append(gradient);names.append('chroma_gradient')
    return maps,names

def main():
    cv2.setNumThreads(2);dest=OUT/'sparse_context_trials';dest.mkdir(exist_ok=True)
    frozen=json.loads((OUT/'sparse_original_review/points_frozen.json').read_text());points=[p for p in frozen['points'] if p['role']=='development']
    cases={r['page']:r for r in json.loads((OUT/'cases_frozen.json').read_text())['cases']}
    books=sorted({p['book'] for p in points});heldout=set(json.loads((OUT/'cases_frozen.json').read_text())['validation_books'])
    assert not heldout&set(books)
    # Fixed settings; no threshold tuning on selected points or held-out outputs.
    protocol=dict(source_sha256=file_hash(__file__),points_sha256=file_hash(OUT/'sparse_original_review/points_frozen.json'),
        training_books=books,never_use_validation_books=sorted(heldout),number_points=len(points),
        classifiers=dict(logistic='StandardScaler + LogisticRegression C1 class_weight balanced max_iter1000',
            forest='256 trees, max_depth5, min_samples_leaf2, balanced class weights, seed524'),
        threshold=.5,annotation_status='Codex tentative sparse original review; not expert GT',
        selection='Predefined hyperparameters; ordinary and spatial ablations',
        book_holdout='Leave-one-of-three known development books out; exploratory, not external validation')
    (dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    outputs=[]
    for spatial in [False,True]:
        xs=[];ys=[];book_labels=[];point_records=[]
        for page in sorted({p['page'] for p in points}):
            r=cases[page];assert file_hash(ROOT/r['original'])==r['source_sha256']
            rgb=np.asarray(Image.open(ROOT/r['original']).convert('RGB'));maps,names=feature_maps(rgb,spatial)
            for point in [p for p in points if p['page']==page]:
                x,y=point['xy_original'];xs.append([m[y,x] for m in maps]);ys.append(int(point['label']=='observed_red_mark'))
                book_labels.append(point['book']);point_records.append(point)
        x=np.asarray(xs,np.float32);y=np.asarray(ys);b=np.asarray(book_labels)
        for classifier in ['logistic','forest']:
            def new_model():
                if classifier=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=1000,random_state=524))
                return RandomForestClassifier(n_estimators=256,max_depth=5,min_samples_leaf=2,class_weight='balanced',random_state=524,n_jobs=4)
            name=('spatial_' if spatial else 'color_')+classifier
            cv=[]
            for book in books:
                model=new_model();train=b!=book;test=~train;model.fit(x[train],y[train]);pred=model.predict_proba(x[test])[:,1]>=.5
                truth=y[test].astype(bool)
                cv.append(dict(book=book,train_points=int(train.sum()),test_points=int(test.sum()),
                    mark_selected=int((pred&truth).sum()),mark_points=int(truth.sum()),
                    nonmark_selected=int((pred&~truth).sum()),nonmark_points=int((~truth).sum())))
            model=new_model();model.fit(x,y)
            path=dest/(name+'.pkl');path.write_bytes(pickle.dumps(model,protocol=4))
            record=dict(name=name,spatial=spatial,classifier=classifier,features=names,training_points=len(points),
                development_book_holdout=cv,model_sha256=file_hash(path),model_path=str(path),training_on_tentative_labels=True)
            if classifier=='forest':record['feature_importances']=dict(zip(names,model.feature_importances_.tolist()))
            outputs.append(record);print(name,cv,flush=True)
    (dest/'models.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2))
    print('Sparse exploratory classifiers frozen:',len(outputs),flush=True)

if __name__=='__main__':main()
