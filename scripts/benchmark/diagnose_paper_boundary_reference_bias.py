"""Observe frozen donor boundaries and median/raw qualification mismatch."""
import gc
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import probe_directional_paper_abstention as previous
import restore_unseen_directional_margin_paper as fresh
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=previous.ROOT
OUT=ROOT/'tmp_work/paper_boundary_reference_bias_diagnostic_20261001'


def summarize(values):
    if not len(values):return dict(samples=0)
    return dict(samples=len(values),mean=values.mean(axis=0).tolist(),median=np.median(values,axis=0).tolist(),
        q10=np.quantile(values,.1,axis=0).tolist(),q90=np.quantile(values,.9,axis=0).tolist())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [previous.OUT,fresh.OUT]:check_bundle(folder)
    old=json.loads((previous.OUT/'diagnostics.json').read_text())['regions']
    rows=[{**r,'local':str((previous.OUT/'native'/r['id']).relative_to(ROOT)),
           'applied_file':'effective_applied_context.png','output_file':'output.png'} for r in old]
    rows.extend({**r,'local':str((fresh.OUT/'native'/r['id']/'candidate').relative_to(ROOT)),
                 'applied_file':'applied_context.png','output_file':'output.png'} for r in json.loads((fresh.OUT/'diagnostics.json').read_text()))
    originals={r['page']['page']:r['page']['original'] for r in previous.texture.history.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    originals.update({r['page']:r['original'] for r in json.loads((fresh.INPUT/'baseline_review.json').read_text())['cases']})
    assert len(rows)==90 and len(originals)==29
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str(Path(previous.diagnostic.references.prior.mask.pale.prior.anchor.prior.mask.prior.MARGIN.first.__file__).relative_to(ROOT)):sha(Path(previous.diagnostic.references.prior.mask.pale.prior.anchor.prior.mask.prior.MARGIN.first.__file__))}
    for folder in [previous.OUT,fresh.OUT]:
        for n in ['manifest.json','complete.json']:deps[str((folder/n).relative_to(ROOT))]=sha(folder/n)
    for path in originals.values():deps[path]=sha(ROOT/path)
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=deps,pages=29,ROIs=90,
        mask_output_reference_changes=0,hold_excluded=True,pixel_GT=False,
        comparison='adjacent known boundary vs other known paper within64px, original Lab minus median-derived paper estimate',
        raw_test='same L50/drop[-6,4]/abs_da2/abs_db4 applied to raw Lab instead of median Lab; diagnostic only')
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');results=[];pages=[]
    paper_estimate=previous.diagnostic.references.prior.mask.pale.prior.anchor.prior.mask.prior.MARGIN.first.paper_estimate
    control={'new_brown_stamp','extra_header_strokes','fresh_red_seal','directional_fibrous_paper','directional_attached_seal','directional_gray_fold'}
    for index,(key,path) in enumerate(originals.items(),1):
        original=read(ROOT/path);lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        median=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB);estimate=paper_estimate(median)
        residual=lab-estimate;drop=-residual[:,:,0]
        raw_ok=(drop<=4)&(drop>=-6)&(lab[:,:,0]>=50)&(np.abs(residual[:,:,1])<=2)&(np.abs(residual[:,:,2])<=4)
        for row in [r for r in rows if r['page']==key]:
            rid=row['id'];local=ROOT/row['local'];x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh']
            cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
            known=read(local/'reference_context.png',True);app=read(local/row['applied_file'],True)
            count=cv2.filter2D(app.astype(np.uint8),cv2.CV_16U,np.array([[0,1,0],[1,0,1],[0,1,0]],np.uint8),borderType=cv2.BORDER_CONSTANT)
            boundary=known&(count>0);violates=known&~raw_ok[cs]
            n,labels,stats,_=cv2.connectedComponentsWithStats(app.astype(np.uint8),connectivity=4);components=[]
            for k in range(1,n):
                if not np.any(labels[s]==k):continue
                component=labels==k
                adjacency=cv2.filter2D(component.astype(np.uint8),cv2.CV_16U,np.array([[0,1,0],[1,0,1],[0,1,0]],np.uint8),borderType=cv2.BORDER_CONSTANT)
                border=known&(adjacency>0);near=cv2.distanceTransform((~component).astype(np.uint8),cv2.DIST_L2,5)<=64
                ring=known&near&~border
                rlab=residual[cs];blab=lab[cs]
                components.append(dict(id=k,xywh=stats[k,:4].tolist(),area=int(stats[k,4]),roi_applied=int(component[s].sum()),
                    adjacent_reference_pixels=int(border.sum()),adjacent_reference_pairs=int(adjacency[border].sum()),
                    boundary_raw_Lab=summarize(blab[border]),ring_raw_Lab=summarize(blab[ring]),
                    boundary_paper_residual=summarize(rlab[border]),ring_paper_residual=summarize(rlab[ring]),
                    raw_gate_failed_boundary=int((border&violates).sum()),raw_gate_failed_ring=int((ring&violates).sum()),
                    failures=dict(too_dark=int((border&(drop[cs]>4)).sum()),too_bright=int((border&(drop[cs]<-6)).sum()),
                        L_below50=int((border&(blab[:,:,0]<50)).sum()),a=int((border&(np.abs(rlab[:,:,1])>2)).sum()),
                        b=int((border&(np.abs(rlab[:,:,2])>4)).sum()))))
            record={**row,'reference_diagnostics':dict(applied_ROI=int(app[s].sum()),known_context=int(known.sum()),
                raw_gate_failed_context=int(violates.sum()),boundary_reference=int(boundary.sum()),
                raw_gate_failed_boundary=int((boundary&violates).sum()),components=components)}
            results.append(record)
            if app[s].any() or rid in control:
                crop=original[y:y+h,x:x+w];base=read(local/row['output_file']);overlay=crop.copy()
                overlay[known[s]]=(90,200,90);overlay[boundary[s]]=(255,140,30);overlay[(boundary&violates)[s]]=(30,30,230)
                figure([crop,base,overlay],['原画像','現行補修・方向保留あり','緑＝参照、青＝境界、赤＝生Lab資格外'],OUT/'comparisons'/(rid+'.png'),row['title']+' · 観測紙面参照の診断',
                    '赤は中央値で資格を得たが元の生画素では同じLab条件を外れる観測点。墨やシミの正解ではない。',
                    'マスク・参照・補修は変更0。境界と周囲64pxの紙色推定残差を保存。')
                for name,item in dict(reference_context=known,applied_context=app,boundary_context=boundary,raw_gate_failed_context=violates).items():write(OUT/'native'/rid/(name+'.png'),item)
        pages.append(dict(page=key,original_sha256=sha(ROOT/path)))
        print('Paper reference bias diagnostic:',index,'/29',key,flush=True)
        del original,lab,median,estimate,residual,raw_ok;gc.collect()
    for path,h in deps.items():assert sha(ROOT/path)==h
    dump(OUT/'diagnostics.json',dict(pages=pages,regions=results))
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,mask_reference_and_output_changes=0,
        observed_original_values_only=True,diagnostic_not_raw_gate_adoption=True,pixel_GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:fixed-reference boundary and raw/median qualification diagnostic',flush=True)


if __name__=='__main__':main()
