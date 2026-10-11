"""Direct-neighbor audit of diagnostic reference bias, no restoration changes."""
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import diagnose_paper_boundary_reference_bias as prior
from build_npj_v6_comparison import dump,read,sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/paper_boundary_reference_bias_review_20261001'


def check_summary(values,note):
    assert note['samples']==len(values)
    if not len(values):return
    for name,value in [('mean',np.mean(values,axis=0)),('median',np.median(values,axis=0)),('q10',np.quantile(values,.1,axis=0)),('q90',np.quantile(values,.9,axis=0))]:
        assert np.allclose(value,note[name],atol=1e-6,rtol=0)


def adjacency(component):
    count=np.zeros(component.shape,np.uint16)
    count[1:]+=component[:-1];count[:-1]+=component[1:];count[:,1:]+=component[:,:-1];count[:,:-1]+=component[:,1:]
    return count


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    data=json.loads((prior.OUT/'diagnostics.json').read_text())
    originals={r['page']['page']:r['page']['original'] for r in prior.previous.texture.history.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    originals.update({r['page']:r['original'] for r in json.loads((prior.fresh.INPUT/'baseline_review.json').read_text())['cases']})
    estimate=prior.previous.diagnostic.references.prior.mask.pale.prior.anchor.prior.mask.prior.MARGIN.first.paper_estimate
    checked=[]
    for key,path in originals.items():
        original=read(ROOT/path);lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        median=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB);residual=lab-estimate(median)
        raw=(residual[:,:,0]>=-4)&(residual[:,:,0]<=6)&(lab[:,:,0]>=50)&(np.abs(residual[:,:,1])<=2)&(np.abs(residual[:,:,2])<=4)
        for row in [r for r in data['regions'] if r['page']==key]:
            rid=row['id'];local=ROOT/row['local'];cx,cy,cw,ch=row['context_xywh'];x,y,w,h=row['xywh']
            cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
            known=read(local/'reference_context.png',True);app=read(local/row['applied_file'],True);bad=known&~raw[cs]
            count=adjacency(app);boundary=known&(count>0);d=row['reference_diagnostics']
            assert d['applied_ROI']==int(app[s].sum()) and d['known_context']==int(known.sum())
            assert d['raw_gate_failed_context']==int(bad.sum()) and d['boundary_reference']==int(boundary.sum()) and d['raw_gate_failed_boundary']==int((boundary&bad).sum())
            n,labels,stats,_=cv2.connectedComponentsWithStats(app.astype(np.uint8),connectivity=4)
            assert [c['id'] for c in d['components']]==[k for k in range(1,n) if np.any(labels[s]==k)]
            for c in d['components']:
                k=c['id'];component=labels==k;neighbor=adjacency(component);border=known&(neighbor>0)
                near=cv2.distanceTransform((~component).astype(np.uint8),cv2.DIST_L2,5)<=64;ring=known&near&~border
                assert c['xywh']==stats[k,:4].tolist() and c['area']==int(stats[k,4])
                assert c['adjacent_reference_pixels']==int(border.sum()) and c['adjacent_reference_pairs']==int(neighbor[border].sum())
                assert c['raw_gate_failed_boundary']==int((border&bad).sum()) and c['raw_gate_failed_ring']==int((ring&bad).sum())
                for name,values in [('boundary_raw_Lab',lab[cs][border]),('ring_raw_Lab',lab[cs][ring]),('boundary_paper_residual',residual[cs][border]),('ring_paper_residual',residual[cs][ring])]:check_summary(values,c[name])
                r=residual[cs];l=lab[cs][:,:,0]
                assert c['failures']==dict(too_dark=int((border&(r[:,:,0]<-4)).sum()),too_bright=int((border&(r[:,:,0]>6)).sum()),
                    L_below50=int((border&(l<50)).sum()),a=int((border&(np.abs(r[:,:,1])>2)).sum()),b=int((border&(np.abs(r[:,:,2])>4)).sum()))
            checked.append(dict(id=rid,applied=d['applied_ROI'],failed_boundary=d['raw_gate_failed_boundary']))
    assert len(checked)==90
    (OUT/'comparisons').mkdir(exist_ok=True)
    for p in (prior.OUT/'comparisons').glob('*.png'):shutil.copyfile(p,OUT/'comparisons'/p.name)
    focus=['unseen_thin_crack','extra_top_branch','extra_second_branch','hole_edge_new','new_margin_hole','new_brown_stamp','extra_isolated_marks','directional_gray_fold']
    for i,rid in enumerate(focus,1):
        stack([OUT/'comparisons'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 境界の原紙面参照',
            ['生画素と中央値で同じ紙面資格条件が一致しない点を赤で表示。','これは資格の整合性・周辺色の診断。物理的な墨やシミのラベルではない。'])
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,records=checked,
        independent_four_neighbor_boundary_count=True,observed_residuals_and_quantiles_reproduced=True,mask_reference_output_changes=0,pixel_GT=False))
    notes=ROOT/'tmp_work/paper_boundary_reference_bias_review_notes_20261001.md';(OUT/'report.md').write_text(notes.read_text())
    pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==38
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==len(pngs)
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    deps={str(notes.relative_to(ROOT)):sha(notes)}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=len(pngs),dependencies=deps,
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered reference-bias diagnostic:38 PNGs /90 records',flush=True)


if __name__=='__main__':main()
