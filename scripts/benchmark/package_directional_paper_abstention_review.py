"""Audit observed gradients and original-pixel abstention, package paired PNGs."""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import probe_directional_paper_abstention as prior
from build_npj_v6_comparison import dump,read,sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/directional_paper_abstention_review_20261001'


def audit_direction(light,folder):
    known=read(folder/'known.png',True);eligible=read(folder/'eligible.png',True);core=read(folder/'core.png',True)
    domain=read(folder/'domain.png',True);guard=read(folder/'guard.png',True);edit=read(folder/'edit.png',True)
    retained=read(folder/'retained.png',True);assert np.array_equal(retained,domain&~guard&~edit)
    assert np.array_equal(eligible,edit&domain&~guard)
    labels=np.load(folder/'labels.npz',allow_pickle=False)['labels'];n,actual,stats,_=cv2.connectedComponentsWithStats(eligible.astype(np.uint8),connectivity=4)
    assert np.array_equal(labels,actual);overlap=np.bincount(labels.ravel(),weights=core.ravel(),minlength=n)
    notes=json.loads((folder/'components.json').read_text());expected_ids=[k for k in range(1,n) if stats[k,4]>=64 and overlap[k]>=32]
    assert [note['id'] for note in notes]==expected_ids
    gradients={};valid={};poisoned=light.copy();poisoned[~retained]=-200
    for sigma in [0,1,2]:
        values=[]
        for data in [light,poisoned]:
            sm=cv2.GaussianBlur(data,(6*sigma+1,6*sigma+1),sigma) if sigma else data
            values.append((np.clip(cv2.Sobel(sm,cv2.CV_32F,1,0,ksize=3,scale=1/8),-5,5),
                           np.clip(cv2.Sobel(sm,cv2.CV_32F,0,1,ksize=3,scale=1/8),-5,5)))
        gradients[sigma]=values[0]
        for name,reference in [('known',known&retained),('retained',retained)]:
            mask=cv2.erode(reference.astype(np.uint8),np.ones((6*sigma+3,6*sigma+3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            valid[name,sigma]=mask
            for a,b in zip(values[0],values[1]):assert np.array_equal(a[mask],b[mask])
    matched=np.zeros_like(eligible);checks=[];H,W=eligible.shape
    for note in notes:
        k=note['id'];x,y,w,h,area=map(int,stats[k]);assert note['xywh']==[x,y,w,h] and note['area']==area and note['core_overlap']==int(overlap[k])
        x0,y0=max(0,x-64),max(0,y-64);x1,y1=min(W,x+w+64),min(H,y+h+64);local=labels[y0:y1,x0:x1]==k
        yy,xx=np.nonzero(local);coordinates=np.stack([xx,yy],axis=1).astype(np.float64);center=coordinates.mean(axis=0);centered=coordinates-center
        cov=centered.T@centered/len(xx);shape=note['shape'];assert np.allclose(cov,shape['covariance'],atol=1e-6,rtol=0)
        vals,vec=np.linalg.eigh(cov);ratio=float((vals[1]-vals[0])/max(vals.sum(),1e-6));angle=float(np.degrees(np.arctan2(vec[1,1],vec[0,1]))%180)
        assert abs(ratio-shape['anisotropy'])<1e-6 and abs(angle-shape['orientation_degrees'])<1e-6
        assert np.allclose(center+[x0,y0],shape['center_xy'],atol=1e-6,rtol=0)
        near=cv2.distanceTransform((~local).astype(np.uint8),cv2.DIST_L2,5)<=64
        for scope in ['known','retained']:
            for sigma in [0,1,2]:
                support=valid[scope,sigma][y0:y1,x0:x1]&near;metric=note['direction'][scope][str(sigma)]
                assert int(support.sum())==metric['samples'] and bool(support.sum()>=256)==metric['available']
                if not metric['available']:continue
                gx,gy=gradients[sigma];a=gx[y0:y1,x0:x1][support].astype(np.float64);b=gy[y0:y1,x0:x1][support].astype(np.float64)
                observed=np.array([[a@a,a@b],[a@b,b@b]])/len(a);tensor=np.asarray(metric['tensor'])
                assert np.allclose(observed,tensor,atol=1e-5,rtol=0)
                val,vector=np.linalg.eigh(tensor);energy=float(val.sum());anisotropy=float((val[1]-val[0])/max(energy,1e-6))
                orientation=float(np.degrees(np.arctan2(vector[1,0],vector[0,0]))%180)
                alignment=min(abs(orientation-angle),180-abs(orientation-angle))
                assert np.allclose(val,metric['eigenvalues'],atol=1e-6,rtol=0)
                assert abs(energy-metric['gradient_energy'])<1e-6 and abs(anisotropy-metric['anisotropy'])<1e-6
                assert abs(orientation-metric['orientation_degrees'])<1e-6 and abs(alignment-metric['alignment_degrees'])<1e-6
        d=note['direction']['retained']['1'];size=area<=.01*H*W and max(w,h)<=.2*min(H,W)
        expected=bool(d['available'] and d['anisotropy']>=.55 and d['gradient_energy']>=.05 and ratio>=.8 and d['alignment_degrees']<=15 and size)
        assert size==note['size_ok'] and expected==note['provisional_matched']
        if expected:matched[y0:y1,x0:x1]|=local
        checks.append(dict(component=k,matched=expected,shape_anisotropy=ratio,retained_sigma1=d))
    assert np.array_equal(matched,read(folder/'provisional_matches.png',True))
    return matched,checks


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [prior.OUT,prior.diagnostic.OUT,prior.texture.OUT]:check_bundle(folder)
    data=json.loads((prior.OUT/'diagnostics.json').read_text());assert len(data['regions'])==77
    unique_pages={}
    for row in data['pages']:
        if row['page'] in unique_pages:assert unique_pages[row['page']]==row
        unique_pages[row['page']]=row
    assert len(unique_pages)==25
    descriptors={r['page']['page']:r for r in prior.texture.history.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    page_checks=[];region_checks=[];unchanged=[]
    for pg in unique_pages.values():
        key=pg['page'];original=read(ROOT/descriptors[key]['page']['original']);light=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        diagnostic_folder=prior.diagnostic.OUT/'pages'/key;hold,checks=audit_direction(light,diagnostic_folder)
        frozen=prior.texture.history.previous.mask.OUT/'pages'/key
        edit=read(frozen/'short_upper/edit.png',True);domain=read(frozen/'domain.png',True);guard=read(frozen/'guard.png',True)
        assert np.array_equal(read(diagnostic_folder/'edit.png',True),edit)
        assert np.array_equal(read(diagnostic_folder/'domain.png',True),domain) and np.array_equal(read(diagnostic_folder/'guard.png',True),guard)
        known,_=prior.diagnostic.references.full_reference(descriptors[key],edit)
        assert np.array_equal(known,read(diagnostic_folder/'known.png',True))
        folder=prior.OUT/'pages'/key;assert int(hold.sum())==pg['hold_pixels']
        for name,value in dict(base_edit=edit,paper_hold=hold,effective_edit=edit&~hold,effective_eligible=edit&domain&~guard&~hold,known=known).items():
            assert np.array_equal(read(folder/(name+'.png'),True),value)
        assert not np.any(known&hold);page_checks.append(dict(page=key,held_pixels=int(hold.sum()),components=checks))
        for row in [r for r in data['regions'] if r['page']==key]:
            rid=row['id'];x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
            native=prior.OUT/'native'/rid;source=ROOT/row['baseline_local'];base=read(source/'output.png');crop=original[y:y+h,x:x+w]
            app=read(source/'retained_applied_context.png',True);reference=read(source/'reference_context.png',True);value=read(native/'output.png')
            local=hold[cs];reverted=app&local;effective=app&~local;expected=base.copy();expected[reverted[s]]=crop[reverted[s]]
            assert np.array_equal(value,expected) and np.array_equal(crop,read(native/'original.png')) and np.array_equal(base,read(native/'baseline.png'))
            assert np.array_equal(read(native/'adaptive_texture.png'),read(prior.texture.OUT/'native'/rid/'adaptive/output.png'))
            for name,item in dict(paper_hold_context=local,reverted_context=reverted,baseline_applied_context=app,effective_applied_context=effective,reference_context=reference,effective_edit=edit[y:y+h,x:x+w]&~hold[y:y+h,x:x+w]).items():
                assert np.array_equal(read(native/(name+'.png'),True),item)
            assert not np.any(reference&local) and not np.any(effective&local)
            assert np.array_equal(value[local[s]],crop[local[s]]) and np.array_equal(value[~local[s]],base[~local[s]])
            stat=row['abstention_diagnostics'];assert stat['hold_candidate_pixels']==int(local[s].sum()) and stat['paper_reverted_pixels']==int(reverted[s].sum())
            assert stat['remaining_paper_applied']==int(effective[s].sum()) and stat['changed_vs_base']==int(np.any(value!=base,axis=2).sum())
            if not stat['changed_vs_base']:unchanged.append(rid)
            region_checks.append(dict(id=rid,**stat))
    for source,name in [(prior.diagnostic.OUT/'comparisons','direction'),(prior.OUT/'comparisons','paper')]:
        (OUT/name).mkdir(exist_ok=True)
        for p in source.glob('*.png'):shutil.copyfile(p,OUT/name/p.name)
    focus=['200021712_00082_1_external_rim_core_10','unseen_fold','extra_top_branch','extra_second_branch','unseen_thin_crack','hole_edge_new','extra_header_strokes','fresh_red_seal','new_brown_stamp']
    for i,rid in enumerate(focus,1):
        stack([OUT/'direction'/(rid+'.png'),OUT/'paper'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 方向診断と補修保留',
              ['上：原画像・基準補修・形状と資格紙面／保持画素の方向。下：原画像・基準・適応模様・保留。',
               '保留は原画素へ戻すため損傷も残り得る。物理原因を断定する判定ではない。'])
    dump(OUT/'verification.json',dict(status='passed',pages=25,ROIs=77,whole_page_direction_and_geometry_reconstruction=page_checks,
        independent_finite_stencil_poison_invariance=True,regions=region_checks,unchanged_ROIs=unchanged,
        source_page_rows=len(data['pages']),unique_pages=25,repeated_source_page_rows_verified_identical=True,
        hold_exact_original=True,outside_hold_exact_baseline=True,held_original_pixels_never_used_as_paper_references=True,
        outside_effective_edit_changes=0,new_solve=False,new_DDRM=False,pixel_GT=False))
    notes=ROOT/'tmp_work/directional_paper_abstention_review_notes_20261001.md';(OUT/'report.md').write_text(notes.read_text())
    pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==163
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==len(pngs)
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=len(pngs),
        dependencies={str(notes.relative_to(ROOT)):sha(notes),**{str((folder/n).relative_to(ROOT)):sha(folder/n) for folder in [prior.OUT,prior.diagnostic.OUT,prior.texture.OUT] for n in ['manifest.json','complete.json']}},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered directional paper abstention:',len(pngs),'PNGs',flush=True)


if __name__=='__main__':main()
