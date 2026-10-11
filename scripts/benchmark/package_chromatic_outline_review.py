"""Audit original chromatic-support maps and package diagnostic PNGs."""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import diagnose_chromatic_repair_outlines as prior
from build_npj_v6_comparison import dump,read,sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/chromatic_repair_outline_review_20261001'


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    data=json.loads((prior.OUT/'diagnostics.json').read_text());assert len(data['pages'])==25 and len(data['regions'])==73
    source_rows={r['id']:r for r in json.loads((prior.prior.OUT/'diagnostics.json').read_text())}
    inputs={r['page']['page']:r['page'] for r in prior.prior.mask.pale.prior.descriptors()}
    checks=[]
    for pg in data['pages']:
        key=pg['page'];dest=prior.OUT/'pages'/key;frozen=prior.prior.mask.OUT/'pages'/key
        features=dict(np.load(dest/'original_features.npz',allow_pickle=False));lab=features['median_Lab'];paper=features['paper_estimate_Lab']
        original=read(ROOT/inputs[key]['original']);expected_lab=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        assert np.array_equal(lab,expected_lab)
        difference=lab-paper;drop=-difference[:,:,0]
        edit=read(dest/'edit.png',True);core=read(dest/'core.png',True);domain=read(dest/'domain.png',True);guard=read(dest/'guard.png',True)
        assert np.array_equal(edit,read(frozen/'short_upper/edit.png',True))
        assert np.array_equal(core,read(frozen/'core.png',True)|read(frozen/'short_upper/strong_core.png',True))
        assert np.array_equal(domain,read(frozen/'domain.png',True)) and np.array_equal(guard,read(frozen/'guard.png',True))
        warm=(difference[:,:,1]>=1)&(difference[:,:,2]>=2)&(lab[:,:,2]>=5)&(drop>=-2)
        assert np.array_equal(warm,read(dest/'warm_support.png',True));values={}
        for radius in prior.RADII:
            near=cv2.dilate(core.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
            expected=near&~edit&domain&~guard&warm;guarded=near&~edit&domain&guard&warm
            saved=read(dest/f'warm{radius}.png',True);assert np.array_equal(expected,saved)
            assert np.array_equal(guarded,read(dest/f'guarded_warm{radius}.png',True));values[radius]=saved
            note=pg['diagnostics'][str(radius)]
            assert note['unedited_warm']==int(saved.sum()) and note['guarded_warm_retained']==int(guarded.sum())
            assert note['below_previous_rim_drop2']==int((saved&(drop<2)).sum())
            if saved.any():assert np.allclose(np.percentile(difference[saved],[10,50,90],axis=0),note['original_Lab_delta_quantiles'],rtol=0,atol=1e-6)
            checks.append(dict(page=key,radius=radius,**note))
        assert not np.any(values[6]&~values[12])
        for row in [r for r in data['regions'] if r['page']==key]:
            rid=row['id'];old=source_rows[rid];assert row['xywh']==old['xywh'];x,y,w,h=row['xywh'];cx,cy,cw,ch=old['context_xywh']
            s=(slice(y,y+h),slice(x,x+w));local=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
            native=prior.prior.OUT/'native'/rid/'short_upper';base=read(native/'output.png');applied=read(native/'retained_applied_context.png',True)
            distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5);inside=(applied&(distance<=3))[local]
            assert int(inside.sum())==row['filled_inside_edge_pixels']
            for radius in prior.RADII:
                note=row['outside'][str(radius)];candidate=values[radius][s];assert note['count']==int(candidate.sum())
                if candidate.any():assert np.allclose(np.percentile(difference[s][candidate],[10,50,90],axis=0),note['original_Lab_delta_quantiles'],rtol=0,atol=1e-6)
            if inside.any():
                outlab=cv2.cvtColor(cv2.medianBlur(base,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
                for actual,note in [(difference[s],row['original_inside_Lab_delta_quantiles']),(outlab-paper[s],row['output_inside_Lab_delta_quantiles'])]:
                    assert np.allclose(np.percentile(actual[inside],[10,50,90],axis=0),note,atol=1e-6,rtol=0)
    (OUT/'comparisons').mkdir(exist_ok=True)
    for p in (prior.OUT/'comparisons').glob('*.png'):shutil.copyfile(p,OUT/'comparisons'/p.name)
    focus=['extra_top_branch','extra_second_branch','hole_edge_new','new_margin_hole','unseen_fold','extra_isolated_marks','new_brown_stamp','extra_header_strokes']
    for i,rid in enumerate(focus,1):
        stack([OUT/'comparisons'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 補修内と未編集輪郭を分ける',
              ['原画像と固定した補修結果。青＝未編集の暖色差、緑＝補間内の境界3px。',
               '診断だけでマスク・復元は変更していない。青を損傷の正解と扱わない。'])
    uncertain=[r for r in data['regions'] if r['page']=='200021712_00082_1' and '_component_' in r['id']]
    stack([OUT/'comparisons'/(r['id']+'.png') for r in uncertain],OUT/'09_unknown_old_components.png',
          '原因未確定の5曲線にも輪郭診断を同じ条件で適用',
          ['6px／12pxの色差支持を可視化。損傷の正解として数えない。'])
    dump(OUT/'verification.json',dict(status='passed',pages=25,ROIs=73,checks=checks,mask_and_output_changes=0,
        diagnostic_support_reconstructed=True,source_features_original_only=True,pixel_GT=False,new_DDRM=False))
    notes=ROOT/'tmp_work/chromatic_repair_outline_review_notes_20261001.md';(OUT/'report.md').write_text(notes.read_text())
    pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==107
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==len(pngs)
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=len(pngs),
        dependencies={str(notes.relative_to(ROOT)):sha(notes),**{str((prior.OUT/n).relative_to(ROOT)):sha(prior.OUT/n) for n in ['manifest.json','complete.json']}},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered chromatic repair outlines:',len(pngs),'PNGs',flush=True)


if __name__=='__main__':main()
