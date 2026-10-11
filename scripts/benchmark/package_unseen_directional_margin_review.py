"""Audit fixed new-book controls and package original-first comparison PNGs."""
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import validate_unseen_directional_margin_candidate as masks
import restore_unseen_directional_margin_paper as restoration
from build_npj_v6_comparison import dump,read,sha
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_seam_texture_review import verify_texture
from package_paper_blending_review import check_bundle

ROOT=masks.ROOT
INPUT=masks.INPUT
OUT=INPUT/'review'


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(masks.OUT);check_bundle(restoration.OUT)
    selected=json.loads((INPUT/'baseline_review.json').read_text());records=[];residuals=[];textures=[];pages=[]
    assert selected['original_only'] and selected['selection_before_new_v6_and_candidate']
    assert len({p['book'] for p in selected['cases']})==4
    assert not {p['book'] for p in selected['cases']}&set(selected['excluded_books'])
    for n,h in selected['inputs'].items():assert sha(ROOT/n)==h
    md=json.loads((masks.OUT/'diagnostics.json').read_text());rows={r['id']:r for r in json.loads((restoration.OUT/'diagnostics.json').read_text())}
    for pg in selected['cases']:
        key=pg['page'];original=read(ROOT/pg['original']);folder=masks.OUT/'pages'/key
        edit=read(folder/'edit.png',True);domain=read(folder/'domain.png',True);guard=read(folder/'guard.png',True);hold=read(folder/'paper_hold.png',True)
        v6=read(INPUT/'v6_masks'/key/'edit.png',True);core=read(folder/'core.png',True)|read(folder/'strong_core.png',True)
        assert np.array_equal(edit,v6) and not core.any() and not hold.any()
        eligible=edit&domain&~guard
        n,labels,stats,_=cv2.connectedComponentsWithStats(eligible.astype(np.uint8),connectivity=4)
        assert np.array_equal(labels,np.load(folder/'labels.npz',allow_pickle=False)['labels'])
        overlaps=np.bincount(labels.ravel(),weights=core.ravel(),minlength=n)
        expected=[k for k in range(1,n) if stats[k,4]>=64 and overlaps[k]>=32];assert expected==[]
        assert json.loads((folder/'direction.json').read_text())==[]
        assert np.array_equal(read(folder/'effective_edit.png',True),edit)
        page_stat=next(r for r in md['pages'] if r['page']==key);assert page_stat['core']==0 and page_stat['added_vs_v6']==0 and page_stat['held_pixels']==0
        pages.append(page_stat)
        for roi in pg['rois']:
            rid=roi['id'];row=rows[rid];x,y,w,h=roi['xywh'];crop=original[y:y+h,x:x+w]
            cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
            lab=cv2.cvtColor(original[cs].astype(np.float32)/255,cv2.COLOR_BGR2LAB);native=restoration.OUT/'native'/rid
            for mode in restoration.MODES:
                local=native/mode;known=read(local/'reference_context.png',True);active=read(local/'eligible_context.png',True)
                app=read(local/'applied_context.png',True);used=read(local/'used_context.png',True);value=read(local/'output.png')
                assert np.array_equal(read(local/'original.png'),crop)
                assert np.array_equal(known,read(restoration.OUT/'reference_pages'/key/(mode+'_known.png'),True)[cs])
                assert np.array_equal(active,read(restoration.OUT/'reference_pages'/key/(mode+'_eligible.png'),True)[cs])
                assert not np.any(app&~active) and not np.any(known&active)
                assert not np.any(np.any(value!=crop,axis=2)&~app[s])
                assert row['diagnostics'][mode]['applied']==int(app[s].sum())
                assert row['diagnostics'][mode]['changed_vs_original']==int(np.any(value!=crop,axis=2).sum())
                if app.any():
                    fields=np.load(local/'harmonic_fields_Lab.npz',allow_pickle=False)
                    mass=cv2.GaussianBlur(known.astype(np.float32),(0,0),6)
                    smooth=cv2.GaussianBlur(np.where(known[:,:,None],lab,0),(0,0),6)/np.maximum(mass[:,:,None],1e-6)
                    for name,target in [('boundary_raw',lab),('boundary_smooth',smooth)]:
                        error,actual=verify_field(fields[name],app,known,target);assert np.array_equal(actual,used)
                        residuals.append(dict(id=rid,mode=mode,field=name,residual=error))
                    donor=json.loads((local/'donors.json').read_text())
                    if donor['components']:textures.append(dict(id=rid,mode=mode,**verify_texture(lab,known,app,local)))
                    else:
                        assert len(donor['patch_xy'])<4 and not np.load(local/'texture_seam.npy',allow_pickle=False).any()
                        assert np.all(np.load(local/'source_x.npy',allow_pickle=False)==-1)
                    dist=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5);weight=np.exp(-(np.maximum(dist-1,0)/8.)**2)*app
                    field=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]
                    field[:,:,0]+=np.load(local/'texture_seam.npy',allow_pickle=False)*np.clip((dist-1)/4.,0,1)*app
                    assert np.array_equal(field,np.load(local/'result_field_Lab.npy',allow_pickle=False))
                    assert np.array_equal(value,restoration.fill.frozen.compose(field,app,crop,s))
                else:assert np.array_equal(value,crop)
            current=read(native/'candidate/output.png');assert np.array_equal(current,crop)
            assert np.array_equal(read(native/'v6/output.png'),crop) and np.array_equal(read(native/'abstention/output.png'),crop)
            assert not read(native/'abstention/reverted_context.png',True).any()
            assert np.array_equal(read(native/'abstention/reference_context.png',True),read(native/'candidate/reference_context.png',True))
            records.append(dict(id=rid,**row['diagnostics']))
    sources=[(INPUT/'classification_comparisons','classification'),(masks.OUT/'comparisons','masks'),
             (restoration.OUT/'comparisons','paper'),(INPUT/'original_ROIs','original')]
    for source,name in sources:
        (OUT/name).mkdir(exist_ok=True)
        for p in source.glob('*.png'):shutil.copyfile(p,OUT/name/p.name)
    focus=['directional_fibrous_paper','directional_diagonal_paper','directional_gray_fold','directional_black_spot']
    for i,rid in enumerate(focus,1):
        stack([OUT/'classification'/(rid+'.png'),OUT/'masks'/(rid+'.png'),OUT/'paper'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 固定条件の別冊対照',
              ['上：npj／v6分類、中：余白追加／保留、下：同じ紙面補間。','原画像を各段に併記。新候補は全例追加0で、有効性の陽性検証ではない。'])
    dump(OUT/'verification.json',dict(status='passed',pages=4,ROIs=13,original_selection_first=True,
        page_checks=pages,records=records,independent_no_core_no_direction_target=True,
        residuals=residuals,texture_provenance=textures,candidate_and_v6_outputs_exact_original=True,
        novel_hold_effect=False,no_rule_pages=[r['page'] for r in pages if r['rule_y'] is None],new_DDRM=False,pixel_GT=False))
    notes=INPUT/'review_notes.md';(OUT/'report.md').write_text(notes.read_text())
    pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==64
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==64
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    deps={str(notes.relative_to(ROOT)):sha(notes),str((INPUT/'baseline_review.json').relative_to(ROOT)):sha(INPUT/'baseline_review.json')}
    for folder in [masks.OUT,restoration.OUT]:
        for n in ['manifest.json','complete.json']:deps[str((folder/n).relative_to(ROOT))]=sha(folder/n)
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=64,dependencies=deps,
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered4 new-books control validation:64 PNGs',flush=True)


if __name__=='__main__':main()
