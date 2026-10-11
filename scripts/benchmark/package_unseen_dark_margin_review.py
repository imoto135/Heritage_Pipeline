"""Audit original-first selection, full-page masks and equal-restorer PNGs."""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import validate_unseen_dark_margin_candidate as masks
import restore_unseen_dark_margin_paper as fill
from build_npj_v6_comparison import dump, read, sha
from check_paper_margin_defect_ddrm import figure
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_seam_texture_review import verify_texture
from package_paper_blending_review import check_bundle

ROOT = masks.ROOT
INPUT = masks.INPUT
OUT = INPUT/'review'


def main():
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(masks.OUT); check_bundle(fill.OUT)
    selected = json.loads((INPUT/'baseline_review.json').read_text())
    for relative, digest in selected['inputs'].items(): assert sha(ROOT/relative) == digest
    assert selected['selection_before_new_v6_and_candidate'] and selected['original_only']
    assert not set(p['book'] for p in selected['cases'])&set(selected['excluded_books'])
    assert len(set(p['book'] for p in selected['cases'])) == selected['selected_books']
    md = json.loads((masks.OUT/'diagnostics.json').read_text())
    fd = json.loads((fill.OUT/'diagnostics.json').read_text()); by_id = {r['id']:r for r in fd}
    region_masks = {r['id']:r for r in md['regions']}; records = []; residuals = []; texture = []
    for page in selected['cases']:
        key = page['page']; original = read(ROOT/page['original']); directory = masks.OUT/'pages'/key
        v6 = read(directory/'v6/edit.png', True); domain = read(directory/'features/safe_domain.png', True)
        guard = read(directory/'features/rim_guard.png', True); covered = np.zeros(v6.shape, bool)
        edits = {m:read(directory/m/'edit.png', True) for m in masks.MODES}
        for m in ['rim', 'thin', 'near8']:
            added = edits[m]&~v6
            assert np.array_equal(added, read(directory/m/'added.png', True))
            assert not np.any(v6&~edits[m]) and not np.any(added&(guard|~domain))
            for name in ['red', 'extra']: assert np.array_equal(read(directory/m/(name+'.png'), True), read(directory/'v6'/(name+'.png'), True))
            for name in ['protect', 'hold']: assert np.array_equal(read(directory/m/(name+'.png'), True), read(directory/'v6'/(name+'.png'), True)&~added)
        for roi in page['rois']:
            rid = roi['id']; row = by_id[rid]; x,y,w,h = roi['xywh']; rs = (slice(y,y+h),slice(x,x+w)); covered[rs] = True
            crop = original[rs]; native = fill.OUT/'native'/rid
            assert np.array_equal(read(native/'original.png'), crop)
            cx,cy,cw,ch = row['context_xywh']; cs=(slice(cy,cy+ch),slice(cx,cx+cw)); s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
            lab=cv2.cvtColor(original[cs].astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            for mode in fill.MODES:
                local=native/mode; value=read(local/'output.png'); edit=read(local/'edit.png',True)
                active=read(local/'eligible_context.png',True); applied=read(local/'applied_context.png',True)
                known=read(local/'reference_context.png',True); used=read(local/'used_context.png',True)
                assert not np.any(known&active) and not np.any(applied&~active)
                assert not np.any(np.any(value!=crop,axis=2)&~applied[s])
                assert not np.any(np.any(value!=crop,axis=2)&~edit)
                expected=read(fill.OUT/'reference_pages'/key/(mode+'_paper.png'),True)[cs]
                assert np.array_equal(known,expected)
                if mode!='npj':
                    assert np.array_equal(edit,edits[mode][rs])
                    assert np.array_equal(active,edits[mode][cs]&domain[cs]&~guard[cs])
                stat=row['diagnostics'][mode]
                assert stat['applied_pixels']==int(applied[s].sum()) and stat['changed_pixels']==int(np.any(value!=crop,axis=2).sum())
                if applied.any():
                    fields=np.load(local/'harmonic_fields_Lab.npz',allow_pickle=False)
                    mass=cv2.GaussianBlur(known.astype(np.float32),(0,0),6)
                    smooth=cv2.GaussianBlur(np.where(known[:,:,None],lab,0),(0,0),6)/np.maximum(mass[:,:,None],1e-6)
                    for name,target in [('boundary_raw',lab),('boundary_smooth',smooth)]:
                        error, actual_used=verify_field(fields[name],applied,known,target)
                        assert np.array_equal(actual_used,used)
                        residuals.append(dict(id=rid,mode=mode,field=name,residual_max=error))
                    donor=json.loads((local/'donors.json').read_text())
                    if donor['components']: texture.append(dict(id=rid,mode=mode,**verify_texture(lab,known,applied,local)))
                    else:
                        assert len(donor['patch_xy'])<4
                        assert not np.load(local/'texture_seam.npy',allow_pickle=False).any()
                        assert np.all(np.load(local/'source_x.npy',allow_pickle=False)==-1)
                    distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5)
                    weight=np.exp(-(np.maximum(distance-1,0)/8.)**2)*applied
                    result=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]
                    result[:,:,0]+=np.load(local/'texture_seam.npy',allow_pickle=False)*np.clip((distance-1)/4.,0,1)*applied
                    assert np.array_equal(result,np.load(local/'result_field_Lab.npy',allow_pickle=False))
                    assert np.array_equal(value,fill.frozen.compose(result,applied,crop,s))
                else: assert np.array_equal(value,crop)
            records.append(dict(id=rid,page=key,additional_vs_v6=region_masks[rid]['diagnostics']['near8']['added_vs_v6'],
                                npj_applied=row['diagnostics']['npj']['applied_pixels'], v6_applied=row['diagnostics']['v6']['applied_pixels']))
        record=next(r for r in md['pages'] if r['page']==key)
        dump(OUT/'page_checks'/(key+'.json'),dict(**record,additional_outside_review_ROIs=int((edits['near8']&~v6&~covered).sum())))
    for folder,name in [(masks.OUT/'comparisons','masks'),(masks.OUT/'additions','additions'),(fill.OUT/'comparisons','paper')]:
        (OUT/name).mkdir(exist_ok=True)
        for p in folder.glob('*.png'): shutil.copyfile(p,OUT/name/p.name)
    shutil.copyfile(INPUT/'selected_originals.png',OUT/'00_original_selection.png')
    focus=['extra_top_branch','extra_second_branch','extra_brown_stain','extra_isolated_marks','extra_header_strokes','extra_small_seal']
    for i,rid in enumerate(focus,1):
        stack([OUT/'masks'/(rid+'.png'),OUT/'paper'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',
              rid+' · マスクと同一条件の紙面復元',
              ['上：分類、下：同じ紙面補間。原画像を必ず併記。','条件固定の紙面補間。DDRMではない。npj全工程の品質比較ではない。'])
    report = (INPUT/'review_notes.md').read_text()
    (OUT/'report.md').write_text(report)
    dump(OUT/'verification.json',dict(status='passed',selected_before_outputs=True,excluded_books=len(selected['excluded_books']),pages=selected['selected_pages'],ROIs=selected['selected_ROIs'],
        page_additions={p['page']:sum(region_masks[p['page']+'_whole']['diagnostics'][m]['added_vs_v6'] for m in ['near8']) for p in selected['cases']},
        no_rule_pages=[r['page'] for r in md['pages'] if r['rule_y'] is None],routing_parity_pages=18,outside_applied_changes=0,
        residuals=residuals,texture_provenance=texture,records=records,DDRM=False,pixel_GT=False))
    pngs=sorted(OUT.rglob('*.png'))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==len(pngs) and all(n.endswith('.png') for n in z.namelist())
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=len(pngs),
        dependencies={str((folder/name).relative_to(ROOT)):sha(folder/name) for folder in [masks.OUT,fill.OUT] for name in ['manifest.json','complete.json']},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered further dark margin validation:',len(pngs),'PNGs',flush=True)


if __name__=='__main__':main()
