"""Fixed color and masks; independently verify adaptive-scale donor provenance."""
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import probe_adaptive_paper_texture_scale_v2 as prior
from build_npj_v6_comparison import dump,read,sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle
ROOT=prior.ROOT
OUT=ROOT/'tmp_work/adaptive_paper_texture_scale_review_20261001'

def verify_texture(lab, known, applied, native):
    """Check each saved seam pixel against a recorded donor tile, without quilting."""
    metadata = json.loads((native / 'donors.json').read_text())
    size=int(metadata['patch_size']);overlap=size//3
    seam = np.load(native / 'texture_seam.npy', allow_pickle=False)
    average = np.load(native / 'texture_average.npy', allow_pickle=False)
    sx = np.load(native / 'source_x.npy', allow_pickle=False)
    sy = np.load(native / 'source_y.npy', allow_pickle=False)
    if not metadata['patch_xy']:
        assert not applied.any()
        assert not seam.any() and not average.any()
        assert (sx == -1).all() and (sy == -1).all()
        return dict(pixels=0, average_error=0., distinct_donors=0)
    assert np.array_equal(sx >= 0, applied)
    assert np.array_equal(sy >= 0, applied)
    assert known[sy[applied], sx[applied]].all()
    assert not seam[~applied].any() and not average[~applied].any()
    light = lab[:, :, 0]
    weights = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 6)
    low = cv2.GaussianBlur(np.where(known, light, 0), (0, 0), 6) / np.maximum(weights, 1e-6)
    signal = light - low
    patches = []
    for px, py in metadata['patch_xy']:
        assert known[py:py+size, px:px+size].shape == (size, size)
        assert known[py:py+size, px:px+size].all()
        patch = signal[py:py+size, px:px+size].copy()
        patches.append(np.clip(patch - float(patch.mean()), -3, 3))
    assert float(np.abs(seam).max()) <= 3
    n, labels, stats, _ = cv2.connectedComponentsWithStats(applied.astype(np.uint8), connectivity=4)
    assert len(metadata['components']) == n - 1
    axis = np.array([min(1., (i+1)/overlap, (size-i)/overlap) for i in range(size)], np.float32)
    taper = axis[:, None] * axis[None, :]
    coverage = np.zeros_like(applied)
    reconstructed = np.zeros_like(average)
    donors_used = set()
    for component in metadata['components']:
        index = component['id']
        x, y, w, h, area = map(int, stats[index])
        assert component['xywh'] == [x, y, w, h] and component['area'] == area
        canvas_shape = (max(h, size)+size, max(w, size)+size)
        total = np.zeros(canvas_shape, np.float32)
        mass = np.zeros(canvas_shape, np.float32)
        for tile in component['tiles']:
            tx, ty, index = tile['x'], tile['y'], tile['donor']
            px, py = metadata['patch_xy'][index]
            assert tile['source_xy'] == [px, py]
            donors_used.add(index)
            patch = patches[index]
            total[ty:ty+size, tx:tx+size] += patch * taper
            mass[ty:ty+size, tx:tx+size] += taper
            tw, th = min(size, w-tx), min(size, h-ty)
            if tw <= 0 or th <= 0:
                continue
            ys, xs = slice(y+ty, y+ty+th), slice(x+tx, x+tx+tw)
            yy, xx = np.indices((th, tw))
            # A seam pixel must equal a placed, unaveraged observed donor pixel.
            matches = ((sx[ys, xs] == px+xx) & (sy[ys, xs] == py+yy)
                       & np.isclose(seam[ys, xs], patch[:th, :tw], rtol=0, atol=1e-6))
            coverage[ys, xs] |= matches & (labels[ys, xs] == component['id'])
        local = labels[y:y+h, x:x+w] == component['id']
        reconstructed[y:y+h, x:x+w][local] = (total[:h, :w] / np.maximum(mass[:h, :w], 1e-6))[local]
    assert coverage[applied].all()
    error = float(np.abs(reconstructed - average).max())
    assert error < 1e-6, error
    qualified_union=np.zeros_like(known)
    for px,py in metadata['patch_xy']:qualified_union[py:py+size,px:px+size]=True
    return dict(pixels=int(applied.sum()), average_error=error, distinct_donors=len(donors_used),
                unique_qualified_source_pixels=int(qualified_union.sum()))


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    for path,value in json.loads((prior.OUT/'final_dependencies.json').read_text()).items():assert sha(ROOT/path)==value
    rows=json.loads((prior.OUT/'diagnostics.json').read_text());assert len(rows)==77
    originals={r['page']['page']:r['page']['original'] for r in prior.history.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    current=None;checks=[];unchanged={m:[] for m in prior.MODES}
    for row in rows:
        rid=row['id'];key=row['page']
        if current!=key:original=read(ROOT/originals[key]);current=key
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        native=prior.OUT/'native'/rid;source=ROOT/row['baseline_local'];crop=original[y:y+h,x:x+w]
        lab=cv2.cvtColor(original[cy:cy+ch,cx:cx+cw].astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        known=read(native/'reference_context.png',True);retained=read(native/'retained_applied_context.png',True)
        applied=read(native/'texture_applied_context.png',True);edit=read(native/'edit.png',True);base=read(native/'baseline.png')
        assert np.array_equal(crop,read(native/'original.png')) and np.array_equal(base,read(source/'output.png'))
        assert np.array_equal(known,read(source/'reference_context.png',True))
        assert np.array_equal(retained,read(source/'retained_applied_context.png',True)) and np.array_equal(edit,read(source/'edit.png',True))
        assert not np.any(applied&~retained)
        counts={}
        if applied[s].any():
            h0,w0=known.shape;integral=cv2.integral(known.astype(np.uint8))
            for size in prior.SIZES:
                total=0
                for yy in range(0,h0-size+1,size//3):
                    for xx in range(0,w0-size+1,size//3):
                        total+=int(integral[yy+size,xx+size]-integral[yy,xx+size]-integral[yy+size,xx]+integral[yy,xx]==size*size)
                counts[str(size)]=dict(total=total,selected=min(512,total))
            assert counts==json.loads((native/'patch_counts.json').read_text())
            adaptive=next((k for k in prior.SIZES if counts[str(k)]['selected']>=4),None)
            fields=dict(np.load(native/'baseline_fields_Lab.npz',allow_pickle=False))
            seam=np.load(native/'baseline_texture_seam.npy',allow_pickle=False);color=np.load(native/'baseline_color_Lab.npy',allow_pickle=False)
            base_field=np.load(native/'baseline_field_Lab.npy',allow_pickle=False)
            dist=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5);bw=np.exp(-(np.maximum(dist-1,0)/8.)**2)*applied;tw=np.clip((dist-1)/4.,0,1)*applied
            expected=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*bw[:,:,None]
            assert np.array_equal(color,expected);expected=expected.copy();expected[:,:,0]+=seam*tw
            assert np.array_equal(base_field,expected) and np.array_equal(prior.history.fill.frozen.compose(base_field,applied,base,s),base)
        for mode in prior.MODES:
            value=read(native/mode/'output.png');stat=row['texture_scale_diagnostics'][mode];check=dict(id=rid,mode=mode)
            if applied[s].any():
                size=12 if mode=='uniform12' and counts['12']['selected']>=4 else None
                if mode=='adaptive':size=adaptive
                assert size==stat['selected_patch_size']
                replaced=size is not None and size!=24;assert replaced==stat['texture_replaced']
                if replaced:
                    md=native/mode;check['provenance']=verify_texture(lab,known,applied,md)
                    metadata=json.loads((md/'donors.json').read_text());assert metadata['patch_size']==size
                    assert len(metadata['patch_xy'])==counts[str(size)]['selected'] and metadata['total_qualified_patches']==counts[str(size)]['total']
                    newseam=np.load(md/'texture_seam.npy',allow_pickle=False);field=np.load(md/'result_field_Lab.npy',allow_pickle=False)
                    expected=color.copy();expected[:,:,0]+=newseam*tw
                    assert np.array_equal(expected,field) and np.array_equal(field[:,:,1:],base_field[:,:,1:])
                    assert np.array_equal(field[dist<=1],base_field[dist<=1])
                    update=applied&(tw>0);assert np.array_equal(update,read(md/'update_context.png',True))
                    assert np.array_equal(prior.history.fill.frozen.compose(field,update,base,s),value)
                    assert not np.any(np.any(value!=base,axis=2)&~update[s])
                    assert abs(float(np.std(newseam[applied]))-stat['seam_L_std'])<1e-6
                else:assert np.array_equal(value,base)
            else:assert np.array_equal(value,base)
            assert stat['changed_vs_base']==int(np.any(value!=base,axis=2).sum())
            assert stat['paper_applied']==int(retained[s].sum())
            assert not np.any(np.any(value!=base,axis=2)&~applied[s]) and not np.any(np.any(value!=crop,axis=2)&~edit)
            if stat['changed_vs_base']==0:unchanged[mode].append(rid)
            checks.append(check)
    (OUT/'paper').mkdir(exist_ok=True)
    for p in (prior.OUT/'comparisons').glob('*.png'):shutil.copyfile(p,OUT/'paper'/p.name)
    focus=['200021712_00082_1_external_rim_core_10','hole_edge_new','new_margin_hole','extra_top_branch','extra_second_branch','unseen_fold','unseen_thin_crack','extra_isolated_marks','new_brown_stamp']
    for i,rid in enumerate(focus,1):
        stack([OUT/'paper'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 紙模様の参照パッチ尺度',
              ['原画像・固定色場＋24px・一律12px・24→16→12の参照確保。色・マスクは固定。',
               '実際の資格ある原画素だけ。原模様・物理原因・画素GTは未確定。'])
    dump(OUT/'verification.json',dict(status='passed',pages=25,ROIs=77,variants=154,checks=checks,unchanged_ROIs=unchanged,
        masks_reference_applied_color_fixed=True,float_ab_exactly_unchanged=True,first_inside_pixel_unchanged=True,
        independent_donor_pixel_and_average_reconstruction=True,outside_edit_changes=0,pixel_GT=False,new_DDRM=False))
    notes=ROOT/'tmp_work/adaptive_paper_texture_scale_review_notes_20261001.md';(OUT/'report.md').write_text(notes.read_text())
    pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==86
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==len(pngs)
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=len(pngs),
        dependencies={str(notes.relative_to(ROOT)):sha(notes),**{str((prior.OUT/n).relative_to(ROOT)):sha(prior.OUT/n) for n in ['manifest.json','complete.json','final_dependencies.json']}},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered adaptive paper texture scale:',len(pngs),'PNGs',flush=True)


if __name__=='__main__':main()
