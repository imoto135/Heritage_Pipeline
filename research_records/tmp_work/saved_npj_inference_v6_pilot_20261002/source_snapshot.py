"""Reuse actual saved npj outputs; run only the fixed-v6 edit-only candidate.

This is a comparison of final processing methods, not a mask-only ablation.
The historical npj outputs are never regenerated or modified.
"""
import json
import os
from pathlib import Path
import shutil
import time

from build_npj_v6_comparison import ROOT, read, write, sha, dump
import compare_npj_v6_ddrm_edits as engine
import cv2
import numpy as np

OUT = ROOT / 'tmp_work/saved_npj_inference_v6_pilot_20261002'
V6 = ROOT / 'tmp_work/warm_ink_hold_v6_20260930'
SPECS = [
    dict(id='upper_crop', page='200010454_00002_2',
         original='data/split_dataset/test/200010454/200010454_00002_2.jpg',
         saved_input='tmp_work/mask_ablation_200010454_selfref/input/200010454_00002_2_crop.png',
         saved_npj='tmp_work/mask_ablation_200010454_selfref/output_v1/200010454_00002_2_crop/result/200010454_00002_2_crop.png',
         frame_xywh=[550, 0, 550, 500], baseline_label='保存npj：旧v1マスク・selfref',
         original_csc='data/split_dataset_csc_mask/test/200010454/200010454_00002_2',
         saved_masks='tmp_work/mask_ablation_200010454_selfref/mask_v1/200010454_00002_2_crop',
         other_npj='tmp_work/mask_ablation_200010454_extref/output_v1/200010454_00002_2_crop/result/200010454_00002_2_crop.png'),
    dict(id='full_page', page='200018243_00003_2',
         original='data/split_dataset/test/200018243/200018243_00003_2.jpg',
         saved_input='tmp_work/full_page_compare/input/200018243_00003_2.jpg',
         saved_npj='tmp_work/full_page_compare/ddrm_v1true_out/200018243_00003_2/result/200018243_00003_2.png',
         frame_xywh=[0, 0, 2073, 3203], baseline_label='保存npj：v1true・ページ出力',
         historical_log='tmp_work/full_page_compare/log_v1true.txt'),
]
REGIONS = [
    dict(id='01_upper_faint_stroke', frame='upper_crop', xywh=[190,325,180,175], title='上部の薄い筆画'),
    dict(id='02_upper_dense_control', frame='upper_crop', xywh=[5,335,175,160], title='濃い筆画の対照'),
    dict(id='03_upper_brown_stain', frame='upper_crop', xywh=[180,235,160,120], title='筆画の上方の茶色い斑点'),
    dict(id='04_upper_pale_stain', frame='upper_crop', xywh=[330,120,150,160], title='淡い茶色の広がり'),
    dict(id='05_page_faint_left', frame='full_page', xywh=[170,700,235,450], title='左列の薄い筆画'),
    dict(id='06_page_faint_middle', frame='full_page', xywh=[760,1870,220,480], title='中央列の細い筆画'),
    dict(id='07_page_dense_control', frame='full_page', xywh=[1510,1650,300,440], title='右列の濃い筆画の対照'),
    dict(id='08_page_long_stroke', frame='full_page', xywh=[440,1850,240,650], title='左寄りの長い細線'),
]


def dependency(p, dependencies):
    p = Path(p)
    dependencies[str(p.relative_to(ROOT))] = sha(p)


def main():
    assert not (OUT / 'complete.json').exists(), 'Frozen pilot already exists'
    OUT.mkdir(parents=True, exist_ok=True)
    dependencies = {}
    source_manifest = json.loads((engine.OUT / 'manifest.json').read_text())
    # Verify the reused inference kernel/model implementation without rerunning npj.
    for rel, digest in source_manifest['dependencies'].items():
        if rel.startswith('modules/') or rel == str(Path(engine.__file__).relative_to(ROOT)):
            assert sha(ROOT / rel) == digest, rel
            dependencies[rel] = digest
    prior_out = engine.OUT
    (OUT / 'snapshots').mkdir(exist_ok=True)
    for name in ['denoising.py', 'svd_replacement.py']:
        p = prior_out / 'snapshots' / name
        assert sha(p) == sha(engine.MODULE / 'functions' / name)
        dependency(p, dependencies)
        shutil.copyfile(p, OUT / 'snapshots' / name)
    dependency(Path(__file__), dependencies)
    for name in ['manifest.json', 'complete.json', 'candidate_freeze.json']:
        dependency(V6 / name, dependencies)
    assert json.loads((V6 / 'candidate_freeze.json').read_text())['variant'] == 'split_direction'
    frozen = json.loads((V6 / 'complete.json').read_text())['outputs']
    frames = {}
    for spec in SPECS:
        x,y,w,h = spec['frame_xywh']
        full = read(ROOT / spec['original'])
        stored = read(ROOT / spec['saved_input'])
        assert np.array_equal(stored, full[y:y+h,x:x+w]), spec['id']
        baseline = read(ROOT / spec['saved_npj'])
        assert baseline.shape == stored.shape
        masks = {}
        for name, filename in [('edit','split_direction_edit_allowed.png'), ('extra','split_direction_extra_hold.png'),
                               ('hold','split_direction_hold.png'), ('red','red.png'), ('protect','protect.png')]:
            p = V6 / spec['page'] / filename
            assert sha(p) == frozen[str(p.relative_to(V6))]
            a = read(p, True)
            assert a.shape == full.shape[:2]
            masks[name] = a[y:y+h,x:x+w].copy()
            dependency(p, dependencies)
        for name in ['hold','red','protect']:
            assert not (masks['edit'] & masks[name]).any(), name
        if 'saved_masks' in spec:
            for name in ['02_text_black.png', '03_text_red.png']:
                cropped = ROOT / spec['saved_masks'] / name
                whole = ROOT / spec['original_csc'] / name
                assert np.array_equal(cv2.imread(str(cropped),0), cv2.imread(str(whole),0)[y:y+h,x:x+w])
                dependency(cropped, dependencies); dependency(whole, dependencies)
        for name in ['original','saved_input','saved_npj','other_npj','historical_log']:
            if name in spec: dependency(ROOT / spec[name], dependencies)
        native = OUT / 'native' / spec['id']
        write(native / 'original.png', stored)
        write(native / 'saved_npj.png', baseline)
        for name,a in masks.items(): write(native / (name+'.png'), a)
        if 'other_npj' in spec:
            other = read(ROOT / spec['other_npj'])
            assert other.shape == stored.shape
            write(native / 'saved_npj_other_reference.png', other)
        frames[spec['id']] = (stored, masks)
    settings = dict(candidate=engine.SETTINGS, frozen_masks='split_direction v6, unchanged',
                    candidate_composition='generated pixels in v6 edit only; all other original pixels retained',
                    inference_frame='same stored input extent as the saved npj output',
                    historical_npj_rerun=False, baseline='actual saved npj inference images, unchanged',
                    matched_historical_noise=False, comparison='final method comparison, not a mask-only effect',
                    historical_run_source_snapshot_available=False, pixel_GT=False)
    dump(OUT / 'manifest.json', dict(settings=settings, frames=SPECS, regions=REGIONS,
         dependencies=dependencies, source_sha256=sha(Path(__file__)),
         selection='Existing original/npj outputs inspected for illustrative developmental ROIs before new v6 inference'))
    shutil.copyfile(__file__, OUT / 'source_snapshot.py')
    engine.OUT = OUT
    os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')
    os.environ['TORCH_HOME'] = str(OUT / 'torch_cache')
    runner = engine.DDRM('cuda:0')
    weight = engine.blend_weight()
    records = []
    repeat_done = False
    start = time.monotonic()
    for spec in SPECS:
        stored, masks = frames[spec['id']]
        H,W = stored.shape[:2]
        padded = cv2.copyMakeBorder(stored,0,128,0,128,cv2.BORDER_REFLECT_101)
        edit = np.pad(masks['edit'], ((0,128),(0,128)))
        accum = np.zeros((H+128,W+128,3), 'float32')
        wacc = np.zeros((H+128,W+128), 'float32')
        coords = [(y,x) for y in range(0,H,96) for x in range(0,W,96)]
        sampled = 0
        ox,oy,_,_ = spec['frame_xywh']
        for i,(y,x) in enumerate(coords):
            patch = padded[y:y+128,x:x+128]
            missing = edit[y:y+128,x:x+128]
            seed = engine.patch_seed(spec['page'], ox+x, oy+y)
            result = runner.sample(patch, missing, seed)
            sampled += int(missing.any())
            if missing.any() and not repeat_done:
                assert np.array_equal(result, runner.sample(patch, missing, seed))
                dump(OUT / 'repeat_verification.json', dict(status='passed', same_mask_same_noise_exact=True,
                     frame=spec['id'], patch_xy=[x,y], page_patch_xy=[ox+x,oy+y]))
                repeat_done = True
            accum[y:y+128,x:x+128] += result.astype('float32') * weight[...,None]
            wacc[y:y+128,x:x+128] += weight
            if (i+1)%100 == 0:
                print(spec['id'], i+1, '/', len(coords), 'tiles; sampled', sampled, flush=True)
        assert (wacc[:H,:W]>0).all()
        raw = np.clip(np.rint(accum[:H,:W] / wacc[:H,:W,None]),0,255).astype('uint8')
        candidate = raw.copy()
        candidate[~masks['edit']] = stored[~masks['edit']]
        changed = np.any(candidate != stored, axis=2)
        assert not (changed & ~masks['edit']).any()
        native = OUT / 'native' / spec['id']
        for name,a in [('candidate_raw',raw),('candidate',candidate),('candidate_changed',changed)]:
            write(native / (name+'.png'),a)
            assert np.array_equal(read(native / (name+'.png'), a.dtype==bool),a)
        record = dict(frame=spec['id'], input_exact_original_crop=True,
                      saved_npj_sha256=sha(ROOT / spec['saved_npj']),
                      candidate_edit_pixels=int(masks['edit'].sum()), candidate_changed_pixels=int(changed.sum()),
                      outside_edit_changed_pixels=0, total_tiles=len(coords), sampled_tiles=sampled)
        records.append(record)
        print('Completed', spec['id'], json.dumps(record), flush=True)
    for rel,digest in dependencies.items(): assert sha(ROOT/rel)==digest,rel
    dump(OUT / 'verification.json', dict(status='passed', records=records,
         real_saved_npj_outputs=True, npj_rerun=False, same_original_input_verified=True,
         candidate_mask_unchanged=True, candidate_outside_edit_exact_original=True,
         historical_pipeline_superiority=False, pixel_GT=False, model_calls=runner.model_calls,
         elapsed_seconds=time.monotonic()-start))
    dump(OUT / 'complete.json', dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('Completed saved-output comparison pilot; npj outputs reused exactly.',flush=True)


if __name__ == '__main__': main()
