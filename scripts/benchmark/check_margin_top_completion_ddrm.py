"""Same saved-DDRM operator/noise, two top-boundary mask ablations, all 22 ROIs."""
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import probe_margin_top_completion as mask_probe
import check_paper_margin_defect_ddrm as prior_ddrm
import probe_adjacent_paper_boundary as blending
from build_npj_v6_comparison import dump, read, sha, write
from package_paper_blending_review import check_bundle

ROOT = mask_probe.ROOT
OUT = ROOT/'tmp_work/margin_top_completion_ddrm_20261001'
driver = prior_ddrm.driver
MODES = mask_probe.MODES


def preflight():
    for folder in [mask_probe.OUT, prior_ddrm.OUT, blending.OUT]:
        check_bundle(folder)
    for source in mask_probe.old.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key = page['page']; old_dir = mask_probe.old.OUT/'candidate_pages'/key
            baseline = {k:read(old_dir/'dark_support'/(k+'.png'),True) for k in ['edit','protect','hold','red']}
            maps = {k:read(mask_probe.OUT/'pages'/key/'features'/(k+'.png'),True)
                    for k in ['top_margin','safe_domain','long_line_guard','warm_anchor_guard']}
            for mode in MODES:
                proposed = {k:read(mask_probe.OUT/'pages'/key/mode/(k+'.png'),True)
                            for k in ['edit','added','protect','hold','red']}
                assert np.array_equal(proposed['edit'],baseline['edit']|proposed['added'])
                assert np.array_equal(proposed['added'],proposed['edit']&~baseline['edit'])
                assert np.array_equal(proposed['protect'],baseline['protect']&~proposed['added'])
                assert np.array_equal(proposed['hold'],baseline['hold']&~proposed['added'])
                assert np.array_equal(proposed['red'],baseline['red'])
                assert not np.any(proposed['added']&maps['warm_anchor_guard'])
                allowed = maps['top_margin'] if mode=='in_band' else maps['safe_domain']&~maps['long_line_guard']
                assert not np.any(proposed['added']&~allowed)
                assert not np.any(proposed['edit']&(proposed['protect']|proposed['hold']|proposed['red']))
    dump(OUT/'mask_preflight.json',dict(status='passed',pages=14,old_edits_retained=True,
        conservation_checked_on_full_pages=True,ROI_gating=False))


def main():
    OUT.mkdir(parents=True,exist_ok=True);preflight()
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        str(Path(prior_ddrm.__file__).relative_to(ROOT)):sha(Path(prior_ddrm.__file__)),
        str(Path(mask_probe.__file__).relative_to(ROOT)):sha(Path(mask_probe.__file__)),
        str(Path(driver.__file__).relative_to(ROOT)):sha(Path(driver.__file__)),
        str(Path(blending.__file__).relative_to(ROOT)):sha(Path(blending.__file__)),
        str(Path(blending.previous.__file__).relative_to(ROOT)):sha(Path(blending.previous.__file__)),
        str(driver.CKPT.relative_to(ROOT)):sha(driver.CKPT)}
    for folder in [mask_probe.OUT, prior_ddrm.OUT, blending.OUT]:
        for name in ['manifest.json','complete.json']:dependencies[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    manifest=dict(settings=driver.SETTINGS,dependencies=dependencies,pages=14,regions=22,
        masks=MODES,source_sha256=sha(Path(__file__)),
        fixed_blending='same adjacent boundary-only algorithm and frozen old top-margin scope; recompute references for each mask',
        newly_added_above_old_margin='DDRM only; outside frozen blending eligibility',
        changes='preprocessing masks only; raw DDRM comparison is primary',pixel_GT=False)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');(OUT/'snapshots').mkdir(exist_ok=True)
    for name in ['denoising.py','svd_replacement.py']:
        shutil.copyfile(prior_ddrm.OUT/'snapshots'/name,OUT/'snapshots'/name)
        assert sha(OUT/'snapshots'/name)==sha(driver.MODULE/'functions'/name)
    driver.OUT=OUT;os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
    runner=driver.DDRM('cuda:0');repeat=dict(passed=False);rows=[];started=time.monotonic()
    for source in mask_probe.old.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key=page['page'];original=read(ROOT/page['original']);H,W=original.shape[:2]
            old_dir=mask_probe.old.OUT/'candidate_pages'/key
            band=read(old_dir/'features/top_margin.png',True)
            masks={'old':read(old_dir/'dark_support/edit.png',True),
                   **{m:read(mask_probe.OUT/'pages'/key/m/'edit.png',True) for m in MODES}}
            protected=np.zeros((H,W),bool)
            for name in ['protect','hold','red','extra']:protected|=read(source/'v6_masks'/key/(name+'.png'),True)
            references={m:blending.previous.clean_reference(original,edit,protected&~edit,band)[1]
                        for m,edit in masks.items()}
            for roi in page['rois']:
                rid=roi['id'];x,y,w,h=roi['xywh'];x0,y0=max(0,x-64),max(0,y-64)
                x1,y1=min(W,x+w+64),min(H,y+h+64)
                context=original[y0:y1,x0:x1];local=(slice(y-y0,y-y0+h),slice(x-x0,x-x0+w))
                crop=context[local];old_native=prior_ddrm.OUT/'native'/rid
                assert np.array_equal(crop,read(old_native/'original.png'))
                old_final=read(old_native/'candidate.png');old_raw=read(old_native/'candidate_raw.png')
                old_context_mask=masks['old'][y0:y1,x0:x1]
                old_corrected=blending.infer(context,references['old'][y0:y1,x0:x1],old_final,
                    band[y:y+h,x:x+w],old_context_mask,(x-x0,y-y0))[0]['boundary']
                assert np.array_equal(old_corrected,read(blending.OUT/'native'/rid/'boundary.png'))
                finals={'old':old_final};raws={'old':old_raw};corrected={'old':old_corrected}
                sampled={};reused={};post_counts={}
                for mode in MODES:
                    missing=masks[mode][y0:y1,x0:x1]
                    if np.array_equal(missing,old_context_mask):
                        finals[mode]=old_final.copy();raws[mode]=old_raw.copy()
                        sampled[mode]=0;reused[mode]='same_context_mask_as_saved_old'
                    else:
                        full,raw,count=prior_ddrm.restore(runner,context,missing,key,x0,y0,repeat)
                        finals[mode]=full[local];raws[mode]=raw[local]
                        sampled[mode]=count;reused[mode]=False
                    post,adjusted,_,_=blending.infer(context,references[mode][y0:y1,x0:x1],finals[mode],
                        band[y:y+h,x:x+w],missing,(x-x0,y-y0))
                    corrected[mode]=post['boundary'];post_counts[mode]=int(adjusted.sum())
                    if reused[mode]:assert np.array_equal(corrected[mode],old_corrected)
                    assert not np.any(np.any(finals[mode]!=crop,axis=2)&~missing[local])
                    assert not np.any(np.any(corrected[mode]!=crop,axis=2)&~missing[local])
                dest=OUT/'native'/rid
                write(dest/'original.png',crop)
                for mode in ['old']+MODES:
                    for name,value in [(mode+'.png',finals[mode]),(mode+'_raw.png',raws[mode]),
                        (mode+'_boundary.png',corrected[mode]),(mode+'_edit.png',masks[mode][y:y+h,x:x+w])]:
                        write(dest/name,value);assert np.array_equal(read(dest/name,value.dtype==bool),value)
                note='前回の余白試作からマスクだけを変更。重み・20段・画像空間ノイズは固定。'
                prior_ddrm.figure([crop]+[finals[m] for m in ['old']+MODES],
                    ['原画像','前回のマスク → DDRM','帯内の解除 → DDRM','全成分の回収 → DDRM'],
                    OUT/'comparisons'/(rid+'.png'),roi['title']+' · '+key,note,'同じDDRMで上端除外の解除を比較。')
                prior_ddrm.figure([crop]+[corrected[m] for m in ['old']+MODES],
                    ['原画像','前回＋境界補正','帯内解除＋境界補正','全成分回収＋境界補正'],
                    OUT/'blended_comparisons'/(rid+'.png'),roi['title']+' · 同じ境界補正',
                    '上方の新しい編集領域は旧余白帯の外なのでDDRMのみ。境界補正の対象範囲は固定。',
                    '補助比較：前回と同じ隣接境界補正を使用。')
                rows.append(dict(id=rid,page=key,xywh=roi['xywh'],title=roi['title'],
                    added_pixels={m:int((masks[m][y:y+h,x:x+w]&~masks['old'][y:y+h,x:x+w]).sum()) for m in MODES},
                    changed_vs_old={m:int(np.any(finals[m]!=old_final,axis=2).sum()) for m in MODES},
                    sampled_patches=sampled,reused=reused,postprocess_pixels=post_counts,outside_edit_changed_pixels=0))
                dump(OUT/'progress.json',dict(completed=len(rows),total=22,latest=rid))
                print('Top-completion DDRM:',len(rows),'/22',rid,sampled,flush=True)
            del original,band,protected,references,masks;gc.collect()
    dump(OUT/'diagnostics.json',rows)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',pages=14,regions=22,actual_DDRM=True,
        frozen_weight_sampler=True,spatial_noise_coupled=True,same_mask_repeat_passed=repeat['passed'],
        old_boundary_output_reproduced=True,outside_edit_changed_pixels=0,model_forward_calls=runner.model_calls,
        elapsed_seconds=time.monotonic()-started,pixel_GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json'] and p.name not in ['manifest.json','complete.json']}))
    print('Complete: two mask ablations, actual fixed DDRM, 22 ROIs',flush=True)


if __name__=='__main__':main()
