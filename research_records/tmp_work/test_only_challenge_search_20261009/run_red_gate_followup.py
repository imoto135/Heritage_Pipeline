"""Fixed Howe/DDRM on F04: experimental red gate removal versus saved v1 red."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import run_ddrm_pilot as driver
import validate_red_retention as review
import probe_howe_csc_fusion as probe

BASE=Path(__file__).resolve().parent
OUT=BASE/'red_detection_followup'
ROOT=driver.ROOT
PAGE='200021063_00002_2'

def main():
    driver.OUT=OUT
    src=ROOT/'data/test/200021063'/(PAGE+'.jpg')
    raw=np.asarray(Image.open(src).convert('RGB'))
    case=dict(id='F04',page=PAGE,source=driver.relative(src),input_size=[raw.shape[1],raw.shape[0]])
    howe=review.load(BASE/'fixed_holdout_validation/masks/howe'/PAGE/'02_text_black.png')
    candidates={'howe_local_red':review.load(OUT/PAGE/'red_without_global_gate.png'),
                'howe_v1_red':review.load(ROOT/'data/split_dataset_csc_mask/test/200021063'/PAGE/'03_text_red.png')}
    records=[]
    for name,red in candidates.items():
        d=OUT/'masks'/name/PAGE
        probe.save(d/'02_text_black.png',howe&~red);probe.save(d/'03_text_red.png',red)
        records.append(dict(method=name,black_pixels=int((howe&~red).sum()),red_pixels=int(red.sum()),
            red_howe_overlap_removed=int((howe&red).sum()),
            masks_sha256={driver.relative(q):driver.digest(q) for q in d.glob('*.png')}))
    stage=OUT/'inputs'/PAGE;stage.mkdir(parents=True,exist_ok=True)
    if not (stage/src.name).exists():(stage/src.name).symlink_to(src)
    protocol=dict(case=case,methods=records,howe_raw_reused=driver.relative(BASE/'fixed_holdout_validation/masks/howe'/PAGE/'02_text_black.png'),
        howe_raw_sha256=driver.digest(BASE/'fixed_holdout_validation/masks/howe'/PAGE/'02_text_black.png'),
        input_sha256=driver.digest(src),checkpoint_sha256=driver.digest(driver.CHECKPOINT),
        pipeline_sha256=driver.digest(driver.PIPELINE),background_sha256=driver.digest(driver.BACKGROUND),
        parameters=dict(seed=999,steps=20,patch=128,overlap=32,shared_initial_noise=True),
        limitations='Exploratory F04 comparison. Different black masks do not guarantee identical stepwise random noise. Red pixels copy original RGB; no seal reconstruction.')
    driver.save_json(OUT/'ddrm_protocol.json',protocol)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(driver.run,case,name,gpu) for gpu,name in enumerate(candidates)]
        outputs=[f.result() for f in futures]
    driver.save_json(OUT/'ddrm_results.json',outputs)
    arrays=[raw];titles=['原画像']
    old=BASE/'fixed_holdout_validation/outputs/howe'/PAGE/'result'/(PAGE+'.png')
    arrays.append(np.asarray(Image.open(old).convert('RGB')));titles.append('Howe＋現行CSC赤')
    audits=[]
    for name,red in candidates.items():
        result=next(r for r in outputs if r['method']==name)
        a=np.asarray(Image.open(ROOT/result['output']).convert('RGB'))
        assert not np.any(a[red]!=raw[red])
        log=(OUT/'logs'/('F04_'+name+'.log')).read_text()
        assert '[ERROR]' not in log
        audits.append(dict(method=name,red_source_copy_mismatch_pixels=0,red_pixels=int(red.sum()),log_has_error=False))
        arrays.append(a);titles.append('Howe＋平均aゲート除外赤' if name=='howe_local_red' else 'Howe＋CSCv1赤')
    review.figure(OUT/'F04_朱印検出_DDRM比較.png',arrays,titles,[1650,2500,400,620],'F04')
    driver.save_json(OUT/'ddrm_output_audit.json',audits)
    print('Validated and rendered F04 full-native DDRM comparison',flush=True)

if __name__=='__main__':main()
