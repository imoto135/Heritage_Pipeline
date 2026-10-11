"""One genuine frozen black restoration per page, no red overlay.

All candidate red masks will share these exact output pixels for fair comparison.
"""
import argparse
from pathlib import Path
import json
import shutil
import subprocess
import sys
import os
import time
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--pages',nargs='+',required=True)
    p.add_argument('--wait-local-training',action='store_true');a=p.parse_args()
    dest=OUT/'sauvola_ddrm_base';dest.mkdir(exist_ok=True);queue=dest/(a.device.replace(':','_')+'_queue.json')
    status=dict(status='waiting_training' if a.wait_local_training else 'running',pages=a.pages,completed=[],device=a.device,
        started_utc=datetime.now(timezone.utc).isoformat())
    queue.write_text(json.dumps(status,ensure_ascii=False,indent=2))
    if a.wait_local_training:
        path=OUT/'training_queues'/(a.device.replace(':','_')+'.json')
        while not path.exists() or json.loads(path.read_text()).get('status')=='running':time.sleep(10)
    ckpt=ROOT/'modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'
    background=ROOT/'modules/ddrm_npj_v2/background/isemonogatari.png'
    snap=ROOT/'tmp_work/npj_dirt_control_20261002/source_snapshot/ddrm_npj_v2'
    pages={r['page']:r for r in json.loads((OUT/'cases_frozen.json').read_text())['cases']}
    status.update(status='running',checkpoint_sha256=file_hash(ckpt),reference_background_sha256=file_hash(background),
        seed=999,steps=20,patch=128,overlap=32,black='fixed Sauvola w61 k.2 R.5',red='all white = no raw red overlay',
        inference_code={str(f.relative_to(ROOT)):file_hash(f) for f in snap.rglob('*.py')},
        sharing='All red variants use exact same saved base output; no repeat diffusion noise differences')
    queue.write_text(json.dumps(status,ensure_ascii=False,indent=2))
    for page in a.pages:
        r=pages[page];d=dest/page;d.mkdir(exist_ok=True);done=d/'complete.json'
        if done.exists():
            record=json.loads(done.read_text());assert file_hash(Path(record['result']))==record['result_sha256']
            status['completed'].append(page);continue
        mask=OUT/'fullpage_baselines'/page/'sauvola_black.png'
        while not mask.exists():time.sleep(10)
        assert file_hash(ROOT/r['original'])==r['source_sha256']
        original=Image.open(ROOT/r['original']).convert('RGB');inp=d/'input';inp.mkdir(exist_ok=True);original.save(inp/(page+'.png'))
        md=d/'masks'/page;md.mkdir(parents=True,exist_ok=True)
        positive=np.asarray(Image.open(mask).convert('L'))>127;assert positive.shape==np.asarray(original).shape[:2]
        Image.fromarray(np.where(positive,0,255).astype(np.uint8)).save(md/'02_text_black.png')
        Image.new('L',original.size,255).save(md/'03_text_red.png')
        command=[sys.executable,'-B',str(snap/'scripts/pipeline_ddrm.py'),'--input_dir',str(inp),
            '--output_dir',str(d/'output'),'--tmp_dir',str(d/'infer_tmp'),'--mask_dir',str(d/'masks'),
            '--ckpt',str(ckpt),'--back_ground',str(background),'--device',a.device,
            '--patch_size','128','--overlap','32','--no_wandb']
        work=d/'work';work.mkdir(exist_ok=True)
        status['current']=page;queue.write_text(json.dumps(status,ensure_ascii=False,indent=2))
        print('DDRM base',page,a.device,flush=True);start=time.monotonic()
        with (d/'infer.log').open('w') as log:
            run=subprocess.run(command,cwd=work,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        assert run.returncode==0,f'DDRM failed: {d}/infer.log'
        result=d/'output'/page/'result'/(page+'.png');assert result.exists() and Image.open(result).size==original.size
        record=dict(page=page,original=r['original'],source_sha256=r['source_sha256'],
            result=str(result),result_sha256=file_hash(result),command=command,seconds=time.monotonic()-start,
            black_mask_sha256=file_hash(md/'02_text_black.png'),red_mask_sha256=file_hash(md/'03_text_red.png'),
            no_original_red_overlay=True,real_restoration=True,reading_GT=False,
            frozen_source_checkpoint_sha256=status['checkpoint_sha256'])
        done.write_text(json.dumps(record,ensure_ascii=False,indent=2));status['completed'].append(page)
        queue.write_text(json.dumps(status,ensure_ascii=False,indent=2));print('DDRM completed',page,round(record['seconds'],1),'sec',flush=True)
    status.update(status='completed',current=None,finished_utc=datetime.now(timezone.utc).isoformat())
    queue.write_text(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
