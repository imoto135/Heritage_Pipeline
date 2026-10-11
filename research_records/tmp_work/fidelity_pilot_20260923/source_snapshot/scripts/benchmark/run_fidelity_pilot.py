"""Run an auditable developmental fidelity pilot; inference and scoring are separate.

Default: 2 pages x intensities 0/.3/1, rescue on. Use --rescue off in a separate
run with identical inputs for the matched CSC ablation. No publication claims
are made from the automatic evaluation-region proxies.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'4')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import cv2
import numpy as np
from fidelity_core import read_image, read_mask, write_image, evaluation_proxies, evaluate
ROOT=Path(__file__).resolve().parents[2]
BENCH=ROOT/'tmp_work/makura_benchmark'
STAGE=Path(__file__).with_name('fidelity_stage.py')
CKPT=ROOT/'modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def save_json(path,data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)); tmp.replace(path)


def write_rows(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def invoke(command,log,expected,signature):
    done=log.with_suffix('.complete.json')
    if done.exists():
        prior=json.loads(done.read_text())
        if prior['run_signature']!=signature or prior['command']!=command:
            raise RuntimeError(f'Resume configuration mismatch: {log}')
        if all(Path(p).exists() and sha(p)==h for p,h in prior['outputs'].items()):
            return prior['wall_seconds']
        raise RuntimeError(f'Completed output modified: {log}')
    # The caller removes no outputs. Partial inference must be examined explicitly.
    if any(Path(p).exists() for p in expected):
        raise RuntimeError(f'Unverified partial outputs exist; use a fresh run directory: {log}')
    start=time.monotonic()
    with open(log,'w') as f:
        f.write(json.dumps(command)+'\n'); f.flush()
        subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    for p in expected: read_image(p)
    wall=time.monotonic()-start
    save_json(done,{'run_signature':signature,'command':command,'wall_seconds':wall,
                    'outputs':{str(p):sha(p) for p in expected}})
    return wall


def review_sheet(clean,observed,outputs,destination):
    """Fixed coordinate raw-scale crops plus downsized context; no outcome selection."""
    h,w=clean.shape[:2]
    entries=[('clean',clean),('input',observed)]+[(name,read_image(p)) for name,p in outputs.items()]
    # Crop coordinates fixed by geometry, not by metric outcomes.
    crops=[(int(w*.2),int(h*.2)),(int(w*.5),int(h*.5)),(int(w*.7),int(h*.75))]
    tile=256; panel=np.full((len(entries)*(tile+24),len(crops)*tile,3),255,np.uint8)
    for row,(name,image) in enumerate(entries):
        for col,(x,y) in enumerate(crops):
            x=min(x,w-tile); y=min(y,h-tile)
            crop=image[y:y+tile,x:x+tile]
            top=row*(tile+24)
            panel[top+24:top+24+tile,col*tile:(col+1)*tile]=crop
            cv2.putText(panel,name,(col*tile+4,top+17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
    write_image(destination,panel)
    save_json(destination.with_suffix('.json'),{'crop_size':tile,'origins':crops,
               'purpose':'Unannotated, fixed geometric ROI for visual review; not text ground truth'})


def worker(gpu,stem,args,manifest,signature):
    clean=read_image(BENCH/'clean'/f'{stem}.png')
    all_rows=[]
    for intensity in args.intensities:
        item=manifest['cases'][f'{stem}/{intensity}']
        observed=read_image(item['input'])
        if clean.shape!=observed.shape: raise ValueError('Input/GT dimensions differ')
        case=args.output/stem/intensity
        case.mkdir(parents=True,exist_ok=True)
        input_dir=case/'input'; input_dir.mkdir(exist_ok=True)
        local=input_dir/f'{stem}.png'
        if not local.exists(): local.symlink_to(item['input'])
        maskdir=case/'masks'/stem
        maskdir.mkdir(parents=True,exist_ok=True)
        prepared=case/'prepared'
        command=[sys.executable,'-B',str(STAGE),'prepare','--input',str(local),
                 '--ace',item['ace'],'--output',str(prepared),'--masks',str(maskdir),'--rescue',args.rescue]
        timing={}
        timing['prepare']=invoke(command,case/'prepare.log',[prepared/f'{n}.png' for n in ('B0','P0','P1')],signature)
        print(f'{stem} {intensity}: references prepared ({timing["prepare"]:.1f}s)',flush=True)
        common=['--input',str(local),'--masks',str(maskdir),'--ckpt',str(CKPT),
                '--device',f'cuda:{gpu}','--seed',str(args.seed)]
        r1=case/'R1'/'output.png'
        command=[sys.executable,'-B',str(STAGE),'ddnm']+common+['--reference',str(prepared/'P1.png'),'--output',str(r1.parent)]
        timing['R1']=invoke(command,case/'R1.log',[r1],signature)
        outputs={'P0':prepared/'P0.png','P1':prepared/'P1.png','R1':r1}
        # Fp isolates classical-reference processing from upstream diffusion.
        for name,reference in [('F0',prepared/'B0.png'),('Fp',prepared/'P1.png'),('F1',r1)]:
            output=case/name/'output.png'
            command=[sys.executable,'-B',str(STAGE),'ddrm']+common+['--reference',str(reference),'--output',str(output.parent)]
            timing[name]=invoke(command,case/f'{name}.log',[output],signature)
            outputs[name]=output
            print(f'{stem} {intensity}: {name} done ({timing[name]:.1f}s)',flush=True)
        # GT is introduced only below, outside inference worker commands.
        stain=np.zeros(clean.shape[:2],bool) if intensity=='0.00' else read_mask(item['stain_gt'],clean.shape[:2])
        if not np.array_equal(clean[~stain],observed[~stain]):
            raise ValueError('Synthetic input changed pixels outside stain GT')
        regions=evaluation_proxies(clean,stain)
        eval_dir=case/'evaluation'; eval_dir.mkdir(exist_ok=True)
        for name,mask in regions.items(): write_image(eval_dir/f'{name}.png',mask.astype(np.uint8)*255)
        start=time.monotonic()
        rows=[]
        for name,path in [('X',local)]+list(outputs.items()):
            score=evaluate(clean,read_image(path),regions)
            rows.append(dict(stem=stem,intensity=intensity,csc_rescue=args.rescue,seed=args.seed,
                             method=name,annotation='automatic_proxy_only',**score))
        input_error=rows[0]['stain_mae']
        for row in rows:
            row['stain_mae_improvement']=input_error-row['stain_mae'] if input_error is not None else None
        timing['evaluate']=time.monotonic()-start
        write_rows(case/'metrics.csv',rows)
        review_sheet(clean,observed,outputs,case/'review.png')
        save_json(case/'completed.json',{'run_signature':signature,'timing':timing,'rows':len(rows),
                                       'outputs':{k:str(v) for k,v in outputs.items()}})
        all_rows+=rows
        print(f'{stem} {intensity}: complete {timing}',flush=True)
    return all_rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stems',nargs='+',default=['200017197_00070_1','200017197_00357_1'])
    p.add_argument('--intensities',nargs='+',default=['0.00','0.30','1.00'])
    p.add_argument('--rescue',choices=['on','off'],default='on')
    p.add_argument('--seed',type=int,default=999)
    p.add_argument('--prepare-only',action='store_true',help='Validate and freeze manifest without inference')
    args=p.parse_args(); args.output=args.output.resolve()
    if not args.output.is_relative_to(ROOT/'tmp_work'): p.error('output must be under project tmp_work')
    if len(args.stems)!=len(set(args.stems)) or len(args.intensities)!=len(set(args.intensities)):
        p.error('Duplicate cases')
    if not set(args.intensities)<= {'0.00','0.30','0.50','0.70','1.00'}: p.error('Invalid intensity')
    args.output.mkdir(parents=True,exist_ok=True)
    tmp=args.output/'tmp'; tmp.mkdir(exist_ok=True)
    for key in ('TMPDIR','TMP','TEMP'): os.environ[key]=str(tmp)
    cases={}; inputs={}
    for stem in args.stems:
        clean_path=BENCH/'clean'/f'{stem}.png'; clean=read_image(clean_path)
        inputs[str(clean_path)]=sha(clean_path)
        for intensity in args.intensities:
            if intensity=='0.00':
                source=clean_path
                ace=ROOT/f'data/makura_csc_mask/{stem}/00_input.png'
                original=ROOT/f'data/makura_csc_mask/{stem}/00_original.png'
                if not np.array_equal(read_image(original),clean):
                    raise ValueError(f'Clean ACE provenance mismatch: {stem}')
                inputs[str(original)]=sha(original)
                gt=None
            else:
                source=BENCH/f'degraded/intensity_{intensity}/{stem}.png'
                ace=BENCH/f'masks/intensity_{intensity}/{stem}/00_input.png'
                gt=BENCH/f'stain_gt/intensity_{intensity}/{stem}.png'
                mask=read_mask(gt,clean.shape[:2])
                if not np.array_equal(read_image(source)[~mask],clean[~mask]): raise ValueError('GT support mismatch')
                inputs[str(gt)]=sha(gt)
            if read_image(source).shape!=clean.shape or read_image(ace).shape!=clean.shape:
                raise ValueError('Input or ACE dimensions mismatch')
            for f in (source,ace): inputs[str(f)]=sha(f)
            cases[f'{stem}/{intensity}']={'input':str(source),'ace':str(ace),'stain_gt':str(gt) if gt else None}
    files=set(Path(__file__).parent.glob('fidelity_*.py'))|{Path(__file__)}
    for module in ('ddrm_npj','ddrm_npj_v2','ddrm_npj_v3','ddnm_npj','stain_removal'):
        files.update((ROOT/'modules'/module).rglob('*.py'))
    hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted(files)}
    manifest={'recipe':'fidelity_v1','seed':args.seed,'csc_seed':42,'rescue':args.rescue,'cases':cases,
              'source_hashes':hashes,'input_hashes':inputs,'checkpoint_sha256':sha(CKPT),
              'environment':{k:importlib.metadata.version(k) for k in ('numpy','torch','scikit-image','scikit-learn','Pillow')},
              'python':platform.python_version(),'opencv':cv2.__version__,
              'selection':'First and last sorted clean stems from previous pilot; developmental selection',
              'limitations':['Automatic regions only; no human text-shape annotations',
                             'No held-out inference or hypothesis test',
                             'Existing ACE input provenance incompletely documented for stained cases',
                             'Checkpoint training-file provenance not established',
                             'fidelity_v1 differs from legacy v15: no gain/flat-field/residual sweep',
                             'DDNM and DDRM may alter original ink; measure rather than assume preservation'],
              'methods':{'P0':'Simple self-background + original ink','P1':'Dilated-degradation inpaint + original ink',
                         'R1':'DDNM full_kept on P1, one pass','F0':'DDRM with B0',
                         'Fp':'DDRM with P1','F1':'DDRM with R1'}}
    signature=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mf=args.output/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest: raise RuntimeError('Existing run manifest differs; use new output')
    save_json(mf,manifest)
    snap=args.output/'source_snapshot'
    for f in sorted(files):
        dest=snap/f.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): dest.write_bytes(f.read_bytes())
    (args.output/'git_diff.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
    if args.prepare_only:
        print(f'Manifest ready: {len(cases)} cases, {len(files)} source files; no inference run')
        return
    save_json(args.output/'status.json',{'status':'running','pid':os.getpid(),'run_signature':signature})
    started=time.monotonic()
    try:
        rows=[]
        # Same-sized independent GPUs; work queue keeps GPU ownership per worker.
        def batch(gpu,stems):
            combined=[]
            for stem in stems: combined+=worker(gpu,stem,args,manifest,signature)
            return combined
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(batch,gpu,args.stems[gpu::2]) for gpu in range(2)]
            for f in concurrent.futures.as_completed(futures): rows+=f.result()
        write_rows(args.output/'metrics.csv',rows)
        save_json(args.output/'status.json',{'status':'complete','rows':len(rows),'wall_seconds':time.monotonic()-started,'run_signature':signature})
    except Exception:
        save_json(args.output/'status.json',{'status':'failed','error':traceback.format_exc(),'run_signature':signature})
        raise

if __name__=='__main__': main()
