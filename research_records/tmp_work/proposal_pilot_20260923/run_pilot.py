"""Exploratory paired pilot. All artifacts stay under this run directory."""
import os
from pathlib import Path
ROOT = Path('/home/imoto/Heritage_Pipeline')
OUT = ROOT / 'tmp_work/proposal_pilot_20260923'
for key in ('TMPDIR', 'TMP', 'TEMP'):
    os.environ[key] = str(OUT / 'tmp')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import concurrent.futures, csv, hashlib, importlib.util, json, subprocess, sys, time, traceback
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
BENCH = ROOT / 'tmp_work/makura_benchmark'
PYTHON = '/home/imoto/miniconda3/envs/diffusion/bin/python'
CKPT = ROOT / 'modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'

def read(path, gray=False):
    a = cv2.imread(str(path), 0 if gray else 1)
    if a is None:
        raise ValueError(f'Cannot read {path}')
    return a

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def load_csc(name, module):
    path = ROOT / f'modules/{module}/scripts/csc.py'
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

# Same stored ACE image for both CSC methods: no ACE implementation confound.
OLD = load_csc('pilot_old', 'ddrm_npj')
NEW = load_csc('pilot_new', 'ddrm_npj_v3')
WRAPPER = "import sys,random,numpy as np,torch,runpy; p=sys.argv.pop(1); sys.path.insert(0,str(__import__('pathlib').Path(p).parent)); random.seed(999); np.random.seed(999); torch.manual_seed(999); torch.cuda.manual_seed_all(999); runpy.run_path(p,run_name='__main__')"

def invoke(module, script, args, log, expected):
    start = time.monotonic()
    cmd = [PYTHON, '-B', '-c', WRAPPER, str(ROOT / 'modules' / module / 'scripts' / script)] + list(map(str, args))
    with open(log, 'w') as f:
        f.write(json.dumps(cmd, ensure_ascii=False) + '\n'); f.flush()
        subprocess.run(cmd, cwd=ROOT / 'modules' / module, stdout=f, stderr=subprocess.STDOUT, check=True)
    read(expected)
    return round(time.monotonic() - start, 2)

def metrics(gt, result, stain, text):
    if gt.shape != result.shape:
        raise ValueError('Dimension mismatch')
    g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    err = np.abs(g.astype(float) - r.astype(float))
    mean = lambda mask: float(err[mask].mean()) if mask.any() else None
    return dict(psnr_rgb=float(peak_signal_noise_ratio(gt,result,data_range=255)),
                ssim_gray=float(structural_similarity(g,r,data_range=255)),
                stain_mae=mean(stain), text_mae=mean(text), nonstain_mae=mean(~stain),
                text_lightening=float(np.maximum(r.astype(float)-g,0)[text].mean()))

def worker(gpu, stem):
    summary = []
    for intensity in ('0.30', '1.00'):
        gt = read(BENCH / 'clean' / f'{stem}.png')
        degraded_path = BENCH / f'degraded/intensity_{intensity}/{stem}.png'
        inp = read(degraded_path)
        stain = read(BENCH / f'stain_gt/intensity_{intensity}/{stem}.png', True) > 127
        # Fixed, source-derived proxy; not a hand-annotated text ground truth.
        text = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) < 150
        ace_path = BENCH / f'masks/intensity_{intensity}/{stem}/00_input.png'
        ace = read(ace_path)
        if gt.shape != inp.shape or ace.shape != gt.shape or stain.shape != gt.shape[:2]:
            raise ValueError('Input dimension mismatch')
        case = OUT / stem / intensity
        case.mkdir(parents=True)
        input_dir = case / 'input'; input_dir.mkdir()
        (input_dir / f'{stem}.png').symlink_to(degraded_path)
        rows = [dict(csc='none', method='degraded', **metrics(gt,inp,stain,text))]
        for label, model in [('original', OLD), ('improved', NEW)]:
            work = case / label; work.mkdir()
            maskroot = work / 'masks'; masks = maskroot / stem; masks.mkdir(parents=True)
            start = time.monotonic()
            result = model.color_space_clustering(ace, **({'ink_rescue_by_lightness':True} if label == 'improved' else {}))
            arrays = {'02_text_black.png':255-result.black_text_mask,
                      '03_text_red.png':255-result.red_text_mask,
                      '04_degradation.png':result.degradation_mask,
                      '05_background.png':result.background_mask}
            for name,a in arrays.items():
                if not cv2.imwrite(str(masks/name), a): raise RuntimeError(name)
            ratios = {k:float((a>127).mean()) for k,a in arrays.items()}
            black_ratio=float((arrays['02_text_black.png']<127).mean())
            if not 0 < black_ratio < .8: raise ValueError(f'Unexpected black ratio {black_ratio}')
            timings={'csc':round(time.monotonic()-start,2)}
            simple = cv2.inpaint(inp, 255-result.background_mask,15,cv2.INPAINT_TELEA)
            simple_path = work / 'simple.png'; cv2.imwrite(str(simple_path),simple)
            common = ['--ddrm_style','--degradation_only','--full_kept','--enhance',
                      '--residual_thresh','10','--no_final_correction','--ckpt',CKPT,
                      '--num_res_blocks','2','--eta','0.5','--mask_dir',maskroot,
                      '--device',f'cuda:{gpu}','--patch_size','128','--overlap','32','--ext','.png']
            previous = input_dir
            for stage in ('preprocess_only','v15_pass1','v15_pass2'):
                output = work/stage
                source = input_dir if stage != 'v15_pass2' else work/'v15_pass1'
                args = common+['--input_dir',source,'--output_dir',output]
                if stage=='preprocess_only': args+=['--y0_only']
                path = output/f'{stem}.png'
                timings[stage]=invoke('ddnm_npj','pipeline_ddnm.py',args,work/f'{stage}.log',path)
                rows.append(dict(csc=label,method=stage,**metrics(gt,read(path),stain,text)))
            for ref, path in [('simple',simple_path),('refined',work/'v15_pass2'/f'{stem}.png')]:
                output=work/f'ddrm_{ref}'
                expected=output/stem/'result'/f'{stem}.png'
                args=['--input_dir',input_dir,'--output_dir',output,'--mask_dir',maskroot,
                      '--ckpt',CKPT,'--back_ground',path,'--device',f'cuda:{gpu}',
                      '--patch_size','128','--overlap','32','--no_wandb']
                timings[f'ddrm_{ref}']=invoke('ddrm_npj_v2','pipeline_ddrm.py',args,work/f'ddrm_{ref}.log',expected)
                rows.append(dict(csc=label,method=f'ddrm_{ref}',**metrics(gt,read(expected),stain,text)))
            (work/'completed.json').write_text(json.dumps(dict(timings=timings,mask_white_ratios=ratios,black_ratio=black_ratio),indent=2))
            print(f'Completed {stem} intensity={intensity} CSC={label}: {timings}',flush=True)
        for row in rows: row.update(stem=stem,intensity=intensity)
        with open(case/'metrics.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summary.extend(rows)
    return summary

def main():
    stems=sorted(p.stem for p in (BENCH/'clean').glob('*.png'))
    assert len(stems)==20, len(stems)
    inventory={}
    for intensity in ('0.30','0.50','0.70','1.00'):
        for folder in ('degraded','stain_gt','masks'):
            p=BENCH/f'{folder}/intensity_{intensity}'
            found=sorted(x.name if x.is_dir() else x.stem for x in p.iterdir() if x.is_dir() or x.suffix=='.png')
            assert found==stems,(folder,intensity)
            inventory[f'{folder}/{intensity}']=len(found)
    selected=[stems[0],stems[-1]]
    paths=[ROOT/'modules'/m/'scripts'/s for m,s in [('ddrm_npj','csc.py'),('ddrm_npj_v3','csc.py'),('ddrm_npj_v2','pipeline_ddrm.py'),('ddnm_npj','pipeline_ddnm.py')]]
    paths += [CKPT,Path(__file__)]
    for stem in selected:
        paths.append(BENCH/'clean'/f'{stem}.png')
        for intensity in ('0.30','1.00'):
            paths.extend([BENCH/f'degraded/intensity_{intensity}/{stem}.png',BENCH/f'stain_gt/intensity_{intensity}/{stem}.png',BENCH/f'masks/intensity_{intensity}/{stem}/00_input.png'])
    manifest=dict(selected=selected,selection='First and last sorted current clean stems; exploratory, no outcome-based selection',
                  seed=999,inventory=inventory,hashes={str(p):sha(p) for p in paths},
                  limitations=['Text mask is source grayscale<150 proxy, not manual GT','Uses stored ACE input; historical provenance not fully established','Original vs improved changes multiple CSC components','v15 pass2 reuses first input masks; frozen exploratory recipe','Single seed, two pages: no significance or generalization claim'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Audit passed; starting two GPU workers: '+str(selected),flush=True)
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,gpu,stem) for gpu,stem in enumerate(selected)]
        for f in concurrent.futures.as_completed(futures): rows.extend(f.result())
    with open(OUT/'metrics.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT/'status.json').write_text(json.dumps({'status':'complete','rows':len(rows)}))

if __name__=='__main__':
    try: main()
    except Exception:
        (OUT/'status.json').write_text(json.dumps({'status':'failed','error':traceback.format_exc()}))
        raise
