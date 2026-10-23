"""Audit provenance, nativeGT patches and saved validation masks; no training."""
import json,shutil,re,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from compare_v5_aligned_fade_control import sha,dump

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/ragan_reproduction_foundation_20261023'
def read(name):return json.loads((OUT/name).read_text())


def main():
    assert not (OUT/'final_artifact_check.json').exists()
    manifest=read('dense_benchmark_manifest.json');io=read('patch_io.json');contracts=read('contracts.json')
    for result,plan_name in [(manifest,'dense_benchmark_plan.json'),(io,'patch_io_plan.json'),(contracts,'contract_plan.json')]:
        assert result['plan_sha256']==sha(OUT/plan_name)
        p=read(plan_name);script=ROOT/'scripts/benchmark'/dict(
            **{'dense_benchmark_plan.json':'prepare_ragan_dense_benchmark.py','patch_io_plan.json':'prepare_ragan_reference_patch_io.py','contract_plan.json':'audit_ragan_reference_contracts.py'})[plan_name]
        assert p['script_sha256']==sha(script)
    for f,h in read('contract_plan.json')['source_files'].items():assert sha(ROOT/f)==h,f
    assert read('loader_variant_verification.json')['script_sha256']==sha(ROOT/'scripts/benchmark/ragan_dense_loader_variants.py')
    for row in read('downloads.json'):assert sha(ROOT/row['path'])==row['sha256']
    pages={p['id']:p for p in manifest['pages']};seen=set();loaded={}
    for p in pages.values():
        assert sha(ROOT/p['image'])==p['image_sha256'] and sha(ROOT/p['gt'])==p['gt_sha256']
        image=np.asarray(Image.open(ROOT/p['image']).convert('RGB'));gt=np.asarray(Image.open(ROOT/p['gt']).convert('L'))
        assert image.shape[:2]==gt.shape and set(np.unique(gt))=={0,255}
        assert p['rgb_content_sha256'] not in seen;seen.add(p['rgb_content_sha256'])
        loaded[p['id']]=(image,gt)
    assert manifest['page_counts']==dict(train=16,val=4,reserved_test=10)
    files_checked=0
    for row in io['rows']:
        assert pages[row['page']]['split']==row['split'] and row['split']!='reserved_test'
        x,y=row['xy'];w,h=row['valid_wh'];raw,gt=loaded[row['page']]
        for kind,meta in row['files'].items():
            f=ROOT/meta['path'];assert sha(f)==meta['sha256'];a=np.asarray(Image.open(f));files_checked+=1
            assert a.shape[:2]==(256,256)
            assert np.all(a[h:]==255) and np.all(a[:,w:]==255)
            if kind in ['img','gt']:
                original=raw if kind=='img' else gt
                assert np.array_equal(a[:h,:w],original[y:y+h,x:x+w])
            if kind in ['gt','ostu']:assert set(np.unique(a)).issubset({0,255})
    assert len(io['rows'])==610 and io['train_patches']==497 and io['val_patches']==113
    controls=read('validation_controls.json');rescored=[]
    for row in controls['rows']:
        assert pages[row['page']]['split']=='val'
        f=ROOT/row['mask_file'];assert sha(f)==row['mask_sha256']
        pred=np.asarray(Image.open(f))==0;truth=loaded[row['page']][1]==0
        # Independent joint histogram, rather than the scorer's boolean sums.
        tn,fp,fn,tp=np.bincount((truth.astype(np.uint8)*2+pred).ravel(),minlength=4)
        for key,value in dict(tn=tn,fp=fp,fn=fn,tp=tp).items():assert row[key]==int(value)
        f1=float(tp/(tp+.5*(fp+fn)));mse=np.mean(np.logical_xor(pred,truth))
        assert abs(f1-row['ink_f1'])<1e-12
        assert abs(-10*np.log10(mse)-row['binary_psnr_db'])<1e-10
        rescored.append(dict(page=row['page'],method=row['method'],f1=f1))
    for method,value in controls['macro_ink_f1'].items():assert abs(value-np.mean([r['f1'] for r in rescored if r['method']==method]))<1e-12
    figures=dict(controls['figures']);figures['figures/公開ローダーの回転とSobel.png']=contracts['figure_sha256']
    for f,h in figures.items():assert sha(OUT/f)==h
    snapshots=OUT/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for name in ['prepare_ragan_dense_benchmark.py','audit_ragan_reference_contracts.py','prepare_ragan_reference_patch_io.py','ragan_dense_loader_variants.py',Path(__file__).name]:
        shutil.copy2(ROOT/'scripts/benchmark'/name,snapshots/name)
    links=[]
    for f in OUT.glob('*.md'):
        for href in re.findall(r'\]\(([^)]+)\)',f.read_text()):
            if '://' in href or href.startswith('#'):continue
            if Path(href).name!='final_artifact_check.json':assert (f.parent/href).exists(),(f,href)
            links.append([f.name,href])
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],capture_output=True,text=True)
    dump(OUT/'final_artifact_check.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        no_training=True,no_model_updates=True,reserved_test_predictions=0,reserved_test_metrics=0,
        verified_original_GT_pairs=len(pages),verified_materialized_patches=len(io['rows']),verified_patch_files=files_checked,
        independently_rescored_validation_masks=len(rescored),validation_readouts=rescored,comparison_pngs=len(figures),figures=figures,
        source_snapshots={f.name:sha(f) for f in snapshots.glob('*.py')},
        result_files={f.name:sha(f) for f in OUT.glob('*.json') if f.name!='final_artifact_check.json'},
        documents={f.name:sha(f) for f in OUT.glob('*.md')},links_checked=links,
        gpu_query_returncode=gpu.returncode,gpu_query_output=gpu.stdout+gpu.stderr))
    print('Verified30 original/GT pairs,610patches/2440files,8full-page masks,5PNGs; no training',flush=True)


if __name__=='__main__':main()
