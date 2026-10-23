"""Acquire organizer-provided denseGT; freeze a small development split, no training."""
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
import numpy as np
import requests
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/ragan_reproduction_foundation_20261023'
DATA=ROOT/'data/reproduction_benchmarks'
sys.path.insert(0,str(OUT/'dependencies'))
import py7zr


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def download(url,path):
    assert not path.exists(),path
    path.parent.mkdir(parents=True,exist_ok=True)
    r=requests.get(url,timeout=(15,60));r.raise_for_status()
    assert len(r.content)<100_000_000
    path.write_bytes(r.content)
    return dict(url=url,final_url=r.url,path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=len(r.content))


def extract(path,dest):
    assert not dest.exists()
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as z:
            names=z.namelist()
            assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
            z.extractall(dest)
    else:
        with py7zr.SevenZipFile(path) as z:
            names=z.getnames()
            assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
            z.extractall(path=dest)


def files_by_id(folder):
    result={}
    for f in folder.rglob('*'):
        if f.suffix.lower() not in ['.png','.bmp','.tif','.tiff','.jpg','.jpeg'] or f.name.startswith('._'):continue
        key=str(int(re.findall(r'\d+',f.stem)[0]));assert key not in result,(key,f)
        result[key]=f
    return result


def main():
    assert not (OUT/'dense_benchmark_manifest.json').exists()
    plan=dict(script_sha256=sha(__file__),sources={
        'DIBCO2017':dict(base='https://vc.ee.duth.gr/dibco2017/benchmark/',image='DIBCO2017_Dataset.7z',gt='DIBCO2017_GT.7z',expected=20),
        'HDIBCO2018':dict(base='https://vc.ee.duth.gr/h-dibco2018/benchmark/',image='dibco2018_Dataset.zip',gt='dibco2018-GT.zip',expected=10)},
        split='DIBCO2017: sorted numeric ids, NumPy default_rng20261023 permutation; first4 validation, remaining16 train. HDIBCO2018 all10 reserved_test.',
        patch='Native nonoverlap256, right/bottom white padding; all children inherit page split; coordinates only, not duplicate files.',
        scope='Small denseGT development foundation. Not the paper full training corpus or an independent final benchmark. No test predictions or metrics.',no_training=True)
    assert not (OUT/'dense_benchmark_plan.json').exists();dump(OUT/'dense_benchmark_plan.json',plan)
    downloads=[];rows=[];patches=[];seen={}
    val=set(map(str,np.random.default_rng(20261023).permutation(np.arange(1,21))[:4]))
    for name,c in plan['sources'].items():
        dirs={}
        for kind in ['image','gt']:
            f=DATA/'archives'/c[kind];downloads.append(download(c['base']+c[kind],f))
            dest=DATA/name/kind;extract(f,dest);dirs[kind]=files_by_id(dest)
            print(name,kind,len(dirs[kind]),flush=True)
        assert dirs['image'].keys()==dirs['gt'].keys() and len(dirs['image'])==c['expected']
        for key in sorted(dirs['image'],key=int):
            image,gt=dirs['image'][key],dirs['gt'][key]
            a=np.asarray(Image.open(image).convert('RGB'));g=np.asarray(Image.open(gt).convert('L'))
            assert a.shape[:2]==g.shape and set(np.unique(g)).issubset({0,255})
            assert 0<int((g==0).sum())<g.size
            split='reserved_test' if name=='HDIBCO2018' else ('val' if key in val else 'train')
            page=f'{name}_{key}';content=hashlib.sha256(a.tobytes()+str(a.shape).encode()).hexdigest()
            assert content not in seen,(page,seen.get(content));seen[content]=page
            rows.append(dict(id=page,dataset=name,split=split,image=str(image.relative_to(ROOT)),gt=str(gt.relative_to(ROOT)),
                image_sha256=sha(image),gt_sha256=sha(gt),rgb_content_sha256=content,width=a.shape[1],height=a.shape[0],
                dense_binary_gt=True,polarity='black0=ink,white255=paper',gt_origin='competition organizer',gt_values=[0,255]))
            for y in range(0,a.shape[0],256):
                for x in range(0,a.shape[1],256):
                    patches.append(dict(page=page,split=split,xy=[x,y],valid_wh=[min(256,a.shape[1]-x),min(256,a.shape[0]-y)]))
    dump(OUT/'downloads.json',downloads)
    dump(OUT/'dense_benchmark_manifest.json',dict(plan_sha256=sha(OUT/'dense_benchmark_plan.json'),pages=rows,patches=patches,
        page_counts={s:sum(r['split']==s for r in rows) for s in ['train','val','reserved_test']},
        patch_counts={s:sum(r['split']==s for r in patches) for s in ['train','val','reserved_test']},
        no_duplicate_original_content=True,no_training=True,no_test_predictions=True))
    print('DenseGT manifest complete:',len(rows),'pages,',len(patches),'native patch coordinates',flush=True)


if __name__=='__main__':main()
