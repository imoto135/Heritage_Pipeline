"""Compare saved local masks and intermediates against exported MATLAB files."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import cv2
from PIL import Image

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from audit_input_parity import edge_map


def main():
    ref=HERE/'matlab_reference'
    frozen=json.loads((HERE/'frozen_probes.json').read_text())
    rows=[]
    for source in frozen['source_files'].values():
        page=source['page']
        needed=[ref/(page+'_'+name+'.png') for name in ('gray','edges','fixed','auto')]
        if not all(p.exists() for p in needed):
            raise FileNotFoundError(f'MATLAB export incomplete for {page}: {needed}')
        path=ROOT/source['original']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
        rgb=np.asarray(Image.open(path).convert('RGB'))
        local_gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        local_edges=edge_map(local_gray)
        arrays={
            'gray':local_gray,
            'edges':local_edges,
            'fixed':np.asarray(Image.open(SOURCE/'masks'/page/'fixed.png').convert('L')),
            'auto':np.asarray(Image.open(SOURCE/'masks'/page/'auto.png').convert('L')),
        }
        for name,local in arrays.items():
            matlab=np.asarray(Image.open(ref/(page+'_'+name+'.png')).convert('L'))
            if name=='edges':matlab=matlab>0
            assert matlab.shape==local.shape,(page,name,matlab.shape,local.shape)
            diff=matlab!=local
            rows.append(dict(page=page,stage=name,pixels=int(diff.size),different=int(diff.sum()),
                             fraction=float(diff.mean()),max_abs_difference=(int(np.max(np.abs(matlab.astype(int)-local.astype(int)))) if name=='gray' else None)))
        parameter_path=ref/(page+'_auto_parameters.txt')
        if not parameter_path.exists():raise FileNotFoundError(parameter_path)
        exported=dict(line.split('=',1) for line in parameter_path.read_text().splitlines())
        local=json.loads((SOURCE/'results'/(page+'.json')).read_text())
        rows.append(dict(page=page,stage='auto_parameters',
                         matlab_c=float(exported['c']),local_c=float(local['c']),
                         matlab_thi=float(exported['thi']),local_thi=float(local['thi']),
                         parameters_match=bool(np.isclose(float(exported['c']),local['c']) and
                                               np.isclose(float(exported['thi']),local['thi']))))
    output=dict(scope='Actual MATLAB runtime exports compared to saved local full pages',rows=rows)
    (HERE/'matlab_parity_comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    for row in rows:print(row)


if __name__=='__main__':main()
