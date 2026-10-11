"""Measure a documented grayscale input difference; this is not MATLAB parity."""
from pathlib import Path
import hashlib
import json
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import canny

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
ROOT=HERE.parents[2]
COEFF=np.array([0.298936021293775,0.587043074451121,0.114020904255103])


def edge_map(gray):
    g=gray.astype('float32')/255
    sm=ndi.gaussian_filter(g,.6,mode='reflect',truncate=4)
    mag=np.hypot(ndi.sobel(sm,axis=0,mode='reflect'),ndi.sobel(sm,axis=1,mode='reflect'))
    maximum=float(mag.max())
    return canny(g,sigma=.6,low_threshold=.1*maximum,
                 high_threshold=.4*maximum,mode='reflect')


def main():
    frozen=json.loads((HERE/'frozen_probes.json').read_text())
    rows=[]
    for case,source in frozen['source_files'].items():
        path=ROOT/source['original']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
        rgb=np.asarray(Image.open(path).convert('RGB'))
        local=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        # A formula proxy using MathWorks' published coefficients and nearest
        # integer. It is not a MATLAB runtime result or an asserted exact cast.
        proxy=np.clip(np.rint(rgb.astype('float64')@COEFF),0,255).astype('uint8')
        different=local!=proxy
        e_local=edge_map(local);e_proxy=edge_map(proxy)
        rows.append(dict(case=case,page=source['page'],pixels=int(local.size),
                         grayscale_different=int(different.sum()),
                         grayscale_max_abs_difference=int(np.abs(local.astype(int)-proxy.astype(int)).max()),
                         local_edge_pixels=int(e_local.sum()),proxy_edge_pixels=int(e_proxy.sum()),
                         edge_disagreement_pixels=int((e_local!=e_proxy).sum())))
        print(rows[-1],flush=True)
    result=dict(scope='Input sensitivity to a published coefficient formula proxy; NOT MATLAB rgb2gray or canny_old runtime parity',
                coefficient_source='https://www.mathworks.com/help/matlab/ref/rgb2gray.html',
                edge_source='https://www.mathworks.com/help/images/ref/edge.html',rows=rows)
    (HERE/'input_parity_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
