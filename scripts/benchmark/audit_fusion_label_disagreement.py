"""Audit whether provisional paper disagreements are remote background or stroke edges."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'tmp_work/makura_v5_exposure_loss_control_20261021'
OUT=EXP/'fusion_label_audit'
PATCH='200017197_00172_0_p00'

def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def gray(p):return np.asarray(Image.open(p).convert('L'))
def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source_path=ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val'/PATCH/'clean.png'
    source=rgb(source_path)
    paper=cv2.morphologyEx(source,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    g=lambda a:cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.int16)
    contrast=g(paper)-g(source)
    source_dark=contrast>=20
    distance=distance_transform_edt(~source_dark)
    remote=(contrast<=5)&(distance>=8)
    label_root=ROOT/'tmp_work/makura_supervised_smoke_20261021/labels_v2'/PATCH
    target=gray(label_root/'paper_target.png');valid=gray(label_root/'valid.png')>0
    provisional_paper=valid&(target==255)
    methods={}
    for folder,names in [('howe_unet_fusion',['howe','unet_020','howe_plus_r3','howe_plus_r8','howe_plus_connected']),
                         ('ridge_gate',['ridge6_r3','ridge10_r3','contrast16_ridge6_r3','contrast16_ridge6_r8'])]:
        for name in names:
            methods[name]=EXP/folder/'synthetic/alpha_0.70'/(name+'.png')
    howe=gray(methods['howe'])==0
    rows=[]
    for name,path in methods.items():
        pred=gray(path)==0
        discordant=pred&provisional_paper
        added=pred&~howe
        rows.append(dict(method=name,prediction_sha256=sha(path),ink_pixels=int(pred.sum()),
            provisional_paper_disagreement=int(discordant.sum()),
            added_ink=int(added.sum()),remote_source_paper_ink=int((pred&remote).sum()),
            beyond_source_dark_1px=int((pred&(distance>1)).sum()),
            beyond_source_dark_2px=int((pred&(distance>2)).sum()),
            beyond_source_dark_3px=int((pred&(distance>3)).sum()),
            added_disagreement_source_contrast_median=float(np.median(contrast[added&provisional_paper])) if (added&provisional_paper).any() else None))
    # Overlay for checking the location of disagreements; red is a provisional label, not expert truth.
    for name in ('howe_plus_r3','ridge6_r3'):
        pred=gray(methods[name])==0
        disagreed=(pred&~howe)&provisional_paper
        overlay=source.copy();overlay[howe]=((overlay[howe].astype(float)*.5)+np.array([0,115,240])*.5).astype(np.uint8)
        overlay[disagreed]=[240,40,50]
        dest=OUT/'figures'/(name+'_原本上の暫定紙ラベル不一致.png')
        dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(overlay).save(dest)
    report=dict(case='alpha_0.70',source_sha256=sha(source_path),provisional_paper_n=int(provisional_paper.sum()),
        source_dark_n=int(source_dark.sum()),remote_source_paper_n=int(remote.sum()),
        source_dark_rule='morphological close 41x41 of clean source RGB minus source grayscale >=20',
        remote_source_paper_rule='clean-source local contrast <=5 and distance >=8 from source-dark proxy',
        source_dark_is_proxy_not_expert_truth=True,source_used_for_diagnostic_not_inference=True,
        rows=rows,formal_accuracy=False)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for row in rows:print(row['method'],'provisional',row['provisional_paper_disagreement'],
        'remote',row['remote_source_paper_ink'],'beyond2',row['beyond_source_dark_2px'])
if __name__=='__main__':main()
