"""Read-only source/shape checks of the author's RA-GAN implementation.

No optimizer step, saved learned weights or quality comparison is performed.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
REF=ROOT/"tmp_work/ra_gan_reference_20261021/source"
OUT=REF.parent


def main():
    sys.path.insert(0,str(OUT/"dependencies"))
    sys.path.insert(0,str(REF))
    from models.Generator import RA_GAN
    from models.Conv import GGC
    from models.Discriminator import Discriminator
    torch.manual_seed(20261021)
    probe=GGC(3,4,kernel_size=3,stride=1,padding=1)
    optimizer=torch.optim.Adam(probe.parameters(),lr=3e-5)
    registered_before={n:id(p) for n,p in probe.named_parameters()}
    optimizer_ids={id(p) for group in optimizer.param_groups for p in group["params"]}
    x=torch.rand(1,3,16,16,requires_grad=True)
    y=probe(x);y.mean().backward()
    registered_after={n:id(p) for n,p in probe.named_parameters()}
    replaced=[n for n in registered_before if registered_before[n]!=registered_after[n]]
    missing=[n for n,p in probe.named_parameters() if id(p) not in optimizer_ids]
    audit=dict(replaced_parameter_names=replaced,new_parameters_missing_from_optimizer=missing,
        replaced_weight_has_gradient=probe.diff_conv.weight.grad is not None,
        explanation="GGC.forward installs a new nn.Parameter in diff_conv.weight after optimizer construction. Forward/backward can run while optimizer still references the old weight.",
        resolution="Not patched here. Preserve author source and verify a documented fix before actual training.")
    assert "diff_conv.weight" in replaced and "diff_conv.weight" in missing
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=RA_GAN(5).to(device).eval();discriminator=Discriminator(4).to(device).eval()
    # Real generated input, using native center crops rather than downsampling.
    pilot=ROOT/"tmp_work/makura_supervised_pilot_20261021"
    results=sorted((pilot/"page_results").glob("*.json"));assert results
    row=json.loads(results[0].read_text())["records"]
    sample=next(r for r in row if r["damage_family"]=="stain_overlap" and r["damage_strength"]==.5)
    rgb=np.asarray(Image.open(ROOT/sample["input"]).convert("RGB"))
    shapes=[]
    for size in (256,512):
        h,w=rgb.shape[:2];q=rgb[(h-size)//2:(h+size)//2,(w-size)//2:(w+size)//2]
        gray=np.asarray(Image.fromarray(q).convert("L"))
        _,otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        sobel=np.rint(np.clip(cv2.magnitude(gx,gy),0,255)).astype(np.uint8)
        image=torch.from_numpy(q.copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
        aux=[torch.from_numpy(v.copy()).unsqueeze(0).unsqueeze(0).float().to(device)/255 for v in (otsu,sobel)]
        with torch.inference_mode():
            pred=model(image,*aux);judgment=discriminator(pred,image)
        assert pred.shape==(1,1,size,size) and torch.isfinite(pred).all() and torch.isfinite(judgment).all()
        shapes.append(dict(native_crop=size,generator_output=list(pred.shape),discriminator_output=list(judgment.shape),
                           prediction_range=[float(pred.min()),float(pred.max())]))
    commit=subprocess.check_output(["git","-C",str(REF),"rev-parse","HEAD"],text=True).strip()
    hashes={str(p.relative_to(REF)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(REF.rglob("*.py"))}
    metadata=dict(source="https://github.com/bzlmh/RA-GAN",commit=commit,code_sha256=hashes,
        torch=torch.__version__,device=str(device),input_contract="RGB + input-derived Otsu + input-derived clipped Sobel magnitude; 5 channels total",
        label_polarity="Author dataset loads L/255: paper=1, black ink=0",
        sample=sample["input"],native_shapes=shapes,optimizer_parameter_issue=audit,
        augmentation_review="Author loader rotates RGB/Otsu/GT but not Sobel in training; paired geometric transforms and input-derived auxiliaries must be synchronized in a local adapter.",
        model_parameters=sum(p.numel() for p in model.parameters()),trained=False,optimizer_steps=0)
    (OUT/"reference_audit.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:metadata[k] for k in ["commit","native_shapes","optimizer_parameter_issue","trained"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
