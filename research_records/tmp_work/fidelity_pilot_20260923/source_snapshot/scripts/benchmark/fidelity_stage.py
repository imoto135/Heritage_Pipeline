"""Isolated inference worker. Only observed image, ACE, masks, references and weights.
No clean target, stain GT, or evaluation regions are accepted by this CLI.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import cv2
import numpy as np
from fidelity_core import read_image, write_image, load_csc, build_references
ROOT=Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m
    spec.loader.exec_module(m)
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','ddnm','ddrm'])
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--masks',type=Path,required=True,help='Per-page CSC directory')
    p.add_argument('--ace',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--ckpt',type=Path)
    p.add_argument('--seed',type=int,default=999)
    p.add_argument('--csc-seed',type=int,default=42)
    p.add_argument('--rescue',choices=['on','off'],default='on')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    start=time.monotonic()
    image=read_image(args.input)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare':
        if args.ace is None: p.error('prepare requires --ace')
        ace=read_image(args.ace)
        if ace.shape != image.shape: raise ValueError('ACE dimensions differ')
        csc=load_module('fidelity_csc',ROOT/'modules/ddrm_npj_v3/scripts/csc.py')
        result=csc.color_space_clustering(ace,seed=args.csc_seed,ink_rescue_by_lightness=args.rescue=='on')
        masks={'02_text_black.png':255-result.black_text_mask,
               '03_text_red.png':255-result.red_text_mask,
               '04_degradation.png':result.degradation_mask,
               '05_background.png':result.background_mask}
        for name,mask in masks.items(): write_image(args.masks/name,mask)
        loaded=load_csc(args.masks,image.shape[:2])
        black_fraction=float(loaded['black'].mean())
        if not 0 < black_fraction < .8: raise ValueError(f'Unexpected black fraction {black_fraction}')
        refs,holes=build_references(image,loaded)
        for name,ref in refs.items(): write_image(args.output/f'{name}.png',ref)
        for name,mask in holes.items(): write_image(args.output/f'{name}.png',mask.astype(np.uint8)*255)
        meta={'recipe':'fidelity_v1','black_fraction':black_fraction,
              'csc_seed':args.csc_seed,'lightness_rescue':args.rescue,
              'mask_fractions':{k:float(v.mean()) for k,v in loaded.items()},
              'mask_overlap_pixels':int((sum(v.astype(np.uint8) for v in loaded.values())>1).sum()),
              'inpaint_radius':15,'dilation_kernel':5,'tone_normalization':False,'text_gain':1.0,
              'residual_sweep':False}
    else:
        if args.reference is None or args.ckpt is None: p.error('inference requires --reference and --ckpt')
        reference=read_image(args.reference)
        if reference.shape!=image.shape: raise ValueError('Reference dimensions differ')
        masks=load_csc(args.masks,image.shape[:2])
        import torch
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.cuda.reset_peak_memory_stats(args.device)
        if args.stage=='ddnm':
            module=load_module('fidelity_ddnm',ROOT/'modules/ddnm_npj/scripts/pipeline_ddnm.py')
            model=module.load_model(str(args.ckpt),args.device,num_res_blocks=2)
            # Reset after model construction: seed refers to sampling, not initialization.
            torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            result=module.infer_page(image,model,module.get_beta_schedule(),args.device,
                                    patch_size=128,overlap=32,bg_image_bgr=reference,
                                    mask01=masks['black'].astype(np.uint8),timesteps=20,
                                    sigma_0=.05,eta=.5,use_shape_guidance=False,
                                    final_correction=False,full_kept=True)
            write_image(args.output/'raw.png',result)
            result[masks['red']]=image[masks['red']]
        else:
            os.chdir(ROOT/'modules/ddrm_npj_v2')
            module=load_module('fidelity_ddrm',ROOT/'modules/ddrm_npj_v2/scripts/pipeline_ddrm.py')
            runner_args=module._Args(str(args.ckpt),args.device,str(args.reference))
            runner_args.seed=args.seed
            runner=module.Diffusion(runner_args,module._Config())
            # Eager load, then reset sampling RNG for pairing across references.
            runner._get_model()
            torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            native=args.output/'native'
            module.process_page(str(args.input),str(native),runner,str(args.output/'tmp'),128,32,
                                mask_dir=str(args.masks.parent))
            result=read_image(native/args.input.stem/'result'/f'{args.input.stem}.png')
        if result.shape!=image.shape: raise ValueError('Output dimension mismatch')
        write_image(args.output/'output.png',result)
        meta={'seed':args.seed,'device':args.device,'peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved(args.device),
              'timesteps':20,'eta':.5,'sigma_0':.05,'patch_size':128,'overlap':32,
              'red_overlay':'CSC red only, original input; shared across stages'}
    meta.update(stage=args.stage,elapsed_seconds=time.monotonic()-start)
    (args.output/'stage.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)

if __name__=='__main__': main()
