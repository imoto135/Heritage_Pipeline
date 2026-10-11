"""Paired DDRM inpainting on frozen npj/v6 edit masks, at native resolution.

This is an explicit edit-mask adapter, not a reproduction of the historical
npj text-reconstruction pipeline. The repository DDRM kernel/model are reused;
spatial Gaussian fields are shared before each mask's SVD permutation.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/npj_v6_ddrm_edit_comparison_20261001'
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')
os.environ['TORCH_HOME'] = str(OUT / 'torch_cache')
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import dump, read, write, sha, FONT_PATH

MODULE = ROOT / 'modules/ddrm_npj_v2'
CKPT = ROOT / 'modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'
SOURCES = [ROOT / 'tmp_work/npj_failure_collection_20261001',
           ROOT / 'tmp_work/npj_failure_followup_20261001']
SETTINGS = dict(operator='repository efficient_generalized_steps / DDRM inpainting',
    checkpoint=str(CKPT.relative_to(ROOT)), seed=999, timesteps=list(range(0, 1000, 50)),
    beta_start=0.0001, beta_end=0.02, diffusion_steps=1000,
    sigma_0=0.05, etaB=1.0, etaA=0.5, etaC=0.5,
    patch_size=128, overlap=32, context_native_px=64, resize=False,
    input_range='RGB [-1,1], matching model training', precision='float32',
    masking=False, masking_reason='standard inpainting; legacy text-specific xt suppression disabled for both',
    outside_edit='exact original pixels after raw DDRM blend',
    baseline_missing='saved npj v1 04_degradation > 0',
    v6_missing='frozen v6 edit > 0; protect/hold/red all known and preserved',
    reference='same original image for both; no Telea reference/pre-fill',
    noise='same spatial Gaussian fields for observation, initialization and each of 3 step draws',
    empty_edit_patch='exact identity, no sampling required',
    mask_dilation=False, historical_npj_pipeline_reproduction=False,
    new_method=False, mask_accuracy=False, pixel_GT=False)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SpatialRandomTorch:
    """Delegate torch, coupling the kernel's three noise draws in image space.

    Kernel draw order is full V-coordinates, cond_after (all kept or empty for
    binary inpainting), then observed U-coordinates. Draw dimensions otherwise
    depend on the mask, so resetting a seed alone would not couple the pixels.
    """
    def __init__(self, operator, seed, device):
        self.operator = operator
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.device = device
        self.calls = 0

    def __getattr__(self, name):
        return getattr(torch, name)

    def randn_like(self, value):
        spatial = torch.randn((1, 3, 128, 128), generator=self.generator,
                              device=self.device, dtype=value.dtype)
        position = self.calls % 3
        self.calls += 1
        if position == 0:
            mapped = self.operator.Vt(spatial)
        elif position == 1 and value.numel() == 0:
            mapped = value.clone()
        else:
            mapped = self.operator.H(spatial)
        assert mapped.shape == value.shape, (position, mapped.shape, value.shape)
        return mapped


def patch_seed(page, x, y):
    value = f'{SETTINGS["seed"]}:{page}:{x}:{y}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little') % (2**63 - 200)


class DDRM:
    def __init__(self, device):
        self.device = torch.device(device)
        torch.set_num_threads(4)
        torch.cuda.set_device(self.device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        sys.path.insert(0, str(MODULE))
        from codes.openai.unet import UNetModel
        self.kernel = load_module('paired_ddrm_kernel', OUT / 'snapshots/denoising.py')
        self.kernel.tqdm = lambda iterable: iterable
        self.svd = load_module('paired_ddrm_svd', OUT / 'snapshots/svd_replacement.py')
        self.model = UNetModel(in_channels=3, model_channels=64, out_channels=3,
                              num_res_blocks=2, attention_resolutions=[16, 8])
        state = torch.load(CKPT, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.to(self.device).eval()
        self.beta = torch.from_numpy(np.linspace(0.0001, 0.02, 1000)).float().to(self.device)
        self.args = SimpleNamespace(masking=False, device=self.device)
        self.model_calls = 0
        self.self_check()

    def operator(self, missing):
        pixels = torch.from_numpy(np.flatnonzero(missing).copy()).long().to(self.device)
        indices = torch.cat([pixels * 3, pixels * 3 + 1, pixels * 3 + 2])
        return self.svd.Inpainting(3, 128, indices, self.device)

    def self_check(self):
        field = torch.arange(3*128*128, device=self.device).float().reshape(1,3,128,128)
        mask = np.zeros((128,128), bool); mask[15:50,25:65] = True
        first = self.operator(mask)
        mask[70:90,50:80] = True
        second = self.operator(mask)
        for operator in [first, second]:
            assert torch.equal(operator.V(operator.Vt(field)), field.reshape(1,-1))
            expected = field.permute(0,2,3,1).reshape(1,-1)[:, operator.kept_indices]
            assert torch.equal(operator.H(field), expected)
        # The full and known-coordinate draws represent the same physical field.
        for operator in [first, second]:
            proxy = SpatialRandomTorch(operator, 1234, self.device)
            full = proxy.randn_like(torch.empty(1,3*128*128,device=self.device))
            generator = torch.Generator(device=self.device).manual_seed(1234)
            spatial = torch.randn((1,3,128,128),generator=generator,device=self.device)
            assert torch.equal(operator.V(full).reshape_as(spatial), spatial)
            proxy.randn_like(torch.empty(1,0,device=self.device))
            known = proxy.randn_like(torch.empty(1,len(operator.kept_indices),device=self.device))
            torch.randn((1,3,128,128),generator=generator,device=self.device)
            expected = operator.H(torch.randn((1,3,128,128),generator=generator,device=self.device))
            assert torch.equal(known, expected)
        dump(OUT / 'operator_verification.json', dict(status='passed',
             svd_roundtrip=True, known_channel_indices=True, spatial_noise_coupling=True))

    @torch.inference_mode()
    def sample(self, image, missing, seed):
        if not missing.any():
            return image.copy()
        operator = self.operator(missing)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()
        original = torch.from_numpy(rgb).float().unsqueeze(0).to(self.device) / 127.5 - 1
        generator = torch.Generator(device=self.device).manual_seed(seed)
        observation_noise = torch.randn(original.shape,generator=generator,device=self.device)
        initial_noise = torch.randn(original.shape,generator=generator,device=self.device)
        observation = operator.H(original) + SETTINGS['sigma_0'] * operator.H(observation_noise)
        initial = operator.Vt(initial_noise).reshape_as(original)
        proxy = SpatialRandomTorch(operator, seed+101, self.device)
        self.kernel.torch = proxy
        try:
            steps, _, _ = self.kernel.efficient_generalized_steps(initial, SETTINGS['timesteps'],
                self.model, self.beta, operator, observation, SETTINGS['sigma_0'], self.device,
                etaB=1., etaA=.5, etaC=.5, args=self.args)
        finally:
            self.kernel.torch = torch
        assert proxy.calls == 60
        result = steps[-1]
        assert torch.isfinite(result).all()
        self.model_calls += 20
        rgb = ((result[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render(images, path, title, note):
    h,w = images[0].shape[:2]; scale = min(620/w,1200/h)
    dw,dh = round(w*scale),round(h*scale)
    canvas = Image.new('RGB',(3*dw+104,dh+252),(250,248,243))
    draw=ImageDraw.Draw(canvas); large=ImageFont.truetype(FONT_PATH,26)
    small=ImageFont.truetype(FONT_PATH,21)
    draw.text((28,16),title,font=large,fill='black')
    draw.text((28,55),'同一DDRM・同一重み・同一ノイズ。編集マスクだけを変更。',font=small,fill=(45,45,45))
    for i,(im,label) in enumerate(zip(images,['原画像','npj編集マスク → DDRM','v6編集マスク → DDRM'])):
        x=28+i*(dw+24);draw.text((x,96),label,font=large,fill='black')
        thumb=cv2.resize(im,(dw,dh),interpolation=cv2.INTER_NEAREST)
        canvas.paste(Image.fromarray(cv2.cvtColor(thumb,cv2.COLOR_BGR2RGB)),(x,136))
    draw.text((28,dh+154),'編集候補を欠損領域に接続したDDRM。元npj全工程の再現ではありません。',font=small,fill=(45,45,45))
    draw.text((28,dh+188),note,font=small,fill=(45,45,45))
    canvas.save(path)


def blend_weight():
    axis=np.maximum(np.hanning(128),.01).astype(np.float32)
    return axis[:,None]*axis[None,:]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    dependencies={};specs=[]
    for source in SOURCES:
        for name in ['baseline_review.json','comparison_manifest.json','comparison_complete.json']:
            dependencies[str((source/name).relative_to(ROOT))]=sha(source/name)
        manifest=json.loads((source/'comparison_manifest.json').read_text())
        for rel,digest in manifest['dependencies'].items():
            assert sha(ROOT/rel)==digest,rel
        complete=json.loads((source/'comparison_complete.json').read_text())['outputs']
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            edit=source/'v6_masks'/page['page']/'edit.png'
            assert sha(edit)==complete[str(edit.relative_to(source))]
            for path in [ROOT/page['original'],ROOT/page['npj_degradation'],edit]:
                dependencies[str(path.relative_to(ROOT))]=sha(path)
            specs.extend(dict(source=source,page=page,roi=roi) for roi in page['rois'])
    for path in [CKPT,Path(__file__),MODULE/'functions/denoising.py',MODULE/'functions/svd_replacement.py',
                 ROOT/'modules/stain_removal/src/train_common.py']:
        dependencies[str(path.relative_to(ROOT))]=sha(path)
    for path in sorted((MODULE/'codes/openai').rglob('*.py')):
        dependencies[str(path.relative_to(ROOT))]=sha(path)
    manifest=dict(settings=SETTINGS,dependencies=dependencies,pages=11,regions=15,
                  selection='all 15 frozen ROIs, no outcome selection')
    path=OUT/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text())==manifest,'Run settings/dependencies changed; use a new output directory'
    else:
        dump(path,manifest)
    (OUT/'snapshots').mkdir(exist_ok=True)
    for name in ['denoising.py','svd_replacement.py']:
        shutil.copyfile(MODULE/'functions'/name,OUT/'snapshots'/name)
    return specs,manifest


def run(device):
    specs,manifest=prepare(); runner=DDRM(device)
    weight=blend_weight();rows=[];start=time.monotonic(); repeat_checked=False
    for spec in specs:
        page,roi,source=spec['page'],spec['roi'],spec['source']; id=roi['id']
        dest=OUT/'native'/id;cache=dest/'complete.json'
        if cache.exists():
            data=json.loads(cache.read_text())
            for rel,digest in data['outputs'].items(): assert sha(dest/rel)==digest
            rows.append(data['diagnostics']);print('Verified cached ROI:',id,flush=True);continue
        begin=time.monotonic(); original=read(ROOT/page['original'])
        npjmask=read(ROOT/page['npj_degradation'],True);v6mask=read(source/'v6_masks'/page['page']/'edit.png',True)
        H,W=original.shape[:2];x,y,w,h=roi['xywh'];pad=64
        x0,y0=max(0,x-pad),max(0,y-pad);x1,y1=min(W,x+w+pad),min(H,y+h+pad)
        context=original[y0:y1,x0:x1].copy();ch,cw=context.shape[:2]
        coords=[(yy,xx) for yy in range(0,ch,96) for xx in range(0,cw,96)]
        padded=cv2.copyMakeBorder(context,0,128,0,128,cv2.BORDER_REFLECT_101)
        accum={m:np.zeros((ch+128,cw+128,3),np.float32) for m in ['npj','v6']}
        masks={m:np.pad(a[y0:y1,x0:x1],((0,128),(0,128))) for m,a in [('npj',npjmask),('v6',v6mask)]}
        wacc=np.zeros((ch+128,cw+128),np.float32);sampled={m:0 for m in masks}
        for index,(yy,xx) in enumerate(coords):
            patch=padded[yy:yy+128,xx:xx+128];seed=patch_seed(page['page'],x0+xx,y0+yy)
            for method in ['npj','v6']:
                missing=masks[method][yy:yy+128,xx:xx+128]
                result=runner.sample(patch,missing,seed)
                sampled[method]+=int(missing.any())
                if not repeat_checked and missing.any():
                    assert np.array_equal(result,runner.sample(patch,missing,seed)), 'Same-mask deterministic repeat failed'
                    dump(OUT/'repeat_verification.json',dict(status='passed',same_mask_same_noise_exact_pixels=True,
                        page=page['page'],patch_xy=[x0+xx,y0+yy]))
                    repeat_checked=True
                accum[method][yy:yy+128,xx:xx+128]+=result.astype(np.float32)*weight[:,:,None]
            wacc[yy:yy+128,xx:xx+128]+=weight
            if (index+1)%20==0:
                print(f'{id}: patches {index+1}/{len(coords)} ({time.monotonic()-begin:.1f}s)',flush=True)
        assert (wacc[:ch,:cw]>0).all()
        metrics={};results={};ys=slice(y-y0,y-y0+h);xs=slice(x-x0,x-x0+w)
        input_crop=context[ys,xs];write(dest/'original.png',input_crop)
        for method in ['npj','v6']:
            raw=np.clip(np.rint(accum[method][:ch,:cw]/wacc[:ch,:cw,None]),0,255).astype(np.uint8)
            edit=masks[method][:ch,:cw];final=raw.copy();final[~edit]=context[~edit]
            results[method]=final[ys,xs];crop_edit=edit[ys,xs]
            changed=np.any(results[method]!=input_crop,axis=2)
            assert not np.any(changed&~crop_edit)
            metrics[method]=dict(edit_pixels=int(crop_edit.sum()),changed_pixels=int(changed.sum()),
                nonedit_changed_pixels=0,sampled_patches=sampled[method],
                raw_nonedit_changed_pixels=int(np.any(raw[ys,xs]!=input_crop,axis=2)[~crop_edit].sum()))
            for name,value in [(method+'.png',results[method]),(method+'_raw.png',raw[ys,xs]),
                               (method+'_edit.png',crop_edit),(method+'_changed.png',changed)]:
                write(dest/name,value)
                assert np.array_equal(read(dest/name,value.dtype==bool),value)
        (OUT/'comparisons').mkdir(exist_ok=True)
        note=f'矩形内の変更画素：npj {metrics["npj"]["changed_pixels"]:,} / v6 {metrics["v6"]["changed_pixels"]:,}（精度・除去率ではありません）'
        render([input_crop,results['npj'],results['v6']],OUT/'comparisons'/(id+'.png'),
               roi['title']+' · '+page['page'],note)
        row=dict(id=id,page=page['page'],book=page['book'],xywh=roi['xywh'],
            context_xywh=[x0,y0,x1-x0,y1-y0],baseline_failure=roi['baseline_failure'],
            failure_type=roi['failure_type'],total_patches=len(coords),elapsed_seconds=time.monotonic()-begin,
            **{method+'_'+key:value for method,metric in metrics.items() for key,value in metric.items()})
        dump(cache,dict(diagnostics=row,outputs={p.name:sha(p) for p in dest.glob('*.png')}))
        rows.append(row);dump(OUT/'progress.json',dict(completed_rois=len(rows),total_rois=15,
                            elapsed_seconds=time.monotonic()-start,latest=id))
        print(f'Completed DDRM ROI {len(rows)}/15: {id}; sampled {sampled}',flush=True)
    assert len(rows)==15
    dump(OUT/'diagnostics.json',rows)
    with (OUT/'diagnostics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for rel,digest in manifest['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',DDRM=True,Telea_used=False,
        pages=11,regions=15,source_and_code_hashes_checked=True,native_pixels=True,
        outside_edit_unchanged=True,spatial_noise_coupled=True,pixel_GT=False,
        historical_npj_pipeline_reproduction=False,model_forward_calls_this_run=runner.model_calls,
        device=str(runner.device),torch_version=str(torch.__version__),cv2_version=cv2.__version__,
        elapsed_seconds=time.monotonic()-start))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*.png')}))
    print('DDRM comparison completed: 11 pages / 15 frozen ROIs',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();run(args.device)
