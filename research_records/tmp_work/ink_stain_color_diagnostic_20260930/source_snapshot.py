"""Check a next-step assumption: do automatic broad and ridge colors differ?

Neither sample group is a material label. Uses only the same three development
pages, excludes target ROIs and ring support, and never fits an editing mask.
"""
from pathlib import Path
import os
import json
import shutil
import cv2
import numpy as np
import probe_spatial_ink_decomposition as spatial

base = spatial.base
ROOT = base.ROOT
OUT = ROOT/'tmp_work/ink_stain_color_diagnostic_20260930'
os.environ['MPLCONFIGDIR'] = str(OUT/'plot_cache')


def main():
    if (OUT/'complete.json').exists():
        raise RuntimeError('Completed color diagnostic is frozen')
    OUT.mkdir(parents=True, exist_ok=True)
    init = json.loads((spatial.OUT/'initializations.json').read_text())
    settings = dict(scales_half_resolution=[8.,16.,32.], ring_radius_sigmas=3.,
        sampling_grid=32, max_candidates=600, minimum_score=.002, minimum_chromatic_contrast=.008,
        ROI_exclusion_native=192, page_border_fraction=.05,
        blob_condition='both Hessian eigenvalues negative; |hi| >= .3 |lo|',
        split='same native 128px tile split as v3; candidates are not material labels',
        purpose='Check separate-color premise after same-color morphology failures; no tuning from this diagnostic',
        pixel_GT=False, neural_training=False)
    dependencies = {str(p.relative_to(ROOT)):base.sha(p) for p in [Path(spatial.__file__), spatial.OUT/'complete.json']}
    for page in init:
        for p in [base.V6/page/'original.png', spatial.OUT/page/'initialization_samples.npz']:
            dependencies[str(p.relative_to(ROOT))] = base.sha(p)
    manifest = dict(settings=settings, source_sha256=base.sha(Path(__file__)),dependencies=dependencies)
    base.dump(OUT/'manifest.json',manifest); shutil.copyfile(__file__,OUT/'source_snapshot.py')
    results = {}
    for page in init:
        original = base.read(base.V6/page/'original.png'); oh, ow = original.shape[:2]
        obs = spatial.colors(spatial.reduced(cv2.medianBlur(original,3)))
        h,w = obs.shape[:2]; sy,sx = oh/h,ow/w
        strength = np.minimum(obs[...,0],obs[...,1]).astype(np.float32)
        score = np.zeros((h,w),np.float32); scale = np.zeros_like(score)
        for sigma in settings['scales_half_resolution']:
            sm = cv2.GaussianBlur(strength,(0,0),sigma)
            xx, yy, xy = [cv2.Sobel(sm,cv2.CV_32F,a,b,ksize=3,scale=sigma*sigma/4.) for a,b in [(2,0),(0,2),(1,1)]]
            disc = np.sqrt((xx-yy)**2+4*xy*xy)
            lo,hi = (xx+yy-disc)/2,(xx+yy+disc)/2
            response = np.maximum(-hi,0.)*np.clip(np.abs(hi)/(np.abs(lo)+1e-9),0,1)
            response[(hi>=0)|(np.abs(hi)<.3*np.abs(lo))] = 0
            take = response > score; score[take] = response[take]; scale[take] = sigma
        px=(np.arange(w)+.5)*sx-.5; py=(np.arange(h)+.5)*sy-.5
        valid=(px[None,:]>=max(.05*ow,192))&(px[None,:]<ow-max(.05*ow,192))& \
              (py[:,None]>=max(.05*oh,192))&(py[:,None]<oh-max(.05*oh,192))
        for target,_,(x,y,rw,rh) in base.ROIS:
            if target==page:
                valid &= ~((px[None,:]>=x-192)&(px[None,:]<x+rw+192)&(py[:,None]>=y-192)&(py[:,None]<y+rh+192))
        points=[]
        for y in range(0,h,32):
            for x in range(0,w,32):
                block=np.where(valid[y:y+32,x:x+32],score[y:y+32,x:x+32],-1)
                dy,dx=np.unravel_index(int(np.argmax(block)),block.shape)
                if block[dy,dx]>=.002:
                    points.append((x+dx,y+dy,float(block[dy,dx])))
        points.sort(key=lambda p:(-p[2],p[1],p[0])); points=points[:600]
        coords=np.asarray(points,float).reshape(-1,3)
        ix,iy=coords[:,0].astype(int),coords[:,1].astype(int)
        ring=[]
        for theta in np.arange(8)*np.pi/4:
            mx=(ix+3*scale[iy,ix]*np.cos(theta)).astype(np.float32)[:,None]
            my=(iy+3*scale[iy,ix]*np.sin(theta)).astype(np.float32)[:,None]
            ring.append(cv2.remap(obs.astype(np.float32),mx,my,cv2.INTER_LINEAR)[:,0])
        contrast=obs[iy,ix]-np.median(ring,axis=0)
        usable=(contrast>.008).all(1)
        native=np.column_stack([px[ix],py[iy]])
        confirm=((native[:,0].astype(int)//128+3*(native[:,1].astype(int)//128))%4)==0
        angles=np.degrees(np.arctan2(contrast[:,1],contrast[:,0]))
        ridge=np.load(spatial.OUT/page/'initialization_samples.npz')
        ridge_est=ridge['usable']&~ridge['confirmation_split']
        ridge_angles=np.degrees(ridge['angle_radians'][ridge_est])
        broad_angles=angles[usable&~confirm]
        def quant(v):
            return np.quantile(v,[.1,.25,.5,.75,.9]).tolist() if len(v) else None
        results[page]=dict(broad_candidates=len(points),broad_estimation_points=int((usable&~confirm).sum()),
            broad_confirmation_points=int((usable&confirm).sum()),
            broad_estimation_angle_quantiles=quant(broad_angles),broad_confirmation_angle_quantiles=quant(angles[usable&confirm]),
            ridge_estimation_points=int(ridge_est.sum()),ridge_estimation_angle_quantiles=quant(ridge_angles),
            angle_quantile_levels=[.1,.25,.5,.75,.9], groups_are_material_labels=False)
        np.savez_compressed(OUT/(page+'_samples.npz'),native_points=native,usable=usable,confirmation_split=confirm,
            contrast=contrast,angle_degrees=angles,scale_analysis=scale[iy,ix])
        display=cv2.resize(original,(int(ow*min(1,1000/oh)),int(oh*min(1,1000/oh))))
        factor=display.shape[0]/oh
        for point in native[usable]:
            cv2.circle(display,tuple(np.rint(point*factor).astype(int)),3,(0,120,255),1)
        base.write(OUT/(page+'_broad_points.png'),display)
        print(page,json.dumps(results[page]),flush=True)
    base.dump(OUT/'summary.json',results)
    for name,expected in dependencies.items():
        assert base.sha(ROOT/name)==expected,name
    assert base.sha(Path(__file__))==manifest['source_sha256']
    base.dump(OUT/'verification.json',dict(status='passed',source_and_dependencies_unchanged=True,
        target_ROIs_and_ring_support_excluded=True,pages=len(results),pixel_GT=False,classification_accuracy_evaluated=False))
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['complete.json','run.log'] and 'plot_cache' not in p.parts]
    base.dump(OUT/'complete.json',dict(status='complete',outputs={str(p.relative_to(OUT)):base.sha(p) for p in sorted(files)}))


if __name__=='__main__':
    main()
