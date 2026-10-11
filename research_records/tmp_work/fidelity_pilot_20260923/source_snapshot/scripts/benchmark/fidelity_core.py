"""Fidelity-only references and region metrics; no model or inference GT access."""
from pathlib import Path
import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected readable 3-channel uint8 image: {path}')
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write {path}')


def read_mask(path, shape, foreground=255):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.dtype != np.uint8 or mask.shape != shape:
        raise ValueError(f'Invalid mask dimensions/type: {path}')
    if not np.isin(mask, [0, 255]).all():
        raise ValueError(f'Mask must be binary 0/255: {path}')
    if foreground not in (0, 255):
        raise ValueError('foreground must be 0 or 255')
    return mask == foreground


def load_csc(directory, shape):
    directory = Path(directory)
    specs = {'black': ('02_text_black.png', 0), 'red': ('03_text_red.png', 0),
             'degradation': ('04_degradation.png', 255), 'background': ('05_background.png', 255)}
    return {key: read_mask(directory / name, shape, fg) for key, (name, fg) in specs.items()}


def validate_masks(masks, shape):
    for key, mask in masks.items():
        if mask.dtype != bool or mask.shape != shape:
            raise ValueError(f'{key}: expected bool mask with shape {shape}')


def build_references(image, masks, radius=15, dilate_kernel=5):
    """New recipe fidelity_v1, NOT legacy v15. No tone/gain/residual sweep.

    B0 fills non-background; P0 restores input ink. P1 fills dilated predicted
    degradation plus ink, then restores original ink. Pixels outside fill masks
    are unchanged. CSC errors remain observable rather than hidden using GT.
    """
    validate_masks(masks, image.shape[:2])
    if radius <= 0 or dilate_kernel < 1 or dilate_kernel % 2 == 0:
        raise ValueError('Invalid inpaint/dilation parameters')
    ink = masks['black'] | masks['red']
    simple_hole = ~masks['background'] | ink
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    refined_hole = cv2.dilate(masks['degradation'].astype(np.uint8), kernel) > 0
    refined_hole |= ink
    if simple_hole.all() or refined_hole.all():
        raise ValueError('No known background remains for inpainting')
    b0 = cv2.inpaint(image, simple_hole.astype(np.uint8)*255, radius, cv2.INPAINT_TELEA)
    p0 = b0.copy(); p0[ink] = image[ink]
    p1 = cv2.inpaint(image, refined_hole.astype(np.uint8)*255, radius, cv2.INPAINT_TELEA)
    p1[ink] = image[ink]
    return {'B0': b0, 'P0': p0, 'P1': p1}, {'simple_hole':simple_hole, 'refined_hole':refined_hole}


def evaluation_proxies(clean, stain):
    """Development proxies, not human annotations; never passed to inference."""
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    red = ((hsv[:,:,0] <= 12) | (hsv[:,:,0] >= 168)) & (hsv[:,:,1] >= 100) & (hsv[:,:,2] >= 60)
    text = (gray < 150) & ~red
    # Light-ink candidate region only; paper texture may be included.
    local_paper = cv2.dilate(gray, np.ones((15,15),np.uint8))
    thin = (gray >= 150) & (gray < 200) & (local_paper.astype(float)-gray >= 20) & ~red
    guard = cv2.dilate((text|thin|red).astype(np.uint8), np.ones((5,5),np.uint8)) > 0
    background = ~stain & ~guard
    return {'stain':stain, 'text_proxy':text, 'thin_proxy':thin,
            'red_proxy':red, 'background_proxy':background, 'nonstain':~stain}


def segmentation_scores(prediction, target, valid=None):
    if prediction.dtype != bool or target.dtype != bool or prediction.shape != target.shape:
        raise ValueError('Binary segmentation shape/type mismatch')
    if valid is None:
        valid = np.ones(target.shape, dtype=bool)
    validate_masks({'valid':valid}, target.shape)
    tp=int((prediction & target & valid).sum())
    fp=int((prediction & ~target & valid).sum())
    fn=int((~prediction & target & valid).sum())
    return {'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def evaluate(clean, restored, regions):
    if clean.dtype != np.uint8 or restored.dtype != np.uint8 or clean.shape != restored.shape:
        raise ValueError('Images must have identical uint8 dimensions; no automatic resizing')
    if clean.ndim != 3 or clean.shape[2] != 3 or min(clean.shape[:2]) < 7:
        raise ValueError('Expected BGR image with minimum side >= 7')
    validate_masks(regions, clean.shape[:2])
    g=cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY)
    r=cv2.cvtColor(restored,cv2.COLOR_BGR2GRAY)
    signed=r.astype(np.float32)-g.astype(np.float32)
    error=np.abs(signed)
    out={'psnr_rgb':float(peak_signal_noise_ratio(clean,restored,data_range=255)),
         'ssim_gray':float(structural_similarity(g,r,data_range=255,win_size=7))}
    # Float RGB conversion: sRGB/D65 assumptions of skimage; not calibrated colorimetry.
    delta=deltaE_ciede2000(rgb2lab(cv2.cvtColor(clean,cv2.COLOR_BGR2RGB).astype(np.float32)/255),
                          rgb2lab(cv2.cvtColor(restored,cv2.COLOR_BGR2RGB).astype(np.float32)/255))
    for name, mask in regions.items():
        n=int(mask.sum()); out[f'{name}_n']=n
        out[f'{name}_mae']=float(error[mask].mean()) if n else None
        out[f'{name}_delta_e00']=float(delta[mask].mean()) if n else None
        out[f'{name}_lightening']=float(np.maximum(signed[mask],0).mean()) if n else None
        out[f'{name}_darkening']=float(np.maximum(-signed[mask],0).mean()) if n else None
    # Separate, explicitly proxy-only whole-page threshold diagnostic; no shape claim.
    scores=segmentation_scores(r<150,g<150)
    out.update({f'dark150_proxy_{k}':v for k,v in scores.items()})
    return out
