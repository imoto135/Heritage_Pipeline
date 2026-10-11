"""Diagnose T01's saved CSC v1 black mask without modifying pipeline or data."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'v1_audit'
OUT.mkdir(exist_ok=True)
CODE = ROOT / 'modules/ddrm_npj/scripts/csc.py'
spec = importlib.util.spec_from_file_location('audited_csc_v1', CODE)
csc = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = csc
spec.loader.exec_module(csc)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    a = cv2.imread(str(p))
    assert a is not None, p
    return a

paths = {
    'raw': ROOT/'data/split_dataset/train/hnsd00000/hnsd007_023.jpg',
    'ace': ROOT/'data/split_dataset_csc/train/hnsd00000/hnsd007_023.jpg',
    'v1_input': ROOT/'data/split_dataset_csc_mask/train/hnsd00000/hnsd007_023/00_input.png',
    'v2_input': ROOT/'data/split_dataset_csc_mask_v2/train/hnsd00000/hnsd007_023/00_input.png',
    'v1': ROOT/'data/split_dataset_csc_mask/train/hnsd00000/hnsd007_023/02_text_black.png',
    'wolf': ROOT/'tmp_work/classical_black_comparison_20261008/masks/T01/wolf61.png',
    'code': CODE,
    'ace_code': ROOT/'modules/ddrm_npj/scripts/ace.py',
    'generator': ROOT/'scripts/generate_csc_masks.py',
}
hashes = {k: sha(p) for k,p in paths.items()}
raw, ace, saved = read(paths['raw']), read(paths['v1_input']), read(paths['v1'])[:,:,0]
h,w = ace.shape[:2]
assert raw.shape == ace.shape and saved.shape == (h,w)
lab = cv2.cvtColor(ace, cv2.COLOR_BGR2LAB).astype(np.float32)
luv = cv2.cvtColor(ace, cv2.COLOR_BGR2LUV).astype(np.float32)
gray = cv2.cvtColor(ace, cv2.COLOR_BGR2GRAY).astype(np.float32)
L = lab[:,:,0].reshape(-1)
border = np.zeros((h,w), bool)
mh,mw = int(h*.04),int(w*.04)
border[:mh,:]=True; border[-mh:,:]=True
border[:,:mw]=True; border[:,-mw:]=True
nonborder=~border.reshape(-1)
roi=[812,421,450,450]
x,y,rw,rh=roi; sl=np.s_[y:y+rh,x:x+rw]
annotation=json.loads((ROOT/'tmp_work/classical_black_comparison_20261008/annotations_frozen.json').read_text())
patches=next(c['nonink_regions_xywh'] for c in annotation['cases'] if c['id']=='T01')
records={}; masks={}; models=[]

class RecordingGMM(GaussianMixture):
    def fit(self, X, y=None):
        result=super().fit(X,y)
        models.append(self)
        return result

def fit_variant(key, scale=100., n_init=1, source=ace):
    start=time.time()
    lab_src=cv2.cvtColor(source,cv2.COLOR_BGR2LAB).astype(np.float32)
    luv_src=cv2.cvtColor(source,cv2.COLOR_BGR2LUV).astype(np.float32)
    gray_src=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY).astype(np.float32)
    features=np.stack([lab_src[:,:,0]/scale, luv_src[:,:,0]/scale, gray_src/255.],axis=-1).reshape(-1,3)
    def factory(**kwargs):
        kwargs['n_init']=n_init
        return RecordingGMM(**kwargs)
    csc.GaussianMixture=factory
    labels=csc._fit_gmm_clusters(features,n_clusters=3,n_components_pca=3,seed=42,max_samples=50000)
    model=models[-1]
    local_L=lab_src[:,:,0].reshape(-1)
    means=[float(local_L[(labels==c)&nonborder].mean()) if np.any((labels==c)&nonborder) else -1. for c in range(3)]
    black=int(np.argmin(means))
    foreground=(labels==black)&nonborder
    mask=np.where(foreground.reshape(h,w),0,255).astype(np.uint8)
    cv2.imwrite(str(OUT/(key+'.png')),mask)
    clusters=[]
    for c in range(3):
        v=local_L[(labels==c)&nonborder]
        clusters.append(dict(label=c,pixels=len(v),mean_L_u8=means[c],L_quantiles_u8=np.percentile(v,[0,10,50,90,100]).tolist()))
    counts=[]
    for px,py,pw,ph in patches:
        a=mask[y+py:y+py+ph,x+px:x+px+pw]
        counts.append(dict(roi_relative_xywh=[px,py,pw,ph],black_pixels=int((a==0).sum()),pixels=a.size,black_fraction=float((a==0).mean())))
    records[key]=dict(black_label=black,clusters=clusters,gmm_converged=bool(model.converged_),gmm_iterations=int(model.n_iter_),
        gmm_sample_log_likelihood=float(model.lower_bound_),nonink_patch_counts=counts,
        saved_mask_different_pixels=int((mask!=saved).sum()),seconds=time.time()-start,
        luminance_denominator=scale,n_init=n_init,input='raw' if source is raw else 'saved ACE')
    masks[key]=mask
    print(key,json.dumps(records[key],ensure_ascii=False),flush=True)

fit_variant('v1_recomputed')
fit_variant('u8_scale_corrected',scale=255.)
fit_variant('n_init_10',n_init=10)
fit_variant('raw_input_control',source=raw)

color_probe=np.array([[[0,0,0],[255,255,255]]],np.uint8)
probe=dict(input_black_white=color_probe.tolist(),lab=cv2.cvtColor(color_probe,cv2.COLOR_BGR2LAB).tolist(),
           luv=cv2.cvtColor(color_probe,cv2.COLOR_BGR2LUV).tolist())
result=dict(roi_xywh=roi,input_saved_vs_ace_pixels_equal=bool(np.array_equal(ace,read(paths['ace']))),
    input_v1_vs_v2_pixels_equal=bool(np.array_equal(ace,read(paths['v2_input']))),
    mask_polarity='0 = foreground, 255 = background',stage1_variants=records,u8_color_space_probe=probe,
    source_sha256={str(paths[k].relative_to(ROOT)):v for k,v in hashes.items()},
    patch_scope='Previously frozen exploratory non-ink rectangles; not dense ground truth or page accuracy.',
    exact_saved_black_mask_reproduced=bool(np.array_equal(masks['v1_recomputed'],saved)))
patch_input_stats=[]
for px,py,pw,ph in patches:
    stats=dict(roi_relative_xywh=[px,py,pw,ph])
    for name,source in [('raw',raw),('ace',ace)]:
        image_L=cv2.cvtColor(source,cv2.COLOR_BGR2LAB)[:,:,0]
        values=image_L[y+py:y+py+ph,x+px:x+px+pw]
        stats[name]=dict(mean_L_u8=float(values.mean()),L_quantiles_u8=np.percentile(values,[0,10,50,90,100]).tolist())
    patch_input_stats.append(stats)
result['nonink_patch_input_luminance']=patch_input_stats
for k,p in paths.items():assert sha(p)==hashes[k]
result['source_hashes_unchanged']=True
(OUT/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))

font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(s):return ImageFont.truetype(font_path,s)
items=[('原画像',raw[sl][:,:,::-1]),('CSC v1に渡されたACE入力',ace[sl][:,:,::-1]),
       ('保存済み CSC v1（ACE入力）',saved[sl]),('CSC v1（原画像入力・ACEなし）',masks['raw_input_control'][sl]),
       ('輝度の0〜1正規化を補正',masks['u8_scale_corrected'][sl]),('Wolf（原画像入力・前回設定）',read(paths['wolf'])[sl][:,:,0])]
canvas=Image.new('RGB',(972,1782),'#faf9f5'); d=ImageDraw.Draw(canvas)
d.text((18,14),'T01：CSC v1の原因切り分け',font=font(25),fill='#20313b')
d.text((18,53),'同じ範囲・原寸表示。原画像入力のv1と正規化補正は診断用。',font=font(17),fill='#20313b')
d.text((18,79),'黒文字分類は第1段階で確定し、第2段階では再判定しない。',font=font(17),fill='#20313b')
for i,(label,a) in enumerate(items):
    xx=24+(i%2)*474; yy=118+(i//2)*518
    d.text((xx,yy),label,font=font(20),fill='#20313b')
    tile=Image.fromarray(a).convert('RGB')
    canvas.paste(tile,(xx,yy+44))
    assert np.array_equal(np.asarray(canvas)[yy+44:yy+494,xx:xx+450],np.asarray(tile))
d.text((18,1701),'入力だけの変更と正規化だけの変更を比較。既存マスク・実装は変更していない。',font=font(16),fill='#20313b')
d.text((18,1727),'保存済み黒マスクとの全ページ照合、入力照合、クラスタ統計はdiagnostics.json。',font=font(16),fill='#20313b')
d.text((18,1753),'画像の拡縮・輪郭補正なし。診断画像から全ページの精度は算出しない。',font=font(16),fill='#20313b')
canvas.save(OUT/'T01_CSCv1_原因切り分け.png')
print('complete',flush=True)
