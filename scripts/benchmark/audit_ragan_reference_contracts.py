"""Execute source-level loader/loss contracts on dense organizerGT. No training."""
import ast
import importlib.util
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/ragan_reproduction_foundation_20261023'
REF=ROOT/'tmp_work/ra_gan_reference_20261021/source'
sys.path.insert(0,str(REF))
from compare_v5_aligned_fade_control import dump,sha,save
from apply_ragan_real_regions import panel
from loss.loss import Loss_Doc
from ragan_pilot_support import supervised_loss


def load_reference():
    spec=importlib.util.spec_from_file_location('pinned_author_loader',REF/'data/Binarizationdataloder.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    assert not (OUT/'contracts.json').exists()
    torch.set_num_threads(2)
    manifest=json.loads((OUT/'dense_benchmark_manifest.json').read_text())
    page=next(p for p in manifest['pages'] if p['split']=='train')
    raw=np.asarray(Image.open(ROOT/page['image']).convert('RGB'));gt=np.asarray(Image.open(ROOT/page['gt']).convert('L'))
    patches=[p for p in manifest['patches'] if p['page']==page['id']]
    chosen=max(patches,key=lambda p:int((gt[p['xy'][1]:p['xy'][1]+256,p['xy'][0]:p['xy'][0]+256]==0).sum()))
    x,y=chosen['xy'];h,w=raw.shape[:2];a=np.full((256,256,3),255,np.uint8);g=np.full((256,256),255,np.uint8)
    hh,ww=min(256,h-y),min(256,w-x);a[:hh,:ww]=raw[y:y+hh,x:x+ww];g[:hh,:ww]=gt[y:y+hh,x:x+ww]
    gray=np.asarray(Image.fromarray(a).convert('L'));otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    edge=np.rint(np.clip(cv2.magnitude(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3),cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)),0,255)).astype(np.uint8)
    fixture=OUT/'loader_fixture';name='audit_0_0.png'
    for kind,arr in [('img',a),('gt',g),('ostu',otsu),('sobel',edge)]:
        d=fixture/kind;d.mkdir(parents=True,exist_ok=True);Image.fromarray(arr).save(d/name)
    source_files={str((REF/f).relative_to(ROOT)):sha(REF/f) for f in ['train.py','models/Generator.py','models/ga.py','models/Conv.py','models/Discriminator.py','loss/loss.py','data/Binarizationdataloder.py']}
    dump(OUT/'contract_plan.json',dict(script_sha256=sha(__file__),source_files=source_files,
        manifest_sha256=sha(OUT/'dense_benchmark_manifest.json'),fixture_page=page['id'],fixture_xy=[x,y],
        selection='First sorted training page; most-ink native256patch, no model predictions',angle=7.5,
        tests=['Actual official loader no-augmentation identity','Force only official7.5degree rotation, inspect every channel',
               'Execute official train.py discriminator-loss assignment AST on controlled logits',
               'Author dense losses versus local full-valid masked losses'],no_training=True,no_optimizer=True))
    loader=load_reference()
    identity=loader.DocData(str(fixture/'img'),(256,256),training=False,only_one=True)[0]
    expected=[a.transpose(2,0,1),otsu[None],edge[None],g[None]]
    for tensor,array in zip(identity[:4],expected):assert np.array_equal(tensor.numpy(),array.astype(np.float32)/255)
    ds=loader.DocData(str(fixture/'img'),(256,256),training=True,only_one=True)
    # Exact author branches: take rotation, angle7.5, skip photometric jitter.
    with patch.object(loader.random,'random',side_effect=[0.,.875,1.]):rot=ds[0]
    angle=7.5;rgb_rot=np.asarray(loader.random_rotate(Image.fromarray(a),0,angle))
    gt_rot=np.asarray(loader.random_rotate(Image.fromarray(g),(255,255,255),angle))
    edge_rot=np.asarray(loader.random_rotate(Image.fromarray(edge),0,angle))
    assert np.array_equal(rot[0].numpy(),rgb_rot.transpose(2,0,1).astype(np.float32)/255)
    assert np.array_equal(rot[3][0].numpy(),gt_rot.astype(np.float32)/255)
    assert np.array_equal(rot[2][0].numpy(),edge.astype(np.float32)/255)
    mismatch=int((edge_rot!=edge).sum());assert mismatch>0
    contract=dict(no_augmentation_identity=True,rotation_angle=angle,
        rgb_matches_rotated=True,gt_matches_rotated=True,sobel_matches_unrotated=True,
        sobel_pixels_different_from_same_rotation=mismatch,
        sobel_mean_abs_difference=float(np.abs(edge.astype(np.float32)-edge_rot).mean()),
        rotated_gt_unique_values=len(np.unique(gt_rot)),
        limitation='Fixture uses a training crop. Corrected Sobel panel is the same geometric rotation of precomputed Sobel, not recomputation after photometric jitter. GT softening may be intentional interpolation, not alone proof of an error.')
    overlay=rgb_rot.copy();overlay[edge>96]=[255,0,70];overlay[edge_rot>96]=[0,180,255]
    panel(OUT/'figures/公開ローダーの回転とSobel.png','公開ローダー実行 / 同じ7.5度回転・学習なし',[
        ('原画像',a),('ローダーRGB回転後',rgb_rot),('教師回転後',gt_rot),
        ('実際のSobel：未回転',edge),('同じ角度を適用したSobel',edge_rot),('赤=未回転 青=回転',overlay)])
    # Execute the actual assignments, without importing train.py (which trains).
    tree=ast.parse((REF/'train.py').read_text());nodes=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['D_real_c','D_fake_c','D_loss'] for t in node.targets):nodes.append(node)
    nodes.sort(key=lambda n:n.lineno);assert len(nodes)==5
    real_logit=torch.tensor([[-3.,0.,3.]],requires_grad=True);fake_logit=torch.tensor([[-3.,0.,3.]],requires_grad=True)
    rs,fs=real_logit.sigmoid(),fake_logit.sigmoid();answers=iter([rs,fs])
    scope=dict(discriminator=lambda *a:next(answers),gt=torch.zeros(1),inputImg=torch.zeros(1),bin_out=torch.zeros(1),F=F,torch=torch)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(REF/'train.py'),'exec'),scope)
    author=scope['D_loss'];equivalent=2-rs.mean()+fs.mean();assert torch.equal(author,equivalent)
    dr,df=torch.autograd.grad(author,(rs,fs));assert torch.allclose(dr,torch.full_like(dr,-1/3)) and torch.allclose(df,torch.full_like(df,1/3))
    unbounded=F.relu(1-real_logit).mean()+F.relu(1+fake_logit).mean()
    loss_contract=dict(author_assignments=[ast.unparse(n) for n in nodes],author_loss=float(author),
        equivalent_on_sigmoid_range='2-mean(Dreal)+mean(Dfake)',unbounded_elementwise_hinge_same_logits=float(unbounded),
        gradients_wrt_real_scores=dr.tolist(),gradients_wrt_fake_scores=df.tolist(),
        interpretation='For scores strictly between0and1, both hinge terms remain active. This differs from standard hinge on unconstrained scores; it is not proof that this code cannot learn.')
    p=torch.linspace(.05,.95,256).reshape(1,1,16,16);t=torch.ones_like(p);t[:,:,5:9,4:12]=0
    author_parts=Loss_Doc(None)(torch.zeros(1,3,16,16),p,t)
    author_total=10*author_parts[0]+sum(author_parts[1:]);ours=supervised_loss(p,t,torch.ones_like(t))
    assert torch.allclose(author_parts[0],(p-t).abs().mean()) and torch.allclose(author_parts[1],F.binary_cross_entropy(p,t))
    dense=dict(author_l1_bce_dice_jaccard=[float(z) for z in author_parts],author_total=float(author_total),local_total=float(ours),
        same_l1_bce=True,total_not_equivalent=not torch.allclose(author_total,ours),
        reason='Author Dice uses squared probabilities and per-image averaging on paper=1; local Dice uses ink=1-p,linear denominator,batch-global masked sums. Masking alone is not the only change.')
    assert dense['total_not_equivalent']
    dump(OUT/'contracts.json',dict(plan_sha256=sha(OUT/'contract_plan.json'),loader=contract,discriminator_loss=loss_contract,
        supervised_loss=dense,figure_sha256=sha(OUT/'figures/公開ローダーの回転とSobel.png'),no_training=True,no_optimizer=True))
    print('Reference loader/loss contracts complete:',mismatch,'Sobel pixels displaced; dense losses differ',flush=True)


if __name__=='__main__':main()
