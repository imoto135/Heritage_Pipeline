"""Pinned author loader, plus an explicit one-statement Sobel rotation variant.

The variant is a controlled correction, not a claimed paper reproduction.
Author files, interpolation, photometric jitter and losses are unchanged.
"""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'tmp_work/ra_gan_reference_20261021/source'


def loader_class(mode='official'):
    if mode not in ['official','aligned_sobel']:
        raise ValueError(mode)
    source=REFERENCE/'data/Binarizationdataloder.py'
    expected=json.loads((REFERENCE.parent/'reference_audit.json').read_text())['code_sha256']['data/Binarizationdataloder.py']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    sys.path.insert(0,str(REFERENCE))
    spec=importlib.util.spec_from_file_location('pinned_dense_loader',source)
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    if mode=='official':return author.DocData
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DocData')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
    branches=[n for n in ast.walk(method) if isinstance(n,ast.If) and any(
        isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='angle' for t in s.targets) for s in n.body)]
    assert len(branches)==1
    branches[0].body.append(ast.parse('sobel = random_rotate(sobel, 0, angle)').body[0])
    namespace=dict(author.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source)+'[aligned_sobel]','exec'),namespace)
    return type('SobelAlignedDocData',(author.DocData,),{'__getitem__':namespace['__getitem__']})


def verify():
    import numpy as np
    from PIL import Image
    from unittest.mock import patch
    from compare_v5_aligned_fade_control import dump,sha
    out=ROOT/'tmp_work/ragan_reproduction_foundation_20261023'
    assert not (out/'loader_variant_verification.json').exists()
    classes={m:loader_class(m) for m in ['official','aligned_sobel']}
    sets={m:c(str(out/'loader_fixture/img'),(256,256),training=True,only_one=True) for m,c in classes.items()}
    results={}
    for m,ds in sets.items():
        with patch('random.random',side_effect=[0.,.875,1.]):results[m]=ds[0]
    a,b=results['official'],results['aligned_sobel']
    for index in [0,1,3,4,5]:assert np.array_equal(a[index].numpy(),b[index].numpy())
    source_edge=Image.open(out/'loader_fixture/sobel/audit_0_0.png')
    # Obtain the same reference rotation helper used by both generated methods.
    rotate=sets['official'].__getitem__.__globals__['random_rotate']
    expected=np.asarray(rotate(source_edge,0,7.5)).astype(np.float32)/255
    assert np.array_equal(b[2][0].numpy(),expected) and not np.array_equal(a[2].numpy(),b[2].numpy())
    identities=[]
    for m,ds in sets.items():
        with patch('random.random',side_effect=[.5,1.]):identities.append(ds[0])
    assert all(np.array_equal(x.numpy(),y.numpy()) for x,y in zip(identities[0][:6],identities[1][:6]))
    dump(out/'loader_variant_verification.json',dict(script_sha256=sha(__file__),
        reference_loader_sha256=sha(REFERENCE/'data/Binarizationdataloder.py'),
        only_added_statement='sobel = random_rotate(sobel, 0, angle)',
        unchanged_RGB_Otsu_GT_gray_gtSobel=True,aligned_Sobel_matches_same_geometric_rotation=True,
        no_rotation_variants_bitexact=True,not_paper_reproduction=True,no_training=True))
    print('Verified official/aligned_sobel: only Sobel changes when rotated; other tensors bitexact')


if __name__=='__main__':verify()
