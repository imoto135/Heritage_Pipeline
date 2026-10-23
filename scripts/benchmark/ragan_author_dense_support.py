"""Source-extracted author objectives and cooperative GPU guard.

No attention, loss, normalization or forward adapters are installed.
"""
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'tmp_work/ra_gan_reference_20261021/source'


class GPUUnavailable(RuntimeError):
    pass


def gpu_window(query=None, own_pid=None):
    """Require no foreign compute PID on any GPU, even when selected GPU is free."""
    own_pid=os.getpid() if own_pid is None else own_pid
    if query is None:
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                             text=True,capture_output=True,timeout=10)
    if query.returncode:
        raise GPUUnavailable('Cannot establish an idle GPU window: '+(query.stderr or query.stdout).strip())
    pids=[]
    for line in query.stdout.splitlines():
        if not line.strip():continue
        if not line.strip().isdigit():raise GPUUnavailable('Unrecognized GPU process record: '+line)
        pids.append(int(line))
    foreign=sorted(set(pids)-{own_pid})
    if foreign:raise GPUUnavailable('Foreign GPU compute processes present: '+','.join(map(str,foreign)))
    return dict(own_pid=own_pid,foreign_compute_pids=[],all_compute_pids=sorted(set(pids)))


def author_objectives():
    """Compile only assignments from pinned train.py, never import its loop."""
    tree=ast.parse((REFERENCE/'train.py').read_text())
    dnames={'D_real_c','D_fake_c','D_loss'}
    gnames={'fin_l1_loss','fin_BCEloss','fin_dice_loss','fin_jaccard_loss','fake_preds','adversarial_loss','G_loss'}
    groups={k:[] for k in ['d','g']}
    for node in ast.walk(tree):
        if not isinstance(node,ast.Assign):continue
        names={n.id for t in node.targets for n in ast.walk(t) if isinstance(n,ast.Name)}
        if names & dnames:groups['d'].append(node)
        if names & gnames:groups['g'].append(node)
    assert len(groups['d'])==5 and len(groups['g'])==4
    functions={}
    for kind,nodes in groups.items():
        args=['discriminator','gt','inputImg','bin_out'] if kind=='d' else ['discriminator','loss_doc','inputImg','bin_out','gt']
        result='D_loss, D_real_c, D_fake_c' if kind=='d' else 'G_loss, fin_l1_loss, fin_BCEloss, fin_dice_loss, fin_jaccard_loss, adversarial_loss'
        template=ast.parse('def objective('+','.join(args)+'):\n    pass').body[0]
        template.body=sorted(nodes,key=lambda n:n.lineno)+[ast.parse('return '+result).body[0]]
        namespace=dict(torch=torch,F=F)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[template],type_ignores=[])),str(REFERENCE/'train.py')+'[extracted]','exec'),namespace)
        functions[kind]=namespace['objective']
    return functions


def make_models(seed=1028):
    import sys
    import random
    import numpy as np
    for p in [REFERENCE,REFERENCE.parent/'dependencies']:
        sys.path.insert(0,str(p))
    from models.Generator import RA_GAN
    from models.Discriminator import Discriminator
    from loss.loss import Loss_Doc
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    # Instantiate both even in no-GAN arm, so RNG consumption and G state agree.
    g=RA_GAN(5);d=Discriminator(4)
    return g,d,Loss_Doc(g)


def state_digest(module):
    h=hashlib.sha256()
    for name,t in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_digest(tensors):
    h=hashlib.sha256()
    for t in tensors:
        a=t.detach().cpu().contiguous().numpy();h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
    return h.hexdigest()


def gradient_norm(loss,params,retain_graph=True):
    grads=torch.autograd.grad(loss,params,retain_graph=retain_graph,allow_unused=True)
    return float(torch.sqrt(sum(g.detach().double().square().sum() for g in grads if g is not None)))
