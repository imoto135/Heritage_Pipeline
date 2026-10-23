"""Equivalent reflection via slices/cat, avoiding CUDA reflection-pad backward."""
import types
import torch
from torch.nn import functional as F

def reflect_by_slices(x,padding):
    left,right,top,bottom=padding
    assert x.ndim==4 and 0<=left<x.shape[-1] and 0<=right<x.shape[-1] and 0<=top<x.shape[-2] and 0<=bottom<x.shape[-2]
    xs=[]
    if left:xs.append(x[...,1:left+1].flip(-1))
    xs.append(x)
    if right:xs.append(x[...,-right-1:-1].flip(-1))
    x=torch.cat(xs,dim=-1) if len(xs)>1 else x
    ys=[]
    if top:ys.append(x[...,1:top+1,:].flip(-2))
    ys.append(x)
    if bottom:ys.append(x[...,-bottom-1:-1,:].flip(-2))
    return torch.cat(ys,dim=-2) if len(ys)>1 else x

def conv_forward(self,x,weight,bias):
    return F.conv2d(reflect_by_slices(x,self._reversed_padding_repeated_twice),weight,bias,self.stride,(0,0),self.dilation,self.groups)

def install(model):
    names=[]
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Conv2d) and module.padding_mode=='reflect':
            module._conv_forward=types.MethodType(conv_forward,module);names.append(name)
    return names

def checks():
    torch.manual_seed(731);rows=[]
    for padding in [(3,3,3,3),(1,2,3,0),(0,0,0,0)]:
        x=torch.randn(2,3,11,13,dtype=torch.float64,requires_grad=True);weight=torch.randn(2,3,11+padding[2]+padding[3],13+padding[0]+padding[1],dtype=torch.float64)
        a=F.pad(x,padding,mode='reflect');b=reflect_by_slices(x,padding);assert torch.equal(a,b)
        ga=torch.autograd.grad((a*weight).sum(),x)[0];gb=torch.autograd.grad((b*weight).sum(),x)[0];torch.testing.assert_close(ga,gb,rtol=1e-12,atol=1e-12)
        rows.append(dict(padding=list(padding),forward_exact=True,max_gradient_delta=float((ga-gb).abs().max())))
    return rows
