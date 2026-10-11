"""Restrict the unchanged architecture to source-estimated relative OD channels."""
import torch
from train_layer_selector import Selector
class RelativeOnlySelector(Selector):
    def forward(self,x):
        x=x.clone();x[:,:3]=0;x[:,6:9]=0
        return super().forward(x)
def load_relative_only(path,device):
    ck=torch.load(path,map_location='cpu');cfg=ck['config'];assert cfg['relative_only'] and cfg['mode']=='relative'
    model=RelativeOnlySelector(9,cfg['pointwise']);model.load_state_dict(ck['model']);model.to(device).eval();return model,cfg,ck
