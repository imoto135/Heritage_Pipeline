"""Synthetic, book-separated test. Known layers do not imply real stain GT."""
import argparse
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,LayerDataset,input_features,file_hash
from train_layer_selector import Selector,CONFIGS

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def load_selector(path,device):
    c=torch.load(path,map_location='cpu');cfg=c['config']
    model=Selector(3 if cfg['mode']=='rgb' else 9,cfg['pointwise']);model.load_state_dict(c['model'])
    model.to(device).eval();return model,cfg,c

def summarize(confusion):
    result=[]
    for c,(tp,fp,fn,tn) in enumerate(confusion):
        result.append(dict(layer=['mark','stain'][c],TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),
            precision=float(tp/max(tp+fp,1)),recall=float(tp/max(tp+fn,1)),
            F1=float(2*tp/max(2*tp+fp+fn,1)),IoU=float(tp/max(tp+fp+fn,1))))
    return result

@torch.no_grad()
def run(model,cfg,ds,device):
    loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=4,pin_memory=True)
    totals=np.zeros((2,4),np.float64);groups={};pair_sums=[];collapse=[]
    for batch_index,(x,y) in enumerate(loader):
        logits=model(x.to(device));prob=logits.sigmoid().cpu().numpy();target=y.numpy();pred=prob>=.5;truth=target>=.15
        for ii in range(len(x)):
            idx=batch_index*8+ii
            rgb,_,_,meta=ds.sample(idx);key=(meta['kind'],meta['same_color'],meta['mark_opacity']<.1)
            group=groups.setdefault(str(key),np.zeros((2,4),np.float64))
            conf=np.zeros((2,4),np.float64)
            for cc in range(2):
                p=pred[ii,cc];t=truth[ii,cc]
                conf[cc]=[(p&t).sum(),(p&~t).sum(),(~p&t).sum(),(~p&~t).sum()]
            totals+=conf;group+=conf
            collapse.append(dict(mark_pixels=int(pred[ii,0].sum()),stain_pixels=int(pred[ii,1].sum()),
                GT_mark_pixels=int(truth[ii,0].sum()),GT_stain_pixels=int(truth[ii,1].sum())))
        # Same geometry, a second known color assignment. Evaluated, not fit.
        other=np.stack([input_features(ds.sample(batch_index*8+i,variant=1)[0],cfg['mode']) for i in range(len(x))])
        other_prob=model(torch.from_numpy(other).to(device)).sigmoid().cpu().numpy()
        pair_sums.append(float(np.mean(np.abs(prob-other_prob))))
    return dict(overall=summarize(totals),groups={k:summarize(v) for k,v in groups.items()},
        color_pair_mean_abs_probability_change=float(np.mean(pair_sums)),
        predicted_zero_mark_samples=sum(r['mark_pixels']==0 and r['GT_mark_pixels']>0 for r in collapse),
        predicted_zero_stain_samples=sum(r['stain_pixels']==0 and r['GT_stain_pixels']>0 for r in collapse),
        samples=len(ds),threshold=.5,GT_support_alpha_threshold=.15)

def preview(model,cfg,ds,device,path):
    sheet=Image.new('RGB',(5*256,4*292),'#faf9f5');d=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,16)
    names=['合成入力','既知の朱の支持','予測・朱の支持','既知のシミ支持','予測・シミ支持']
    for row,i in enumerate([2,4,6,11]):
        rgb,y,clean,meta=ds.sample(i)
        with torch.no_grad():p=model(torch.from_numpy(input_features(rgb,cfg['mode']))[None].to(device)).sigmoid()[0].cpu().numpy()
        panels=[rgb,np.repeat((y[0]>=.15)[:,:,None]*255,3,2),np.repeat((p[0]>=.5)[:,:,None]*255,3,2),
            np.repeat((y[1]>=.15)[:,:,None]*255,3,2),np.repeat((p[1]>=.5)[:,:,None]*255,3,2)]
        for j,(a,n) in enumerate(zip(panels,names)):
            sheet.paste(Image.fromarray(np.asarray(a,dtype=np.uint8)),(j*256,row*292+32));d.text((j*256+4,row*292+4),n,font=font,fill='#21313a')
    sheet.save(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=list(CONFIGS));p.add_argument('--device',default='cuda:0')
    p.add_argument('--samples',type=int,default=768);p.add_argument('--output-label',default='synthetic_fixed_test')
    a=p.parse_args();torch.set_num_threads(4)
    dest=OUT/a.output_label;dest.mkdir(exist_ok=True)
    protocol=dict(checkpoint='Fixed last.pth at 12000 steps as primary; best.pth only logged training diagnostic',
        checkpoints_frozen_before_any_real_outputs=True,seed=20261010,test_background_books='Separate prior source-bank test books',
        samples_per_condition=a.samples,conditions=['overlapping colors','overlapping colors + folds/dots stress'],
        genuine_real_stain_GT=False,source_sha256=file_hash(__file__))
    (dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    results=json.loads((dest/'results.json').read_text()) if (dest/'results.json').exists() else []
    for name in a.names:
        path=OUT/'learned_trials'/f'{name}_seed413'/'last.pth'
        assert (path.parent/'complete.json').exists(),'Wait for frozen full training completion'
        prior=next((r for r in results if r['name']==name),None)
        if prior is not None:
            assert prior['checkpoint_sha256']==file_hash(path)
            assert all(c['samples']==a.samples for c in prior['conditions'].values())
            (dest/(name+'.json')).write_text(json.dumps(prior,ensure_ascii=False,indent=2))
            print('Reuse completed synthetic test record',name,flush=True)
            continue
        model,cfg,checkpoint=load_selector(path,a.device);assert checkpoint['step']==12000
        record=dict(name=name,checkpoint_sha256=file_hash(path),step=checkpoint['step'],conditions={})
        for stress in [False,True]:
            ds=LayerDataset('test',a.samples,mode=cfg['mode'],seed=20261010,stress=stress,training='overlap')
            start=time.monotonic();key='stress_folds_dots' if stress else 'standard_overlapping_colors'
            record['conditions'][key]=run(model,cfg,ds,a.device)
            record['conditions'][key]['seconds']=time.monotonic()-start
            preview(model,cfg,ds,a.device,dest/(name+'_'+key+'.png'))
            print(name,key,[(m['layer'],round(m['F1'],3)) for m in record['conditions'][key]['overall']],flush=True)
        # Anti-degeneracy controls on exactly the same known layers.
        results.append(record);(dest/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
        (dest/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2))
        del model
    (dest/'complete.json').write_text(json.dumps(dict(status='completed',models=len(results),synthetic_only=True,
        real_restoration_or_readability_claim=False),indent=2))

if __name__=='__main__':main()
