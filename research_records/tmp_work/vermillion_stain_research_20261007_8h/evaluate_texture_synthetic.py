"""Frozen synthetic test for transfer trials; no real GT claim."""
import argparse,json,time
from synthetic_layers import OUT,LayerDataset,file_hash
from evaluate_synthetic_selectors import load_selector,run,preview
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=['continue_control','paper_texture','paper_texture_sparse']);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    dest=OUT/'texture_synthetic_fixed_test';dest.mkdir(exist_ok=True)
    for name in a.names:
        trial=OUT/'texture_transfer_trials'/f'{name}_seed614';assert (trial/'complete.json').exists();path=trial/'last.pth';target=dest/(name+'.json')
        if target.exists():assert json.loads(target.read_text())['checkpoint_sha256']==file_hash(path);continue
        model,cfg,ck=load_selector(path,a.device);assert ck['step']==10000;start=time.monotonic();conditions={}
        for stress in [False,True]:
            ds=LayerDataset('test',768,mode='relative',training='overlap',seed=20261010,stress=stress)
            conditions['stress' if stress else 'standard']=run(model,cfg,ds,a.device);preview(model,cfg,ds,a.device,dest/(name+('_stress' if stress else '_standard')+'.png'))
        record=dict(name=name,conditions=conditions,checkpoint_sha256=file_hash(path),script_sha256=file_hash(__file__),
            GT='Digitally added alpha supports only; source contrast shapes are not real ink GT',samples_per_condition=768,
            same_fixed_test_as_first_six=True,test_used_for_training=False,final_checkpoint=True,seconds=time.monotonic()-start)
        target.write_text(json.dumps(record,ensure_ascii=False,indent=2));print('Synthetic transfer test',name,round(record['seconds'],1),flush=True)
    records=[json.loads((dest/(n+'.json')).read_text()) for n in a.names];(dest/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
