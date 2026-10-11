"""Global-size controls for StainRestorer; known resizing loss kept explicit."""
import argparse,json,time,sys
import cv2,numpy as np,torch
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash
from run_direct_literature_baselines import load,invoke

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--sizes',nargs='+',type=int,default=[256,512]);a=p.parse_args()
    cv2.setNumThreads(2);torch.set_num_threads(4);model,resources=load('stain',a.device)
    cases=json.loads((OUT/'cases_frozen.json').read_text())['cases'];pages={r['page']:{**r,'path':ROOT/r['original']} for r in cases}
    for r in json.loads((OUT/'external_originals/frozen_cases/cases_frozen.json').read_text())['cases']:pages['external_'+r['page']]={**r,'path':OUT/r['original']}
    for size in a.sizes:
        dest=OUT/'direct_literature_baselines'/f'stain_global{size}';dest.mkdir(exist_ok=True)
        protocol=dict(method=f'stain_global{size}',resources={str(f):file_hash(f) for f in resources},source_sha256=file_hash(__file__),
            config_sha256=file_hash(ROOT/'tmp_work/npj_literature_models_20261002/repos/StainRestorer/config.yml'),
            geometry=f'Whole original resized to {size} square with OpenCV bilinear, no tiles; model RGB output upsized bilinearly to native shape',
            author_default_config_size=256,size_matches_default_config=size==256,exact_author_benchmark=False,
            input_channel='RGB float32 /255 as author PIL loader',weight='Normal StainDoc, not seal/mark removal',
            limitation='Interpolation restores dimensions, not fine details lost during shrinking. Display diagnostic only, not native-resolution reconstruction quality',
            labels_used=False,expert_GT=False)
        (dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
        for page,r in pages.items():
            folder=dest/page;folder.mkdir(exist_ok=True);done=folder/'complete.json'
            if done.exists():
                old=json.loads(done.read_text());assert old['source_sha256']==r['source_sha256'];assert old['result_sha256']==file_hash(folder/'result.png');continue
            assert file_hash(r['path'])==r['source_sha256'];rgb=np.asarray(Image.open(r['path']).convert('RGB'));h,w=rgb.shape[:2];start=time.monotonic()
            bgr=cv2.resize(rgb[:,:,::-1].copy(),(size,size),interpolation=cv2.INTER_LINEAR);value=invoke(model,bgr,'stain',a.device)
            low=np.rint(value*255).astype(np.uint8)[:,:,::-1].copy();Image.fromarray(low).save(folder/'model_output_RGB.png')
            result=cv2.resize(low,(w,h),interpolation=cv2.INTER_LINEAR);Image.fromarray(result).save(folder/'result.png')
            done.write_text(json.dumps(dict(page=page,source_sha256=r['source_sha256'],result_sha256=file_hash(folder/'result.png'),protocol_sha256=file_hash(dest/'protocol.json'),
                seconds=time.monotonic()-start,shape=list(rgb.shape),method=protocol['method'],resizing_not_detail_recovery=True),indent=2))
            print('Stain global',size,page,round(time.monotonic()-start,1),flush=True)
        (dest/'queue_complete.json').write_text(json.dumps(dict(pages=list(pages),size=size),indent=2))

if __name__=='__main__':main()
