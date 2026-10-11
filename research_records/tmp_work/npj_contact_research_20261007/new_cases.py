from common import *
import shutil,sys,subprocess,time,os
from concurrent.futures import ThreadPoolExecutor
from selective import detect,combine
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_npj_literature_models as ext
SNAP=ROOT/'tmp_work/npj_dirt_control_20261002/source_snapshot/ddrm_npj_v2'
CKPT=ROOT/'modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'
PAPER=ROOT/'modules/ddrm_npj_v2/background/isemonogatari.png'
OBS={1:'挿絵：主対象外',2:'広い紙色むら：黒文字マスクの大きな塊はなく低優先',3:'表紙・本文なし：対象外',
 4:'装丁：対象外',5:'折れ・重なった紙：シミ単独の評価に不適',6:'挿絵・着色：対象外',7:'挿絵・着色：対象外',
 8:'表紙：対象外',9:'挿絵：対象外',10:'薄い像と広い茶色の跡が重なる：表の筆画か裏写りか未確定',
 11:'本文に茶色い背景むら：本文マスクは大部分独立、補助対照',12:'題扉・朱印・意匠：主対象外',
 13:'シミは主に枠線より上、本文を覆わない対照',14:'表紙：対象外',15:'本文に茶色い跡：局所的なマスク膨張候補',
 16:'挿絵：対象外',17:'挿絵：対象外',18:'本文なしの余白と挿絵：対象外',19:'角・装丁の濃い範囲：別課題',
 20:'薄い茶色の範囲が本文付近：重なりの程度と材質は未確定',21:'余白の細い跡：本文との重なりなし',
 22:'茶色の円形跡が細い筆画に重なる：接触シミの新候補',23:'下端の色むらと細い筆画：薄い筆画保持の対照',
 24:'余白の細い跡：本文との重なりなし',25:'細い茶色線・裏写り混在：材質未確定の対照',26:'細い茶色線・裏写り混在：材質未確定の対照'}
SELECTION=[('N01',22,'細い曲線に茶色の円形跡が重なる'),('N02',15,'本文に茶色い跡が重なる'),
 ('N03',10,'薄い像とシミ・筆記材の確認が必要'),('N04',20,'薄い茶色い範囲と本文・重なりの確認'),
 ('N05',23,'薄い筆画の正常対照'),('N06',26,'茶色線と裏写り・材質不確定対照')]
def prepare():
 out=HERE/'new_cases';out.mkdir(exist_ok=True)
 if (out/'selection.json').exists():return json.loads((out/'selection.json').read_text())
 source=json.loads((HERE/'other_book_screen/interim/selection.json').read_text());rows=source['selected'];review=[]
 for i,r in enumerate(rows,1):review.append(dict(id=r['review_id'],page=r['page'],finding=OBS[i]))
 dump(HERE/'other_book_screen/interim/visual_review.json',dict(review=review,complete=True,method='original and saved black-mask visual review; no restoration output'))
 selected=[]
 for id,n,reason in SELECTION:
  r=dict(rows[n-1]);r.update(id=id,reason=reason,detail_xywh=[120,100,520,560],source_sha256={k:sha(ROOT/r['source'][k]) for k in ['original','npj_black','npj_red']})
  x,y,w,h=r['crop_xywh'];a=read(ROOT/r['original']);write(out/'inputs'/id/(r['page']+'.png'),a[y:y+h,x:x+w])
  md=out/'masks'/r['page'];md.mkdir(parents=True,exist_ok=True)
  for k,fn in [('npj_black','02_text_black.png'),('npj_red','03_text_red.png')]:
   im=Image.open(ROOT/r['source'][k]).convert('L');im.crop((x,y,x+w,y+h)).save(md/fn)
  r['input']=str((out/'inputs'/id/(r['page']+'.png')).relative_to(ROOT));selected.append(r)
 manifest=dict(cases=selected,source_sha256=sha(__file__),created=time.time(),selected_before_new_case_inference=True,
  settings='same frozen local NPJ-derived implementation seed999 20steps native768 patch128 overlap32',
  material_GT=False,reading_GT=False,all_prior_methods_fixed_before_new_outputs=True,
  code_hashes={str(p.relative_to(ROOT)):sha(p) for p in SNAP.rglob('*.py')},checkpoint_sha256=sha(CKPT),background_sha256=sha(PAPER))
 dump(out/'selection.json',manifest);return manifest
def main():
 manifest=prepare();out=HERE/'new_cases'
 def group(gpu):
  model,deps=ext.load('docres',f'cuda:{gpu}');pns=ext.prompts();stats=[]
  for r in manifest['cases'][gpu::2]:
   id=r['id'];result=out/'output'/r['page']/'result'/(r['page']+'.png');a=read(ROOT/r['input']);cwd=out/'work'/id;cwd.mkdir(parents=True,exist_ok=True)
   log=out/'logs'/f'{id}.log';log.parent.mkdir(exist_ok=True)
   cmd=[sys.executable,'-B',str(SNAP/'scripts/pipeline_ddrm.py'),'--input_dir',str(out/'inputs'/id),'--output_dir',str(out/'output'),
     '--tmp_dir',str(out/'infer_tmp'/id),'--mask_dir',str(out/'masks'),'--ckpt',str(CKPT),'--back_ground',str(PAPER),
     '--device',f'cuda:{gpu}','--patch_size','128','--overlap','32','--no_wandb']
   print('new case start',id,r['page'],flush=True);start=time.monotonic()
   if not result.exists():
    with log.open('w') as f:q=subprocess.run(cmd,cwd=cwd,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
    assert q.returncode==0,(id,log)
   b=read(result);assert a.shape==b.shape
   bgr=a[:,:,::-1].copy();inp=np.concatenate([bgr,pns['binarization_promptv2'](bgr)],axis=2)
   prob,_=ext.tiled(model,inp,'docres',f'cuda:{gpu}');np.save(out/'inputs'/id/'docres_probability.npy',prob[:,:,0])
   v=ink_render(sauvola(a).astype('float32'));doc=ink_render((prob[:,:,0]<.5).astype('float32'))
   mask,comp=detect(a,b);local=combine(a,b,mask,(prob[:,:,0]<.5).astype('float32'))
   write(out/'methods'/id/'sauvola.png',v);write(out/'methods'/id/'docres.png',doc);write(out/'methods'/id/'local_docres.png',local);write(out/'methods'/id/'mask.png',mask*255)
   changed=np.any(local!=b,axis=2);assert not (changed&~mask).any()
   stat=dict(id=id,page=r['page'],seconds=time.monotonic()-start,mask_pixels=int(mask.sum()),changed_pixels=int(changed.sum()),outside_changed=0,
    baseline_sha256=sha(result),components=comp);dump(out/'logs'/f'{id}.json',stat);stats.append(stat)
   x,y,w,h=r['detail_xywh'];arrays=[a,b,v,doc,local]
   card(out/'comparisons'/f'{id}.png',[aa[y:y+h,x:x+w] for aa in arrays],
    ['原画像','NPJ系基準','Sauvola w=61','DocRes 二値化','自動範囲＋DocRes'],id+' '+r['page'],
    r['reason']+'。処理出力を見る前に選択。\n全例共通条件。材質・表の筆画・読解改善は未確定。',size=420)
   print('new case complete',id,round(stat['seconds'],1),flush=True)
  return stats
 with ThreadPoolExecutor(max_workers=2) as pool:stats=sum(pool.map(group,[0,1]),[])
 for path,h in manifest['code_hashes'].items():assert sha(ROOT/path)==h
 for r in manifest['cases']:
  for k,h in r['source_sha256'].items():assert sha(ROOT/r['source'][k])==h
 dump(out/'complete.json',dict(stats=stats,sources_unchanged=True))
if __name__=='__main__':main()
