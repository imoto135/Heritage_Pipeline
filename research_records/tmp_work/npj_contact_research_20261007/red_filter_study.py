from common import *
from red_mask_audit import retained
from render_full import records
from full_page import SNAP,CKPT,PAPER
import subprocess,sys,os,time,shutil
from concurrent.futures import ThreadPoolExecutor
def main():
 out=HERE/'red_filter_study';byid={c['id']:c for c in records()};selected=[]
 for id in ['N02','N03','F41']:
  c=byid[id]
  if id=='F41':frame=[0,1200,768,768];detail=[90,180,340,390]
  else:
   old=next(c for c in json.loads((HERE/'new_cases/selection.json').read_text())['cases'] if c['id']==id);frame=old['crop_xywh'];detail=old['detail_xywh']
  selected.append(dict(id=id,page=c['page'],source=c['source'],frame_xywh=frame,detail_xywh=detail))
 protocol=dict(selected=selected,source_sha256=sha(__file__),created=time.time(),diagnostic=True,
  rationale='identify red-overlay mechanism and the cost of the existing absolute-color rejection rule; not a proposed new method',
  variants=['saved_black_saved_red','saved_black_filtered_red','sauvola_black_filtered_red'],
  color_rule='existing HSV hue<=12 or>=168, S>180, component support>=.05; fixed',
  code_hashes={str(p.relative_to(ROOT)):sha(p) for p in SNAP.rglob('*.py')},weights_sha256=sha(CKPT),material_GT=False,reading_GT=False)
 dump(out/'protocol.json',protocol);tasks=[]
 for c in selected:
  a=read(ROOT/c['source']['original']);red=np.asarray(Image.open(ROOT/c['source']['npj_red']).convert('L'))<127;filtered,_=retained(a,red);black=np.asarray(Image.open(ROOT/c['source']['npj_black']).convert('L'))<127
  x,y,w,h=c['frame_xywh'];sl=np.s_[y:y+h,x:x+w];a=a[sl];ip=out/'inputs'/c['id']/(c['page']+'.png');write(ip,a)
  for variant in protocol['variants']:
   md=out/'masks'/variant/c['page'];bm=sauvola(a) if variant.startswith('sauvola') else black[sl];rm=red[sl] if variant.endswith('saved_red') else filtered[sl]
   write(md/'02_text_black.png',np.repeat((~bm)[:,:,None]*255,3,axis=2));write(md/'03_text_red.png',np.repeat((~rm)[:,:,None]*255,3,axis=2));tasks.append((c,variant,ip))
 def group(gpu):
  stats=[]
  for c,variant,ip in tasks[gpu::2]:
   page=c['page'];dest=out/'output'/variant;result=dest/page/'result'/(page+'.png');log=out/'logs'/f"{c['id']}_{variant}.log";log.parent.mkdir(parents=True,exist_ok=True)
   tmp=out/'infer_tmp'/f"{c['id']}_{variant}";cwd=out/'work'/f"{c['id']}_{variant}";cwd.mkdir(parents=True,exist_ok=True)
   cmd=[sys.executable,'-B',str(SNAP/'scripts/pipeline_ddrm.py'),'--input_dir',str(ip.parent),'--output_dir',str(dest),'--tmp_dir',str(tmp),'--mask_dir',str(out/'masks'/variant),'--ckpt',str(CKPT),'--back_ground',str(PAPER),'--device',f'cuda:{gpu}','--patch_size','128','--overlap','32','--no_wandb']
   start=time.monotonic()
   with log.open('w') as f:p=subprocess.run(cmd,cwd=cwd,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
   assert p.returncode==0,(c['id'],variant,log);assert Image.open(result).size==Image.open(ip).size
   stats.append(dict(id=c['id'],variant=variant,seconds=time.monotonic()-start,command=cmd,output=str(result.relative_to(ROOT)),sha256=sha(result)));shutil.rmtree(tmp,ignore_errors=True);print('red filter',c['id'],variant,flush=True)
  return stats
 with ThreadPoolExecutor(max_workers=2) as pool:stats=sum(pool.map(group,[0,1]),[])
 for c in selected:
  a=read(out/'inputs'/c['id']/(c['page']+'.png'));values=[a]+[read(out/'output'/v/c['page']/'result'/(c['page']+'.png')) for v in protocol['variants']]
  x,y,w,h=c['detail_xywh'];card(out/'comparisons'/(c['id']+'.png'),[v[y:y+h,x:x+w] for v in values],['原画像','保存黒＋保存赤','保存黒＋絶対色赤','Sauvola黒＋絶対色赤'],c['id']+' '+c['page'],
   '同じ原寸768pxフレーム、DDRMコード・重み・乱数初期値。マスクだけを変更。\n厳しい赤判定が本来の朱印候補も除く可能性を確認する診断。新規手法・読解評価ではない。',size=440)
 for p,h in protocol['code_hashes'].items():assert sha(ROOT/p)==h
 dump(out/'complete.json',dict(results=stats,source_code_unchanged=True))
if __name__=='__main__':
 while not (HERE/'real_models/complete.json').exists():time.sleep(5)
 main()
