"""Evaluate auxiliary ROI losses against the exact stable-BCE20% cached baseline."""
import argparse,json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_local_region_control import OUT,CONDITIONS,PARENT
from compare_v5_aligned_fade_control import dump,sha
import evaluate_ragan_fine_retention_control as evaluator


def adapter(step):
 """Translate cached baseline condition names; never infer or change their masks."""
 root=OUT/'baseline_cache_adapter'/f'step_{step:05d}';assert not root.exists();root.mkdir(parents=True)
 directory='evaluation' if step==864 else 'evaluation_00432';source=PARENT/directory;dest=root/directory;dest.mkdir()
 mapping={'initial':'initial','stable_exposure_order0':'no_gan_order0','stable_exposure_order1':'no_gan_order1'}
 original=json.loads((source/'results.json').read_text());converted={}
 for kind in ['real','synthetic','traced_fade']:
  converted[kind]=[dict(r,condition=mapping[r['condition']]) for r in original[kind] if r['condition'] in mapping]
 converted['checkpoint_sha256']={mapping[c]:s for c,s in original['checkpoint_sha256'].items() if c in mapping}
 converted['figures']=[];dump(dest/'results.json',converted)
 train=json.loads((source/'training_reviewed_results.json').read_text());dump(dest/'training_reviewed_results.json',dict(rows=[dict(r,condition=mapping[r['condition']]) for r in train['rows'] if r['condition'] in mapping],figures=[]))
 for old,new in mapping.items():
  for kind in ['native','real','synthetic','training_reviewed_masks']:
   folder=dest/kind;folder.mkdir(exist_ok=True);(folder/new).symlink_to((source/kind/old).resolve(),target_is_directory=True)
  folder=dest/'traced_fade';folder.mkdir(exist_ok=True);(folder/(new+'.png')).symlink_to((source/'traced_fade'/(old+'.png')).resolve())
 shutil.copy2(PARENT/'evaluation_protocol.json',root/'evaluation_protocol.json')
 if step==864:
  native=json.loads((source/'training_native_results.json').read_text());native_map={'initial':'initial_order0','stable_exposure_order0':'no_gan_order0','stable_exposure_order1':'no_gan_order1'}
  folder=root/'training_context_audit';folder.mkdir();dump(folder/'results.json',dict(rows=[dict(r,condition=native_map[r['condition']]) for r in native['rows'] if r['condition'] in native_map],figures=[]))
  (folder/'masks').mkdir()
  for old,new in native_map.items():(folder/'masks'/new).symlink_to((source/'training_native_masks'/old).resolve(),target_is_directory=True)
 dump(root/'provenance.json',dict(source=str(source.relative_to(ROOT)),original_results_sha256=sha(source/'results.json'),original_training_results_sha256=sha(source/'training_reviewed_results.json'),condition_name_translation=mapping,no_prediction_recomputation=True,no_mask_edits=True))
 return root

def main(step,device):
 cached=adapter(step)
 # Only this evaluation process redirects cached baseline reads. Dataset paths
 # and crop formulas use their original constants, and no training is performed.
 evaluator.full.OUT=cached;evaluator.OUT=OUT;evaluator.CONDITIONS=CONDITIONS
 evaluator.LABELS={'initial':'再開前10周','baseline_order0':'補助項なし 順序1','baseline_order1':'補助項なし 順序2',
  **{f'{kind}_order{n}':label+f' 順序{n+1}' for n in (0,1) for kind,label in [('inkmean','墨芯平均BCE追加'),('inkpaper','墨芯＋近傍明紙BCE追加')]}}
 evaluator.ORDER=list(evaluator.LABELS);evaluator.main(step,device)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=[432,864],default=864);p.add_argument('--device',default='cuda:0');a=p.parse_args();main(a.step,a.device)
