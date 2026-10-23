"""Count reviewed center-point exposure; pixel counts are not gradient weights."""
import json,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_full_continuation import OUT,read_rows,reviewed,gray
from compare_v5_aligned_fade_control import dump

def main():
 rows=read_rows('train');labels={}
 for row in rows:
  patch=row['patch_id']
  if patch not in labels:labels[patch]=(gray(ROOT/row['labels']['valid'])>0,gray(ROOT/row['labels']['target'])<128)
 reviews=reviewed();totals=defaultdict(lambda:dict(crops=0,valid_pixels=0,valid_ink=0,reviewed_trace_points=0))
 for entry in json.loads((OUT/'crop_schedule.json').read_text()):
  row=rows[entry['index']];x,y=entry['xy'];valid,target=labels[row['patch_id']];v=valid[y:y+256,x:x+256];t=target[y:y+256,x:x+256]
  a=totals[row['damage_family']];a['crops']+=1;a['valid_pixels']+=int(v.sum());a['valid_ink']+=int((v&t).sum())
  if row['patch_id'] in reviews:
   trace=reviews[row['patch_id']][1][y:y+256,x:x+256];a['reviewed_trace_points']+=int((trace&v&t).sum())
 total={k:sum(a[k] for a in totals.values()) for k in next(iter(totals.values()))}
 dump(OUT/'fine_exposure_density.json',dict(by_family=dict(totals),all=total,
  tracked_centers_fraction_valid=total['reviewed_trace_points']/total['valid_pixels'],
  tracked_centers_fraction_ink=total['reviewed_trace_points']/total['valid_ink'],
  interpretation='counts of existing reviewed center points only; does not count all fine strokes or prove their gradient/loss contribution'))
if __name__=='__main__':main()
