"""Reuse the same fixed readout and cached baseline for the BCE numerical fix."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_stable_bce_control import OUT,CONDITIONS
import evaluate_ragan_fine_retention_control as evaluation

def main(step,device):
 evaluation.OUT=OUT;evaluation.CONDITIONS=CONDITIONS
 evaluation.LABELS={'initial':'再開前10周','baseline_order0':'従来BCE 順序1','baseline_order1':'従来BCE 順序2',
  'stable_bce_order0':'安定BCE 順序1','stable_bce_order1':'安定BCE 順序2'}
 evaluation.ORDER=list(evaluation.LABELS)
 evaluation.main(step,device)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=[432,864],default=864);p.add_argument('--device',default='cuda:0');a=p.parse_args();main(a.step,a.device)
