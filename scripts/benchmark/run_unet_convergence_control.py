"""Same 10-epoch dark-stain data and crop schedule for the U-Net baseline."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_convergence_audit import OUT
from compare_v5_aligned_fade_control import dump
import run_contact_stain_training_control as control


def prepare():
    dest=OUT/'unet_contact';assert not dest.exists();dest.mkdir()
    parent=OUT/'ragan_contact'
    for name in ('train_manifest.jsonl','val_manifest.jsonl','label_protocol.json','adapter_verification.json','code_hashes.json'):
        shutil.copyfile(parent/name,dest/name)
    conf=json.loads((parent/'condition.json').read_text());conf.update(kind='unet',steps=2160,
        changed_factor='U-Net architecture; same dark-data/crops/labels/updates/common loss')
    dump(dest/'condition.json',conf)
    protocol=json.loads((OUT/'evaluation_protocol.json').read_text())
    protocol.update(conditions=['unet_contact'],checkpoints=[432,2160],
        checkpoint_model_folder='unet',prediction_kind='unet',
        comparison='same dark-data 10-epoch baseline',only_architecture_changed=True)
    dump(OUT/'unet_evaluation_protocol.json',protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        control.OUT=OUT
        control.train('unet_contact','cuda:1',steps=2160,validate_every=432,keep_checkpoints=True)
