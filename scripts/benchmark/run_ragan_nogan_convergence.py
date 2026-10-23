"""Matched generator-only 10-epoch control using the fixed dark-stain data."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_convergence_audit import OUT
from compare_v5_aligned_fade_control import sha,dump
import run_contact_stain_training_control as control


def prepare():
    dest=OUT/'ra_no_gan_contact';assert not dest.exists();dest.mkdir()
    parent=OUT/'ragan_contact'
    for name in ('train_manifest.jsonl','val_manifest.jsonl','label_protocol.json','adapter_verification.json','code_hashes.json'):
        shutil.copyfile(parent/name,dest/name)
    conf=json.loads((parent/'condition.json').read_text());conf.update(kind='ra_no_gan',
        changed_factor='remove discriminator and adversarial loss only',steps=2160)
    dump(dest/'condition.json',conf)
    protocol=json.loads((OUT/'evaluation_protocol.json').read_text())
    protocol.update(conditions=['ra_no_gan_contact'],checkpoints=[432,2160],
        checkpoint_model_folder='ra_no_gan',comparison='same dark-data GAN run at 432 and 2160',
        only_GAN_removed=True)
    dump(OUT/'nogan_evaluation_protocol.json',protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        control.OUT=OUT
        control.train('ra_no_gan_contact','cuda:0',steps=2160,validate_every=432,keep_checkpoints=True)
