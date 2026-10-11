"""Precompute independent page outputs while safely suspending the queue owner.

The already running children finish normally. Resume the original queue in all
cases; it verifies and reuses completed files and renders the final board.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import time
from PIL import Image
import run_ddrm_pilot as driver
import validate_new_materials as trial


def state(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0]
    except FileNotFoundError:
        return None


def main(parent, children):
    proto=json.loads((trial.OUT/'protocol.json').read_text())
    driver.OUT=trial.OUT
    original_command=Path(f'/proc/{parent}/cmdline').read_bytes()
    assert b'validate_new_materials.py' in original_command and b'infer' in original_command
    for cid,pid in children.items():
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        case=next(c for c in proto['cases'] if c['id']==cid)
        assert case['page'].encode() in cmd and b'pipeline_ddrm.py' in cmd
    os.kill(parent,signal.SIGSTOP)
    try:
        time.sleep(.1)
        assert state(parent) in ('T','t')
        def group(case,gpu):
            cid=case['id'];results=[]
            if cid in children:
                pid=children[cid]
                # A completed child becomes a zombie until the queue is resumed.
                while state(pid) not in ('Z',None):time.sleep(1)
                p=case['page'];output=trial.OUT/'outputs/howe'/p/'result'/(p+'.png')
                assert output.exists() and list(Image.open(output).size)==case['input_size']
                assert '[ERROR]' not in (trial.OUT/'logs'/f'{cid}_howe.log').read_text(errors='replace')
                methods=['isolated']
            else:
                methods=['csc','howe','isolated']
            for method in methods:
                result=driver.run(case,method,gpu)
                assert '[ERROR]' not in (trial.OUT/'logs'/f'{cid}_{method}.log').read_text(errors='replace')
                results.append(result)
            return results
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs=[pool.submit(group,c,i%2) for i,c in enumerate(proto['cases'])]
            results=[r for job in jobs for r in job.result()]
        driver.save_json(trial.OUT/'parallel_precompute_complete.json',dict(
            results=results,parameters_unchanged=True,original_queue_parent=parent,
            gpu_assignment={'V01':0,'V02':1,'V03':0,'V04':1},
            script_sha256=driver.digest(Path(__file__))))
    finally:
        if state(parent) is not None:os.kill(parent,signal.SIGCONT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=int,required=True)
    p.add_argument('--v01-child',type=int,required=True);p.add_argument('--v02-child',type=int,required=True)
    a=p.parse_args();main(a.parent,{'V01':a.v01_child,'V02':a.v02_child})
