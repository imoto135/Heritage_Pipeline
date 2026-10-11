"""Second development probe: conserve v6 networks anchored by red/warm hold.

Keep the first failed probe as read-only evidence. The inference adjustment is
generic across full pages, with no ROI/page whitelist. It does not label ink GT.
"""
from pathlib import Path
import json
import shutil
import cv2
import numpy as np
import probe_defect_context_preprocessing as base
from build_npj_v6_comparison import sha,dump,read

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/defect_context_preprocessing_pilot_v2_20261001'
INITIAL_INFER=base.infer


def infer(original,v6):
    modes,features,components=INITIAL_INFER(original,v6)
    network=v6['protect']|v6['hold']|v6['red']
    # A small gap in the v6 raster should not split one connected warm stroke.
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    connected=cv2.morphologyEx(network.astype(np.uint8),cv2.MORPH_CLOSE,kernel)>0
    n,labels=cv2.connectedComponents(connected.astype(np.uint8),connectivity=8)
    anchor=v6['red']|v6['extra']
    counts=np.bincount(labels.ravel(),weights=anchor.ravel(),minlength=n)
    anchored=counts>0;anchored[0]=False
    anchor_guard=anchored[labels]&connected
    # Preserve the network and a 2px halo; this can also retain connected stains.
    anchor_guard=cv2.dilate(anchor_guard.astype(np.uint8),kernel)>0
    extra_guard=cv2.dilate(features['neutral_ridge'].astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0
    before=modes['context_stroke']['added'].copy()
    added=before&~anchor_guard&~extra_guard
    modes['context_stroke']=dict(edit=v6['edit']|added,added=added,
        protect=v6['protect']&~added,hold=v6['hold']&~added,red=v6['red'])
    features['warm_anchor_guard']=anchor_guard
    features['neutral_ridge_guard_4px']=extra_guard
    features['rejected_by_conservation']=before&~added
    assert not np.any(added&(anchor|anchor_guard|extra_guard))
    return modes,features,components


def main():
    settings=dict(base.SETTINGS)
    settings['runner_source_sha256']=sha(Path(__file__))
    settings['base_source_sha256']=sha(Path(base.__file__))
    settings['conservation']=dict(network='v6 protect OR hold OR red',
        anchors='v6 confirmed red OR extra warm hold',closing_kernel=5,anchor_halo_px=2,
        neutral_ridge_guard_px=4,condition='exclude whole anchored networks from added edit',
        limitation='connected stains may also be preserved; this is not proof of ink identity')
    base.OUT=OUT;base.SETTINGS=settings;base.infer=infer
    base.main()
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest['dependencies'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'runner_snapshot.py')
    verification=json.loads((OUT/'verification.json').read_text())
    verification.update(anchored_network_conservation=True,neutral_ridge_guard_px=4,
                        previous_failed_probe_preserved=True)
    dump(OUT/'verification.json',verification)
    print('Anchored-network probe completed; development only',flush=True)


if __name__=='__main__':main()
