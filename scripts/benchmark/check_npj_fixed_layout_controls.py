"""Additional original-page controls for the frozen layout candidate.

No restored image is available for these inputs in this run: only assess frame
geometry and observed writing witnesses. Never substitute raw input for npj.
"""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, read, write, sha, dump, overlay
from package_saved_npj_role_separated_review import figure
import probe_npj_full_page_layout_v4 as fixed

OUT=ROOT/'tmp_work/npj_layout_fixed_screening_20261004'
FROZEN=ROOT/'tmp_work/npj_full_page_layout_v4_20261004'
ROLES={
    '200006665_00047_2':'本文・完全な枠が見えない対照',
    '200005598_00013_1':'白黒・太い二重枠の対照',
    '200022050_00012_1':'茶色の紙面と密な本文・材質は未確定',
    '200008316_00020_1':'本文枠外の実際の見出し・必須保持対照',
    '200021712_00042_1':'虫食い例と同冊の別ページ・表と綴じ側の対照'}


def main():
    assert not (OUT/'selection.json').exists(), 'Already recorded'
    complete=json.loads((FROZEN/'complete.json').read_text())
    for rel,digest in complete['outputs'].items():assert sha(FROZEN/rel)==digest,rel
    frozen_manifest=json.loads((FROZEN/'manifest.json').read_text())
    for rel,digest in frozen_manifest['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    cases=json.loads((OUT/'pending_selection.json').read_text())['cases']
    previous_pages={c['page'] for c in frozen_manifest['frames']}
    assert not previous_pages & {c['page'] for c in cases}
    deps={c['original']:sha(ROOT/c['original']) for c in cases}
    for p in [Path(__file__),FROZEN/'complete.json',OUT/'pending_selection.json',OUT/'original_screening.png']:
        deps[str(p.relative_to(ROOT))]=sha(p)
    for rel,digest in frozen_manifest['dependencies'].items():
        if rel.startswith('scripts/'):deps[rel]=digest
    dump(OUT/'selection.json',dict(cases=cases,roles=ROLES,dependencies=deps,
         selected_and_roles_recorded_before_layout=True,
         settings=frozen_manifest['settings'],threshold_changes=False,
         baseline_scope='Original-page layout only; no restored npj or new diffusion inference',
         novelty='Pages absent from frozen layout development8; may have been screened or used by previous v6. Not unused-book or independent damage accuracy claim',
         illustration_review='All five originals visually reviewed as writing pages',
         heading_witness_original_xywh=[560,210,1200,430]))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    results=[]
    for c in cases:
        original=read(ROOT/c['original'])
        masks,diagnostic=fixed.detect_frame(original)
        native=OUT/'native'/c['page']
        for k in ['interior','exterior','footprint']:write(native/(k+'.png'),masks[k])
        assert not (masks['interior']&masks['exterior']).any()
        assert not (masks['exterior']&~masks['footprint']).any()
        H,W=original.shape[:2]
        expected=np.ones((H,W),bool)
        chosen=diagnostic['chosen']
        if chosen:
            expected[:]=False;x0,y0,x1,y1=chosen['box']
            expected[max(0,y0-8):min(H,y1+9),max(0,x0-8):min(W,x1+9)]=True
        assert np.array_equal(expected,masks['interior'])
        marked=overlay(original,[('blue',masks['exterior'])])
        figure([original,marked],['原ページ','青＝紙面内で本文外包の外'],OUT/(c['page']+'_layout.png'),
               c['page']+'：固定した配置判定の追加対照',
               ['配置のみ。npj出力・汚れ除去結果の比較ではない。閾値変更なし。',
                '外包の外にも文字があり得る。材質・編集可否のGTは未確定。'],overview=True)
        r=dict(page=c['page'],role=ROLES[c['page']],layout=diagnostic,
               exterior_pixels=int(masks['exterior'].sum()),
               layout_only=True,restoration=False,edit_success=False,
               envelope_independently_reconstructed=True)
        if c['page']=='200008316_00020_1':
            x,y,w,h=560,210,1200,430;s=np.s_[y:y+h,x:x+w]
            assert original[s].shape==(h,w,3)
            local_out=masks['exterior'][s]
            # Visible heading strokes used only as a post-detection witness.
            L=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
            dark=L[s]<30
            r['heading_witness']=dict(original_xywh=[x,y,w,h],
                 dark_pixels=int(dark.sum()),dark_pixels_outside_envelope=int((dark&local_out).sum()),
                 is_pixel_GT=False,scope='Dark pixels in observed heading region; not character segmentation')
            figure([original[s],marked[s]],['原画像・実際の見出し','青＝保持外包の外'],
                   OUT/'heading_outside_frame_witness.png','枠外の見出し：枠外を編集可とみなせるか',
                   ['配置判定だけ。汚れ選別・npj復元・文字消失は実行／測定していない。',
                    '本文枠外に観測された見出しがある。文字の保持は別途必要。'],overview=True)
        results.append(r)
        print(c['page'],chosen['box'] if chosen else None,flush=True)
    dump(OUT/'diagnostics.json',dict(cases=results))
    assert all(sha(ROOT/rel)==digest for rel,digest in deps.items())
    dump(OUT/'verification.json',dict(source_dependencies_unchanged=True,
         original_panels_verified_on_save=True,native_geometry_audited=True,
         thresholds_unchanged=True,no_inference=True,no_edit=True,
         no_pixel_GT=True,illustrations_excluded=True))


if __name__=='__main__':main()
