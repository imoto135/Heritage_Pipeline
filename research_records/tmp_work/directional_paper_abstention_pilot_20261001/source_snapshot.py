"""Hold two orientation-coherent paper components; original pixels, no new solve."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import diagnose_directional_paper_candidates as diagnostic
import probe_adaptive_paper_texture_scale_v2 as texture
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=diagnostic.ROOT
OUT=ROOT/'tmp_work/directional_paper_abstention_pilot_20261001'
SETTINGS=dict(baseline='same frozen upper/horizontal guard77 paper baselines',
    hold='frozen whole-page directional provisional matches, no ROI gating',
    effective_edit='baseline candidate edit minus hold; held pixels remain original and nonreference',
    effective_paper_applied='saved pure-paper applied minus hold',
    outside_hold='previous output bitwise preserved',
    comparator='saved adaptive24/16/12 texture; not combined with hold',
    paper_color_reference='unchanged; held original pixels never added to known-paper references',
    new_solve=False,new_DDRM=False,pages=25,ROIs=77,development=True,pixel_GT=False,
    interpretation='conservative restoration abstention, not proof of natural-paper physical cause')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [diagnostic.OUT,texture.OUT]:check_bundle(folder)
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for folder in [diagnostic.OUT,texture.OUT]:
        for name in ['manifest.json','complete.json']:deps[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    rows=texture.history.regions();assert len(rows)==77
    inputs=texture.history.mask.diagnostic.prior.mask.pale.prior.descriptors();originals={r['page']['page']:r['page']['original'] for r in inputs}
    for path in originals.values():deps[path]=sha(ROOT/path)
    manifest=dict(settings=SETTINGS,dependencies=deps,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');results=[];current=None;pages=[]
    for row in rows:
        rid=row['id'];key=row['page']
        if current!=key:
            original=read(ROOT/originals[key]);current=key;page_dir=diagnostic.OUT/'pages'/key
            hold=read(page_dir/'provisional_matches.png',True);edit=read(page_dir/'edit.png',True);eligible=read(page_dir/'eligible.png',True)
            known=read(page_dir/'known.png',True);assert not np.any(hold&~eligible) and not np.any(hold&known)
            for name,item in dict(base_edit=edit,paper_hold=hold,effective_edit=edit&~hold,
                effective_eligible=eligible&~hold,known=known).items():write(OUT/'pages'/key/(name+'.png'),item)
            pages.append(dict(page=key,hold_pixels=int(hold.sum())));gc.collect()
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        source=ROOT/row['baseline_local'];native=OUT/'native'/rid
        base=read(source/'output.png');crop=original[y:y+h,x:x+w];adaptive=read(texture.OUT/'native'/rid/'adaptive/output.png')
        applied=read(source/'retained_applied_context.png',True);reference=read(source/'reference_context.png',True)
        local_hold=hold[cs];reverted=applied&local_hold;value=base.copy();value[reverted[s]]=crop[reverted[s]]
        effective=applied&~local_hold;effective_edit=edit[y:y+h,x:x+w]&~hold[y:y+h,x:x+w]
        assert not np.any(reference&local_hold) and not np.any(effective&local_hold)
        assert np.array_equal(value[hold[y:y+h,x:x+w]],crop[hold[y:y+h,x:x+w]])
        assert not np.any(np.any(value!=base,axis=2)&~reverted[s])
        assert not np.any(np.any(value!=crop,axis=2)&~effective_edit)
        for name,item in dict(original=crop,baseline=base,adaptive_texture=adaptive,output=value,paper_hold_context=local_hold,
            reverted_context=reverted,baseline_applied_context=applied,effective_applied_context=effective,
            reference_context=reference,effective_edit=effective_edit).items():write(native/(name+'.png'),item)
        notes=dict(hold_candidate_pixels=int(local_hold[s].sum()),paper_reverted_pixels=int(reverted[s].sum()),
            remaining_paper_applied=int(effective[s].sum()),changed_vs_base=int(np.any(value!=base,axis=2).sum()),
            original_preserved_in_hold=True,reference_pixels=int(reference.sum()))
        results.append({**row,'abstention_diagnostics':notes})
        elsewhere='旧14ページ固定ROIの紙面外は保存DDRM。' if row['base_elsewhere']=='saved_DDRM' else '紙面外は原画像。'
        figure([crop,base,adaptive,value],['原画像','固定マスクの紙面補修','同じマスク・適応模様','方向一致成分の補修保留'],OUT/'comparisons'/(rid+'.png'),
            row['title']+' · 紙面と同じ方向の候補を保留する比較',
            f'保留で原画素へ戻す {int(reverted[s].sum()):,}画素。それ以外は基準と完全一致。',
            elsewhere+' 原模様も損傷も残す保守的候補。方向から物理原因を断定しない。')
        print('Directional paper abstention:',len(results),'/77',rid,notes['changed_vs_base'],flush=True)
    for path,value in deps.items():assert sha(ROOT/path)==value
    dump(OUT/'diagnostics.json',dict(pages=pages,regions=results))
    dump(OUT/'verification.json',dict(status='passed',pages=25,ROIs=77,ROI_gating=False,hold_original_preserved=True,
        outside_hold_exact_baseline=True,held_pixels_not_used_as_color_or_texture_references=True,
        outside_effective_edit_changes=0,new_solve=False,new_DDRM=False,pixel_GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:77 original/texture/abstention comparison PNGs',flush=True)


if __name__=='__main__':main()
