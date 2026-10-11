"""Development: dirt-only edits on saved npj, independent of writing frames.

Compare the same broad-dark components with no original-stroke guard, with a
neutral-ridge/warm-network guard, and with an extra brown-color condition.
No stroke generation, no model inference, no automatic adoption or pixel GT.
"""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT,read,write,sha,dump,overlay
from package_saved_npj_role_separated_review import figure
from probe_saved_npj_role_separated_postprocess import frame_guard
import probe_npj_full_page_layout as components
import probe_defect_context_preprocessing as context
import probe_npj_dirt_clean_reference as paper

OUT=ROOT/'tmp_work/npj_contact_dirt_guard_20261004'
PREVIOUS=ROOT/'tmp_work/npj_full_page_layout_v4_20261004'
SETTINGS=dict(
    feature_scope='Entire original page and fixed v6 roles; no ROI names or display boxes in feature inference',
    component_scope='Available aligned saved-npj frame; broad components evaluated before stroke/color exclusions',
    broad_components=components.SETTINGS['broad_damage'],
    saved_npj_residual=components.SETTINGS['saved_npj_residual'],
    residual_edit='Restrict accepted component to saved npj Lab L<30 plus 2px elliptical halo; frame border2 excluded',
    ridge='Original neutral ridge from Oct1: sigma1.5/3/6, response>=1.5, delta_a<=4, a<=6, paper L-drop>=6;4px halo',
    warm_guard='Fixed v6 protect|hold|red closed5px; components touching red|extra, then2px halo',
    mandatory='Fixed-v6 red|extra and aligned existing CSC red where recorded',
    modes=['no_guard','ridge_guard','ridge_and_colour'],
    colour_ablation='Additional pointwise original delta_a>=2.5, a>=2, b>=.9*max(a,0); not a replay of old component classifier',
    paper_reference='Same footprint minus original stroke/warm guard in every mode; existing Oct2 bright/closing/sigma2/4-neighbour solver',
    no_layout_requirement=True,no_stroke_generation=True,no_new_model_inference=True,
    all_cases_development=True,adopted=False,pixel_GT=False,
    limitations=['An original ridge is not proof of writing; defect rims can be guarded',
                 'Low-contrast true writing may fall outside the guard',
                 'Broad dark brown paper and binding shadows are not confirmed stains',
                 'Only saved npj black residuals are tested; faint dirt or existing text loss not solved'])


def source_cases():
    cases=components.sources()
    for c in cases:
        c['previous_output']=str((PREVIOUS/'native'/c['id']/'repaired.png').relative_to(ROOT))
    old=json.loads((ROOT/'tmp_work/npj_literature_trial_20261002/selection.json').read_text())
    choices={'page_200019865':[1100,150,768,768],
             'known_200010454_v2':[1260,1740,620,620],
             'old_full_page':[60,650,768,768]}
    for c in old['frames']:
        if c['id'] not in choices:continue
        assert c['input_reference_exact'] and not c['illustration']
        cases.append(dict(id=c['id'],page=c['page'],original=c['original'],
             saved_npj=c['saved_npj'],v6_roles=c['roles'],xywh=choices[c['id']],
             baseline_full_page=True,baseline_kind=c['baseline_label'],
             original_input=c['input'],npj_red=None,
             role='Saved npj preservation control; faint writing / red seal / margin marks'))
    return cases


def features(original,roles):
    _,f,diagnostic=context.infer(original,roles)
    lab=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    network=roles['protect']|roles['hold']|roles['red']
    k5=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    connected=cv2.morphologyEx(network.astype(np.uint8),cv2.MORPH_CLOSE,k5)>0
    n,labels=cv2.connectedComponents(connected.astype(np.uint8),connectivity=8)
    counts=np.bincount(labels.ravel(),weights=(roles['red']|roles['extra']).ravel(),minlength=n)
    anchored=counts>0;anchored[0]=False
    warm=cv2.dilate((anchored[labels]&connected).astype(np.uint8),k5)>0
    ridge=cv2.dilate(f['neutral_ridge'].astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0
    # Paper footprint alone; no fitted frame or assumption that its outside is dirt.
    bright=(lab[:,:,0]>=35).astype(np.uint8)
    footprint=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,
             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(65,65)))
    contours,_=cv2.findContours(footprint,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    page=np.zeros(bright.shape,np.uint8)
    if contours:cv2.drawContours(page,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
    footprint=cv2.erode(page,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))>0
    colour=(f['delta_a']>=2.5)&(lab[:,:,1]>=2)&(lab[:,:,2]>=.9*np.maximum(lab[:,:,1],0))
    return dict(drop=cv2.GaussianBlur(f['light_drop'],(0,0),8,borderType=cv2.BORDER_REFLECT_101),
         b=lab[:,:,2],ridge_guard=ridge,warm_guard=warm,guard=ridge|warm,
         colour=colour,footprint=footprint),diagnostic


def main():
    assert not OUT.exists(),'Use a new output name'
    OUT.mkdir(parents=True)
    frozen=json.loads((PREVIOUS/'complete.json').read_text())
    for rel,digest in frozen['outputs'].items():assert sha(PREVIOUS/rel)==digest,rel
    for rel,digest in json.loads((PREVIOUS/'manifest.json').read_text())['dependencies'].items():
        assert sha(ROOT/rel)==digest,rel
    cases=source_cases();deps={}
    for p in [Path(__file__),Path(context.__file__),Path(components.__file__),Path(paper.__file__),
              ROOT/'scripts/benchmark/probe_saved_npj_role_separated_postprocess.py',
              ROOT/'scripts/benchmark/package_saved_npj_role_separated_review.py',
              ROOT/'scripts/benchmark/build_npj_v6_comparison.py',PREVIOUS/'complete.json',
              ROOT/'tmp_work/npj_literature_trial_20261002/selection.json']:
        deps[str(p.relative_to(ROOT))]=sha(p)
    for c in cases:
        for key in ['original','saved_npj','previous_output','npj_red','original_input']:
            if c.get(key):deps[c[key]]=sha(ROOT/c[key])
        for key in ['red','extra','protect','hold','edit']:
            p=ROOT/c['v6_roles']/(key+'.png');deps[str(p.relative_to(ROOT))]=sha(p)
    dump(OUT/'manifest.json',dict(settings=SETTINGS,cases=cases,dependencies=deps,
         source_sha256=sha(Path(__file__)),conditions_recorded_before_outputs=True))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    records=[]
    for c in cases:
        original=read(ROOT/c['original'])
        if c.get('original_input'):assert np.array_equal(original,read(ROOT/c['original_input']))
        x,y,w,h=c['xywh'];sl=np.s_[y:y+h,x:x+w]
        base=read(ROOT/c['saved_npj'])
        if c.get('baseline_full_page'):base=base[sl]
        assert original[sl].shape==base.shape==(h,w,3)
        full_roles={k:read(ROOT/c['v6_roles']/(k+'.png'),True) for k in ['red','extra','protect','hold','edit']}
        assert all(v.shape==original.shape[:2] for v in full_roles.values())
        f,old_colour_components=features(original,full_roles)
        roles={k:v[sl] for k,v in full_roles.items()}
        mandatory=roles['red']|roles['extra']
        if c.get('npj_red'):mandatory|=cv2.imread(str(ROOT/c['npj_red']),0)[sl]<127
        f={k:v[sl] for k,v in f.items()}
        broad,component_diagnostics=components.select_update(base,f['drop'],f['b'],f['footprint'],mandatory)
        L=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        dark=L<30
        residual=cv2.dilate(dark.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
        candidate=broad&residual&frame_guard((h,w),2)&~mandatory
        updates=dict(no_guard=candidate,ridge_guard=candidate&~f['guard'],
                     ridge_and_colour=candidate&~f['guard']&f['colour'])
        domain=f['footprint']&~f['guard']&frame_guard((h,w),2)
        native=OUT/'native'/c['id']
        for key,a in dict(original=original[sl],baseline=base,mandatory=mandatory,
             guard=f['guard'],ridge_guard=f['ridge_guard'],warm_guard=f['warm_guard'],
             colour=f['colour'],footprint=f['footprint'],broad=broad,
             candidate=candidate,domain=domain).items():write(native/(key+'.png'),a)
        methods={};results={}
        for name,update in updates.items():
            known,threshold=paper.paper_reference(base,update,domain,mandatory)
            repaired,applied,used,solves=paper.fill(base,update,known)
            poison=base.copy();poison[~known]=[255,0,255]
            tested,again,used_again,_=paper.fill(poison,update,known)
            assert np.array_equal(tested[applied],repaired[applied])
            assert np.array_equal(again,applied) and np.array_equal(used_again,used)
            assert np.array_equal(repaired[~applied|mandatory],base[~applied|mandatory])
            if name!='no_guard':assert np.array_equal(repaired[f['guard']],base[f['guard']])
            for k,a in dict(update=update,reference=known,repaired=repaired,applied=applied,used_reference=used).items():
                write(native/name/(k+'.png'),a)
            after=cv2.cvtColor(repaired.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
            results[name]=repaired
            methods[name]=dict(candidate_pixels=int(update.sum()),applied_pixels=int(applied.sum()),
                 changed_pixels=int(np.any(repaired!=base,2).sum()),
                 removed_dark_pixels=int((applied&dark&(after>60)).sum()),
                 candidates_on_original_guard=int((update&f['guard']).sum()),
                 candidates_on_v6_keep=int((update&(roles['protect']|roles['hold'])).sum()),
                 reference_L_threshold=threshold,solves=solves,
                 excluded_reference_poison_invariant=True,outside_applied_exact=True,mandatory_exact=True)
        previous=read(ROOT/c['previous_output']) if c.get('previous_output') else base
        write(native/'previous.png',previous)
        figure([original[sl],base,previous,results['no_guard'],results['ridge_guard']],
               ['原画像','固定npj基準','前回の枠外補修／基準','線保持なし・汚れ候補','原画像の線を保持'],
               OUT/(c['id']+'_comparison.png'),c['page']+'：枠内の残留黒を対象にする開発比較',
               ['追加DDRM・文字生成なし。原画像の全ページから線と紙面を推定。',
                '右2列は同じ広い暗部＋npj黒残留。線保持の有無だけを変える。',
                '線は汚れでも応答する。弱い本物の線が外れる可能性もある。自動採用しない。'],overview=True)
        figure([original[sl],results['ridge_guard'],results['ridge_and_colour']],
               ['原画像','線保持・色差条件なし','同じ範囲＋茶色色差条件'],
               OUT/(c['id']+'_colour_ablation.png'),c['page']+'：茶色色差による追加選別',
               ['同じ広い暗部／残留／線保持条件。色条件の追加だけを比較。',
                '旧版の連結成分選別の再現ではない。無変更は除去成功ではない。'],overview=True)
        marked=overlay(original[sl],[('edit',updates['ridge_guard']),('blue',f['guard']),('hold',mandatory)])
        figure([original[sl],marked],['原画像','黄＝候補、青＝線保持、紫＝必須保持'],
               OUT/(c['id']+'_scope.png'),c['page']+'：黒残留の選別と保持',
               ['原画像の濃淡線は保持の手掛かり。材質や文字のGTではない。',
                '候補・保持マスクの内外で、保存npjを土台にする。'],overview=True)
        records.append(dict(id=c['id'],page=c['page'],components=component_diagnostics,
             previous_original_colour_components=old_colour_components,methods=methods))
        print(c['id'],{k:(v['candidate_pixels'],v['applied_pixels'],v['removed_dark_pixels']) for k,v in methods.items()},flush=True)
        del original,full_roles,f;gc.collect()
    assert all(sha(ROOT/rel)==digest for rel,digest in deps.items())
    dump(OUT/'diagnostics.json',dict(cases=records))
    dump(OUT/'verification.json',dict(source_dependencies_unchanged=True,
         reference_poison_invariant=True,mandatory_and_outside_applied_exact=True,
         guarded_modes_preserve_guard_exact=True,comparison_panels_exact_on_save=True,
         conditions_recorded_before_outputs=True,no_new_inference=True,no_stroke_generation=True,
         pixel_GT=False,adopted=False))


if __name__=='__main__':main()
