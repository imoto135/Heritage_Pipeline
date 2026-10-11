"""Fixed dirt policies on two books absent from registered experiment cases.

Mixed illustration book: only two visually reviewed writing pages enter.
Missing archived CSC masks are generated using the frozen Oct2 local source;
these controls are explicitly not historical corpus-v1 output or paper replay.
"""
import argparse
import csv
import gc
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT,read,write,sha,dump,overlay
from package_saved_npj_role_separated_review import figure
from probe_saved_npj_role_separated_postprocess import frame_guard
import validate_npj_dirt_transfer as inference
import probe_npj_contact_dirt_guard as contact
import probe_npj_contact_dirt_guard_v2 as chroma
import probe_npj_full_page_layout_v4 as layout
import probe_npj_full_page_layout as broad
import probe_npj_dirt_clean_reference as paper

OUT=ROOT/'tmp_work/npj_dirt_new_book_20261004'
FROZEN=ROOT/'tmp_work/npj_dirt_control_20261002'
SPECS=[
    dict(id='new_warm_upper_loss',page='200021925_00007_1',book='200021925',xywh=[500,0,768,768],
         role='visible_paper_loss_and_frame_and_writing',material='Visible torn/missing paper near top rule; rim shade not pixel GT',new_masks=True),
    dict(id='new_warm_lower_loss',page='200021925_00007_1',book='200021925',xywh=[1050,1171,768,768],
         role='visible_paper_loss_contacting_writing_and_rule',material='Visible missing paper near bottom rule, including writing; do not reconstruct letters',new_masks=True),
    dict(id='new_warm_faint_writing',page='200021925_00004_1',book='200021925',xywh=[600,550,768,768],
         role='faint_writing_and_folded_paper_control',material='Observed writing on textured/folded paper; no stain identity asserted',new_masks=True),
    dict(id='new_gray_binding_mark',page='200004148_00062_1',book='200004148',xywh=[1100,2050,768,768],
         role='monochrome_binding_mark_and_lower_rule',material='Heavy printed/binding-side marks; material unconfirmed',new_masks=False),
    dict(id='new_gray_writing',page='200004148_00062_1',book='200004148',xywh=[600,900,768,768],
         role='monochrome_writing_control',material='Writing; fixed warm gate is expected to abstain',new_masks=False),
    dict(id='known_saved_red_seal',page='200010454_00002_2',book='200010454',xywh=[1164,1740,620,620],
         role='known_actual_saved_npj_red_seal_control',material='Actual red seal and writing; known development regression control',new_masks=False,
         saved_full='tmp_work/speed_test_single/ddrm_out/200010454_00002_2/result/200010454_00002_2.png')]


def registry():
    books=set();sources={}
    def walk(value):
        if isinstance(value,dict):
            if value.get('book'):books.add(str(value['book']))
            if value.get('page'):
                stem=str(value['page'])
                if re.match(r'^\d{9}_',stem):books.add(stem.split('_')[0])
            for v in value.values():walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
        elif isinstance(value,str):
            books.update(re.findall(r'data/(?:split_dataset(?:_csc(?:_mask)?)?|(?:train|test|val))/(?:train/|val/|test/)?([^/]+)/',value))
    for d in sorted((ROOT/'tmp_work').iterdir()):
        if not d.is_dir() or d==OUT:continue
        for name in ['manifest.json','baseline_review.json','selection.json']:
            p=d/name
            if not p.exists():continue
            value=json.loads(p.read_text());sources[str(p.relative_to(ROOT))]=sha(p);walk(value)
    return dict(registered_books=sorted(books),source_files=sources,
         scope='Top-level experimental manifests, selections and baseline reviews; original-only inventory/screening may contain these books; model training independence unknown')


def prepare():
    assert not OUT.exists(),'Use a new output name'
    r=registry();assert not {'200021925','200004148'}&set(r['registered_books'])
    inference.verified_bundle(FROZEN)
    inference.verified_bundle(ROOT/'tmp_work/npj_contact_dirt_guard_v2_20261004')
    inference.verified_bundle(ROOT/'tmp_work/npj_full_page_layout_v4_20261004')
    OUT.mkdir();dump(OUT/'book_registry_before.json',r)
    inventory={c['page']:c for c in csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv').open())}
    cases=[];deps={}
    for spec in SPECS:
        if spec['new_masks']:
            page=spec['page'];folder=OUT/'full_npj_masks'/page
            c=dict(split='train',original=f'data/train/200021925/{page}.jpg',
                 npj_input=str((folder/'00_input.png').relative_to(ROOT)),
                 npj_black=str((folder/'02_text_black.png').relative_to(ROOT)),
                 npj_red=str((folder/'03_text_red.png').relative_to(ROOT)),
                 npj_degradation=str((folder/'04_degradation.png').relative_to(ROOT)),
                 npj_background=str((folder/'05_background.png').relative_to(ROOT)),
                 mask_provenance='New full-page ACE/CSC from frozen Oct2 local source; NOT archived corpus-v1 masks')
        else:
            c=dict(inventory[spec['page']],mask_provenance='Archived corpus-v1 full-page CSC masks, copied without recomputation')
        c.update(spec);c.update(contains_illustration=False,rois=[],
             baseline_kind='Actual historical saved npj crop' if c.get('saved_full') else 'New frozen-code crop control; not historical output or whole-page DDRM',
             new_v6_book=c['book'] in {'200021925','200004148'})
        original=read(ROOT/c['original']);x,y,w,h=c['xywh']
        assert 0<=x and 0<=y and x+w<=original.shape[1] and y+h<=original.shape[0]
        deps[c['original']]=sha(ROOT/c['original'])
        if c.get('saved_full'):
            base=read(ROOT/c['saved_full']);assert base.shape==original.shape
            matching=ROOT/'tmp_work/speed_test_single/input/200010454_00002_2.jpg'
            assert np.array_equal(original,read(matching))
            deps[c['saved_full']]=sha(ROOT/c['saved_full']);deps[str(matching.relative_to(ROOT))]=sha(matching)
            write(OUT/'npj_output'/c['id']/'result'/(c['id']+'.png'),base[y:y+h,x:x+w])
        else:write(OUT/'input'/(c['id']+'.png'),original[y:y+h,x:x+w])
        cases.append(c)
    for p in [Path(__file__),Path(contact.__file__),Path(chroma.__file__),Path(layout.__file__),Path(broad.__file__),
              Path(paper.__file__),Path(inference.__file__),ROOT/'tmp_work/npj_contact_dirt_guard_v2_20261004/complete.json',
              ROOT/'tmp_work/npj_full_page_layout_v4_20261004/complete.json',FROZEN/'complete.json',
              ROOT/'tmp_work/npj_next_book_original_screen_20261004/unregistered_book_originals.png']:
        deps[str(p.relative_to(ROOT))]=sha(p)
    dump(OUT/'selection.json',dict(cases=cases,dependencies=deps,
         selection_before_new_masks_inference_and_v6=True,
         illustration_review='200021925_00007_1 and00004_1 reviewed as writing-only; all illustration/cover/title-decoration pages excluded',
         no_method_threshold_changes=True,adopted=False,material_pixel_GT=False,
         two_new_books_plus_one_known_seal_regression=True,
         limitation='One warm new book and one monochrome new book, three fresh full pages; new-book red seal unavailable; known positive seal is not an independent new-book result'))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    print('Selected 5 fresh crop controls in two books, plus 1 actual saved red-seal control',flush=True)


def masks():
    assert not (OUT/'full_mask_complete.json').exists(),'Already prepared'
    selection=json.loads((OUT/'selection.json').read_text())
    warm={c['page']:c for c in selection['cases'] if c['new_masks']}
    source=OUT/'full_original';source.mkdir()
    for c in warm.values():(source/(c['page']+'.jpg')).symlink_to(ROOT/c['original'])
    command=[sys.executable,'-B',str(FROZEN/'source_snapshot/ddrm_npj_v2/scripts/generate_mask_csc.py'),
             '--input_dir',str(source),'--output_dir',str(OUT/'full_npj_masks'),'--save_ace']
    dump(OUT/'mask_command.json',dict(command=command,
         mask_scope='Entire original page, then exact crop; new local classification not archived masks'))
    with (OUT/'mask.log').open('w') as f:
        subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    for c in selection['cases']:
        original=read(ROOT/c['original']);x,y,w,h=c['xywh']
        assert np.array_equal(read(ROOT/c['npj_input']),original)
        for key,filename in [('npj_black','02_text_black.png'),('npj_red','03_text_red.png')]:
            full=cv2.imread(str(ROOT/c[key]),0);assert full.shape==original.shape[:2]
            write(OUT/'masks'/c['id']/filename,full[y:y+h,x:x+w])
    outputs={str(p.relative_to(OUT)):sha(p) for folder in [OUT/'full_npj_masks',OUT/'masks'] for p in sorted(folder.rglob('*.png'))}
    dump(OUT/'full_mask_complete.json',dict(outputs=outputs,no_existing_mask_overwritten=True))
    print('Fresh two full-page CSC sets plus archived-mask exact crops verified',flush=True)


def infer():
    inference.OUT=OUT;inference.infer()
    p=OUT/'inference_complete.json';m=json.loads(p.read_text())
    m['scope']='5 newly inferred crop controls;1 actual historical saved crop copied without inference'
    dump(p,m)


def classify():
    import compare_saved_npj_v6 as classifier
    selection=json.loads((OUT/'selection.json').read_text());target=OUT/'classification'
    assert not (target/'complete.json').exists(),'Already classified'
    unique={c['page']:c for c in selection['cases']}
    dump(target/'baseline_review.json',dict(cases=list(unique.values()),
         selection_protocol='Original-only visual review; 3 new pages in 2 books plus known red-seal page; fixed v6'))
    classifier.OUT=target;classifier.render=lambda:None;classifier.compute()
    dump(target/'complete.json',dict(outputs={str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*.png'))},
         full_original_scope=True,thresholds_unchanged=True))
    print('Fixed v6 full-page masks ready',flush=True)


def compare():
    assert not (OUT/'comparison_complete.json').exists(),'Already compared'
    selection=json.loads((OUT/'selection.json').read_text())
    for rel,digest in selection['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    for folder,name in [(OUT,'full_mask_complete.json'),(OUT,'inference_complete.json'),(OUT/'classification','complete.json')]:
        for rel,digest in json.loads((folder/name).read_text())['outputs'].items():assert sha(folder/rel)==digest,rel
    records=[]
    for c in selection['cases']:
        original=read(ROOT/c['original']);x,y,w,h=c['xywh'];sl=np.s_[y:y+h,x:x+w]
        base=read(OUT/'npj_output'/c['id']/'result'/(c['id']+'.png'))
        roles={k:read(OUT/'classification/v6_masks'/c['page']/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        feature,old=context_features=contact.features(original,roles)
        geom,layout_diag=layout.detect_frame(original)
        roles={k:v[sl] for k,v in roles.items()};f={k:v[sl] for k,v in feature.items()}
        mandatory=roles['red']|roles['extra']|(cv2.imread(str(ROOT/c['npj_red']),0)[sl]<127)
        L=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        broad_update,_=broad.select_update(base,f['drop'],f['b'],f['footprint'],mandatory)
        residual=cv2.dilate((L<30).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
        guarded=broad_update&residual&~f['guard']&~mandatory&frame_guard((h,w),2)
        frame_update,_=broad.select_update(base,f['drop'],f['b'],geom['exterior'][sl],mandatory)
        keep=roles['protect']|roles['hold']|mandatory
        keepguard=cv2.dilate(keep.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
        v6update=roles['edit']&~keepguard&frame_guard((h,w),1)
        modes=dict(v6=(v6update,~keep&frame_guard((h,w),1),paper.paper_reference),
             full_page_layout=(frame_update,geom['exterior'][sl],paper.paper_reference),
             contact_guard=(guarded,f['footprint']&~f['guard']&frame_guard((h,w),2),chroma.paper_reference))
        n=OUT/'native'/c['id'];results={};methods={}
        for k,a in dict(original=original[sl],baseline=base,mandatory=mandatory,guard=f['guard'],
             frame_interior=geom['interior'][sl],broad=broad_update).items():write(n/(k+'.png'),a)
        for method,(update,domain,reference_fn) in modes.items():
            known,t=reference_fn(base,update,domain,mandatory)
            repaired,applied,used,solves=paper.fill(base,update,known)
            poison=base.copy();poison[~known]=[255,0,255]
            tested,again,_,_=paper.fill(poison,update,known)
            assert np.array_equal(tested[applied],repaired[applied]) and np.array_equal(again,applied)
            assert np.array_equal(repaired[~applied|mandatory],base[~applied|mandatory])
            if method=='contact_guard':assert np.array_equal(repaired[f['guard']],base[f['guard']])
            if method=='full_page_layout':assert np.array_equal(repaired[geom['interior'][sl]],base[geom['interior'][sl]])
            for k,a in dict(update=update,domain=domain,reference=known,repaired=repaired,applied=applied,used_reference=used).items():write(n/method/(k+'.png'),a)
            after=cv2.cvtColor(repaired.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            methods[method]=dict(update=int(update.sum()),applied=int(applied.sum()),
                 changed=int(np.any(repaired!=base,2).sum()),
                 dark_to_bright=int((applied&(L<30)&(after[:,:,0]>60)).sum()),
                 bright_green_in_applied=int((applied&(after[:,:,0]>60)&(after[:,:,1]<-8)).sum()),
                 reference_L_threshold=t,solves=solves,mandatory_and_outside_applied_exact=True,
                 poison_invariant=True)
            results[method]=repaired
        prior=read(n/'full_page_layout/applied.png',True)
        additional=read(n/'contact_guard/applied.png',True)&~prior
        combined=results['full_page_layout'].copy();combined[additional]=results['contact_guard'][additional]
        write(n/'combined.png',combined);write(n/'additional_applied.png',additional)
        assert np.array_equal(combined[prior],results['full_page_layout'][prior])
        figure([original[sl],base,results['v6'],results['full_page_layout'],combined],
             ['原画像','固定ローカルnpj対照','固定v6領域の紙面補修','固定：全ページ枠外','枠外維持＋線保持補修'],
             OUT/(c['id']+'_comparison.png'),c['page']+'：別冊に設定変更なしで適用',
             ['新規対照は原画素crop、同じ凍結コード・モデル・背景。過去出力／全ページDDRMではない。',
              '暖色冊は今回full-page CSC、白黒冊は保存CSC。既知の朱印対照だけは実保存npj。',
              '原画像から候補を選んでから評価。編集0や黒減少を成功率・材質GTとしない。'],overview=True)
        figure([original[sl],base,results['contact_guard']],['原画像','同じnpj基準','線保持＋参照色条件'],
             OUT/(c['id']+'_guard_only.png'),c['page']+'：枠に依存しない補修を単独確認',
             ['前回と同じ広い暗部・npj黒残留・中性線と暖色接続保持・参照ab距離4。',
              '文字生成・色の全体補正なし。未使用冊だが学習データの独立性は不明。'],overview=True)
        records.append(dict(id=c['id'],page=c['page'],role=c['role'],layout=layout_diag,
             methods=methods,additional_applied=int(additional.sum()),original_only_selection=True))
        print(c['id'],{k:(v['applied'],v['dark_to_bright']) for k,v in methods.items()},flush=True)
        del original,feature,roles,geom;gc.collect()
    dump(OUT/'diagnostics.json',dict(cases=records))
    dump(OUT/'comparison_complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'native').rglob('*.png'))},
         method_thresholds_unchanged=True,reference_poison_invariant=True,no_stroke_generation=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','masks','infer','classify','compare'])
    {'prepare':prepare,'masks':masks,'infer':infer,'classify':classify,'compare':compare}[parser.parse_args().stage]()
