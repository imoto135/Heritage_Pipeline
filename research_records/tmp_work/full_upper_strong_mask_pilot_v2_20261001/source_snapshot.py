"""Compare full-upper strong cores with and without a horizontal-shape guard."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import diagnose_unanchored_upper_cores as diagnostic
import probe_anchored_pale_margin_masks as pale
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=pale.ROOT
OUT=ROOT/'tmp_work/full_upper_strong_mask_pilot_v2_20261001'
MODES=['raw_upper','short_upper']
SETTINGS=dict(base='frozen pale6 edit on25 development pages',
    source='same full-safe-domain drop10/L80/seed32/peak6 strong-core diagnostic',
    raw_upper='only qualifying strong components with zero overlap with the frozen old core',
    short_upper='reject newly unanchored strong components whose width exceeds3*height',
    existing_core_changes=False,domain_and_guards_fixed=True,
    new_core_growth='same radius6/drop2 halo plus frozen one-pass pale6 support',
    no_recursive_chaining=True,ROI_gating=False,development=True,pages=25,pixel_GT=False,
    paper_edge_counterexample='200021802 upper horizontal physical page edge; show both modes',
    training=False,restoration=False,new_DDRM=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [diagnostic.OUT,pale.OUT]:check_bundle(folder)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for folder in [diagnostic.OUT,pale.OUT]:
        for name in ['manifest.json','complete.json']:dependencies[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    for mod in [diagnostic,pale]:dependencies[str(Path(mod.__file__).relative_to(ROOT))]=sha(Path(mod.__file__))
    inputs=pale.prior.descriptors()
    for r in inputs:dependencies[r['page']['original']]=sha(ROOT/r['page']['original'])
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=dependencies,settings=SETTINGS)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');pages=[];regions=[]
    for i,r in enumerate(inputs,1):
        page=r['page'];key=page['page'];original=read(ROOT/page['original']);source=r['source']
        v6={k:read(source/'v6_masks'/key/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        features,_=pale.prior.anchor.prior.mask.prior.MARGIN.load_features(key,original,v6)
        frozen=read(pale.OUT/'pages'/key/'drop6/edit.png',True);core=r['core'];domain=r['domain'];guard=r['guard']
        raw,components=diagnostic.infer(original,features,core,domain,guard)
        assert np.array_equal(raw,read(diagnostic.OUT/'pages'/key/'full_core.png',True))
        # Already anchored components retain the previous mask, including its frozen halo.
        support=(features['light_drop']>=10)&(cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]<=80)&domain&~guard
        support=cv2.morphologyEx(support.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0;support&=domain&~guard
        _,labels=cv2.connectedComponents(support.astype(np.uint8),connectivity=8)
        ids=[n['id'] for n in components if n['accepted'] and n['old_core_overlap']==0]
        raw=np.isin(labels,ids)&support
        rejected=np.zeros_like(core);rejection_notes=[]
        for n in components:
            x,y,w,h=n['xywh']
            if n['accepted'] and n['old_core_overlap']==0 and w>3*h:
                # Isolate the exact connected component within its bounding box.
                selected=labels==n['id'];assert int(selected.sum())==n['area']
                rejected|=selected;rejection_notes.append(n)
        short=raw&~rejected;values={};mode_notes={};dest=OUT/'pages'/key
        for mode,new_core in [('raw_upper',raw),('short_upper',short)]:
            combined_core=core|new_core
            connected,details=pale.candidate(original,features,combined_core,domain,guard,6)
            halo=cv2.dilate((new_core&~core).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))>0
            addition=(connected|(halo&(features['light_drop']>=2)))&domain&~guard&~frozen
            values[mode]=addition;mode_notes[mode]=details
            assert not np.any(addition&(guard|~domain))
            for name,item in dict(edit=frozen|addition,added=addition,strong_core=new_core,new_strong_core=new_core&~core).items():write(dest/mode/(name+'.png'),item)
        for name,item in dict(base_edit=frozen,core=core,domain=domain,guard=guard,rejected_horizontal=rejected).items():write(dest/(name+'.png'),item)
        assert not np.any(values['short_upper']&~values['raw_upper'])
        pages.append(dict(page=key,additions={m:int(v.sum()) for m,v in values.items()},
                          rejected_horizontal=rejection_notes,strong_components=components,pale_components=mode_notes))
        rois=[dict(id=key+'_whole',title='ページ全体',xywh=[0,0,original.shape[1],original.shape[0]])]+page['rois']
        for roi in rois:
            rid=roi['id'];x,y,w,h=roi['xywh'];s=(slice(y,y+h),slice(x,x+w));crop=original[s]
            before=crop.copy();before[frozen[s]]=(0,200,255);shows=[]
            for mode in MODES:
                show=before.copy();show[values[mode][s]]=(255,120,0);shows.append(show)
            figure([crop,before]+shows,['原画像','黄＝前回の差6編集','青＝上方強コアも回収','青＝横長の新コアは除外'],OUT/'comparisons'/(rid+'.png'),
                roi['title']+' · 旧余白帯より上の強コア',
                '青は今回の追加。元の保護を維持し、上方の強コアから同じ輪郭・淡い支持へ接続。復元なし。',
                '紙端の反例も表示。横長除外は新しい開発条件で独立した評価ではない。')
            regions.append(dict(id=rid,page=key,xywh=roi['xywh'],additions={m:int(v[s].sum()) for m,v in values.items()}))
        print('Full upper strong:',i,'/25',key,pages[-1]['additions'],flush=True)
        del original,v6,features,raw,short,values,support,labels;gc.collect()
    for p,d in dependencies.items():assert sha(ROOT/p)==d
    dump(OUT/'diagnostics.json',dict(pages=pages,regions=regions))
    dump(OUT/'verification.json',dict(status='passed',pages=25,raw_vs_horizontal_guard=True,
        old_edits_retained=True,domain_guards_preserved=True,ROI_gating=False,restoration=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json','.py'] and p.name not in ['manifest.json','complete.json']}))
    print('Complete:full-upper strong masks with paper-edge ablation',flush=True)


if __name__=='__main__':main()
