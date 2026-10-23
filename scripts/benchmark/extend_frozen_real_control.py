"""Quarantine a coordinate-invalid control; freeze a new source-only control."""
import argparse, importlib.util, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import frozen_real_transfer as f
D=f.OUT/'additional_control'
def prepare():
    assert not (D/'annotation.json').exists()
    a=dict(id='R08',ink_paths=[
      dict(id='left_vertical',confidence='high',xy=[[200,281],[201,310],[201,350],[200,380]]),
      dict(id='left_horizontal',confidence='high',xy=[[172,325],[190,322],[213,321],[223,321]]),
      dict(id='right_vertical',confidence='high',xy=[[374,289],[376,316],[376,348],[378,383]]),
      dict(id='right_horizontal',confidence='high',xy=[[347,324],[369,318],[392,314],[406,314]])],
      nonink_polygons=[dict(id='column_gap',confidence='high',xy=[[253,279],[319,279],[319,359],[253,359]]),
        dict(id='upper_paper',confidence='high',xy=[[90,160],[151,160],[151,221],[90,221]])])
    f.dump(D/'annotation.json',a);c=json.loads((D/'source.json').read_text());labels,paths=f.partial.raster(c,a)
    Image.fromarray(labels).save(D/'labels.png');raw=f.rgb(D/'R08_original.png');overlay=raw.copy();overlay[labels==1]=[0,100,255];overlay[labels==2]=np.rint(.45*raw[labels==2]+.55*np.array([0,200,80])).astype(np.uint8)
    f.real.panel(D/'R08_原画像と部分注釈.png','R08 別ページの対照：青=墨芯 緑=非文字',[('原画像',raw),('Codex暫定注釈・専門家未確認',overlay)])
    f.dump(f.OUT/'annotation_review.json',dict(utc=datetime.now(timezone.utc).isoformat(),invalid_cases=['R07'],reason='R07 right_long_stroke is drawn in blank paper and clear_lower_gap overlaps visible small text; coordinate errors visible on source overlay, not a method-specific failure.',
      handling='Retain all original annotations, masks and scores; invalidate all R07 numerical method comparisons and exclude from primary totals. Source-only new page R08 chosen as simple-cross control and frozen before its predictions.',
      timing='R07 overlay was created before inference but visually reviewed after its scores were printed. This is a documented post-result annotation invalidation, not a fully blinded adjudication.',
      other_cases='R01–R06 overlays reviewed; provisional, expert-unconfirmed. R02 remains conditional secondary diagnostic.'))
def freeze():
    assert not (D/'protocol.json').exists();f.load()
    c=json.loads((D/'source.json').read_text());c['source_sha256']=f.sha(f.ROOT/c['source'])
    x,y,w,h=c['roi'];assert np.array_equal(f.rgb(f.ROOT/c['source'])[y:y+h,x:x+w],f.rgb(D/'R08_original.png'))
    files=[D/'source.json',D/'annotation.json',D/'labels.png',D/'R08_original.png',Path(__file__),f.OUT/'protocol.json',f.OUT/'annotation_review.json']
    f.dump(D/'protocol.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),case=c,files={str(p.relative_to(f.ROOT)):f.sha(p) for p in files},
      methods=f.NAMES,inherit=f.sha(f.OUT/'protocol.json'),source_overlay_visually_reviewed=True,no_R08_predictions_seen=True,
      selection='New source page100249371_00021_1 from pre-output source survey; simple dark cross strokes and paper gaps as control. Chosen after invalid R07 review; no model outputs from R08 used.'))
def run():
    p=json.loads((D/'protocol.json').read_text());assert not (D/'results.json').exists();f.load()
    for n,h in p['files'].items():assert f.sha(f.ROOT/n)==h
    c=p['case'];page=f.rgb(f.ROOT/c['source']);x,y,w,h=c['roi'];a=json.loads((D/'annotation.json').read_text());labels,paths=f.partial.raster(c,a)
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);rows=[]
    for m in f.NAMES:
        if m=='howe_fixed':
            spec=importlib.util.spec_from_file_location('howe_additional_control',f.ROOT/'tmp_work/howe_diagnosis_20261008/experiment.py');howe=importlib.util.module_from_spec(spec);spec.loader.exec_module(howe)
            pred_page=f.howe_mask(page,D/'scratch','howe',howe);pred=pred_page[y:y+h,x:x+w];f.save(D/'full_page_howe.png',pred_page)
        else:
            kind=m.split('_')[0];model=f.make_model(kind).to(device);q=torch.load(f.MODELS/m/'step_03024.pt',map_location='cpu',weights_only=False);model.load_state_dict(q['model']);del q;model.eval();before=f.digest_state(model)
            pred=f.real.infer_roi(model,kind,page,c['roi'],device)<.5;assert before==f.digest_state(model);del model;torch.cuda.empty_cache()
        target=D/'masks'/f'{m}.png';f.save(target,pred)
        row=dict(case='R08',book=c['book'],family='clean_control',method=m,secondary=False,prediction_sha256=f.sha(target),**f.partial.measure(pred,labels,paths));rows.append(row)
        print(m,'R08',row['ink_exact'],row['ink_r2'],'/',row['ink_n'],'nontext',row['nonink_false'],'/',row['nonink_n'],flush=True)
    f.dump(D/'results.json',dict(rows=rows,protocol_sha256=f.sha(D/'protocol.json'),unchanged_weights=True,completed_utc=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','run']);args=p.parse_args();globals()[args.action]()
