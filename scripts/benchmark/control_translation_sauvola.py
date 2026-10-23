"""Local-method control for the frozen512 translation diagnostic. No training."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import isolate_translation_global_gates as gate
from compare_v5_aligned_fade_control import dump,sha,save
from apply_ragan_real_regions import panel


def sauvola(a):
    # Same formula as the archived classical baseline: normalized gray, R=.5.
    g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype('float32')/255
    m=cv2.boxFilter(g,-1,(61,61),borderType=cv2.BORDER_REFLECT_101)
    second=cv2.boxFilter(g*g,-1,(61,61),borderType=cv2.BORDER_REFLECT_101)
    s=np.sqrt(np.maximum(second-m*m,0))
    return g < m*(1+.2*(s/.5-1))


def main():
    p=gate.check();out=gate.OUT;assert not (out/'classical_control_plan.json').exists()
    ref=gate.ROOT/'research_records/tmp_work/classical_black_comparison_20261008/compare.py'
    dump(out/'classical_control_plan.json',dict(
        status='Adaptive control after natural translations and before classical outputs',
        script_sha256=sha(__file__),baseline_formula_reference=str(ref.relative_to(gate.ROOT)),baseline_source_sha256=sha(ref),
        parent_plan_sha256=sha(out/'plan.json'),method='Sauvola w61,k.2,R.5 on normalizedfloat32 gray, reflect101 border',
        cases=p['parent_plan']['cases'],shifts=p['parent_plan']['shifts_xy'],
        purpose='Check translation/alignment/partial-label scoring using a radius30 local classical method. Not a new accuracy benchmark or threshold search.',no_training=True))
    rows=[];figures=[]
    for case in p['parent_plan']['cases']:
        k=case['id'];base=gate.prev.OUT/'inputs'/k;a=np.asarray(Image.open(base/'original.png'))
        labels={s:np.asarray(Image.open(base/f'{s}.png'))>0 for s in ['core','paper']}
        selected=labels['core']|labels['paper'];zero=None;x0,y0,x1,y1=case['figure_box'];sl=np.s_[y0:y1,x0:x1]
        items=[('原画像',gate.pad(a[sl]))]
        for dx,dy in p['parent_plan']['shifts_xy']:
            shifted=np.roll(a,(dy,dx),axis=(0,1));pred=np.roll(sauvola(shifted),(-dy,-dx),axis=(0,1))
            if zero is None:zero=pred.copy()
            # Paper-probability proxy0/1 used only to reuse the same binary scorer.
            metrics=gate.prev.parent.score((~pred).astype(np.float32),labels,.5)
            dest=out/'classical_control'/k/f'dx{dx}_dy{dy}.png';save(dest,pred)
            reload=np.asarray(Image.open(dest))<128;assert np.array_equal(pred,reload)
            row=dict(case=k,shift_xy=[dx,dy],**metrics,selected_disagreement=int(((pred!=zero)&selected).sum()),
                mask_file=str(dest.relative_to(gate.ROOT)),mask_sha256=sha(dest));rows.append(row)
            items.append((f'x{dx} y{dy} 墨{row["core_r2"]} 黒{row["paper_ink"]}',gate.pad(np.where(pred[sl],0,255).astype(np.uint8))))
        fig=out/'figures'/f'Sauvola_{k}.png';panel(fig,k[:3]+' Sauvola61/.2 / 入力移動の対照',items);figures.append(str(fig.relative_to(out)))
    dump(out/'classical_control.json',dict(plan_sha256=sha(out/'classical_control_plan.json'),rows=rows,
        figures={f:sha(out/f) for f in figures},all_selected_binary_predictions_unchanged=all(r['selected_disagreement']==0 for r in rows),no_training=True))
    print('Saved27 Sauvola controls,3 PNGs; selected disagreements:',sum(r['selected_disagreement'] for r in rows),flush=True)


if __name__=='__main__':main()
