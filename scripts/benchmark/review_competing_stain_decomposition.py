"""Verify and summarize the frozen three-component development experiment."""
from pathlib import Path
import html
import json
import hashlib
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'tmp_work/ink_stain_competition_v4_refined_20260930'
OUT = ROOT/'tmp_work/ink_stain_competition_review_20260930'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    frozen = json.loads((SOURCE/'complete.json').read_text())
    for name, expected in frozen['outputs'].items():
        assert sha(SOURCE/name) == expected, name
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    for name, expected in manifest['dependencies'].items():
        assert sha(ROOT/name) == expected, name
    assert sha(SOURCE/'source_snapshot.py') == manifest['source_sha256']
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []; cards = []; budget_data = []
    for row in json.loads((SOURCE/'diagnostics.json').read_text()):
        folder = SOURCE/row['page']/row['roi']
        x, y, w, h = row['xywh']
        results = {}; family = []
        original = cv2.imread(str(folder/'original.png'))
        edit_path = ROOT/'tmp_work/warm_ink_hold_v6_20260930'/row['page']/'split_direction_edit_allowed.png'
        edit = cv2.imread(str(edit_path), 0)[y:y+h,x:x+w] > 0
        for name, info in row['fits'].items():
            fit = np.load(folder/(name+'.npz'))
            assert np.isfinite(fit['ink']).all() and (fit['ink'] >= 0).all()
            assert np.isfinite(fit['coefficients']).all() and (fit['coefficients'] >= 0).all()
            ink = cv2.resize(fit['ink'], (w+96, h+96), interpolation=cv2.INTER_LINEAR)[48:48+h,48:48+w]
            stain = cv2.resize(fit['stain'], (w+96, h+96), interpolation=cv2.INTER_LINEAR)[48:48+h,48:48+w]
            family.append(ink)
            assert int((ink >= .015).sum()) == info['ink_above_gate']
            assert np.isclose(ink.sum(), info['ink_sum'], rtol=1e-6)
            if name == 'no_stain':
                reference = ink
            results[name] = {key: info[key] for key in ['converged', 'iterations', 'relative_kkt',
                'ink_density_mass_ratio_to_no_stain', 'ink_above_gate', 'ink_above_gate_on_v6_edit',
                'lost_above_gate_from_no_stain', 'old_solver_relative_l1']}
            results[name].update(stain_density_mass=float(stain.sum()),
                combined_mass_ratio=float((ink.sum()+stain.sum())/max(reference.sum(), 1e-12)),
                added_above_gate_from_no_stain=int(((ink >= .015)&(reference < .015)).sum()))
        minimum, maximum = np.min(family, axis=0), np.max(family, axis=0)
        stable = (minimum >= .015)&edit
        sensitive = (maximum >= .015)&edit&~stable
        budget_data.append((reference[edit],int(stable.sum()),minimum[edit]))
        dest = OUT/row['page']/row['roi']; dest.mkdir(parents=True, exist_ok=True)
        old = original.copy(); old[edit] = np.rint(.6*original[edit]+.4*np.array([210,70,220])).astype(np.uint8)
        overlay = original.copy()
        overlay[stable] = np.rint(.4*original[stable]+.6*np.array([40,180,40])).astype(np.uint8)
        overlay[sensitive] = np.rint(.4*original[sensitive]+.6*np.array([0,170,245])).astype(np.uint8)
        assert cv2.imwrite(str(dest/'v6_edit.png'), old)
        assert cv2.imwrite(str(dest/'v6_edit_density_sensitivity.png'), overlay)
        rows.append(dict(page=row['page'], roi=row['roi'], fits=results,
            v6_editable_pixels=row['v6_editable_pixels'], all_fits_above_gate_on_v6_edit=int(stable.sum()),
            some_fits_above_gate_on_v6_edit=int(sensitive.sum()),
            density_gate_sensitivity=[dict(gate=gate, all_fits_above_gate_on_v6_edit=int(((minimum >= gate)&edit).sum()))
                for gate in [.005,.01,.015,.02,.03,.05,.08,.1,.15,.2]]))
        rel = str(dest.relative_to(OUT))
        source_rel = '../'+str(folder.relative_to(ROOT/'tmp_work'))
        figures = [('原画像',source_rel+'/original.png'),('v6の編集候補',rel+'/v6_edit.png'),
            ('編集候補上の成分の安定性',rel+'/v6_edit_density_sensitivity.png'),
            ('汚れ層を加えたときの変化',source_rel+'/both_low_change.png')]
        cards.append('<article><h2>'+html.escape(row['page']+' / '+row['roi'])+'</h2><div class="grid">'+''.join(
            '<figure><figcaption>'+label+'</figcaption><a href="'+url+'"><img src="'+url+'"></a></figure>'
            for label,url in figures)+'</div></article>')
    # Equal-size comparator: simply keep the largest no-stain densities across
    # the exact same existing v6 edit pixels in the five development rectangles.
    values=np.concatenate([v for v,_,_ in budget_data])
    proposed_values=np.concatenate([v for _,_,v in budget_data])
    budget=sum(n for _,n,_ in budget_data)
    selected=np.zeros(len(values),bool)
    selected[np.argsort(-values,kind='stable')[:budget]]=True
    offset=0
    for row,(v,count,_) in zip(rows,budget_data):
        row['equal_budget_no_stain_on_v6_edit']=int(selected[offset:offset+len(v)].sum())
        offset+=len(v)
    comparator=dict(total_existing_edit_pixels=len(values),additional_hold_budget=budget,
        baseline='no_stain density descending, deterministic stable tie order',
        baseline_last_selected_density=float(np.min(values[selected])) if budget else None,
        proposed='minimum ink density over all five fits >= .015',
        scope='Only these five mixed development rectangles; not GT or a deployable threshold')
    allocations=[]
    rankings={'no_stain':np.argsort(-values,kind='stable'),
              'minimum_over_fits':np.argsort(-proposed_values,kind='stable')}
    for count in sorted(set([0,budget,len(values)]+[int(len(values)*f) for f in np.arange(.1,1.,.1)])):
        item={'budget':count}
        for label,ranking in rankings.items():
            active=np.zeros(len(values),bool); active[ranking[:count]]=True
            offset=0; per_roi={}
            for row,(v,_,_) in zip(rows,budget_data):
                per_roi[row['roi']]=int(active[offset:offset+len(v)].sum()); offset+=len(v)
            item[label]=per_roi
        allocations.append(item)
    comparator['allocation_sensitivity']=allocations
    summary = dict(rows=rows, equal_budget_comparison=comparator, pixel_GT=False, improvement_over_v6_demonstrated=False,
        caveat='Density mass ratios over mixed diagnostic rectangles; not ink recall or stain specificity')
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    verification = dict(status='passed', frozen_output_hashes_checked=len(frozen['outputs']),
        dependency_hashes_checked=len(manifest['dependencies']), saved_density_statistics_reproduced=True,
        all_solvers_converged=all(f['converged'] for r in rows for f in r['fits'].values()),
        browser_rendering_tested=False, accuracy_evaluated=False)
    (OUT/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    header = ['roi', 'no_stain pixels', 'blobs mass%', 'creases mass%', 'both low mass%', 'both high mass%', 'both low on v6 edit', 'v6 edit']
    lines = [' | '.join(header)]
    for r in rows:
        f = r['fits']
        cells = [r['roi'], str(f['no_stain']['ink_above_gate'])]
        cells += [f"{100*f[name]['ink_density_mass_ratio_to_no_stain']:.1f}" for name in ['blobs_low','creases_low','both_low','both_high']]
        cells += [str(f['both_low']['ink_above_gate_on_v6_edit']), str(r['v6_editable_pixels'])]
        lines.append(' | '.join(cells))
    (OUT/'diagnostic_table.txt').write_text('\n'.join(lines)+'\n')
    (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>汚れ層の比較分析</title><style>body{font-family:sans-serif;margin:24px;background:#faf8f3}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{max-width:100%}article{margin:32px 0}p{line-height:1.6}</style><h1>汚れ層と、v6の編集候補上での感度診断</h1><p>2列目：紫は既存v6の編集候補。3列目：その画素のうち、緑は全5条件で筆記候補密度0.015以上、橙は条件依存。これは正解ラベルでも新しい編集マスクでもありません。4列目：汚れ層追加後の全域の変化（緑は寄与が残る、紫は診断しきい値未満へ低下）。</p><p><a href="report.md">分析</a> · <a href="summary.json">集計</a> · <a href="verification.json">検証</a></p>'+''.join(cards)+'</html>')
    print('\n'.join(lines))
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
