"""Summarize every fixed-global-gate intervention; no model execution."""
import csv
from pathlib import Path
import shutil
import isolate_translation_global_gates as exp
from compare_v5_aligned_fade_control import dump, sha


def main():
    p=exp.check();out=exp.OUT;r=exp.prev.read(out/'results.json');v=exp.prev.read(out/'verification.json')
    assert v['results_sha256']==sha(out/'results.json')
    rows=r['rows']
    fields=['condition','case','shift_xy','core_exact','core_r2','paper_ink','natural_core_exact','natural_core_r2','natural_paper_ink',
            'natural_vs_fixed_selected_disagreement','natural_vs_fixed_selected_probability_mean_abs','natural_vs_fixed_selected_probability_max_abs']
    with (out/'全条件.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    lines=['# GGCの大域係数を固定した原因診断','','保存済み4モデル、3入力、9配置。追加学習は実行していない。',
           '', '各入力の原点で得たGGCの4係数を、全配置で同じ値に固定した。'
           '入力・畳み込み重み・閾値0.5は変更していない。'
           '比較相手は前回保存した通常の配置別出力で、同じCPU・float32・8スレッド・テンソル配置を使用した。',
           '', '原点の通常出力12条件は前回確率と完全一致し、係数を固定した原点12条件も通常出力と完全一致した。'
           '変更したのはGGCの大域係数の経路だけである。局所特徴や縮小・拡大、境界の経路は残る。',
           '', '## 全条件', '', '墨は半径2被覆。厳密被覆はCSVにも保存した。紙黒は暫定非文字576画素内の黒化数。'
           '結果は既知の学習原画像の診断で、未知画像の精度や特定層だけの原因を確定するものではない。', '']
    summary=[]
    for cp in p['parent_plan']['checkpoints']:
        for case in p['parent_plan']['cases']:
            n,k=cp['condition'],case['id'];rr=[x for x in rows if x['condition']==n and x['case']==k]
            lines += [f'### {n} / {k}', '', f'[元画像＋係数固定の全配置](figures/{n}_{k}.png)', '',
                      '|移動(x,y)|通常 墨/紙黒|係数固定 墨/紙黒|通常との判定反転|', '|---|---|---|---:|']
            for x in rr:
                lines.append(f"|{x['shift_xy']}|{x['natural_core_r2']}/{x['natural_paper_ink']}|{x['core_r2']}/{x['paper_ink']}|{x['natural_vs_fixed_selected_disagreement']}|")
            lines.append('')
            one=[x for x in rr if sum(abs(t) for t in x['shift_xy'])==1]
            aligned=[x for x in rr if sum(abs(t) for t in x['shift_xy'])==32]
            summary.append(dict(condition=n,case=k,
                natural_one_core_r2=[x['natural_core_r2'] for x in one],fixed_one_core_r2=[x['core_r2'] for x in one],
                natural_one_paper_ink=[x['natural_paper_ink'] for x in one],fixed_one_paper_ink=[x['paper_ink'] for x in one],
                fixed_aligned_core_r2=[x['core_r2'] for x in aligned],fixed_aligned_paper_ink=[x['paper_ink'] for x in aligned],
                changed_conditions=sum(x['natural_vs_fixed_selected_disagreement']>0 for x in rr)))
    lines += ['## 限界と記録','','GGC係数を固定しても残る誤りは、この4係数の変動だけでは説明できない。'
              '一方、この介入でGGC自体や全ての大域・境界効果を取り除いたことにはならない。'
              '配置の循環移動には前回と同じ遠方の継ぎ目・境界の交絡がある。', '',
              '[凍結計画](plan.json)・[全条件CSV](全条件.csv)・[独立再採点](verification.json)。']
    (out/'比較結果.md').write_text('\n'.join(lines)+'\n');dump(out/'group_summary.json',summary)
    snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
    for script in [Path(exp.__file__),Path(__file__)]:shutil.copy2(script,snap/script.name)
    dump(out/'report_check.json',dict(inference_script_sha256=sha(exp.__file__),report_script_sha256=sha(__file__),
        verification_sha256=sha(out/'verification.json'),documents={f:sha(out/f) for f in ['全条件.csv','比較結果.md','group_summary.json']},no_training=True))
    print('Saved all108 paired readouts and12 comparison PNG links',flush=True)


if __name__=='__main__':main()
