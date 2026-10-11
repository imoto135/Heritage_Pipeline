"""Package the visual first-batch findings; verdicts are provisional, not GT."""
from collections import Counter
import csv
import html
import json
from pathlib import Path

from build_npj_v6_comparison import dump,sha

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/npj_failure_collection_20261001'
FINDINGS={
 'dense_stain':('improved','汚れの中心が墨候補から編集候補に移る。周囲の本文の主要部は保護。既知の孤立した斑点での改善。',False),
 'thin_black_water':('partial','本文周辺の編集候補は大幅に減り、薄い文字を編集しにくくなる。ただし保護マスクは一部のかすれで途切れており、全筆画の検出回復は断定しない。紙のムラも編集対象から外れる。',False),
 'brown_as_red':('improved','この矩形の朱候補が1,336画素から0へ減り、茶色い斑点の主要部が編集候補へ移る。周囲の本文は保護される。',False),
 'thin_black_loop':('partial','矩形の編集候補は6,155画素から0になる。ループ周辺の筆画保護が増えるが、最も薄い左下の線は保護マスクにも断続的にしか入らない。誤編集の回避と、文字検出の回復を区別する。',False),
 'warm_stroke':('improved','分断されていた長い赤褐色の線がほぼ連続して保留側に入る。墨候補だった主要部は旧npjでも保護済み。保留は朱インクの確定分類ではない。',False),
 'dense_stain_new':('partial','広い誤インク面は縮小するが、汚れの一部が保護・保留に残る。矩形のv6編集候補は39画素しかなく、汚れ全体を編集できる対策にはなっていない。',False),
 'brown_false_red_new':('improved','筆画のない茶色い紙面の朱候補が76,777画素から0になる。一部は編集候補へ移るが、淡い広がりは背景に残る。朱の誤判定改善であり、汚れの全面除去ではない。',False),
 'hole_edge_new':('unresolved','穴・裂けの濃い縁は保護・保留に残る。保護画素は4,144から6,739へ増え、旧npjで編集候補だった薄い縁まで保護側に広がる。v6の追加保留は0なので、土台の処理による誤保護。',True),
 'thin_black_text':('control','主要な本文は両方で保護され、目視で明確な大片欠損の追加は見えない。薄い青灰色の線は本文か裏写りか未確定。',False),
 'control_black':('control','主要な濃い本文は両方で保護される。この切り出しでは、明確な本文の誤編集増加は見えない。',False),
 'control_text_new':('control','通常本文の主要部は維持される。一方、薄い影のような線や周辺の小さなムラも保護に増える。本文の維持だけで汚れ検出の良否を判定しない。',False),
}
LABELS=dict(improved='改善が見える',partial='部分改善・制限あり',unresolved='未解決',control='対照')


def main():
    selected=json.loads((OUT/'baseline_review.json').read_text());diagnostics={r['id']:r for r in json.loads((OUT/'comparison_diagnostics.json').read_text())}
    findings=[]
    for row in selected['cases']:
        for roi in row['rois']:
            verdict,obs,adverse=FINDINGS[roi['id']]
            findings.append(dict(id=roi['id'],page=row['page'],book=row['book'],xywh=roi['xywh'],
                title=roi['title'],failure_type=roi['failure_type'],baseline_failure=roi['baseline_failure'],
                evaluation_group=row['evaluation_group'],verdict=verdict,verdict_label=LABELS[verdict],observation=obs,
                adverse_protection=adverse,diagnostics=diagnostics[roi['id']],reviewer='assistant_visual_review',human_confirmed=False))
    dump(OUT/'v6_visual_review.json',dict(findings=findings,source_sha256=sha(Path(__file__)),pixel_GT=False,
        human_confirmation_pending=True,criteria='Same-ROI visual judgement of ink protection and stain editing; no claim of completed restoration'))
    problems=[r for r in findings if r['baseline_failure']];counts=Counter(r['verdict'] for r in problems)
    screening=json.loads((OUT/'screening.json').read_text());scope=list(csv.DictReader((OUT/'screening_review.csv').open()))
    summary=dict(original_thumbnails_screened=6072,candidate_pages_thumbnail_reviewed=len(scope),
        illustration_pages_excluded=sum(r['scope']=='excluded_illustration' for r in scope),cover_pages_excluded=1,
        comparison_pages=8,comparison_books=7,problem_rois=8,control_rois=3,
        visual_verdict_counts=dict(counts),adverse_protection_rois=1,
        problem_rois_on_new_pages=3,problem_rois_on_development_pages=5,
        new_book_problem_rois=2,v6_extra_active_rois=[r['id'] for r in findings if r['diagnostics']['v6_extra_pixels']>0],
        v6_extra_active_pages=[r['page'] for r in diagnostics.values() if r['id'].endswith('_whole') and r['v6_extra_pixels']>0],
        population_accuracy=False,pixel_GT=False,human_confirmation_pending=True,restoration=False)
    dump(OUT/'review_summary.json',summary)
    columns=['page','book','id','title','failure_type','evaluation_group','verdict_label','observation','adverse_protection']
    with (OUT/'case_review.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(findings)
    table='\n'.join(f'| [{r["title"]}](comparisons/{r["id"]}.png) | {r["page"]} | {r["verdict_label"]} | {r["observation"]} |' for r in findings)
    report='''# 保存済みnpjの失敗収集と固定v6比較：第1回

2026-10-01。原画像6,072ページを縮小画像で走査し、58ページの候補を原画像と保存済み旧npjマスクの一覧で目視した。挿絵・色図・模様を含む13ページと表紙1ページを主評価から除外した。本文ページ8枚・7冊を比較し、**旧npjで問題が見える8箇所は、改善が見える4、部分改善3、未解決1**と暫定記録した。対照は3箇所。

これは選んだ問題の目視結果であり、全6,072ページの精度・対策率ではない。私の目視判定で、利用者確認と画素GTは未実施。既知開発ページからの問題5箇所、新しいページからの問題3箇所を分けて記録している。新しいページのうち1枚は既知開発冊、2枚は開発冊以外。

[比較一覧](index.html)では原画像・旧npj・v6を同じ領域に重ねて確認できる。全ページと切り出しの計19枚。原寸画像と編集マスクにもリンクした。

| 問題または対照 | ページ | 暫定判定 | 所見 |
| --- | --- | --- | --- |
'''+table+'''

濃い汚れについては、孤立した斑点では編集対象へ戻せた。一方、今回新たに見つけた本文・枠線とつながった汚れでは、誤インク面は縮小しても汚れが編集対象にほとんど入らない。孤立斑点での成功だけで汚れ全般に対策できたとは言えない。

穴・裂けの縁は未解決で、保護が広がる方向の副作用がある。薄い黒筆画では編集候補に入ることを避けやすくなるが、最もかすれた筆画が文字マスクとして連続して回復したとは断定できない。対照3箇所で主要本文の明確な誤編集増加は見えなかったが、薄い影やムラまで残す傾向は引き続き確認が必要。

**今回の11切り出しのうち、v6で追加した暖色保留が動いたのは赤褐色の長い筆画1箇所だけ。** この矩形で追加保留は11,065画素。その他の切り出しの比較はv6の土台である固定v2までの処理による違いであり、v6で追加した条件の効果として数えない。ページ全体では、長い赤褐色筆画のページで20,954画素、朱印のある200010454_00002_2で1,997画素の追加保留があり、後者は今回選んだ失敗矩形の外にある。残る6ページの追加保留は0。v6を採用すべきかと、npj全体を置き換えるべきかは分けて判断する。

矩形内の画素数は分類の変化を示す診断値で、文字の救済率や汚れ除去率ではない。汚れを朱候補から背景へ移しただけの画素は、汚れが除去されたわけではない。v6は編集候補がかなり少ないため、文字を安全に残す代わりに必要な汚れ処理も止める可能性がある。復元画像の比較は未実施。

次の検証対象は、本文と接する濃い汚れ、穴・裂けの誤保護、最も薄い筆画の断続的な保護。まずこの失敗集を増やし、改善が見えた箇所と残った箇所を固定した復元器で比較して、文字の損失や汚れの残りへ実際に影響するか確認する。現在の結果からnpjの全面置換や新手法の導入は決めない。

比較基準は `data/split_dataset_csc_mask` の**保存済み旧npj v1**。ACE・CSCは再計算していない。v2や枕草子v3、以前の少数CPU再計算結果は混ぜていない。v6は `split_direction` を固定し、条件変更・学習・原画像編集は行っていない。既知5ページは元画像の画素一致と保存出力のハッシュを確認して再利用し、新規3ページは同じ処理を原寸で実行した。

[入力・コードの検証](verification.json)、[原寸の重ね合わせと旧マスクの排他性](native_overlay_verification.json)、[機械的な画素集計](comparison_diagnostics.json)、[v6を見る前の箇所と所見](baseline_review.json)、[暫定判定CSV](case_review.csv)、[全ページの索引](page_inventory.csv)。

再実行：`/home/imoto/miniconda3/envs/diffusion/bin/python -B scripts/benchmark/compare_saved_npj_v6.py compute`。選定後の比較・検証を再実行し、完成出力はハッシュで確認して再利用する。目視所見の整理は `scripts/benchmark/finalize_saved_npj_review.py`。
'''
    (OUT/'report.md').write_text(report)
    index=(OUT/'index.html').read_text()
    for r in findings:
        marker=f'<section id="{r["id"]}"><h3>'
        if marker in index:
            index=index.replace(marker,f'<section id="{r["id"]}"><p><strong>{r["verdict_label"]}</strong>：{html.escape(r["observation"])}</p><h3>')
    summary_html='<p><strong>第1回の暫定目視：問題8箇所中、改善4・部分改善3・未解決1。対照3箇所。全資料の精度ではありません。</strong></p>'
    if summary_html not in index:index=index.replace('<h1>保存済みnpjの失敗収集と固定v6比較</h1>','<h1>保存済みnpjの失敗収集と固定v6比較</h1>'+summary_html)
    (OUT/'index.html').write_text(index)
    rows=list(csv.DictReader((OUT/'page_inventory.csv').open()));fields=list(rows[0]);by_page={r['page']:r for r in scope}
    for row in rows:
        if row['page'] in by_page:
            s=by_page[row['page']];row['review_status']=s['scope'] if s['scope'].startswith('excluded') else 'thumbnail_reviewed'
        page_results=[r for r in findings if r['page']==row['page']]
        if page_results:
            row.update(review_status='baseline_and_v6_visually_reviewed',
                failure_type=';'.join(sorted({r['failure_type'] for r in page_results if r['baseline_failure']})),
                roi_xywh=json.dumps({r['id']:r['xywh'] for r in page_results}),
                v6_status=';'.join(f'{r["id"]}:{r["verdict"]}' for r in page_results),
                observation=' / '.join(f'{r["title"]}：{r["observation"]}' for r in page_results))
    with (OUT/'page_inventory.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    inventory=json.loads((OUT/'inventory_summary.json').read_text());inventory.update(all_images_visually_reviewed=False,
        images_reviewed_in_this_inventory=True,v6_newly_applied=True,
        original_thumbnails_screened=6072,candidate_pages_thumbnail_reviewed=58,
        baseline_and_v6_reviewed_pages=8,v6_newly_applied_pages=3,v6_cached_pages=5,review_summary='review_summary.json')
    dump(OUT/'inventory_summary.json',inventory)
    dump(OUT/'review_complete.json',dict(outputs={name:sha(OUT/name) for name in ['report.md','index.html','v6_visual_review.json',
        'review_summary.json','case_review.csv','page_inventory.csv','inventory_summary.json','native_overlay_verification.json']},
        judgement='provisional assistant visual review; no population accuracy',source_sha256=sha(Path(__file__))))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
