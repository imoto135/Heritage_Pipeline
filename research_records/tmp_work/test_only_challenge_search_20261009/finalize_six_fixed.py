"""Package six challenge observations, separate text/defect findings and limits."""
import csv,json
from pathlib import Path
import review_six_fixed as v

def main():
    out=v.OUT;proto=json.loads((out/'protocol.json').read_text());audit=json.loads((out/'inference_audit.json').read_text())
    masks=json.loads((out/'mask_audit.json').read_text());reviews=json.loads((out/'visual_review.json').read_text())
    assert len(reviews)==6 and all(r['completed'] for r in reviews)
    assert {r['id'] for r in reviews}=={c['id'] for c in proto['cases']}
    assert v.driver.digest(Path(v.__file__))==proto['script_sha256']
    assert v.driver.digest(v.driver.PIPELINE)==proto['ddrm']['code_sha256']
    assert v.driver.digest(v.driver.CHECKPOINT)==proto['ddrm']['checkpoint_sha256']
    assert v.driver.digest(v.driver.BACKGROUND)==proto['ddrm']['background_sha256']
    runtime_source=v.BASE/'fixed_holdout_validation/ddrm_runtime_manifest.json'
    runtime=json.loads(runtime_source.read_text())
    for item in runtime['files']:assert v.driver.digest(v.ROOT/item['path'])==item['sha256']
    v.driver.save_json(out/'frozen_runtime_audit.json',dict(reference=v.driver.relative(runtime_source),
        reference_sha256=v.driver.digest(runtime_source),verified_files=len(runtime['files']),files=runtime['files']))
    for c in proto['cases']:assert v.driver.digest(v.ROOT/c['source'])==c['source_sha256']
    for r in audit['results']:assert v.driver.digest(v.ROOT/r['output'])==r['sha256']
    for row in masks:
        for method in row['methods'].values():
            for rel,digest in method['mask_sha256'].items():assert v.driver.digest(v.ROOT/rel)==digest
        for rel,digest in row['csc_masks_sha256'].items():assert v.driver.digest(v.ROOT/rel)==digest
    fields=['id','task','text_csc','text_sauvola','text_howe','defect_result','limitation','next_action']
    with (out/'6課題_文字保持と劣化除去.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in reviews:writer.writerow({k:r[k] for k in fields})
    table='\n'.join(f"| {r['id']} {r['task']} | {r['text_howe']} | {r['defect_result']} | {r['next_action']} |" for r in reviews)
    association='\n'.join(f"| {r['id']} | {r['howe_extra_roi_pixels']:,} | {r['howe_extra_unassigned_roi_pixels']:,} | {r['csc_extra_association_roi']['degradation']:,} | {r['csc_extra_association_roi']['background']:,} |" for r in masks)
    optional=json.loads((out/'04_optional_audit.json').read_text())
    iso_proto=json.loads((v.BASE/'howe_csc_fusion/isolated_csc_v2/ddrm_protocol.json').read_text())
    assert iso_proto['ddrm']==proto['ddrm']
    spotcounts='、'.join(f"{s['howe_dark_pixels']}→{s['isolated_dark_pixels']}" for s in optional['spots'])
    report=f'''# 元の6課題：固定条件の文字保持と劣化除去

原画像／CSC文字／Sauvola文字／Howe文字を同じDDRM条件で比較し直した。朱印改善候補・Howe・Sauvola・DDRMの数値は変えていない。全18原寸結果を確認し、入力が以前と全画素一致する15結果はハッシュ確認付きで再利用、03の3結果だけ新たに推論した。

成果物：[6課題の比較](6課題_固定条件DDRM比較.png)、[全ページ](6課題_全ページ比較.png)、[文字保持と劣化除去の比較表](6課題_文字保持と劣化除去.csv)。各事例の原寸ROIはfigures、本文確認はtext_contextに保存した。

## 結論

文字の保持を優先する基本候補はHoweでよい。とくに01・02・03の淡い曲線や短線はCSCで欠け、Howeで連続して残る箇所を確認した。Sauvolaは比較基準として残す。一方、文字以外の黒い像をHoweが拾うこともあり、文字保持と不要像除去の単一順位は付けない。

| 課題 | Howeの文字保持 | 劣化除去・残存 | 次の扱い |
|---|---|---|---|
{table}

02は旧02より弱い代替、04も小さな孤立点の代替であり、課題の強さが同等とは主張しない。02・03は同じ資料で独立した二つの資料ではない。06の既定ROIには本文がほとんどないため、別の本文ROI[x700,y900,w800,h1000]も比較した。

全ページの比較では、02・03・06などのHowe/Sauvola出力に、紙面外周の影・撮影境界に由来するとみられる黒い線や帯も残った。既定本文ROIだけでは見えない問題として記録する。原資料の本文枠線・余白の注記と、撮影由来の外周像を区別する局所判断が必要であり、一律の外周切り落としは今回の設定には加えていない。

## CSCを他の分類に使う際の実装上の境界

凍結DDRMが直接読むのは02_text_black.pngと03_text_red.png。CSCの背景・劣化マスクをそのまま推論へ渡す処理は現在ない。今回の主比較は「黒文字マスクの比較＋共通の固定赤条件」であり、CSCの残りの分類をすべて使った4クラス統合モデルの性能評価ではない。背景・劣化は候補の由来を調べる参照情報として照合した。

Howeで増えた画素の多くは、保存CSCのどのマスクにも属さない領域にあった。CSCには黒文字近傍を劣化から外す後処理があり、未割当の保護帯が残る。未割当を背景・劣化に自動変換してHoweを削るのは避ける。過去のCSC劣化一律差し引きでも薄い表文字を削る条件を棄却済み。

以下は既定ROI内のHowe追加候補の所属数で、文字・汚れの正解数ではない。保存分類は重なる可能性もあり、所属数の合計を精度と見なさない。

| ID | Howe追加画素 | CSC未割当 | CSC劣化所属 | CSC背景所属 |
|---|---:|---:|---:|---:|
{association}

原画像と所属を並べた診断画像はdiagnostic_figuresに保存した。紫＝未割当、橙＝CSC劣化、青＝CSC背景。

## 04の孤立点は追加条件として別に確認

既存のisolated_csc_v2条件を再調整せず参照した。3点の確認矩形で暗画素は{spotcounts}、下部印字ROIの出力変化は{optional['printing_changed_pixels']}画素。これらは記述的な暗画素集計で、全資料のノイズ除去精度ではない。

これはDDRM前に黒文字マスクから限定した孤立成分を除く処理であり、DDRM本体の変更ではない。別資料で独立記号などの役割が不明な候補もあったため、基本設定には一律採用しない。[04の追加条件比較](04_孤立点除去_追加条件比較.png)。

## 03で固定赤条件の副作用を確認

元の6ページの保存CSC赤はいずれも0だった。凍結した朱印改善候補を適用すると03の染み2箇所、計1,211画素が赤保護対象になった。位置は[x1352,y966,w34,h26]と[x1367,y1036,w38,h31]。原画像からは朱印とは確認できず、染み状の色を保護した副作用として記録する。

このため03だけ3条件のDDRMを再推論した。3条件に同じ赤を使うが、その1211画素の色保持を劣化除去成功や朱印保持成功として数えない。[原画像と選択範囲](03_固定赤条件_新規選択範囲.png)。朱印の追加チューニングは行っていない。固定候補を未確認の色像へ無条件に広げる共通設定の採用は見送る。

## 07の残存像

原画像の青灰色の像には直前ページの鏡像と対応する裏移りがある。一方、出力に残る短線・点すべてが裏移りとは確認できない。既存の位置合わせ検証では厳しい除去条件で候補0となり、マスクを変更しなかった。CSCで消えるという理由だけで不要な裏移りとは判定しない。

次は07の残存像を、表文字、対向ページと対応する裏移り候補、点・傷状の像、不明へ分けて小範囲で確認する。確認できた局所候補だけを除去対象とし、Howeの本文全体のパラメータを再調整しない。判定不明は保持する。06は紙面損傷の別タスクとして扱い、枝状像を白くしたことを紙や文字の復元とは数えない。

## 条件・評価範囲

Sauvola窓61/k=.20/R=.5、Howe c160/Canny sigma.6・low.1・high.4（作者基本エネルギーのローカル実装、MATLAB Cannyと自動探索の完全再現ではない）。DDRMは凍結コード、epoch350、seed999、20 steps、patch128、overlap32、共有初期ノイズ、原寸全ページ。同じ初期ノイズでも異なる黒マスク間で全ステップの乱数が一致する保証はない。

出力原寸・入力と出力のハッシュ・エラーログなし・選択赤画素の原画像RGB一致を確認。朱印は推論後の原色コピーであり、退色・欠損の復元ではない。数値や既存の保存CSCマスクを変更していない。

6例はtest由来の開発事例で、選定・調整に既に使用した。独立した最終テストではなく、画素正解・文字列正解・OCR評価もない。CSVは目視の観察記録で、精度やCERのスコアではない。現在の条件を固定したまま、最終的な定量比較は未使用資料と注釈を別途用意して行う。
'''
    (out/'検討結果.md').write_text(report,encoding='utf-8')
    files=[Path(__file__),Path(v.__file__),v.BASE/'infer_six_fixed_parallel.py',out/'protocol.json',out/'mask_audit.json',out/'inference_audit.json',out/'visual_review.json',out/'04_optional_audit.json',out/'frozen_runtime_audit.json',out/'検討結果.md',out/'6課題_文字保持と劣化除去.csv',out/'6課題_固定条件DDRM比較.png',out/'6課題_全ページ比較.png']
    v.driver.save_json(out/'final_review.json',dict(artifact_sha256={v.driver.relative(p):v.driver.digest(p) for p in files},
        outputs=18,new_inferences=audit['new'],reused_outputs=audit['reused'],visual_reviews_completed=6,
        red_condition_retuned=False,text_conditions_retuned=False,final_test=False,
        inference_masks=['black','red'],other_csc_classes_directly_used=False))
    print('Finalized six-case visual review and 18 verified outputs',flush=True)

if __name__=='__main__':main()
