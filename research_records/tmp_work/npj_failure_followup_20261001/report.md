# 追加3ページの固定比較

[比較PNG](comparison_summary.png) / [全8枚のZIP](npj_v6_followup_comparison_images.zip)

原画像と保存済みnpjだけで追加候補13ページを目視し、イラスト3ページを除外。v6を見る前に3ページ・3冊の問題3箇所と本文対照1箇所を固定した。開発冊以外は200014740の1ページ、残り2ページは既知冊の新しいページ。代表抽出ではない。

薄い筆画は部分改善1、本文に接する茶色い部分と別冊の穴・裂けは未解決2。対照の主要筆画は維持された。すべて補助者の暫定目視で、画素GT・全資料の精度ではない。茶色い部分の原因が汚れか損傷かは未確定。

4箇所ともv6固有の追加暖色保留は0画素。今回の差をv6固有の改良効果とは数えず、固定v6一式と旧npjの比較として扱う。暖色保留・基礎処理の閾値は変更していない。

マスクの色だけでは除去品質を判断できないため、前回と合わせた11ページ・15箇所に同一の局所補間を適用した。[筆画の実際の変化](../npj_v6_edit_effect_probe_20261001/02_stroke_edit_effect.png)、[残る汚れ・損傷](../npj_v6_edit_effect_probe_20261001/03_remaining_edit_effect.png)、[補間の所見と次の検証](../npj_v6_edit_effect_probe_20261001/report.md)。

再実行：`python -B scripts/benchmark/run_npj_failure_followup.py`。画像のまとめと所見：`python -B scripts/benchmark/finalize_npj_png_followup.py`。入力・固定コードのハッシュ、分類の排他性、原寸マスクと比較画像の一致を確認済み。原画像・保存済みnpjを変更していない。
