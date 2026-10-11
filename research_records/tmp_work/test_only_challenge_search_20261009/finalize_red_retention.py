"""Verify completed red-control outputs, reproduce the diagnostic, write findings."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import validate_red_retention as v


def main():
    out = v.OUT
    protocol = json.loads((out / 'protocol.json').read_text())
    complete = json.loads((out / 'ddrm_complete.json').read_text())
    diagnostics = json.loads((out / 'mask_diagnostics.json').read_text())
    cases = {c['id']: c for c in protocol['cases']}
    assert len(complete['results']) == 9
    for result in complete['results']:
        path = v.ROOT / result['output']
        assert v.driver.digest(path) == result['sha256']
        assert list(Image.open(path).size) == cases[result['id']]['input_size']
    assert all(a['methods'][m]['red_mask_source_rgb_changed_pixels'] == 0
               for a in complete['red_source_copy_audit'] for m in ['csc', 'howe'])
    control = complete['red_on_off_control']
    assert control['outside_red_changed_pixels'] == 0
    assert control['external_recomposite_vs_pipeline_on_changed_pixels'] == 0

    # This additional ROI diagnoses a visually observed failure after inference.
    case = cases['R03']; page = case['page']; roi = [1550, 2090, 200, 220]
    raw = cv2.imread(str(v.ROOT / case['source']))[:, :, ::-1]
    red = v.load(out / 'masks/howe' / page / '03_text_red.png')
    black = [v.load(out / 'masks' / m / page / '02_text_black.png') for m in ['csc', 'howe']]
    overlays = []
    for mask in black:
        overlay = raw.copy(); overlay[mask] = [25, 70, 250]; overlays.append(overlay)
    outputs = [np.asarray(Image.open(out / 'outputs' / m / page / 'result' / (page + '.png')).convert('RGB'))
               for m in ['csc', 'howe']]
    v.figure(out / 'ddrm_figures/R03_小朱印_黒混入確認.png', [raw] + overlays + outputs,
             ['原画像', 'CSC黒候補（青）', 'Howe黒候補（青）', 'CSC黒＋CSC赤', 'Howe黒＋CSC赤'], roi, 'R03')
    x, y, w, h = roi; sl = np.s_[y:y+h, x:x+w]
    small = dict(page=page, roi_xywh=roi,
        selection='Post-result diagnostic, not frozen evaluation ROI or ink GT',
        red_pixels=int(red[sl].sum()), csc_black_pixels=int(black[0][sl].sum()),
        howe_black_pixels=int(black[1][sl].sum()),
        howe_only_black_pixels=int((black[1][sl] & ~black[0][sl]).sum()))
    v.driver.save_json(out / 'R03_small_seal_black_diagnostic.json', small)

    table = '\n'.join(
        f"| {r['id']} | `{r['page']}` | {r['red_pixels_page']:,} | {r['red_pixels_roi']:,} |"
        for r in diagnostics)
    report = f'''# CSC赤とHowe黒の併用：朱印保持の検証

2026-10-09。**検出された赤画素は原画像どおり保持できるが、Howe併用では赤候補から漏れた朱印の細部が黒くなる例がある。** さらに、保存CSC赤には茶色い綴じ側が含まれる。Howeを黒文字の基本候補にする方針は維持し、赤／黒の境界処理を次の検討にする。現状を完成版としては扱わない。

[4例の比較PNG](4例_CSC赤とHowe併用_DDRM比較.png)は左から原画像、CSC赤候補を緑で表示、CSC黒＋CSC赤のDDRM、Howe黒＋CSC赤のDDRM。濃淡のある表側朱印R01、明瞭な朱印R02、墨と重なる朱印R03、由来を確認する薄い赤像R04を選んだ。R04は表側朱印の正例には数えない。

## 条件と出力の確認

全4ページは`data/test`から取得し、分割後testとバイト一致、資料IDはtrain/valに存在しないことを確認した。[分割確認](split_membership.json)。原画像と保存CSC赤を見て選んだ開発例であり、独立した最終評価用のtest例とは扱わない。正解マスクがないためPrecision/Recallは算出していない。

Howeはc=160、Canny sigma=0.6、低閾値比0.1、高閾値比0.4。DDRMは前回と同じ凍結コード、epoch350、伊勢物語背景、seed999、20段、128pxパッチ、重なり32、共有初期ノイズを使用した。今回の結果を見た再調整や孤立点除去は行っていない。保存CSC赤は変更せず、**両条件とも黒マスクから赤候補を除いて赤を優先**した。この優先処理後の比較であり、元のCSC黒をそのまま渡した条件とは異なる。

新規全ページ推論7出力（3ページ×2方式＋R02赤合成OFF）が完了。R04の2出力は、黒・赤マスク全画素、原画像ハッシュ、DDRM設定の一致を確認して旧06から再利用した。計9出力の存在・原寸サイズ・ハッシュ、新規ログにエラーなしを確認した。[凍結条件](protocol.json)、[マスク診断](mask_diagnostics.json)、[実装スナップショット](implementation_manifest.json)、[完了記録](ddrm_complete.json)。CSC赤は既存保存マスクを評価しており、スナップショット内のCSCで全マスクを再生成して一致確認したわけではない。

| 例 | ページ | 全ページ赤候補画素 | 事前選定ROI内赤候補画素 |
|---|---|---:|---:|
{table}

## 明瞭な朱印とHoweの黒混入

R01～R03の全赤候補について、両方式の出力RGBとOpenCVで復号した原画像RGBは全画素一致した。ただし、これは**赤マスク内だけの保持確認**であり、朱印全体の検出や保持が完全という意味ではない。現行パイプラインは推論後に赤マスク内の原画像をコピーする。赤をDDRMが生成・修復した結果ではない。

R03では原画像の朱印の細部にHowe黒候補が入り、最終出力に黒い点が加わる。赤候補を優先しても、赤マスクに入らなかった部分は黒に残る。[小朱印の原寸診断](ddrm_figures/R03_小朱印_黒混入確認.png)。この追加矩形は出力を見た後の診断用で、事前選定ROIや正解注釈ではない。矩形内の黒候補はCSC {small['csc_black_pixels']:,}画素、Howe {small['howe_black_pixels']:,}画素、Howeだけの黒候補{small['howe_only_black_pixels']:,}画素。矩形には周囲も含むため、この全数を誤検出数とは呼ばない。墨と朱印が実際に重なる箇所もあり、朱印矩形内の黒を一律除去する案は採らない。

個別比較：[R01](ddrm_figures/R01_赤保持比較.png)、[R02](ddrm_figures/R02_赤保持比較.png)、[R03](ddrm_figures/R03_赤保持比較.png)。別の本文矩形も比較PNGに保存したが、密な正解注釈に基づく文字保持の定量評価は行っていない。

## 赤合成ON/OFFによる確認

R02は黒マスク・原画像・DDRM設定を完全にそろえ、赤合成だけをON/OFFにした。赤候補の外で変わった出力は{control['outside_red_changed_pixels']:,}画素、OFF出力に原画像の赤候補を外部合成した画像とON出力の差も{control['external_recomposite_vs_pipeline_on_changed_pixels']:,}画素。今回のON/OFF差が最終合成だけによることを確認できた。黒マスクの異なるCSC/Howe比較については、共有初期ノイズだけで全段の画素別ノイズ一致は保証していない。

朱印の事前選定ROIは赤候補{control['朱印']['red_mask_pixels']:,}画素、ON/OFF差{control['朱印']['changed_output_pixels']:,}画素。[朱印のON/OFF](red_off_control/R02_朱印_赤合成ON_OFF.png)。茶色い綴じ側の確認矩形にも赤候補{control['綴じ側']['red_mask_pixels']:,}画素が入り、ON/OFF差{control['綴じ側']['changed_output_pixels']:,}画素がある。[綴じ側のON/OFF](red_off_control/R02_綴じ側_赤合成ON_OFF.png)。朱印以外の茶色い領域も赤クラスとして原画像合成されることが分かった。紙や綴じをすべて消すべきという意味ではなく、「赤インク」として扱う分類の問題である。

## 旧06の薄い赤像の再検討

R04（旧06）の保存赤は0画素で、両出力で薄い赤像は消える。しかし、直前ページ`200003803_00002_2`の朱印を左右反転すると、上部の輪郭と下部の文字の形が対応する。[隣接原画像の比較](R04_隣接原画像の鏡像比較.png)、[朱印の拡大比較](R04_薄い赤像と直前ページ朱印.png)。単純な反転・リサイズで、精密位置合わせや実物確認は未実施。**裏移りの可能性があり、表側朱印を消した失敗例とは確定できない。** 以前の報告にもこの留保を追記した。薄い赤を一律に保持するために閾値を緩める根拠にはしない。[由来確認記録](R04_provenance_review.json)。

## 今後の検討

1. Howe黒＋CSC赤を候補として維持し、朱印付近で黒が混入する位置に「朱インク／実際の墨／紙・染み」の局所注釈を作る。淡い表側朱印の細部と、隣接ページで確認する赤い裏移りを区別する。
2. その注釈を使い、朱印の赤候補漏れと茶色い領域の赤誤分類を別々に評価する。赤候補周辺の処理を検討する際は、重なる本物の墨を保護する。固定幅の赤マスク膨張や朱印矩形内の黒の一括除去を、そのまま採用しない。
3. 変更候補はまずマスクと局所PNGで比較し、候補を絞ってからDDRMを再推論する。今回と既存の開発例を最終test成績に使わず、最終評価には未使用資料を残す。
'''
    (out / '検討結果.md').write_text(report, encoding='utf-8')
    v.driver.save_json(out / 'final_review.json', dict(
        checked_outputs=9, report=v.driver.relative(out / '検討結果.md'),
        report_sha256=v.driver.digest(out / '検討結果.md'),
        finalizer_sha256=v.driver.digest(Path(__file__)),
        interpretation='Saved red copied exactly; Howe black contamination and brown-region red classification remain',
        pale_R04='Possible bleedthrough; not confirmed front-side red positive'))
    print('Report and 9 outputs verified', flush=True)


if __name__ == '__main__':
    main()
