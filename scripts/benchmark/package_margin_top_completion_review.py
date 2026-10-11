"""Independently check top-boundary masks, fixed DDRM and separate fill results."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import check_margin_top_completion_ddrm as ddrm
import diagnose_margin_top_fill as fill
from build_npj_v6_comparison import dump, read, sha, write
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT = ddrm.ROOT
MASKS = ddrm.mask_probe.OUT
OUT = ROOT/'tmp_work/margin_top_completion_review_20261001'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [MASKS,ddrm.OUT,fill.OUT,ddrm.prior_ddrm.OUT,ddrm.blending.OUT]:check_bundle(folder)
    mask_rows={r['id']:r for r in json.loads((MASKS/'diagnostics.json').read_text())}
    rows=json.loads((ddrm.OUT/'diagnostics.json').read_text());lookup={r['id']:r for r in rows}
    unchanged_pages=[];unchanged_regions=[];checked=[];books=set();pages=[]
    for source in ddrm.mask_probe.old.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key=page['page'];pages.append(key);books.add(page['book'])
            original=read(ROOT/page['original']);H,W=original.shape[:2]
            old_dir=ddrm.mask_probe.old.OUT/'candidate_pages'/key
            old={k:read(old_dir/'dark_support'/(k+'.png'),True) for k in ['edit','protect','hold','red']}
            maps={k:read(MASKS/'pages'/key/'features'/(k+'.png'),True)
                  for k in ['top_margin','safe_domain','long_line_guard','warm_anchor_guard']}
            assert np.array_equal(maps['top_margin'],read(old_dir/'features/top_margin.png',True))
            modes={m:{k:read(MASKS/'pages'/key/m/(k+'.png'),True) for k in ['edit','added','protect','hold','red']}
                   for m in ddrm.MODES}
            domain=maps['safe_domain'];band=maps['top_margin']
            expected=np.zeros_like(domain)
            if band.any():
                yy,xx=np.nonzero(band)
                expected[16:int(yy.max())+1,round(.06*W):round(.94*W)]=True
            assert np.array_equal(domain,expected)
            for mode,values in modes.items():
                assert np.array_equal(values['edit'],old['edit']|values['added'])
                assert np.array_equal(values['added'],values['edit']&~old['edit'])
                for name in ['protect','hold']:assert np.array_equal(values[name],old[name]&~values['added'])
                assert np.array_equal(values['red'],old['red'])
                assert not np.any(values['added']&maps['warm_anchor_guard'])
                assert not np.any(values['edit']&(values['protect']|values['hold']|values['red']))
                allowed=band if mode=='in_band' else domain&~maps['long_line_guard']
                assert not np.any(values['added']&~allowed)
                assert int(values['added'].sum())==mask_rows[key+'_whole']['added_pixels'][mode]
                for name in values:
                    raw=cv2.imread(str(MASKS/'pages'/key/mode/(name+'.png')),cv2.IMREAD_GRAYSCALE)
                    assert set(np.unique(raw))<={0,255}
            assert not np.any(modes['in_band']['added']&~modes['completed']['added'])
            if not any(v['added'].any() for v in modes.values()):
                for values in modes.values():
                    for name in old:assert np.array_equal(values[name],old[name])
                unchanged_pages.append(key)
            for roi in page['rois']:
                rid=roi['id'];x,y,w,h=roi['xywh'];s=(slice(y,y+h),slice(x,x+w));crop=original[s]
                native=ddrm.OUT/'native'/rid
                assert np.array_equal(crop,read(native/'original.png'))
                assert np.array_equal(read(native/'old.png'),read(ddrm.prior_ddrm.OUT/'native'/rid/'candidate.png'))
                assert np.array_equal(read(native/'old_boundary.png'),read(ddrm.blending.OUT/'native'/rid/'boundary.png'))
                x0,y0=max(0,x-64),max(0,y-64);x1,y1=min(W,x+w+64),min(H,y+h+64)
                old_context=old['edit'][y0:y1,x0:x1]
                same=True
                for mode,values in modes.items():
                    assert np.array_equal(read(native/(mode+'_edit.png'),True),values['edit'][s])
                    assert np.array_equal(read(MASKS/'native'/rid/(mode+'_added.png'),True),values['added'][s])
                    assert int(values['added'][s].sum())==lookup[rid]['added_pixels'][mode]
                    for suffix in ['.png','_boundary.png']:
                        value=read(native/(mode+suffix));assert not np.any(np.any(value!=crop,axis=2)&~values['edit'][s])
                    local_fill=read(fill.OUT/'native'/rid/(mode+'_telea.png'))
                    assert not np.any(np.any(local_fill!=crop,axis=2)&~values['edit'][s])
                    equal_context=np.array_equal(values['edit'][y0:y1,x0:x1],old_context)
                    assert bool(lookup[rid]['reused'][mode])==equal_context
                    if equal_context:
                        for suffix in ['.png','_raw.png','_boundary.png']:
                            assert np.array_equal(read(native/(mode+suffix)),read(native/('old'+suffix)))
                        assert np.array_equal(local_fill,read(fill.OUT/'native'/rid/'old_telea.png'))
                    same&=equal_context
                if same:unchanged_regions.append(rid)
                checked.append(rid)
            del original,old,modes,maps;gc.collect()
    assert len(checked)==len(set(checked))==22 and len(pages)==14 and len(books)==11
    assert len(unchanged_pages)==13 and len(unchanged_regions)==21
    rid='new_margin_hole';n=ddrm.OUT/'native'/rid
    selected=read(MASKS/'native'/rid/'completed_added.png',True)
    old_band=read(MASKS/'native'/rid/'top_margin.png',True)
    region_count=int(selected.sum());outside_old_band=int((selected&~old_band).sum())
    metrics={}
    for name,path in dict(original=n/'original.png',old_ddrm=n/'old.png',in_band_ddrm=n/'in_band.png',
        completed_ddrm=n/'completed.png',completed_telea=fill.OUT/'native'/rid/'completed_telea.png').items():
        lab=cv2.cvtColor(read(path).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        light=lab[:,:,0][selected]
        metrics[name]=dict(pixels=len(light),L_le_80_pixels=int((light<=80).sum()),median_L=float(np.median(light)))
    write(OUT/'fixed_darkness_region.png',selected)
    dump(OUT/'diagnostics.json',dict(fixed_region='completed_added, same 6730 pixels for all fill comparisons',
        pixels=region_count,above_old_band_pixels=outside_old_band,metrics=metrics,
        mask_regions=[r for r in mask_rows.values() if not r['id'].endswith('_whole')],ddrm_regions=rows))
    overviews=[('01_mask_ablation.png',MASKS/'comparisons'/(rid+'.png')),
               ('02_same_ddrm.png',ddrm.OUT/'comparisons'/(rid+'.png')),
               ('03_fixed_boundary.png',ddrm.OUT/'blended_comparisons'/(rid+'.png')),
               ('04_same_mask_fill.png',fill.OUT/'comparisons'/(rid+'.png'))]
    for name,path in overviews:shutil.copyfile(path,OUT/name)
    stack([ddrm.OUT/'comparisons'/(key+'.png') for key in ['new_frame_text','new_brown_stamp','new_header_ink','new_mono_text']],
        OUT/'05_controls.png','枠・印影・見出し・本文の対照',
        ['前回から追加0。DDRMの入力周辺マスクも同一で、保存済みの出力と完全一致。'])
    yy,xx=np.nonzero(selected);H,W=selected.shape
    x0,y0=max(0,int(xx.min())-12),max(0,int(yy.min())-12)
    x1,y1=min(W,int(xx.max())+13),min(H,int(yy.max())+13)
    images=[read(path)[y0:y1,x0:x1] for path in [n/'original.png',n/'old.png',n/'completed.png',fill.OUT/'native'/rid/'completed_telea.png']]
    ddrm.prior_ddrm.figure(images,['原画像','前回のDDRM','今回のDDRM','同じマスクのTelea'],
        OUT/'06_top_closeup.png','追加回収した上方の損傷 · 拡大比較',
        '同じ切り出しを最近傍で拡大。DDRMの暗い線と、Teleaのぼけ・紙色ムラを区別する。',
        '切り出し＝追加領域の外接矩形＋12px。マスクの正解領域ではない。')
    paths=[n/'original.png',ddrm.prior_ddrm.OUT/'native'/rid/'npj.png',ddrm.prior_ddrm.OUT/'native'/rid/'v6.png',
           n/'old.png',n/'completed.png',fill.OUT/'native'/rid/'completed_telea.png']
    ddrm.prior_ddrm.figure([read(p) for p in paths],['原画像','npj → DDRM','v6 → DDRM','前回余白 → DDRM','今回余白 → DDRM','同じマスク → Telea'],
        OUT/'07_npj_v6_context.png','保存済みnpj・固定v6から今回まで',
        'npj・v6・前回余白は保存出力を再利用。TeleaはDDRMではない。',
        '同じ原画像。npjは編集候補をDDRMの欠損に接続した比較で、元npj全工程の再現ではない。')
    items=[(OUT/name,'overview/'+name) for name in [n for n,_ in overviews]+['05_controls.png','06_top_closeup.png','07_npj_v6_context.png']]
    for folder,group,prefix in [(MASKS,'comparisons','masks'),(ddrm.OUT,'comparisons','ddrm'),
        (ddrm.OUT,'blended_comparisons','fixed_boundary'),(fill.OUT,'comparisons','same_mask_fill')]:
        items.extend((p,prefix+'/'+p.name) for p in sorted((folder/group).glob('*.png')))
    zip_path=OUT/'margin_top_completion_comparison_images.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for path,name in items:archive.write(path,name)
    with zipfile.ZipFile(zip_path) as archive:
        assert len(archive.namelist())==len(items)==109 and archive.testzip() is None
        for path,name in items:assert hashlib.sha256(archive.read(name)).hexdigest()==sha(path)
    table='\n'.join(f'| [{mask_rows[r["id"]]["title"]}](../{MASKS.name}/comparisons/{r["id"]}.png) · `{r["id"]}` | {r["added_pixels"]["in_band"]:,} | {r["added_pixels"]["completed"]:,} | {"再推論、復元品質は未完成" if r["id"]==rid else "周辺マスク同一・保存出力と完全一致"} |' for r in rows)
    labels=dict(original='原画像',old_ddrm='前回のマスク→DDRM',in_band_ddrm='帯内解除→DDRM',completed_ddrm='全成分回収→DDRM',completed_telea='全成分回収→Telea')
    dark_table='\n'.join(f'| {labels[k]} | {v["L_le_80_pixels"]:,} | {v["median_L"]:.3f} |' for k,v in metrics.items())
    report=f'''# 上余白帯の上端で除外された損傷の回収

2026-10-01。**右上に残っていた損傷を編集対象へ回収できた。前処理は前進したが、同じDDRMでは暗い線状の模様と塗り跡が発生し、自然な修復はまだ未達。** 同じマスクのTeleaでは暗い模様が少なくなる一方、ぼけ・色ムラが残る。今回のマスク候補を開発比較用に保存し、復元品質を改善した完成版とは扱わない。

前回[紙模様の試作](../paper_pattern_review_20261001/report.md)で、濃い残存損傷はマスク外にあると分離した。今回は前処理だけを変更し、全14ページ・11冊・22箇所を保持して比較した。調整済みの開発例であり、独立評価・画素GT・全資料の改善率はない。イラストページは従来どおり対象外。

## 比較PNG

- [マスクのアブレーション](01_mask_ablation.png)：原画像、前回、上端解除の帯内だけ、上方まで全成分を回収。黄＝編集、青＝保護、赤＝朱保護、紫＝保留。
- [同じDDRM後](02_same_ddrm.png)：重み・20段・空間ノイズを固定し、マスクだけ変更した主要比較。
- [前回と同じ境界補正](03_fixed_boundary.png)：補助比較。補正対象の旧余白帯も固定したため、新しく上方へ追加した領域には境界補正を加えない。
- [同じマスクのDDRMとTelea](04_same_mask_fill.png)、[上方の拡大](06_top_closeup.png)。Teleaは局所補間でDDRMではない。
- [本文・枠・印影・見出しの対照](05_controls.png)、[保存済みnpj・固定v6から今回まで](07_npj_v6_context.png)。
- [全比較109PNG ZIP](margin_top_completion_comparison_images.zip)：概要7、全ページのマスク14、ROIマスク22、同じDDRM22、固定境界補正22、復元器診断22。

## 変更した判定

従来は上余白帯のどの端に触れた成分も一括除外していた。問題のページの旧帯はy=127〜307、本文の上枠はy=340。右上の損傷2成分は帯の上端に接し、それ以外の境界には接していなかった。

今回も色の種32画素以上、面積128〜ページ1%、最大半径6px以上、最大辺が短辺20%以下という以前の形状条件を使う。**上端だけ**に触れる成分に限定して除外を解除し、側端・本文側の端に触れた成分は解除しない。帯内のみの条件は元の帯の中で3pxの周囲を回収する。

全成分の条件では同じ成分を、ページ上端16px・左右6%・本文枠の上32pxを除く領域で上方へ追う。暗い支持領域の閾値と3pxの周囲の条件も固定した。横に長い暗い線（ページ幅55%以上）は8pxの保護幅を付けて接続の障壁にする。元の朱・追加暖色保留と暖色接続保護を解除しない。既存の編集領域は全て維持し、その外だけを追加する。

ROIの座標や「穴」のラベルでマスクを作っていない。全ページに同じ規則を適用した。新しい帯を一律に汚れと認定せず、旧帯内で条件を満たした上端接触成分だけを追う。余白の位置を文字なしの保証にする方法ではない。

全14ページのうち変更は`100249416_00024_1`だけ。帯内解除は前回から3,585画素、全成分回収は6,730画素を追加した。全成分のうち{outside_old_band:,}画素は旧帯の上方、{region_count-outside_old_band:,}画素は帯内。全成分化による周囲の形状差で帯内解除から帯内にも1画素の差がある。面積は回収候補の大きさで、正解・検出率ではない。

## 同じDDRMと復元器の切り分け

旧のnpj・v6・余白DDRMはハッシュ確認した保存出力を使用した。今回の2マスクは同じモデル・原画像・128pxパッチ・重複32px・周囲64px・20段・画像空間ガウスノイズで比較した。欠損に原画像を混ぜ戻す処理はなく、編集範囲外は原画素へ厳密に戻す。変更したROIだけ各13パッチを再推論し、ほか21箇所は周辺マスクまで同一なので保存DDRMを再利用した。同一マスク・ノイズの再実行の一致も確認した。

帯内だけの解除では、帯より上にある暗い部分が原画素として残る。全成分回収はそこも大きく薄くする一方、上方に灰色の線・塊と平坦な塗り跡を生む。前回の境界補正を同じ旧帯に適用しても、上方の新しい編集部分には適用されず、暗い生成模様は解決しない。境界補正の範囲や紙模様の方式を同時に変えて結果を混ぜなかった。

DDRMの出力を見てから、同じ原画像・マスク・64pxの文脈をTelea半径3で補間する診断を追加した。全22箇所を扱い、旧マスクのTeleaは以前の保存出力と完全一致した。全成分回収のTeleaではDDRMの上方の灰色の模様は目立たないが、暖色のにじみや平坦な色の塊が残る。従って「編集対象へ戻せない」問題は低減したが、「自然な紙面へ復元する」問題は続いている。

全成分回収で追加した**同じ{region_count:,}画素**に対し、L≤80の画素数とLの中央値を測った。暗い部分の残り方の観測値で、除去精度・損傷の正解率・真正な復元の指標ではない。原画像と前回DDRMが一致するのは、この画素集合が以前は編集範囲外だったため。

| 条件 | L≤80の画素数 | L中央値 |
| --- | ---: | ---: |
{dark_table}

Telea側の暗い画素が少なくても、明るく平滑にするだけで良い品質と数えない。既存の128pxパッチ用DDRMモデルを使い、紙面専用には再学習していない。紙面だけの編集にも同じ接続を用いているが、暗い模様の原因を学習データだけと断定はしない。周囲の暗い枠、欠損形状、モデル・サンプラーの影響を今回分離していない。

## 全22箇所

| ROI | 帯内解除の追加 | 全成分回収の追加 | 実際のDDRM |
| --- | ---: | ---: | --- |
{table}

13ページの全ページマスクと、ほか21ROIの周辺マスク・DDRM出力・固定境界補正出力は以前と完全一致した。印影の以前からの小さな点の変化は利用者レビューで許容されたものとして保持し、今回の追加変化と扱わない。本文と重なるシミ、細い裂け、欠損文字は未解決のまま。

## 確認と次の段階

入力・既存マスク・ソース・モデル重み・DDRM核・保存出力のハッシュ、全ページのマスクの演算と保護、原寸のPNG、編集範囲外の不変、周辺マスクが同一の場合の出力再利用、前回境界補正と旧Teleaの再現、ZIP内109PNGのハッシュを確認した。[確認JSON](verification.json)。

上端接触を回収する候補マスクを次の**開発用の検出基準**として保存する。今回のDDRM経路が自然な紙面を作るとは確認できなかったため、実運用のv6や全資料へ置き換えない。次はこの固定マスクのまま、紙面と確認した領域について、周囲の紙色を使う復元とDDRMの接続を比較する。紙色の低周波と暗い生成模様を分け、損傷を除去しても新たな線を生まない条件を調べる。本文・印影との区別を別に検証し、余白全部を紙面補間へ流さない。紙模様の前回試作をこの比較へ自動的に足さない。

[マスクの固定条件](../{MASKS.name}/manifest.json)・[成分の判定](../{MASKS.name}/pages/100249416_00024_1/components.json)、[DDRM条件](../{ddrm.OUT.name}/manifest.json)、[復元器診断の条件](../{fill.OUT.name}/manifest.json)。実装：[前処理](../../scripts/benchmark/probe_margin_top_completion.py)、[固定DDRM](../../scripts/benchmark/check_margin_top_completion_ddrm.py)、[Telea診断](../../scripts/benchmark/diagnose_margin_top_fill.py)、[検証・梱包](../../scripts/benchmark/package_margin_top_completion_review.py)。
'''
    (OUT/'report.md').write_text(report)
    dump(OUT/'verification.json',dict(status='passed',pages=pages,books=len(books),regions=checked,
        unchanged_full_pages=unchanged_pages,unchanged_context_and_output_regions=unchanged_regions,
        full_page_mask_conservation=True,added_only_in_allowed_domain=True,in_band_subset_completed=True,
        original_pixels_outside_edit=True,actual_DDRM=True,spatial_noise_coupled=True,
        previous_boundary_and_Telea_reproduced=True,zip_PNG_count=109,zip_hashes=True,
        closeup_xywh=[x0,y0,x1-x0,y1-y0],pixel_GT=False))
    dump(OUT/'manifest.json',dict(source_sha256=sha(Path(__file__)),
        dependencies={str((f/n).relative_to(ROOT)):sha(f/n) for f in [MASKS,ddrm.OUT,fill.OUT,ddrm.prior_ddrm.OUT,ddrm.blending.OUT]
                      for n in ['manifest.json','complete.json']},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print('Passed: 14 full-page masks, all 22 ROIs; 109 PNGs packaged',flush=True)


if __name__=='__main__':main()
