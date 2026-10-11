"""Audit display/channel semantics and native versus author-sized colour outputs."""
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
import torch

from run_npj_literature_models import ROOT, TRIAL, EXTERNAL, load, model_call
from build_npj_v6_comparison import read, write, sha, dump
from package_saved_npj_inference_v6_review import check
from package_saved_npj_role_separated_review import figure

OUT = ROOT/'tmp_work/stainrestorer_colour_audit_20261002'
CASES = [('known_200010454_v2', [1370,1880,256,256], '朱印と紙面'),
         ('old_upper_crop', [140,244,256,256], '細い筆画と茶色い紙面'),
         ('crop_hnsd', [170,300,256,256], '枠線と暗い紙面')]


def main():
    assert not (OUT/'manifest.json').exists(), 'Use another version'
    OUT.mkdir(parents=True, exist_ok=True)
    check(TRIAL/'stain')
    check(ROOT/'tmp_work/npj_literature_trial_review_20261002')
    model, resources = load('stain', 'cuda:0')
    dependencies = {str(p.relative_to(ROOT)):sha(p) for p in resources + [Path(__file__),
        Path(__file__).parent/'run_npj_literature_models.py',
        Path(__file__).parent/'package_saved_npj_role_separated_review.py',
        TRIAL/'stain'/'complete.json', TRIAL/'selection_complete.json',
        EXTERNAL/'repos/StainRestorer/data/dataset_RGB.py',
        EXTERNAL/'repos/StainRestorer/config.yml']}
    dump(OUT/'manifest.json',dict(dependencies=dependencies,
        cases=CASES,source_snapshot=True,question='channel/display issue versus model-induced colour',
        no_colour_correction=True,checkpoint_unchanged=True))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    class Identity(torch.nn.Module):
        def forward(self,x): return [x,x]
    swatches = np.array([[[0,0,255],[255,0,0],[0,255,0],[40,120,220]]],dtype=np.uint8)
    identity = np.rint(model_call(Identity(),swatches,'stain','cpu')*255).astype(np.uint8)
    assert np.array_equal(identity,swatches)
    write(OUT/'identity_swatches.png',identity)
    assert np.array_equal(np.asarray(Image.open(OUT/'identity_swatches.png')),swatches[:,:,::-1])
    rows = []
    for fid,box,title in CASES:
        original_path = TRIAL/'inputs'/fid/'original.png'
        original_rgb = np.asarray(Image.open(original_path).convert('RGB'))
        original = original_rgb[:,:,::-1].copy()
        assert np.array_equal(original,read(original_path))
        x,y,w,h = box; cropped_rgb = original_rgb[y:y+h,x:x+w].copy()
        # Independent author-equivalent PIL RGB -> CHW tensor -> direct model.
        native = torch.from_numpy(cropped_rgb.transpose(2,0,1).copy()).float().unsqueeze(0).to('cuda:0')/255.
        # Author DataLoaderVal uses Albumentations Resize default linear, 256x256,
        # RGB ToTensor [0,1]. Bypass dataset pairing/metrics, leave model unchanged.
        resized_rgb = cv2.resize(original_rgb,(256,256),interpolation=cv2.INTER_LINEAR)
        resized = torch.from_numpy(resized_rgb.transpose(2,0,1).copy()).float().unsqueeze(0).to('cuda:0')/255.
        with torch.inference_mode():
            native_heads = model(native)
            resized_heads = model(resized)
        assert len(native_heads)==2 and len(resized_heads)==2
        all_heads = native_heads+resized_heads
        assert all(torch.isfinite(v).all() for v in all_heads)
        outputs = [v[0].permute(1,2,0).cpu().numpy() for v in all_heads]
        adapter = model_call(model,original[y:y+h,x:x+w],'stain','cuda:0')
        native_direct = np.clip(outputs[0],0,1)[:,:,::-1].copy()
        assert np.allclose(adapter,native_direct,atol=1e-6,rtol=0)
        quantized = [np.rint(np.clip(v,0,1)*255).astype(np.uint8)[:,:,::-1].copy() for v in outputs]
        for key,value in zip(['native_final','native_coarse','resized_final','resized_coarse'],quantized):
            write(OUT/'native'/fid/(key+'.png'),value)
        # Raw tensor saved in RGB, no clamping; shows colour change before PNG conversion.
        (OUT/'native'/fid).mkdir(parents=True,exist_ok=True)
        np.save(OUT/'native'/fid/'native_final_RGB_float.npy',outputs[0])
        np.save(OUT/'native'/fid/'resized_final_RGB_float.npy',outputs[2])
        prior = read(TRIAL/'stain'/fid/'output.png')[y:y+h,x:x+w]
        figure([original[y:y+h,x:x+w],prior,quantized[0],quantized[1]],
            ['原画像・原寸', '前回512px・生出力', '原寸256px・最終', '原寸256px・中間'],
            OUT/('native_'+fid+'.png'),title+'：入力範囲と出力段階を確認',
            ['全列RGBを正しく保存。色補正なし。256px切り出しは色・処理経路の診断用。',
             '前回512px列は保存済み結果そのもの。今回256pxは同じ領域を単独推論。',
             'ここでの色差はGT精度や汚れ除去率ではない。'])
        figure([resized_rgb[:,:,::-1].copy(),quantized[2],quantized[3]],
            ['原画像・256へ縮小', '256px設定・最終出力', '256px設定・中間出力'],
            OUT/('resized_'+fid+'.png'),title+'：公式入力サイズ・RGBでの追加対照',
            ['参照画像全体を256×256へ線形縮小。公式データローダーのサイズ・RGB範囲に合わせる。',
             '縮小で細い筆画が失われるため、筆画保持の比較には使わない。色の診断用。',
             'pairedデータセットの公式評価・PSNRの再現ではない。'])
        # Green dominance is a raw channel diagnostic, not restoration accuracy.
        deltas = []
        for name,out,input_rgb in [('native_final',outputs[0],cropped_rgb),
                                  ('native_coarse',outputs[1],cropped_rgb),
                                  ('resized_final',outputs[2],resized_rgb),
                                  ('resized_coarse',outputs[3],resized_rgb)]:
            inp = input_rgb.astype(np.float32)/255
            neutral_green = out[:,:,1]-.5*(out[:,:,0]+out[:,:,2])
            original_green = inp[:,:,1]-.5*(inp[:,:,0]+inp[:,:,2])
            deltas.append(dict(output=name,mean_raw_RGB_delta_255=((out-inp).mean(axis=(0,1))*255).tolist(),
                mean_green_balance_change_255=float((neutral_green-original_green).mean()*255),
                raw_outside_0_1_fraction=float(((out<0)|(out>1)).mean())))
        rows.append(dict(frame=fid,adapter_direct_RGB_equal=True,stats=deltas))
    for rel,digest in dependencies.items(): assert sha(ROOT/rel)==digest,rel
    dump(OUT/'diagnostics.json',rows)
    dump(OUT/'verification.json',dict(identity_red_blue_green_exact=True,
        PIL_RGB_cv2_BGR_exact=True,independent_direct_tensor_adapter_equal=True,
        saved_figures_pixel_exact=True,finite=True,previous_outputs_unchanged=True,
        pixel_GT=False,no_colour_correction=True))
    dump(OUT/'inference_complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
