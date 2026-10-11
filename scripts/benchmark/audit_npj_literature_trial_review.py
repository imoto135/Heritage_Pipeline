"""Separate audit: decode originals afresh, compare PNG panel pixels and frozen hashes."""
import json
from pathlib import Path
import subprocess
from PIL import Image
import numpy as np
from build_npj_v6_comparison import ROOT, sha, dump
from package_saved_npj_inference_v6_review import check


def main():
    trial = ROOT/'tmp_work/npj_literature_trial_20261002'
    review = ROOT/'tmp_work/npj_literature_trial_review_20261002'
    out = ROOT/'tmp_work/npj_literature_trial_independent_audit_20261002'
    assert not (out/'complete.json').exists()
    manifest = check(review)
    for method in ['stain','docres','structural','matched_stain','matched_docres']:
        check(trial/method)
    frames = json.loads((trial/'selection.json').read_text())['frames']
    renderings = {r['file']:r for r in json.loads((review/'diagnostics.json').read_text())['rendering']}
    native_panels = 0; exact_frames = 0; matched_panels = 0
    def rgb(p): return np.array(Image.open(p).convert('RGB'))
    def audit_panel(figure_name, arrays):
        nonlocal native_panels
        canvas = rgb(review/figure_name)
        dw,dh = renderings[figure_name]['display_size']
        for i, a in enumerate(arrays):
            h,w = a.shape[:2]
            assert dw % w == 0 and dh % h == 0 and dw//w == dh//h
            expected = np.array(Image.fromarray(a).resize((dw,dh),Image.Resampling.NEAREST))
            left = 24+i*(dw+24)
            assert np.array_equal(canvas[120:120+dh,left:left+dw],expected),figure_name
            native_panels += 1
    for frame in frames:
        fid = frame['id']; original = rgb(ROOT/frame['original'])
        if frame['box']:
            x,y,w,h = frame['box']; original = original[y:y+h,x:x+w]
        assert np.array_equal(original,rgb(trial/'inputs'/fid/'original.png'))
        saved_input = rgb(ROOT/frame['input'])
        assert np.array_equal(saved_input,rgb(trial/'inputs'/fid/'saved_input.png'))
        saved_npj = rgb(ROOT/frame['saved_npj'])
        assert np.array_equal(saved_npj,rgb(trial/'inputs'/fid/'saved_npj.png'))
        exact_frames += 1
        outputs = [original,saved_npj,rgb(trial/'stain'/fid/'output.png'),rgb(trial/'docres'/fid/'output.png')]
        for region in frame['regions']:
            x,y,w,h = region['xywh']; s = np.s_[y:y+h,x:x+w]
            audit_panel(region['id']+'.png',[a[s] for a in outputs])
            for key,a in zip(['original','saved_npj','stain','docres'],outputs):
                assert np.array_equal(a[s],rgb(review/'native'/region['id']/(key+'_image.png')))
        if not frame['input_reference_exact']:
            p = np.load(trial/'matched_docres'/fid/'background_probability.npy',allow_pickle=False)
            assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all()
            assert not (p==.5).any()
            matched = [saved_input,saved_npj,rgb(trial/'matched_stain'/fid/'output.png'),
                       rgb(trial/'matched_docres'/fid/'output.png')]
            assert np.array_equal(matched[-1][:,:,0],(p>.5).astype(np.uint8)*255)
            for region in frame['regions']:
                x,y,w,h = region['xywh']; s = np.s_[y:y+h,x:x+w]
                audit_panel('matched_'+region['id']+'.png',[a[s] for a in matched])
                matched_panels += 4
    assert exact_frames == 6 and native_panels == 104 and matched_panels == 24
    clean = {}
    for repo in ['StainRestorer','DocRes']:
        folder = ROOT/'tmp_work/npj_literature_models_20261002/repos'/repo
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=folder).strip()
        clean[repo] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=folder,text=True).strip()
    dump(out/'verification.json',dict(status='passed',fresh_source_originals=6,
        native_restoration_panels_PIL_exact=native_panels,matched_input_panels=matched_panels,
        all_input_model_source_output_hashes=True,external_repos_unmodified=clean,
        matched_docres_probabilities_and_decode=True,pixel_GT=False,adopted=False,
        report_sha256=sha(review/'report.md')))
    dump(out/'manifest.json',dict(dependencies={str(p.relative_to(ROOT)):sha(p)
        for p in [Path(__file__),review/'manifest.json',review/'complete.json']}))
    dump(out/'complete.json',dict(outputs={str(p.relative_to(out)):sha(p)
        for p in sorted(out.rglob('*')) if p.is_file()}))
    print('Independent audit passed: 6 actual inputs, 104 native panels, both official repos clean.')


if __name__ == '__main__': main()
