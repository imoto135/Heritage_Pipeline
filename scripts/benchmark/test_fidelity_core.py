"""Scientific sanity tests: identity, lost/added strokes, color and region safety."""
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from fidelity_core import build_references, evaluate, read_mask, segmentation_scores


class FidelityTests(unittest.TestCase):
    def setUp(self):
        self.image=np.full((32,32,3),210,np.uint8)
        self.image[8:24,10:12]=70
        self.text=np.zeros((32,32),bool); self.text[8:24,10:12]=True
        self.stain=np.zeros((32,32),bool); self.stain[20:25,20:25]=True
        self.regions={'text':self.text,'stain':self.stain,'background':~(self.text|self.stain)}
        self.masks={'black':self.text,'red':np.zeros((32,32),bool),
                    'degradation':self.stain,'background':~(self.text|self.stain)}

    def test_identity(self):
        m=evaluate(self.image,self.image,self.regions)
        self.assertTrue(np.isinf(m['psnr_rgb']))
        self.assertEqual(m['ssim_gray'],1)
        self.assertEqual(m['text_mae'],0)
        self.assertEqual(m['stain_delta_e00'],0)

    def test_missing_stroke(self):
        out=self.image.copy(); out[self.text]=210
        m=evaluate(self.image,out,self.regions)
        self.assertGreater(m['text_lightening'],0)
        self.assertEqual(m['dark150_proxy_recall'],0)

    def test_extra_stroke(self):
        out=self.image.copy(); out[5:15,25:27]=50
        m=evaluate(self.image,out,self.regions)
        self.assertLess(m['dark150_proxy_precision'],1)
        self.assertGreater(m['background_darkening'],0)

    def test_ink_darkening(self):
        out=self.image.copy(); out[self.text]=30
        m=evaluate(self.image,out,self.regions)
        self.assertEqual(m['text_darkening'],40)
        self.assertEqual(m['text_lightening'],0)

    def test_nearly_equal_luminance_color_change(self):
        clean=np.full((32,32,3),128,np.uint8)
        out=np.empty_like(clean); out[:]=[148,128,120]
        m=evaluate(clean,out,{'all':np.ones((32,32),bool)})
        self.assertLessEqual(m['all_mae'],1)
        self.assertGreater(m['all_delta_e00'],1)

    def test_empty_region(self):
        m=evaluate(self.image,self.image,{'empty':np.zeros((32,32),bool)})
        self.assertIsNone(m['empty_mae'])
        self.assertEqual(m['empty_n'],0)

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError): evaluate(self.image,self.image[:20],self.regions)

    def test_all_white_cannot_preserve_ink(self):
        m=evaluate(self.image,np.full_like(self.image,255),self.regions)
        self.assertGreater(m['text_mae'],100)
        self.assertEqual(m['dark150_proxy_recall'],0)

    def test_no_tone_change_and_no_ink_gain(self):
        refs,holes=build_references(self.image,self.masks)
        for name in ('P0','P1'):
            np.testing.assert_array_equal(refs[name][self.text],self.image[self.text])
        np.testing.assert_array_equal(refs['P1'][~holes['refined_hole']],self.image[~holes['refined_hole']])
        self.assertGreater(float(refs['P1'][~holes['refined_hole']].mean()),203)

    def test_empty_degradation_keeps_original(self):
        self.masks['degradation'][:]=False
        refs,_=build_references(self.image,self.masks)
        np.testing.assert_array_equal(refs['P1'],self.image)

    def test_no_background_fails(self):
        self.masks['background'][:]=False
        with self.assertRaises(ValueError): build_references(self.image,self.masks)

    def test_binary_and_polarity(self):
        root=Path(__file__).resolve().parents[2]/'tmp_work/fidelity_tests'
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as d:
            p=Path(d)/'mask.png'; cv2.imwrite(str(p),self.text.astype(np.uint8)*255)
            np.testing.assert_array_equal(read_mask(p,(32,32),255),self.text)
            np.testing.assert_array_equal(read_mask(p,(32,32),0),~self.text)
            cv2.imwrite(str(p),np.full((32,32),127,np.uint8))
            with self.assertRaises(ValueError): read_mask(p,(32,32))

    def test_ignore_region_for_annotation(self):
        pred=self.text.copy(); pred[0,0]=True
        valid=np.ones_like(pred); valid[0,0]=False
        self.assertEqual(segmentation_scores(pred,self.text,valid)['precision'],1)


if __name__=='__main__': unittest.main()
