"""Safety invariants for the experimental exclusion selector."""
import unittest
import cv2
import numpy as np
import probe_evidence_gated_howe as policy

class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.ink=np.zeros((360,360),np.uint8)
        cv2.putText(self.ink,'RR',(80,230),cv2.FONT_HERSHEY_SIMPLEX,3,1,4)
        self.ink=self.ink.astype(bool)
        self.blank=np.zeros_like(self.ink)
        self.gray=np.full(self.ink.shape,220,np.uint8);self.gray[self.ink]=170
        self.donor=np.full(self.ink.shape,220,np.uint8);self.donor[self.ink]=50
        self.path=policy.ROOT/'data/test/synthetic/neighbor.jpg'

    def run_gate(self,gray=None,csc=None,red=None,refs=None):
        return policy.decide(self.gray if gray is None else gray,self.ink,
            self.blank if csc is None else csc,self.blank if red is None else red,
            self.ink,self.blank,[(self.path,self.donor)] if refs is None else refs)

    def test_agreeing_external_shape_can_exclude_faint_candidate(self):
        new,removed,_=self.run_gate()
        self.assertTrue(np.array_equal(removed,self.ink))
        self.assertFalse(new.any())

    def test_no_reference_preserves_candidate(self):
        new,removed,_=self.run_gate(refs=[])
        self.assertFalse(removed.any());self.assertTrue(np.array_equal(new,self.ink))

    def test_single_component_kept_even_when_reference_matches(self):
        _,labels,_,_=cv2.connectedComponentsWithStats(self.ink.astype(np.uint8),8)
        self.ink=labels==1
        self.gray=np.full(self.ink.shape,220,np.uint8);self.gray[self.ink]=170
        self.donor=np.full(self.ink.shape,220,np.uint8);self.donor[self.ink]=50
        _,removed,_=self.run_gate()
        self.assertFalse(removed.any())

    def test_blank_reference_does_not_explain_faint_candidate(self):
        _,removed,_=self.run_gate(refs=[(self.path,np.full_like(self.gray,220))])
        self.assertFalse(removed.any())

    def test_dark_front_ink_is_not_a_faint_candidate(self):
        gray=self.gray.copy();gray[self.ink]=60
        _,removed,_=self.run_gate(gray=gray)
        self.assertFalse(removed.any())

    def test_red_component_is_protected(self):
        red=self.blank.copy();y,x=np.argwhere(self.ink)[0];red[y,x]=True
        _,removed,_=self.run_gate(red=red)
        self.assertFalse(removed.any())

    def test_even_one_csc_black_pixel_protects_component(self):
        csc=self.blank.copy();y,x=np.argwhere(self.ink)[0];csc[y,x]=True
        _,removed,_=self.run_gate(csc=csc)
        self.assertFalse(removed.any())

if __name__=='__main__':unittest.main()
