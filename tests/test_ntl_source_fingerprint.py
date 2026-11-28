import unittest
import numpy as np
from scripts.ntl_source_fingerprint import fingerprint
class TestFingerprint(unittest.TestCase):
    def setUp(self):
        self.labels=[str(i) for i in range(18)]
        self.x=np.full((18,32),-4.)
        self.x[0]=-2.
    def test_shared_token_difficulty(self):
        a=fingerprint(self.x,self.labels)
        b=fingerprint(self.x-np.linspace(0,5,32),self.labels)
        np.testing.assert_allclose(a.local_relative,b.local_relative)
        self.assertEqual(a.prediction,b.prediction)
    def test_mask(self):
        pad=np.column_stack([self.x,np.full((18,8),np.nan)])
        mask=np.r_[np.ones(32,dtype=bool),np.zeros(8,dtype=bool)]
        np.testing.assert_allclose(fingerprint(pad,self.labels,mask).local_relative,fingerprint(self.x,self.labels).local_relative)
    def test_axis_reorder(self):
        p=np.arange(18)[::-1]
        a=fingerprint(self.x,self.labels)
        b=fingerprint(self.x[p],[self.labels[i] for i in p])
        self.assertEqual(a.prediction,b.prediction)
        np.testing.assert_allclose(a.local_relative,b.local_relative[p])
    def test_outlier_can_change_mean_not_robust(self):
        x=self.x.copy();x[0,0]=-100.
        self.assertNotEqual(np.argmax(x.mean(axis=1)),0)
        self.assertEqual(fingerprint(x,self.labels).prediction,"0")
    def test_invalid(self):
        for x,l in [(self.x[:17],self.labels[:17]),(self.x[:,:7],self.labels),(self.x+5,self.labels),(self.x,["x"]*18)]:
            with self.assertRaises(ValueError):fingerprint(x,l)
    def test_source_bias_not_cancelled(self):
        a=fingerprint(self.x,self.labels)
        x=self.x.copy();x[1]+=3.
        self.assertNotEqual(a.prediction,fingerprint(x,self.labels).prediction)
if __name__=="__main__":unittest.main()
