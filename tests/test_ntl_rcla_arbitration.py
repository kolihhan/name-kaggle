import unittest
import numpy as np
from scripts.ntl_rcla_arbitration import arbitrate,select_candidates
F=["gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash"]
class ArbitrationTest(unittest.TestCase):
    def setUp(self):
        self.labels=F+["other-"+str(i) for i in range(15)]
        self.ids=["a","b","c","d","e"]
        self.base=[F[0],F[1],F[0],self.labels[3],F[1]]
        self.z=np.zeros((5,18))
        self.z[0,:2]=[3,2]
        self.z[1,:2]=[2,3]
        self.z[2,[0,3]]=[3,2]
        self.z[3,[0,1]]=[3,2]
        self.z[4,:3]=[2,3,2]
    def test_eligibility(self):
        self.assertEqual([c.row_id for c in select_candidates(self.ids,self.labels,self.z,self.base)],["a","b"])
    def test_consensus_and_disagreement(self):
        out,s=arbitrate(self.ids,self.labels,self.z,self.base,{
            "a":{F[0]:(-3,-4),F[1]:(-2,-3)},
            "b":{F[1]:(-3,-3),F[0]:(-2,-4)}})
        self.assertEqual(out.tolist(),[F[1],*self.base[1:]])
        self.assertEqual(s["changed"],1)
    def test_missing_keeps_base(self):
        out,s=arbitrate(self.ids,self.labels,self.z,self.base,{})
        self.assertEqual(out.tolist(),self.base)
        self.assertEqual(s["fallback_missing"],2)
    def test_bad_id(self):
        with self.assertRaises(ValueError):arbitrate(self.ids,self.labels,self.z,self.base,{"c":{}})
    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):select_candidates(["a"]*5,self.labels,self.z,self.base)
    def test_bad_score_falls_back(self):
        out,s=arbitrate(self.ids,self.labels,self.z,self.base,{"a":{F[0]:(-3,-3),F[1]:(float("nan"),-2)}})
        self.assertEqual(out.tolist(),self.base)
        self.assertEqual(s["fallback_missing"],2)
if __name__=="__main__":unittest.main()
