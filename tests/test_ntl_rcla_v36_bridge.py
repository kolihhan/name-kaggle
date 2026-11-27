"""Test original submission preservation and label-blind ID alignment."""
import tempfile,time,unittest
from pathlib import Path
import numpy as np,pandas as pd
from scripts.ntl_rcla_v36_bridge import apply
F=["gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash"]
class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        p=Path(self.tmp.name);self.out=p/"submission.parquet";self.native=p/"native.npz";self.test=p/"test.csv"
        self.base=pd.DataFrame({"id":["a","b","c"],"model":[F[0],"other-0",F[2]]})
        self.base.to_parquet(self.out,index=False)
        self.original=self.out.read_bytes()
        pd.DataFrame({"id":["c","a","b"],"text":["C","A","B"]}).to_csv(self.test,index=False)
        labs=np.array(F+["other-"+str(i) for i in range(15)])
        z=np.zeros((3,18));z[0,[2,1]]=[4,3];z[1,[0,1]]=[4,3];z[2,[3,0]]=[4,3]
        np.savez(self.native,ids=np.array(["c","a","b"]),labels=labs,logits=z)
    def run_bridge(self,scorer,delta=20):
        return apply(self.out,self.native,self.test,scorer,time.monotonic()+delta)
    def test_alignment(self):
        seen=[]
        def scorer(c,text,deadline):
            seen.append((c.row_id,text))
            if c.row_id=="a":return {F[0]:(-4,-4),F[1]:(-3,-3)}
            return {F[2]:(-4,-2),F[1]:(-3,-3)}
        s=self.run_bridge(scorer)
        self.assertEqual(seen,[("a","A"),("c","C")])
        self.assertEqual(s["changed"],1);self.assertTrue(s["applied"])
        self.assertEqual(pd.read_parquet(self.out).model.tolist(),[F[1],"other-0",F[2]])
    def test_no_scorer(self):
        self.assertFalse(self.run_bridge(None)["applied"])
        self.assertEqual(self.out.read_bytes(),self.original)
    def test_deadline(self):
        self.assertFalse(self.run_bridge(lambda *a:self.fail("unexpected work"),-1)["applied"])
        self.assertEqual(self.out.read_bytes(),self.original)
    def test_error_after_partial_score(self):
        def scorer(c,*args):
            if c.row_id=="c":raise TimeoutError()
            return {F[0]:(-4,-4),F[1]:(-3,-3)}
        self.assertFalse(self.run_bridge(scorer)["applied"])
        self.assertEqual(self.out.read_bytes(),self.original)
    def test_missing_scores(self):
        self.assertFalse(self.run_bridge(lambda *a:None)["applied"])
        self.assertEqual(self.out.read_bytes(),self.original)
    def test_id_mismatch(self):
        pd.DataFrame({"id":["c","a","z"],"text":["C","A","Z"]}).to_csv(self.test,index=False)
        with self.assertRaises(ValueError):self.run_bridge(lambda *a:None)
        self.assertEqual(self.out.read_bytes(),self.original)
if __name__=="__main__":unittest.main()
