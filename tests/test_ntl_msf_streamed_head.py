import unittest
import torch
import torch.nn.functional as F
from scripts.ntl_msf_streamed_head import streamed_logp
class TestStreamed(unittest.TestCase):
    def test_dense_equivalence(self):
        torch.manual_seed(17)
        for dtype,tol in [(torch.float32,1e-5),(torch.bfloat16,.05)]:
            h=torch.randn(18,7,16).to(dtype)
            head=torch.nn.Linear(16,53,bias=False,dtype=dtype)
            y=torch.randint(0,53,(18,7));y[0,:4]=torch.tensor([0,12,13,52])
            expected=-F.cross_entropy(head(h).float().reshape(-1,53),y.reshape(-1),reduction='none').reshape_as(y)
            for tile in (1,13,53,100):
                got=streamed_logp(h,y,head,tile)
                self.assertLessEqual(float((got-expected).abs().max()),tol)
    def test_reject_out_of_vocab(self):
        with self.assertRaises(AssertionError):
            streamed_logp(torch.ones(1,1,4),torch.tensor([[10]]),torch.nn.Linear(4,10,bias=False),4)
if __name__=='__main__':unittest.main()
