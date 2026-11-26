"""C1 real-behavior unit tests: prompt isolation and contrastive negative weights."""
from __future__ import annotations
import unittest
import numpy as np
import torch
from scripts.ntl_c1_prompt_contrastive_20261126 import (
    prompt_supervised_contrast, prompt_group_splits, train_pair
)

class C1Tests(unittest.TestCase):
    def test_group_splits_do_not_leak_prompt(self):
        groups=np.repeat(np.arange(18),4)
        y=np.tile(np.arange(4),18)
        folds=prompt_group_splits(groups,y,n_splits=3)
        seen=set()
        for tr,va in folds:
            self.assertFalse(set(groups[tr]) & set(groups[va]))
            self.assertTrue(set(range(4)).issubset(set(y[va])))
            seen.update(va.tolist())
        self.assertEqual(len(seen),len(groups))

    def test_positive_pairs_need_cross_prompt_same_model(self):
        torch.manual_seed(17)
        z=torch.randn(6,5,requires_grad=True)
        y=torch.tensor([0,1,2,0,1,2])
        prompt=torch.tensor([0,0,0,1,1,1])
        loss=prompt_supervised_contrast(z,y,prompt)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(),0)
        loss.backward()
        self.assertGreater(z.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):
            prompt_supervised_contrast(z,y,torch.zeros(6,dtype=torch.long))

    def test_more_same_prompt_negative_weight_increases_loss(self):
        torch.manual_seed(4)
        z=torch.randn(8,8)
        y=torch.tensor([0,1,2,3,0,1,2,3])
        p=torch.tensor([0,0,0,0,1,1,1,1])
        a=prompt_supervised_contrast(z,y,p,same_prompt_negative_weight=1.)
        b=prompt_supervised_contrast(z,y,p,same_prompt_negative_weight=3.)
        self.assertGreater(b.item(),a.item())

    def test_matched_fitted_heads_are_finite_and_distinct(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(72,16)).astype('float32')
        y=np.tile(np.arange(4),18)
        p=np.repeat(np.arange(18),4)
        tr=np.flatnonzero(p<12);va=np.flatnonzero(p>=12)
        base,c1=train_pair(x,y,p,tr,va,epochs=2,dim=12,seed=3)
        self.assertEqual(base.shape,(24,4))
        self.assertEqual(base.shape,c1.shape)
        self.assertTrue(np.isfinite(base).all())
        self.assertTrue(np.isfinite(c1).all())
        self.assertGreater(np.max(np.abs(base-c1)),1e-6)
        with self.assertRaisesRegex(ValueError,"leak"):
            train_pair(x,y,p,tr,np.array([tr[0]]),epochs=1)

if __name__=="__main__":
    unittest.main()
