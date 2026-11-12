"""CPU-only synthetic regression for safe ordinal-map adapter.

No public/hidden competition rows, model inference, Kaggle GPU or submission.
"""
from __future__ import annotations
import unittest
from scripts.ntl_native_gpu_mapfix_realpath_20261112 import (
    normalize_device_map,MODEL_LINE,INJECT,POLICY,
)

class TestDeviceOrdinalCompat(unittest.TestCase):
    def test_integer_gpu_indices_become_explicit_cuda(self):
        x={"x":0,"y":1,"z":"cpu","q":"disk","v":"cuda:1"}
        self.assertEqual(normalize_device_map(x,2),
                         {"x":"cuda:0","y":"cuda:1","z":"cpu","q":"disk","v":"cuda:1"})
        self.assertEqual(x["x"],0)

    def test_bad_ordinals_never_silently_fall_back(self):
        for value,n in ((-1,2),(2,2),(0,0),(0,-1)):
            with self.assertRaises(ValueError):
                normalize_device_map({"a":value},n)

    def test_no_bool_to_device_and_no_unnecessary_changes(self):
        x={"disabled":True,"none":None,"c":"cpu","string_number":"0"}
        self.assertEqual(normalize_device_map(x,2),x)

    def test_injection_retains_exact_original_load_call_and_does_not_train(self):
        self.assertIn("AutoModel.from_pretrained",MODEL_LINE)
        self.assertIn("device_map='auto'",MODEL_LINE)
        self.assertIn("_ntl_prior_warmup(model, fixed, hf_quantizer)",INJECT)
        self.assertIn("torch.cuda.set_device(0)",INJECT)
        self.assertEqual(INJECT.count(POLICY),2)
        self.assertNotIn(".fit(",INJECT)
        self.assertNotIn("optimizer",INJECT)

if __name__=="__main__":
    unittest.main()
