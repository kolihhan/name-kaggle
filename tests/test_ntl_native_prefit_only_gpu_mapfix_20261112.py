"""Synthetic AST-only mutation regression. No competition data or models."""
import unittest
from scripts.ntl_native_prefit_only_gpu_mapfix_20261112 import disable_gaussian_only

GOOD="CONFIG = {'models': ["+(
    "{'name': 'q14v5', 'gauss': 0.7}, "
    "{'name': 'q8v5', 'gauss': 0.44}, "
    "{'name': 'll8v5', 'keep':24}, "
    "{'name': 'n14b', 'keep':27}, "
    "{'name': 'q25_14', 'keep':32}, "
    "{'name': 'deb'}]}\n"
)
class Tests(unittest.TestCase):
    def test_exact_frozen_weight_update(self):
        changed,flags=disable_gaussian_only(GOOD)
        self.assertEqual(flags['changes'],2)
        self.assertIn("'gauss': 0.0",changed)
        self.assertEqual(changed.count("'gauss': 0.0"),2)
        self.assertIn("{'name': 'deb'}",changed)

    def test_blocks_drift_in_public_gauss_weights(self):
        with self.assertRaises(ValueError):
            disable_gaussian_only(GOOD.replace("0.7","0.71"))

    def test_blocks_one_missing_model(self):
        with self.assertRaises(ValueError):
            disable_gaussian_only(GOOD.replace("'q8v5'","'other'"))

    def test_blocks_double_modified_candidate(self):
        updated,_=disable_gaussian_only(GOOD)
        with self.assertRaises(ValueError):
            disable_gaussian_only(updated)

    def test_blocks_duplicate_gaussian_target(self):
        with self.assertRaises(ValueError):
            disable_gaussian_only(GOOD.replace("'ll8v5'","'q8v5'"))

if __name__=="__main__":
    unittest.main()
