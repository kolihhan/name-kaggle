import unittest
from scripts.ntl_pro_source_heterogeneity import fishertwo,wilson,evaluate

class SourceHeterogeneityTests(unittest.TestCase):
    def test_fisher_identity_and_extremes(self):
        self.assertAlmostEqual(fishertwo(5,10,5,10),1.0,places=9)
        self.assertLess(fishertwo(0,20,20,20),1e-8)
    def test_frozen_real_evidence_and_conclusions(self):
        r=evaluate()
        self.assertEqual([(d["sol_longer"],d["total_prompt_pairs"])
            for d in r["cohorts"]],[(75,150),(195,196),(20,47)])
        self.assertLess(r["heterogeneity_tests"][0]["two_sided_fisher_exact_p_posthoc"],1e-12)
        self.assertLess(r["heterogeneity_tests"][1]["two_sided_fisher_exact_p_posthoc"],1e-12)
        self.assertFalse(r["holdout_fresh"])
        self.assertFalse(r["balanced_accuracy_measured"])
    def test_invalid_counts_rejected(self):
        for args in ((2,1,0,10), (5,0,5,10)):
            with self.assertRaises(ValueError):fishertwo(*args)
        with self.assertRaises(ValueError):wilson(5,3)

if __name__=="__main__":
    unittest.main()
