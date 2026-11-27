"""Historical regression cases: real reported outcomes, not fabricated scores."""
import unittest
from dataclasses import replace
from scripts.ntl_evidence_gate import Evidence, Outcome, classify, require_promotion

class EvidenceGateTest(unittest.TestCase):
    def test_source_cache_429_is_not_scientific_failure(self):
        # 45938678359: 56/140 assets admitted, zero evaluable outer folds.
        e=Evidence(False,False,"nested replay","18class_BA",18)
        self.assertEqual(classify(e),Outcome.BLOCKED)

    def test_green_ci_not_model_quality(self):
        # 46034186246: actual 75-row ID recovery, no new model predictions.
        e=Evidence(True,True,"static ID recovery","none",18)
        self.assertEqual(classify(e),Outcome.ENGINEERING)

    def test_pairwise_prompt_does_not_reject_representation_class(self):
        # 46003869781: 16/32 within-family rankings, spent two prompts.
        e=Evidence(True,True,"Qwen14B anonymous Yes/No prompt","pair_ranking",18,
                   exact_intervention_failed=True)
        self.assertEqual(classify(e),Outcome.PROMPT_RETIRED)

    def test_old_positive_glm_not_current_champion_gain(self):
        # 43548137138: +5/252, older lineage, p=.125.
        e=Evidence(True,True,"old GLM correction","18class_BA",18,
                   same_rows=True,full18_delta=5/252)
        self.assertEqual(classify(e),Outcome.DIAGNOSTIC)
        with self.assertRaises(ValueError):require_promotion(e)

    def test_spent_current_panel_not_fresh(self):
        e=Evidence(True,True,"v36 Fable guard","18class_BA",18,
                   same_rows=True,full18_delta=2/252)
        self.assertEqual(classify(e),Outcome.DIAGNOSTIC)

    def test_hypothetical_pass_and_mutations(self):
        e=Evidence(True,True,"synthetic positive control","18class_BA",18,
                   same_rows=True,independent_source=True,provenance_verified=True,
                   baseline_sha="a"*64,full18_delta=.03,paired_ci_low=.005,
                   runtime_verified=True,diagnostic_only=False)
        require_promotion(e)
        for change in (dict(same_rows=False),dict(independent_source=False),
                       dict(provenance_verified=False),dict(runtime_verified=False),
                       dict(truth_classes=17),dict(paired_ci_low=0),dict(full18_delta=.019),
                       dict(baseline_sha=""),dict(diagnostic_only=True)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                require_promotion(replace(e,**change))

if __name__=="__main__":unittest.main()
