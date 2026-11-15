import unittest
from scripts.ntl_complementarity_gate import evaluate


def actual_shape():
    def row(n, labels, both, neither, rescues, regressions, disagreements):
        return {
            "rows":n,"true_labels":labels,"both_right":both,"both_wrong":neither,
            "rescues":rescues,"regressions":regressions,
            "champ_correct":both+regressions,"wemm_correct":both+rescues,
            "prediction_disagreements":disagreements,
        }
    return {
        "scope":"SPENT_TWO_SOURCE_12_LABEL_DIAGNOSTIC_ONLY",
        "champion_official_submission_ref":"58295802",
        "wemm_folds_evaluated":[0],
        "no_new_training":True,
        "official_submission":False,
        "combined":row(84,12,53,21,3,7,15),
        "per_source":{
            "nimitha":row(48,12,32,12,2,2,7),
            "toan":row(36,12,21,9,1,5,8),
        },
    }


class ComplementarityGateTests(unittest.TestCase):
    def test_existing_counts_reject_and_block_full18(self):
        report=evaluate(actual_shape())
        self.assertEqual(report["net_rescues_minus_regressions"],-4)
        self.assertEqual(report["per_source_net"],{"nimitha":0,"toan":-4})
        self.assertEqual(report["independent_full18_gate"],"BLOCKED_SPENT_12_OF_18")
        self.assertFalse(report["promote"])

    def test_no_promote_even_with_more_rescues_in_spent_source(self):
        sample=actual_shape()
        sample["combined"]["rescues"]=8
        sample["combined"]["wemm_correct"]=61
        sample["combined"]["rows"]=89
        sample["combined"]["prediction_disagreements"]=20
        sample["per_source"]["toan"]["rescues"]=6
        sample["per_source"]["toan"]["wemm_correct"]=27
        sample["per_source"]["toan"]["rows"]=41
        sample["per_source"]["toan"]["prediction_disagreements"]=13
        report=evaluate(sample)
        self.assertEqual(report["candidate_decision"],"PRELIMINARY_ONLY_NOT_PROMOTABLE")
        self.assertFalse(report["promote"])

    def test_tampered_source_totals_fail(self):
        sample=actual_shape()
        sample["combined"]["rescues"]=100
        with self.assertRaises(ValueError):
            evaluate(sample)
        sample=actual_shape()
        sample["per_source"]["nimitha"]["rescues"]=3
        with self.assertRaises(ValueError):
            evaluate(sample)
        sample=actual_shape()
        sample["scope"]="FRESH_18_LABEL"
        with self.assertRaises(ValueError):
            evaluate(sample)


if __name__=="__main__":
    unittest.main()
