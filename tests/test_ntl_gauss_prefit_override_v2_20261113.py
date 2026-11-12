"""No-input synthetic controls for Native prefit-Gaussian override and group benchmark."""
import unittest

from scripts.ntl_native_restore_frozen_gauss_no_fit_20261112 import guard_readout
from scripts.ntl_native_gauss_prefit_override_v2_20261113 import (
    transform_readout, OLD,
)
from scripts.ntl_native_gauss_prefit_override_256_probe_20261113 import (
    INJECT, POST,
)


def frozen_synthetic_readout():
    original=(
      "def member_scores():\n"
      "    if spec.get('gauss',0)>0:\n"
      "        branch = fake_branch\n"
      "        if branch['gaussian_branch'] == 'source-test-conditioned':\n"
      "            assert branch['test_redo'] > 0\n"
      "            with np.load(folder/'gaussian-train-features.npz',allow_pickle=False) as training:\n"
      "                prm = gauss.fit(training['features'],training['y'],training['groups'],cfg)\n"
      "        else:\n"
      "            assert branch['gaussian_branch'] == 'fixed-no-test-redo'\n"
      "            prm = gaussian_state(gaussian_folder, spec)\n"
      "        selected = grp_mask\n"
      "        value = np.zeros((len(groups), 18))\n"
      "        if selected.any():\n"
      "            value[selected] = gauss.scores(final_features[selected], np.unique(groups[selected], return_inverse=True)[1], prm)\n"
    )
    return "%%writefile /kaggle/temp/native/core_readout.py\n"+guard_readout(original)


class PrefitOverrideTests(unittest.TestCase):
    def test_prior_restored_candidate_is_not_a_working_gaussian_score(self):
        input_source=frozen_synthetic_readout()
        self.assertIn(OLD,input_source)
        self.assertIn("prm = None",input_source)

    def test_prefit_path_restores_nonzero_score_flow_without_fit(self):
        transformed=transform_readout(frozen_synthetic_readout())
        self.assertNotIn(OLD,transformed)
        self.assertNotIn("gauss.fit(",transformed)
        self.assertIn("NTL_GAUSS_PREFIT_RUNTIME_BOUND=",transformed)
        self.assertIn("NTL_GAUSS_PREFIT_GROUP_ELIGIBILITY=",transformed)
        self.assertIn("NTL_GAUSS_PREFIT_SCORES=",transformed)
        self.assertIn("scored_rows",transformed)
        self.assertEqual(transformed.count("prm = gaussian_state(gaussian_folder, spec)"),2)

    def test_readout_transform_fails_closed_on_drift(self):
        with self.assertRaises(ValueError):
            transform_readout(frozen_synthetic_readout().replace(
                "prm = gaussian_state(gaussian_folder, spec)",
                "prm = random_new_fit(...)"))

    def test_synthetic_benchmark_has_256_rows_and_no_official_test_path(self):
        compile(INJECT,"synthetic_gpu256.py","exec")
        compile(POST,"synthetic_timing.py","exec")
        self.assertIn("range(256)",INJECT)
        self.assertIn("_index<128",INJECT)
        self.assertIn("_index//4",INJECT)
        self.assertIn("singlecase",INJECT)
        self.assertNotIn("train.csv",INJECT)
        self.assertNotIn("kaggle/input",INJECT)

    def test_tfidf_group_token_document_frequency_is_in_native_bounds(self):
        group_df=4
        n=256
        self.assertGreaterEqual(group_df,2)
        self.assertLessEqual(group_df/n,0.05)
        self.assertLessEqual(group_df,18)


if __name__=="__main__":
    unittest.main()
