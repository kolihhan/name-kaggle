"""Synthetic source guard unit tests; no Kaggle inputs, fitting, or GPU."""
import unittest

from scripts.ntl_native_restore_frozen_gauss_no_fit_20261112 import (
    WANTED, restore_config, guard_readout, guard_worker,
    WORKER_SOURCE_FIT, WORKER_TRAIN_READ, SCORING_CONDITION,
)


def synthetic_config():
    specs=[
        {"name":"q14v5","gauss":0.0},
        {"name":"q8v5","gauss":0.0},
        {"name":"ll8v5"},{"name":"n14b"},{"name":"q25_14"},{"name":"deb"}
    ]
    return "CONFIG = "+repr({"models":specs})


def synthetic_readout():
    return (
      "def member_scores():\n"
      "    if spec.get('gauss',0)>0:\n"
      "        branch = fake_branch\n"
      "        if branch['gaussian_branch'] == 'source-test-conditioned':\n"
      "            assert branch['test_redo'] > 0\n"
      "            with np.load(folder/'gaussian-train-features.npz',allow_pickle=False) as training:\n"
      "                prm = gauss.fit(training['features'],training['y'],training['groups'],cfg)\n"
      "            prm['T1'] = 3.0\n"
      "        else:\n"
      "            assert branch['gaussian_branch'] == 'fixed-no-test-redo'\n"
      "            prm = gaussian_state(gaussian_folder, spec)\n"
      "        selected = grp_mask\n"
      "        if selected.any():\n"
      "            value[selected] = gauss.scores(final_features[selected],groups,prm)\n"
    )


class FrozenHeadTests(unittest.TestCase):
    def test_only_restores_unchanged_original_numeric_heads(self):
        updated,restored=restore_config(synthetic_config())
        self.assertEqual(restored,WANTED)
        self.assertIn("'gauss': 0.7",updated)
        self.assertIn("'gauss': 0.44",updated)
        self.assertNotIn("'gauss': 0.0",updated)

    def test_config_drift_fails_closed(self):
        with self.assertRaises(ValueError):
            restore_config(synthetic_config().replace("'gauss': 0.0","'gauss': 0.1",1))

    def test_readout_frozen_state_unmodified(self):
        original=synthetic_readout()
        result=guard_readout(original)
        self.assertIn("prm = gaussian_state(gaussian_folder, spec)",result)
        self.assertNotIn("gauss.fit(",result)
        self.assertIn("if selected.any() and prm is not None:",result)
        self.assertIn("prm = None",result)

    def test_extra_fit_precondition_never_passes(self):
        worker="\n".join((WORKER_SOURCE_FIT,WORKER_TRAIN_READ))
        result=guard_worker(worker)
        self.assertIn("if False and not train_final",result)
        self.assertIn("train_frame = None",result)
        self.assertNotIn("pd.read_csv",result)
        with self.assertRaises(ValueError):
            guard_worker(result)

    def test_bad_readout_branch_is_rejected(self):
        with self.assertRaises(ValueError):
            guard_readout(synthetic_readout().replace(
                "if branch['gaussian_branch'] == 'source-test-conditioned':",
                "if branch['gaussian_branch'] == 'unrecognized':"))


if __name__=="__main__":
    unittest.main()
