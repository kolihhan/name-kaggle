"""CPU-only fail-closed GLM 64-prompt selector tests."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from ntl_glm_crosssource_private import PAIRS, select_pairs


def row(pid, label, target="80", setting="default"):
    return {
        "label": label, "pid": str(pid), "user_message": "Original prompt " + str(pid),
        "text": f"distinct output for {pid} {label}", "target_words": target,
        "setting": setting, "provider": "zhipu", "returned_model": label,
    }


def panel():
    return [row(i, label) for i in range(PAIRS + 2)
            for label in ("glm-5.3", "glm-5.3-flash")]


class TestSelectExternalGLM(unittest.TestCase):
    def test_deterministic_balanced_prompts_without_label_metadata_drift(self):
        x = panel()
        a = select_pairs(x)
        b = select_pairs(list(reversed(x)))
        self.assertEqual(a, b)
        self.assertEqual(len(a), PAIRS * 2)
        self.assertEqual(len({z["pid"] for z in a}), PAIRS)
        self.assertEqual(sum(z["label"] == "glm-5.3" for z in a), PAIRS)
        self.assertEqual(sum(z["label"] == "glm-5.3-flash" for z in a), PAIRS)

    def test_invalid_served_model_or_insufficient_pairs_rejected(self):
        x = panel()
        x[0]["returned_model"] = "other"
        with self.assertRaisesRegex(ValueError, "served-model"):
            select_pairs(x)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            select_pairs(panel()[:12])


if __name__ == "__main__":
    unittest.main()
