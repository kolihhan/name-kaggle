"""Synthetic invariants only; these do NOT measure attribution accuracy."""
import unittest
import numpy as np

from scripts.matched_panel_evidence import matched_pair_evidence


class MatchedPanelEvidenceTest(unittest.TestCase):
    def test_orientation_reverses_on_label_swap(self):
        # Paired responses share a topic; A and B have opposite identity
        # signatures in the third coordinate.
        t = np.linspace(-1, 1, 24)
        a = np.column_stack([3+t, np.cos(t), np.full(t.size, 0.6)])
        b = np.column_stack([3+t, np.cos(t), np.full(t.size, -0.6)])
        q = np.array([3.1, 0.6, 0.55])
        pos = matched_pair_evidence(q, a, b, top_k=12)
        neg = matched_pair_evidence(q, b, a, top_k=12)
        self.assertGreater(pos.margin, 0)
        self.assertAlmostEqual(pos.margin, -neg.margin, places=12)
        self.assertEqual(pos.panels_used, 12)
        self.assertEqual(pos.agreement, 1.0)

    def test_rejects_unusable_inputs(self):
        a = np.ones((3, 4))
        b = np.ones((3, 4))
        q = np.ones(4)
        with self.assertRaises(ValueError):
            matched_pair_evidence(q, a, b, top_k=4)
        with self.assertRaises(ValueError):
            matched_pair_evidence(q, a[:2], b, top_k=2)
        with self.assertRaises(ValueError):
            matched_pair_evidence(np.zeros(4), a, b, top_k=2)
        with self.assertRaises(ValueError):
            matched_pair_evidence(q, a, b, top_k=2)


if __name__ == "__main__":
    unittest.main()
