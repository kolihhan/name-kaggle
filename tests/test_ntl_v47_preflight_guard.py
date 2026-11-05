"""No-network guards for exact no-training Lightning V47 Kaggle preflight."""
import unittest

from scripts.ntl_v47_full_preflight_submit_once import verify_kaggle_push_accepted


class TestExactPushGate(unittest.TestCase):
    def test_valid_new_kernel_version(self):
        self.assertEqual(
            verify_kaggle_push_accepted(
                0, "Kernel version 6 successfully pushed. Please check progress at Kaggle."
            ),
            6,
        )

    def test_quota_rejection_with_zero_cli_exit_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, "PREFLIGHT_PUSH_REJECTED"):
            verify_kaggle_push_accepted(
                0,
                "Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.",
            )

    def test_http_403_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, "PREFLIGHT_PUSH_REJECTED"):
            verify_kaggle_push_accepted(
                0,
                "403 Client Error: Forbidden for url: "
                "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel",
            )

    def test_old_completed_canary_is_not_new_accepted_version(self):
        with self.assertRaisesRegex(RuntimeError, "PREFLIGHT_PUSH_UNCONFIRMED"):
            verify_kaggle_push_accepted(
                0, 'kolihhan/ntl-private-permission-canary has status "KernelWorkerStatus.COMPLETE"'
            )

    def test_nonzero_exit_fails(self):
        with self.assertRaisesRegex(RuntimeError, "PREFLIGHT_PUSH_REJECTED"):
            verify_kaggle_push_accepted(1, "No permission to push kernel")


if __name__ == "__main__":
    unittest.main()
