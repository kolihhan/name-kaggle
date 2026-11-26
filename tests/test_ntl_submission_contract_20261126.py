"""CPU-only actual Parquet roundtrip and failure tests for NTL code scoring."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from scripts.ntl_submission_contract_20261126 import (
    LABELS, SubmissionContractError, check, verify_files
)

class ContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, n=4000):
        ids = ["0" * 4 + str(i) for i in range(n)]
        test = pd.DataFrame({"id": ids, "text": ["a text"] * n})
        pred = pd.DataFrame({"id": ids, "model": ["gpt-5.5"] * n})
        test.to_csv(self.root / "test.csv", index=False)
        pred.to_parquet(self.root / "submission.parquet", index=False)
        return test, pred

    def test_hidden_size_full_roundtrip(self):
        for n in (1, 3, 252, 4000, 4500):
            self.write(n)
            result = verify_files(self.root / "test.csv", self.root / "submission.parquet")
            self.assertEqual(result["rows"], n)
            self.assertGreater(result["file_bytes"], 0)
            self.assertEqual(len(result["sha256"]), 64)

    def test_all_18_legal(self):
        ids = [str(i) for i in range(18)]
        train = pd.DataFrame({"id": ids})
        pred = pd.DataFrame({"id": ids, "model": sorted(LABELS)})
        self.assertEqual(check(train, pred)["unique_predictions"], 18)

    def test_mismatched_order(self):
        test, pred = self.write(4)
        with self.assertRaisesRegex(SubmissionContractError, "original order"):
            check(test, pred.iloc[::-1].reset_index(drop=True))

    def test_duplicate_output_ids(self):
        test, pred = self.write(4)
        pred.loc[3, "id"] = pred.loc[2, "id"]
        with self.assertRaisesRegex(SubmissionContractError, "duplicate id"):
            check(test, pred)

    def test_illegal_model(self):
        test, pred = self.write(4)
        pred.loc[0, "model"] = "gemini-3.9-flash"
        with self.assertRaisesRegex(SubmissionContractError, "unrecognized"):
            check(test, pred)

    def test_missing_column(self):
        test, pred = self.write(4)
        with self.assertRaisesRegex(SubmissionContractError, "exactly"):
            check(test, pred[["id"]])

    def test_null_prediction(self):
        test, pred = self.write(4)
        pred.loc[0, "model"] = None
        with self.assertRaisesRegex(SubmissionContractError, "null model"):
            check(test, pred)

    def test_blank_and_missing_id(self):
        test, pred = self.write(4)
        pred.loc[0, "id"] = " "
        with self.assertRaisesRegex(SubmissionContractError, "blank id"):
            check(test, pred)

    def test_zero_byte_and_invalid_parquet(self):
        self.write(4)
        path = self.root / "submission.parquet"
        path.write_bytes(b"")
        with self.assertRaisesRegex(SubmissionContractError, "zero-byte"):
            verify_files(self.root / "test.csv", path)
        path.write_bytes(b"not parquet")
        with self.assertRaisesRegex(SubmissionContractError, "invalid parquet"):
            verify_files(self.root / "test.csv", path)

    def test_filename(self):
        self.write(4)
        path = self.root / "unapproved.parquet"
        path.write_bytes((self.root / "submission.parquet").read_bytes())
        with self.assertRaisesRegex(SubmissionContractError, "filename"):
            verify_files(self.root / "test.csv", path)

if __name__ == "__main__":
    unittest.main()
