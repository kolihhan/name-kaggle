"""CPU regression tests for preview bypass; these do not test model inference."""
import copy
import unittest

from scripts import ntl_dual_embedding_wemm_preview as builder


class RealInferencePathTests(unittest.TestCase):
    def force(self, cells):
        self.assertTrue(callable(getattr(builder, "force_real_inference", None)),
                        "the builder must remove the known-answer preview shortcut")
        builder.force_real_inference(cells)

    def test_public_fixture_and_larger_inputs_execute_model_branch(self):
        source = (
            "preview = emit_known_preview(DATA,None,OUTPUT)\n"
            "if not preview:\n"
            "    infer(DATA, OUTPUT)\n"
        )
        for as_lines in (False, True):
            cells = [{"cell_type": "code", "source": source.splitlines(True) if as_lines else source}]
            self.force(cells)
            for size in (1, 3, 64):
                with self.subTest(as_lines=as_lines, rows=size):
                    calls = []
                    def shortcut(*args):
                        self.fail("hardcoded public preview was called")
                    namespace = {
                        "DATA": list(range(size)), "OUTPUT": "synthetic-output",
                        "emit_known_preview": shortcut,
                        "infer": lambda rows, dest: calls.append((len(rows), dest)),
                    }
                    exec("".join(cells[0]["source"]), namespace)
                    self.assertEqual(calls, [(size, "synthetic-output")])

    def test_source_drift_or_duplicate_anchor_aborts_without_mutation(self):
        needle = "preview = emit_known_preview(DATA,None,OUTPUT)"
        for source in ("preview = other_preview()", needle + "\n" + needle):
            cells = [{"cell_type": "code", "source": source}]
            before = copy.deepcopy(cells)
            with self.assertRaisesRegex(RuntimeError, "preview.*anchor"):
                self.force(cells)
            self.assertEqual(cells, before)

    def test_metadata_markdown_and_unrelated_cells_are_preserved(self):
        cells = [
            {"cell_type": "markdown", "source": ["Original author credit"], "metadata": {}},
            {"cell_type": "code", "source": "MODEL = 'pinned'\n", "metadata": {"tag": "keep"}},
            {"cell_type": "code", "source": ["preview = emit_known_preview(DATA,None,OUTPUT)\n"]},
        ]
        before = copy.deepcopy(cells[:2])
        self.force(cells)
        self.assertEqual(cells[:2], before)


if __name__ == "__main__":
    unittest.main()
