"""No-GPU regression for the one-fold frozen WeMM-only Kaggle builder."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import ntl_wemm_isolated_fold0_smoke_20261113 as b


def c(src):
    return {"cell_type":"code","source":src.splitlines(keepends=True)}


class SmokeBuilderTests(unittest.TestCase):
    def fixture(self, root):
        source = root/"src"
        source.mkdir()
        cells = [{"cell_type":"markdown","source":["fixture"]} for _ in range(55)]
        cells[3] = c("# Mount shim\n\ndef _ntl_fix_obj(x): return x\n")
        cells[42] = c("%%writefile /kaggle/temp/ntl-gsq-v27-runtime/stage_dataset_weights.py\nx=1\n")
        cells[51] = c("""%%writefile /kaggle/temp/ntl-gsq-v27-runtime/wemm_embedding_worker.py
x = load_file(str(self.assets / f'fold{fold}/trainables.safetensors'))
""")
        cells[53] = c("""%%writefile /kaggle/temp/ntl-gsq-v27-runtime/wemm_fusion.py
def calibrate_wemm(*args): pass
def load_wemm(*args): pass
""")
        cells[54] = c("""CONFIG['wemm_embedding'] = {'base':'/kaggle/input/base','assets':'/kaggle/input/assets','share':0.07,'deadline_seconds':30240}
""")
        # Exactly 43 code cells; the rest are markdown.
        for i in list(range(0,55)):
            if i in (3,42,51,53,54):
                continue
            if sum(t.get("cell_type")=="code" for t in cells)==43:
                break
            cells[i] = c("safe=1\n")
        book={"cells":cells}
        data=json.dumps(book).encode()
        (source/"notebook.ipynb").write_bytes(data)
        (source/"kernel-metadata.json").write_text(json.dumps({
            "id":b.SOURCE, "is_private":False,"competition_sources":["name-that-llm"],
            "model_sources":[b.WEMM_SOURCE],"dataset_sources":[b.FROZEN_HEADS]
        }))
        code=[b.cell_text(v) for v in cells if v.get("cell_type")=="code"]
        digest=hashlib.sha256(json.dumps(code,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        return source,hashlib.sha256(data).hexdigest(),digest

    def test_output_is_private_no_competition_and_frozen_one_fold(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            src,sha,digest=self.fixture(root)
            with patch.object(b,"SHA256",sha),patch.object(b,"DIGEST",digest):
                record=b.build(src,root/"out")
            meta=json.loads((root/"out"/"kernel-metadata.json").read_text())
            nb=json.loads((root/"out"/meta["code_file"]).read_text())
            self.assertEqual(record["only_frozen_fold"],0)
            self.assertFalse(record["new_training"])
            self.assertFalse(record["official_competition_submission"])
            self.assertEqual(meta["competition_sources"],[])
            self.assertEqual(meta["model_sources"],[b.WEMM_SOURCE])
            self.assertEqual(meta["dataset_sources"],[b.FROZEN_HEADS])
            self.assertTrue(meta["is_private"])
            self.assertEqual(len(nb["cells"]),7)
            joined="\n".join(b.cell_text(x) for x in nb["cells"] if x["cell_type"]=="code")
            self.assertIn("synthetic-only.csv",joined)
            self.assertIn('"--folds","0"',joined)
            self.assertNotIn("CONFIG['wemm_embedding']",joined)

    def test_source_mutation_refused(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            src,sha,_=self.fixture(root)
            (src/"notebook.ipynb").write_text((src/"notebook.ipynb").read_text()+" ")
            with patch.object(b,"SHA256",sha):
                with self.assertRaisesRegex(ValueError,"notebook changed"):
                    b.build(src,root/"out")

    def test_supervised_fit_worker_refused(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            src,_,_=self.fixture(root)
            path=src/"notebook.ipynb"
            book=json.loads(path.read_text())
            book["cells"][51]["source"].append("model.fit(X,y)\n")
            data=json.dumps(book).encode()
            path.write_bytes(data)
            code=[b.cell_text(c) for c in book["cells"] if c.get("cell_type")=="code"]
            digest=hashlib.sha256(json.dumps(code,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
            with patch.object(b,"SHA256",hashlib.sha256(data).hexdigest()),patch.object(b,"DIGEST",digest):
                with self.assertRaisesRegex(ValueError,"fitting/training"):
                    b.build(src,root/"out")


if __name__=="__main__":
    unittest.main()
