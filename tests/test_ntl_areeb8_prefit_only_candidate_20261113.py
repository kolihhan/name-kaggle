"""Synthetic structural regression for Areeb eight-model no-fit Notebook transform."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import ntl_areeb8_prefit_only_candidate_20261113 as build


def src(x):
    return {"cell_type":"code","metadata":{},"source":x.splitlines(keepends=True)}


class AreebNoFitTests(unittest.TestCase):
    def input(self,root):
        inp=root/"src"; inp.mkdir()
        notebook={"cells":[
          src("x=1\n"+build.FAST+"\n"),
          src("%%writefile infer.py\n"
              "from sklearn.svm import LinearSVC\n"
              "from sklearn.linear_model import LogisticRegression\n"
              "_svc.fit(A, y)\n"
              "prm = _ga.fit(Z,y,G,kwargs)\n"
              "v = LogisticRegression(C=10)\n"+
              "# comment\n"*60),
          src("""LAYERS = [2, 4, 6]
MODELS = [{'name':'q14v5',"gauss": 0.70},{'name':'q8v5',"gauss": 0.44}]
cmd = ['--svc_w', '0', '--distort_w', '0']
if FAST_COMMIT:
    print('fixture')
else:
    print('real')
""")
         ]}
        path=inp/"test.ipynb";data=json.dumps(notebook).encode();path.write_bytes(data)
        (inp/"kernel-metadata.json").write_text(json.dumps(
            {"id":build.SOURCE,"title":"test", "is_private":False,"enable_gpu":True,
             "enable_internet":False,"kernel_type":"notebook","code_file":path.name,
             "competition_sources":["name-that-llm"],
             "dataset_sources":["fixture"]*17,"kernel_sources":["fixture"]*8,
             "model_sources":["fixture"]*8}))
        return inp,hashlib.sha256(data).hexdigest()

    def test_exact_one_candidate_without_fitting(self):
        with TemporaryDirectory() as root:
            root=Path(root);p,sha=self.input(root)
            with patch.object(build,"EXPECTED_SHA",sha):
                r=build.build(p,root/"out")
            self.assertTrue(r["runtime_supervised_fit_blocked"])
            self.assertFalse(r["new_training"])
            j=json.loads(next((root/"out"/"kernel").glob("*.ipynb")).read_text())
            text="\n".join(build.source(c) for c in j["cells"])
            self.assertIn("FAST_COMMIT = False",text)
            self.assertNotIn(build.FAST,text)
            self.assertIn("NO_NEW_TRAINING: supervised classifier fitting forbidden",text)
            self.assertEqual(text.count('"gauss": 0.0'),2)
            self.assertNotIn("_ga.fit(",text)
            self.assertIn("NTL_AREEB_NO_SUPERVISED_FIT_GUARD",text)

    def test_source_hash_drift_blocks(self):
        with TemporaryDirectory() as root:
            root=Path(root);p,_=self.input(root)
            with self.assertRaisesRegex(ValueError,"source SHA drift"):
                build.build(p,root/"out")

    def test_runtime_fit_inventory_drift_blocks(self):
        with TemporaryDirectory() as root:
            root=Path(root);p,_=self.input(root)
            path=p/"test.ipynb";b=json.loads(path.read_text())
            s=build.source(b["cells"][1]).replace("_svc.fit(","_svc.retrain(")
            b["cells"][1]["source"]=s.splitlines(keepends=True)
            path.write_text(json.dumps(b))
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(build,"EXPECTED_SHA",sha):
                with self.assertRaisesRegex(ValueError,"supervised fit call inventory"):
                    build.build(p,root/"out")


if __name__=="__main__":unittest.main()
