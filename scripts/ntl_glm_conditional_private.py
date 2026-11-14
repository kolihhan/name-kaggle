"""Render one frozen conditional-GLM private latency/paired check, no competition submit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.ntl_glm_observational_pair import build as build_full, likelihood_payload, ROOT
from scripts.ntl_wemm_isolated_fold0_smoke_20261113 import cell_text, new_code

DEST = "kolihhan/ntl-glm-conditional-compare-nov14"
RUNNER = ROOT / "kaggle" / "ntl_glm_conditional_runner.py"
RESULT = "glm-conditional-summary.json"


def build(champion: Path, out: Path, payload: str) -> dict:
    """Reuse all pinned sources/assets; change only the post-champion GLM worker."""
    provenance = build_full(champion, out, payload)
    [notebook_path] = out.glob("*.ipynb")
    notebook = json.loads(notebook_path.read_text())
    if not notebook["cells"] or notebook["cells"][-1].get("cell_type") != "code":
        raise ValueError("missing frozen benchmark terminal code cell")
    original_last = cell_text(notebook["cells"][-1])
    if not original_last.startswith("ntl_glm_runner.compare(DATA, "):
        raise ValueError("frozen observational entrypoint changed")
    runtime = RUNNER.read_text(encoding="utf-8")
    if "competition_submit_code" in runtime or "fit(" in runtime:
        raise ValueError("unexpected fitting or competition submit in conditional runtime")
    compile(runtime, str(RUNNER), "exec")
    code = (
        "from pathlib import Path\n"
        "Path('/kaggle/temp/ntl_glm_conditional_runner.py').write_text(" + repr(runtime) + ")\n"
        "import ntl_glm_conditional_runner\n"
        "ntl_glm_conditional_runner.run(DATA, " + repr(payload) + ")\n"
    )
    notebook["cells"][-1] = new_code(code, 82)
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False))
    meta_path = out / "kernel-metadata.json"
    metadata = json.loads(meta_path.read_text())
    if not metadata["is_private"] or not metadata["enable_gpu"] or not metadata["competition_sources"] == ["name-that-llm"]:
        raise ValueError("private Kaggle runtime contract drift")
    metadata["id"] = DEST
    metadata["title"] = DEST.split("/", 1)[1]
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        **provenance, "kernel": DEST, "conditional_only": True,
        "conditional_runtime_sha256": hashlib.sha256(runtime.encode()).hexdigest(),
        "score_file": RESULT, "official_submission": False, "new_training": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("NTL_GLM_CONDITIONAL_RENDER=" + json.dumps(
        build(args.champion, args.out, likelihood_payload()), sort_keys=True
    ), flush=True)
