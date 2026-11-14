"""Render a single frozen GLM-only 64-pair external-source private GPU check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.ntl_glm_observational_pair import (
    ROOT, build as build_observational, likelihood_payload,
)
from scripts.ntl_wemm_isolated_fold0_smoke_20261113 import cell_text

DEST = "kolihhan/ntl-glm-crosssource-paired64-nov14"
SOURCE = "lxdlxdlxd/ntl-synth-generations-v1"
RUNTIME = ROOT / "kaggle/ntl_glm_crosssource_private.py"
ROUTER = ROOT / "kaggle/ntl_glm_conditional_runner.py"


def build(champion: Path, out: Path, payload: str) -> dict:
    reference = build_observational(champion, out, payload)
    [notebook_path] = out.glob("*.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    runtime = RUNTIME.read_text(encoding="utf-8")
    router = ROUTER.read_text(encoding="utf-8")
    compile(runtime, str(RUNTIME), "exec")
    compile(router, str(ROUTER), "exec")
    cells = notebook["cells"]
    helper = [
        i for i, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
        and "Path('/kaggle/temp/ntl_glm_runner.py').write_text(" in cell_text(cell)
    ]
    prepare = [
        i for i, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
        and "DATA = str(ntl_glm_runner.prepare(DATA))" in cell_text(cell)
    ]
    if len(helper) != 1 or len(prepare) != 1 or helper[0] >= prepare[0]:
        raise ValueError("champion private diagnostic injection layout drift")
    helper_src = cell_text(cells[helper[0]])
    if helper_src.count("import ntl_glm_runner") != 1:
        raise ValueError("not exact base runtime import")
    helper_src += (
        "\n"
        "Path('/kaggle/temp/ntl_glm_conditional_runner.py').write_text(" + repr(router) + ")\n"
        "Path('/kaggle/temp/ntl_glm_crosssource_private.py').write_text(" + repr(runtime) + ")\n"
        "import ntl_glm_crosssource_private\n"
    )
    cells[helper[0]]["source"] = helper_src.splitlines(keepends=True)

    prepare_src = cell_text(cells[prepare[0]])
    prepare_src = prepare_src.replace(
        "DATA = str(ntl_glm_runner.prepare(DATA))",
        "DATA = ntl_glm_crosssource_private.prepare(DATA)",
    )
    cells[prepare[0]]["source"] = prepare_src.splitlines(keepends=True)
    last = cell_text(cells[-1])
    if not last.startswith("ntl_glm_runner.compare(DATA, "):
        raise ValueError("base observational terminal entrypoint drift")
    cells[-1]["source"] = (
        "ntl_glm_crosssource_private.compare(DATA, " + repr(payload) + ")\n"
    ).splitlines(keepends=True)

    metadata_path = out / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if not metadata["is_private"] or not metadata["enable_gpu"] or metadata["competition_sources"] != ["name-that-llm"]:
        raise ValueError("private kernel settings were weakened")
    metadata["id"] = DEST
    metadata["title"] = DEST.split("/", 1)[1]
    metadata["dataset_sources"] = list(dict.fromkeys([*metadata["dataset_sources"], SOURCE]))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    return {
        "kernel": DEST, "source": SOURCE, "source_sha256": "8339081afee1d4aed0b806a686d91978f0d7815bc67fb8f620860477e0895675",
        "pairs": 64, "rows": 128,
        "champion_code_sha256": reference["champion_code_sha256"],
        "glm_payload_sha256": reference["public_likelihood_payload_sha256"],
        "new_training": False, "official_submission": False,
        "source_independence_proven": False,
        "runtime_sha256": hashlib.sha256(runtime.encode()).hexdigest()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("NTL_GLM_EXTERNAL_PAIR_BUILD=" + json.dumps(
        build(args.champion, args.out, likelihood_payload()), sort_keys=True
    ), flush=True)
