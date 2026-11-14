"""Build one no-fit competition candidate from the exact 0.683 scored source.

This renderer is intentionally CPU-only: a separate one-shot authorized release
workflow must perform a Kaggle code submission, after private runtime and
transfer/quality gates. No submission is triggered here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.ntl_glm_observational_pair import (
    ASSETS, BINARY_SOURCE, CHAMPION, CHAMP_CODE_SHA256, PAYLOAD_SHA,
    likelihood_payload, ROOT
)
from scripts.ntl_wemm_isolated_fold0_smoke_20261113 import cell_text, new_code

DEST = "kolihhan/ntl-glm-conditional-official-candidate"
EXTRA_CODE = {
    "ntl_glm_runner.py": "kaggle/ntl_glm_observational_pair.py",
    "ntl_glm_conditional_runner.py": "kaggle/ntl_glm_conditional_runner.py",
    "ntl_glm_official_runner.py": "kaggle/ntl_glm_official_runner.py",
}
ORIGINAL_ASSET_COUNTS = (17, 8)


def build(champion_dir: Path, out: Path, payload: str) -> dict:
    if hashlib.sha256(__import__("base64").b64decode(payload, validate=True)).hexdigest() != PAYLOAD_SHA:
        raise ValueError("GLM scoring payload drift")
    [notebook_path] = champion_dir.glob("*.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    original_cells = notebook["cells"]
    original_cell_count = len(original_cells)
    source = "\n".join(cell_text(cell) for cell in original_cells if cell.get("cell_type") == "code")
    if hashlib.sha256(source.encode()).hexdigest() != CHAMP_CODE_SHA256:
        raise ValueError("not the 0.683 champion executable")
    metadata = json.loads((champion_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("id") != CHAMPION or metadata.get("is_private") is not True
            or metadata.get("competition_sources") != ["name-that-llm"]
            or (len(metadata.get("dataset_sources", [])), len(metadata.get("model_sources", [])))
               != ORIGINAL_ASSET_COUNTS or
            "FAST_COMMIT = False" not in source or
            source.count('"gauss": 0.0') != 2 or "_ga.fit(" in source or
            "NO_NEW_TRAINING: supervised classifier fitting forbidden" not in source):
        raise ValueError("scored champion assets/no-fit gate changed")
    for name, path in EXTRA_CODE.items():
        code = (ROOT / path).read_text(encoding="utf-8")
        compile(code, path, "exec")
        if "competition_submit_code" in code:
            raise ValueError("official notebook must not self-submit")
        if name == "ntl_glm_official_runner.py" and "scorer.scores(payload)" not in code:
            raise ValueError("official code lost frozen GLM scoring call")
    injected = (
        "from pathlib import Path\n"
        "import sys\n"
        "d=Path('/kaggle/temp')\n"
        + "".join(
            f"(d / {name!r}).write_text({(ROOT / path).read_text(encoding='utf-8')!r})\n"
            for name, path in EXTRA_CODE.items()
        ) +
        "sys.path.insert(0, str(d))\n"
        "import ntl_glm_official_runner\n"
        f"ntl_glm_official_runner.finalize(DATA, {payload!r})\n"
    )
    notebook["cells"].append(new_code(injected, 99))
    metadata.pop("id_no", None)
    metadata.pop("docker_image", None)
    metadata.update(id=DEST, title=DEST.split("/", 1)[1],
                    enable_gpu=True, enable_tpu=False,
                    enable_internet=False, machine_shape="NvidiaTeslaT4",
                    is_private=True)
    metadata["dataset_sources"] = list(dict.fromkeys(
        [*metadata["dataset_sources"], *ASSETS]))
    metadata["kernel_sources"] = list(dict.fromkeys(
        [*metadata.get("kernel_sources", []), BINARY_SOURCE]))
    file_name = metadata["code_file"]
    if Path(file_name).name != file_name or not file_name.endswith(".ipynb"):
        raise ValueError("unexpected notebook path")
    if out.exists():
        raise FileExistsError("refuse overwrite")
    out.mkdir(parents=True)
    (out / file_name).write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "source_champion_sha256": CHAMP_CODE_SHA256,
        "glm_likelihood_payload_sha256": PAYLOAD_SHA,
        "production_code_sha256": {
            name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for name, path in EXTRA_CODE.items()
        },
        "original_code_cells": original_cell_count,
        "appended_cells": len(notebook["cells"]) - original_cell_count,
        "kernel": DEST,
        "new_training": False, "official_submission_created": False,
        "truth_used_by_production": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("NTL_GLM_OFFICIAL_CANDIDATE_STATIC=" + json.dumps(
        build(args.champion, args.out, likelihood_payload()), sort_keys=True
    ), flush=True)
