"""One frozen genuine-inference NTL 0.691 external 18-class transfer pilot.

Build from exact original owned Lightning0691 Notebook V2, changing ONLY:
  * input text route to 216 prompted-source-disjoint externally served examples
  * add the public external dataset mount
No Gaussian fit / Atlas training; output notebook never writes truth labels.
The hidden Kaggle competition test is NOT used or submitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from scripts import ntl_lightning_0694_scale300_private as base

ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "scripts/ntl_oleg18_frozen_selector_20261116.py"
DEST = "kolihhan/ntl-oleg18-frozen0691-probe-nov16"
SOURCE = "kolihhan/lightning-0-694-authentic-no-fit-inference"
DATASET = "olegzholobov/name-that-llm-response-corpus"
EXPECTED_ROWS = 216

# NOTE: This is injected only in the original single input-CSV cell; no
# executable changes to the 37 already fitted prediction modules.
INFER_INPUT = '''
# External public-domain provenance-sensitive quality transfer, labels hidden from input.
from pathlib import Path as _ntl_oleg18_Path
import hashlib as _ntl_oleg18_hashlib
_oleg18_root = _ntl_oleg18_Path(
  "/kaggle/input/datasets/olegzholobov/name-that-llm-response-corpus")
for _oleg18_file, _oleg18_sha in NTL_OLEG18_SHA.items():
    _oleg18_path = _oleg18_root / _oleg18_file
    if not _oleg18_path.is_file():
        raise FileNotFoundError("NTL_EXTERNAL_DATASET_MOUNT_MISSING:" + _oleg18_file)
    if _ntl_oleg18_hashlib.sha256(_oleg18_path.read_bytes()).hexdigest()!=_oleg18_sha:
        raise ValueError("NTL_EXTERNAL_DATASET_SOURCE_SHA_DRIFT")
_oleg18_wide = pd.read_parquet(_oleg18_root / "external_short_wide.parquet")
_oleg18_meta = pd.read_parquet(_oleg18_root / "external_cell_metadata.parquet")
_oleg18_selected = ntl_oleg18_select(_oleg18_wide, _oleg18_meta)
assert len(_oleg18_selected)==216
assert _oleg18_selected["model"].nunique()==18
_oleg18_input = _ntl_oleg18_Path(
  "/kaggle/working/ntl_oleg18_frozen216_UNLABELLED.csv")
_oleg18_selected[["id", "text"]].to_csv(_oleg18_input, index=False)
# Prevent public or model-facing serialization of labels / prompt groups.
del _oleg18_selected, _oleg18_wide, _oleg18_meta
DATA = _oleg18_input
print("NTL_OLeg18_FROZEN_18_CLASS_BLIND_INPUT_ONLY=216", flush=True)
'''


def build(source: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("refuse_duplicate_external_pilot")
    raw_selector = SELECTOR.read_text(encoding="utf-8")
    compile(raw_selector, str(SELECTOR), "exec")
    if "def ntl_oleg18_select(" not in raw_selector:
        raise ValueError("selector_not_versioned")
    if ("ntl_oleg18_select" not in INFER_INPUT
        or '["id", "text"]' not in INFER_INPUT
        or "submission.parquet" in INFER_INPUT):
        raise ValueError("external_input_leakage_or_overwrite")
    injected = "\n" + raw_selector + "\n" + INFER_INPUT
    with tempfile.TemporaryDirectory() as tmp:
        intermediate = Path(tmp) / "frozen300"
        audit = base.build(source, intermediate)
        meta = json.loads(
            (intermediate / "kernel-metadata.json").read_text(encoding="utf-8"))
        file_name = meta["code_file"]
        notebook = json.loads(
            (intermediate / file_name).read_text(encoding="utf-8"))
    if audit.get("source_v2_code_sha") != base.V2_CODE_SHA:
        raise ValueError("not_exact_frozen_scored_champion")
    if meta.get("id") != base.TO or meta.get("dataset_sources") is None:
        raise ValueError("unexpected_frozen300_metadata")
    if meta["dataset_sources"].count(DATASET):
        raise ValueError("external_dataset_already_seen_in_champion_mount")
    counts = {"input": 0}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        code = base.source(cell)
        if base.PILOT in code:
            if code.count(base.PILOT) != 1:
                raise ValueError("ambiguous_pilot_input_cell")
            code = code.replace(base.PILOT, injected, 1)
            cell["source"] = code.splitlines(keepends=True)
            counts["input"] += 1
    if counts["input"] != 1:
        raise ValueError("external_provenance_input_not_installed")
    scientific_code = "\n".join(
        base.source(cell) for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for guard in ("preview = False  # always genuine model inference",
                  "NO_NEW_TRAINING: Gaussian fit blocked",
                  "NO_NEW_TRAINING: Atlas training blocked"):
        if scientific_code.count(guard) != 1:
            raise ValueError("frozen_no_fit_or_full_inference_guard_drift")
    if ("gauss.fit(" in scientific_code
        or "NTL0694_SCALE300_INPUT_TEXT_ONLY" in scientific_code
        or "NTL_OLeg18_FROZEN_18_CLASS_BLIND_INPUT_ONLY" not in scientific_code):
        raise ValueError("unexpected_frozen_competitor_source_change")
    compiled = 0
    for cell in notebook["cells"]:
        code = base.source(cell)
        if cell.get("cell_type") == "code" and code.startswith("%%writefile "):
            if not code.splitlines()[0].endswith(".py"):
                continue
            compile("\n".join(code.splitlines()[1:]),
                    code.splitlines()[0], "exec")
            compiled += 1
    if compiled != 37:
        raise ValueError("frozen_model_stage_modified")
    meta["dataset_sources"] = list(meta["dataset_sources"]) + [DATASET]
    meta.update({
        "id": DEST, "title": DEST.split("/")[1],
        "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
    })
    if len(meta["dataset_sources"]) != 24:
        raise ValueError("unexpected_new_data_asset_count")
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(notebook, ensure_ascii=False)
    (out / file_name).write_text(payload, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8")
    return {
        "source": SOURCE, "source_sha256": base.V2_CODE_SHA,
        "dest": DEST, "external_dataset": DATASET,
        "target_rows": EXPECTED_ROWS, "classes": 18,
        "per_class": 12, "true_labels_in_model_input": False,
        "previous_fit_assets_unchanged": True,
        "model_stage_count": compiled, "dataset_mounts": 24,
        "notebook_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "selector_sha256": hashlib.sha256(raw_selector.encode()).hexdigest(),
        "gpu_run_in_this_step": False, "official_submission": False,
        "no_new_training": True, "independent_transfer_unverified": True,
        "source_checkpoint_prefit_overlap_unknown": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print("NTL_PRO_OLeg18_FROZEN_0691_PRIVATE_PILOT_BUILD=" +
          json.dumps(build(args.source, args.out), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
