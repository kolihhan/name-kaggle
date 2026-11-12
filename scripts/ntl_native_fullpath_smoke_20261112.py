"""Isolated full-inference-path diagnostic for failed official NTL #58268344.

Use the previously submitted own private Notebook, not a third party re-pull.
Only one executable source line is modified: disable 3-row known-preview
shortcut. Preserve all guards and source assets. No training or submission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

SOURCE = "kolihhan/frozen-native-ntl-ensemble-no-new-training"
DEST = "kolihhan/ntl-native-realpath-smoke-20261112"
TARGET_TITLE = "ntl-native-realpath-smoke-20261112"
MARKER = "preview = emit_known_preview(DATA,None,OUTPUT)"
GUARDS = (
    "NO_NEW_TRAINING: supervised Gaussian fitting forbidden",
    "NO_NEW_TRAINING: Atlas training mode forbidden",
    "NO_NEW_TRAINING: core_worker train-final forbidden",
)
OUTPUT_FILENAME = "submission.parquet"


def safe_call(args: list[str]) -> None:
    r = subprocess.run(args, capture_output=True, text=True, timeout=125)
    if r.returncode:
        raise RuntimeError(f"Kaggle read failed for {args[:3]} rc={r.returncode}")


def cell_text(cell: dict) -> str:
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else str(src)


def build(out: Path, *, do_pull: bool=True) -> dict:
    srcdir = out / "source"
    destdir = out / "kernel"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("do not overwrite existing smoke")
    srcdir.mkdir(parents=True)
    destdir.mkdir(parents=True)
    if do_pull:
        safe_call(["kaggle","kernels","pull",SOURCE,"-p",str(srcdir),"-m"])
    candidates = list(srcdir.glob("*.ipynb"))
    if len(candidates) != 1:
        raise RuntimeError("source Notebook missing/ambiguous")
    meta_file = srcdir / "kernel-metadata.json"
    if not meta_file.exists():
        raise RuntimeError("source metadata missing")
    nb = json.loads(candidates[0].read_text(encoding="utf-8"))
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    if metadata.get("competition_sources") != ["name-that-llm"]:
        raise RuntimeError("not original competition source")
    if len(metadata.get("model_sources") or []) != 6:
        raise RuntimeError("pretrained model mounts mismatch")
    if len(metadata.get("dataset_sources") or []) != 21:
        raise RuntimeError("pretrained data mounts mismatch")
    if not metadata.get("enable_gpu") or metadata.get("enable_internet"):
        raise RuntimeError("source GPU or internet policy drift")
    if metadata.get("kernel_type") != "notebook":
        raise RuntimeError("not the prior 40-cell Notebook")
    changed = 0
    before_code = "\n".join(cell_text(c) for c in nb["cells"] if c.get("cell_type")=="code")
    for guard in GUARDS:
        if before_code.count(guard) != 1:
            raise RuntimeError("missing frozen no-supervised-training guard")
    if before_code.count(MARKER) != 1:
        raise RuntimeError("known preview shortcut missing or duplicated")
    if "if not preview:" not in before_code:
        raise RuntimeError("expected alternate full pipeline missing")
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        content = cell_text(cell)
        if MARKER in content:
            content = content.replace(MARKER, "preview = False  # diagnostic: force real inference path", 1)
            cell["source"] = content.splitlines(keepends=True)
            changed += 1
    if changed != 1:
        raise RuntimeError("non-unique shortcut replacement")
    after_code = "\n".join(cell_text(c) for c in nb["cells"] if c.get("cell_type")=="code")
    if after_code != before_code.replace(MARKER, "preview = False  # diagnostic: force real inference path", 1):
        raise RuntimeError("unexpected source modification")
    if "preview = emit_known_preview(DATA,None,OUTPUT)" in after_code:
        raise RuntimeError("source still uses preview shortcut")
    for guard in GUARDS:
        if after_code.count(guard) != 1:
            raise RuntimeError("guard lost")
    meta = dict(metadata)
    meta.pop("id_no",None)
    meta.update({
        "id":DEST,
        "title":TARGET_TITLE,
        "is_private":True,
        "enable_gpu":True,
        "enable_internet":False,
        "machine_shape":"NvidiaTeslaT4",
    })
    outfile = meta.get("code_file")
    if not isinstance(outfile,str) or Path(outfile).name != outfile or not outfile.endswith(".ipynb"):
        raise RuntimeError("unsafe code filename")
    nbfile = destdir / outfile
    data = json.dumps(nb,ensure_ascii=False).encode("utf-8")
    nbfile.write_bytes(data)
    (destdir/"kernel-metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return {
        "source":SOURCE,"destination":DEST,"changed_code_lines":1,
        "source_code_sha256":hashlib.sha256(before_code.encode()).hexdigest(),
        "diagnostic_code_sha256":hashlib.sha256(after_code.encode()).hexdigest(),
        "notebook_sha256":hashlib.sha256(data).hexdigest(),
        "pretrained_model_mounts":6,
        "training_guards_preserved":3,
        "private":True,"no_internet":True,
        "new_training":False,
        "competition_submission":False,
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    report=build(args.root)
    print("NTL_NATIVE_FULLPATH_SMOKE_SOURCE="+json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
