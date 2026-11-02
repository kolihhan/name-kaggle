from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
BASE_DATASET_REF = "kolihhan/name-that-llm-qwen3-8b-base-49e3418f"
RUNTIME_DATASET_REF = "kolihhan/name-that-llm-two-stage-lora24-v1-offline-runtime"
WHEEL_SPECS = (
    "transformers==4.57.1",
    "peft==0.17.1",
    # Gate B did not freeze a safetensors version. Materialization resolves it once;
    # the resulting wheel bytes are then frozen by the runtime manifest.
    "safetensors",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(root: Path) -> tuple[list[dict[str, object]], str]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files: list[dict[str, object]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "size": int(path.stat().st_size),
            }
        )
    if not files:
        raise RuntimeError(f"cannot freeze empty package root: {root}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return files, hashlib.sha256(encoded).hexdigest()


def build_base_lineage(
    *, files: list[dict[str, object]], manifest_sha256: str
) -> dict[str, object]:
    return {
        "purpose": "two_stage_lora24_v1_offline_base_snapshot",
        "dataset_ref": BASE_DATASET_REF,
        "dataset_version_expected": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "files": files,
        "file_manifest_sha256": manifest_sha256,
        "competition_train_rows_included": 0,
        "competition_test_rows_included": 0,
    }


def build_runtime_lineage(
    *, files: list[dict[str, object]], manifest_sha256: str
) -> dict[str, object]:
    return {
        "purpose": "two_stage_lora24_v1_offline_wheelhouse",
        "dataset_ref": RUNTIME_DATASET_REF,
        "dataset_version_expected": 1,
        "wheel_specs": list(WHEEL_SPECS),
        "files": files,
        "file_manifest_sha256": manifest_sha256,
        "competition_train_rows_included": 0,
        "competition_test_rows_included": 0,
    }


def assert_dataset_ref_absent(listing_csv: str, dataset_ref: str) -> None:
    refs = {
        line.split(",", 1)[0].strip().strip('"')
        for line in listing_csv.splitlines()[1:]
        if line.strip()
    }
    if dataset_ref in refs:
        raise RuntimeError(
            f"dataset already exists; refusing overwrite/version: {dataset_ref}"
        )


def validate_flat_package(package_dir: Path) -> None:
    package_dir = Path(package_dir)
    nested = [
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() and path.parent != package_dir
    ]
    if nested:
        raise RuntimeError(f"nested payload files are not allowed for Kaggle create: {nested[:5]}")


def _dataset_metadata(dataset_ref: str, title: str) -> dict[str, object]:
    return {
        "title": title,
        "id": dataset_ref,
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }


def prepare_base_package(package_dir: Path) -> dict[str, object]:
    from huggingface_hub import snapshot_download

    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    if any(package_dir.iterdir()):
        raise RuntimeError(f"base package directory must be empty: {package_dir}")
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(package_dir),
        allow_patterns=["config.json", "model.safetensors", "model.safetensors.index.json", "model-*.safetensors"],
    )
    shutil.rmtree(package_dir / ".cache", ignore_errors=True)
    validate_flat_package(package_dir)
    files, manifest_sha256 = build_file_manifest(package_dir)
    lineage = build_base_lineage(files=files, manifest_sha256=manifest_sha256)
    (package_dir / "base_lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True), encoding="utf-8"
    )
    (package_dir / "dataset-metadata.json").write_text(
        json.dumps(
            _dataset_metadata(BASE_DATASET_REF, "Name That LLM Qwen3 8B Base frozen revision"),
            indent=2,
        ),
        encoding="utf-8",
    )
    return lineage


def prepare_runtime_package(package_dir: Path) -> dict[str, object]:
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    if any(package_dir.iterdir()):
        raise RuntimeError(f"runtime package directory must be empty: {package_dir}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--dest",
            str(package_dir),
            "--no-deps",
            "--only-binary=:all:",
            "--platform",
            "manylinux2014_x86_64",
            "--python-version",
            "311",
            "--implementation",
            "cp",
            *WHEEL_SPECS,
        ]
    )
    validate_flat_package(package_dir)
    files, manifest_sha256 = build_file_manifest(package_dir)
    lineage = build_runtime_lineage(files=files, manifest_sha256=manifest_sha256)
    (package_dir / "runtime_lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True), encoding="utf-8"
    )
    (package_dir / "dataset-metadata.json").write_text(
        json.dumps(
            _dataset_metadata(RUNTIME_DATASET_REF, "Name That LLM LoRA24 v1 offline runtime"),
            indent=2,
        ),
        encoding="utf-8",
    )
    return lineage


def create_private_dataset(package_dir: Path, dataset_ref: str) -> None:
    validate_flat_package(package_dir)
    slug = dataset_ref.split("/", 1)[1]
    listing = subprocess.run(
        ["kaggle", "datasets", "list", "-s", slug, "--csv"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert_dataset_ref_absent(listing.stdout, dataset_ref)
    subprocess.check_call(["kaggle", "datasets", "create", "-p", str(package_dir)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare immutable offline inputs for two-stage LoRA24 v1 submission preflight"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("base", "runtime"):
        child = sub.add_parser(name)
        child.add_argument("--package-dir", type=Path, required=True)
        child.add_argument(
            "--create-kaggle-dataset",
            action="store_true",
            help="Create the fixed private dataset after local package verification.",
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "base":
        lineage = prepare_base_package(args.package_dir)
        ref = BASE_DATASET_REF
    else:
        lineage = prepare_runtime_package(args.package_dir)
        ref = RUNTIME_DATASET_REF
    print(json.dumps(lineage, indent=2, sort_keys=True))
    if args.create_kaggle_dataset:
        create_private_dataset(args.package_dir, ref)


if __name__ == "__main__":
    main()
