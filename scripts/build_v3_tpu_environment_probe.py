from __future__ import annotations

import argparse
import json
from pathlib import Path

KERNEL_REF = "kolihhan/ntl-v3-tpu-env-probe"
KERNEL_TITLE = "NTL V3 TPU Environment Probe"
CODE_FILE = "kernel_main.py"
MACHINE_SHAPE = "TpuV5E8"


def _kernel_source() -> str:
    return '''from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

OUTPUT = Path("/kaggle/working/environment_probe.json")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
started = time.time()
payload = {
    "candidate": "dct-temporal-fingerprint-v3",
    "scope": "tpu_environment_identity_no_model_no_scoring",
    "gate0_pass": False,
    "official_rows_scored": 0,
    "firewall_rows_scored": 0,
    "competition_test_rows_used": 0,
    "public_lb_tuning": False,
    "submission_authorized": False,
}

try:
    import torch
    import torch_xla
    import torch_xla.runtime as runtime

    device = torch_xla.device()
    device_type = str(runtime.device_type())
    real_devices = [str(value) for value in torch_xla.real_devices()]

    x = torch.arange(8, dtype=torch.float32, device=device)
    y = (x * x).sum()
    tensor_result = float(y.cpu().item())

    payload.update({
        "torch_version": str(torch.__version__),
        "torch_xla_version": str(getattr(torch_xla, "__version__", "unknown")),
        "device": str(device),
        "device_type": device_type,
        "real_devices": real_devices,
        "tensor_result": tensor_result,
        "gate0_pass": bool(device_type == 'TPU'),
        "elapsed_seconds": time.time() - started,
    })
except Exception as exc:
    payload.update({
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "elapsed_seconds": time.time() - started,
    })

OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload["gate0_pass"]:
    raise SystemExit("Gate 0 FAIL: Kaggle kernel is not backed by an actual TPU XLA device")
'''


def write_kernel_bundle(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / CODE_FILE).write_text(_kernel_source(), encoding="utf-8")
    metadata = {
        "id": KERNEL_REF,
        "title": KERNEL_TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": False,
        "machine_shape": MACHINE_SHAPE,
        "competition_sources": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_kernel_bundle(args.output_dir)
    print(f"built TPU environment probe: {KERNEL_REF}")


if __name__ == "__main__":
    main()
