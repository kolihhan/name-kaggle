from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import time
from pathlib import Path
from typing import Any

CANDIDATE = "dct-temporal-fingerprint-v3"
SCOPE = "cpu_execution_feasibility_only_no_scoring"
MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
KEEP_LAYERS = 24
HIDDEN_SIZE = 4096
POOLED_LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]
MAX_LENGTH_BEFORE_EOS = 447

_LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.")


def is_required_weight_key(key: str) -> bool:
    if key in {"model.embed_tokens.weight", "embed_tokens.weight", "model.norm.weight", "norm.weight"}:
        return True
    match = _LAYER_RE.match(key)
    return bool(match and int(match.group(1)) < KEEP_LAYERS)


def select_required_shards(weight_map: dict[str, str]) -> list[str]:
    return sorted({shard for key, shard in weight_map.items() if is_required_weight_key(key)})


def validate_probe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    exact = {
        "candidate": CANDIDATE,
        "scope": SCOPE,
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "holdout_predictions_computed": False,
    }
    for key, expected in exact.items():
        observed = payload.get(key)
        if observed != expected:
            raise RuntimeError(f"CPU feasibility lineage drift for {key}: {observed!r} != {expected!r}")
    return {
        "candidate": CANDIDATE,
        "model_loaded": bool(payload.get("model_loaded", False)),
        "backward_step_completed": bool(payload.get("backward_step_completed", False)),
        "scientific_decision_authorized": False,
    }


def _peak_rss_gib() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _disk_summary(path: Path) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    gib = float(1024**3)
    return {
        "total_gib": usage.total / gib,
        "used_gib": usage.used / gib,
        "free_gib": usage.free / gib,
    }


def _download_truncated_snapshot(work_dir: Path) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    work_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename="model.safetensors.index.json",
            revision=MODEL_REVISION,
            local_dir=str(work_dir),
        )
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = dict(index["weight_map"])
    shards = select_required_shards(weight_map)
    if not shards:
        raise RuntimeError("no required model shards selected")

    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    sizes = {
        sibling.rfilename: int(sibling.size or 0)
        for sibling in info.siblings
        if getattr(sibling, "rfilename", None)
    }
    selected_bytes = sum(sizes.get(name, 0) for name in shards)
    disk_before = _disk_summary(work_dir)

    # Keep a safety margin for Python wheels, extracted temporary files and model objects.
    if selected_bytes and selected_bytes > int(disk_before["free_gib"] * (1024**3) * 0.82):
        raise RuntimeError(
            f"selected model shards need {selected_bytes / (1024**3):.2f} GiB, "
            f"only {disk_before['free_gib']:.2f} GiB free before download"
        )

    allow_patterns = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "*.tiktoken",
        *shards,
    ]
    root = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(work_dir),
            allow_patterns=allow_patterns,
        )
    )

    filtered = {
        "metadata": dict(index.get("metadata", {})),
        "weight_map": {key: shard for key, shard in weight_map.items() if is_required_weight_key(key)},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(filtered, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root, {
        "selected_shards": shards,
        "selected_shard_count": len(shards),
        "selected_shard_bytes": int(selected_bytes),
        "selected_shard_gib": selected_bytes / float(1024**3),
        "disk_before": disk_before,
        "disk_after": _disk_summary(work_dir),
    }


def run_probe(work_dir: Path) -> dict[str, Any]:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from scripts.dct_temporal_pool import masked_dct_pool
    from scripts.run_two_stage_lora24_gate_a import gate_a_training_spec

    started = time.time()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "candidate": CANDIDATE,
        "scope": SCOPE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "holdout_predictions_computed": False,
        "model_loaded": False,
        "backward_step_completed": False,
        "torch": str(torch.__version__),
        "cpu_count": int(os.cpu_count() or 0),
        "initial_peak_rss_gib": _peak_rss_gib(),
        "initial_disk": _disk_summary(work_dir),
    }

    snapshot, download = _download_truncated_snapshot(work_dir / "model")
    payload["download"] = download

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    if config.model_type != "qwen3" or int(config.hidden_size) != HIDDEN_SIZE:
        raise RuntimeError("unexpected Qwen3 config in CPU feasibility probe")
    config.num_hidden_layers = KEEP_LAYERS
    config.use_cache = False
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = list(config.layer_types[:KEEP_LAYERS])

    load_started = time.time()
    base = AutoModel.from_pretrained(
        str(snapshot),
        config=config,
        torch_dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    payload["model_load_seconds"] = time.time() - load_started
    payload["peak_rss_after_model_load_gib"] = _peak_rss_gib()
    payload["model_loaded"] = True

    spec = gate_a_training_spec()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    lora_spec = dict(spec["lora"])
    model = get_peft_model(
        base,
        LoraConfig(
            r=int(lora_spec["r"]),
            lora_alpha=int(lora_spec["alpha"]),
            lora_dropout=float(lora_spec["dropout"]),
            bias=str(lora_spec["bias"]),
            target_modules=list(lora_spec["target_modules"]),
        ),
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    layers = [int(value) for value in spec["pooled_layers"]]
    norms = nn.ModuleList([nn.LayerNorm(HIDDEN_SIZE) for _ in layers]).float()
    representation_dim = HIDDEN_SIZE * len(layers) * 2
    plain = nn.Linear(representation_dim, 18).float()
    centered = nn.Linear(representation_dim, 18).float()

    adapter_parameters = [p for p in model.parameters() if p.requires_grad]
    head_parameters = list(norms.parameters()) + list(plain.parameters()) + list(centered.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": float(spec["adapter_lr"])},
            {"params": head_parameters, "lr": float(spec["head_lr"])},
        ],
        weight_decay=float(spec["weight_decay"]),
    )

    # Realistic worst-case triplet shape from the frozen recipe: 3 rows, 447 tokens + EOS.
    eos = int(tokenizer.eos_token_id)
    vocab_size = int(config.vocab_size)
    generator = torch.Generator().manual_seed(int(spec["seed"]))
    body = torch.randint(
        low=10,
        high=max(11, vocab_size - 1),
        size=(3, MAX_LENGTH_BEFORE_EOS),
        generator=generator,
        dtype=torch.long,
    )
    input_ids = torch.cat([body, torch.full((3, 1), eos, dtype=torch.long)], dim=1)
    attention_mask = torch.ones_like(input_ids)
    target = torch.tensor([0, 1, 2], dtype=torch.long)

    def center_triplet(z: torch.Tensor) -> torch.Tensor:
        return z - z.mean(dim=0, keepdim=True)

    step_started = time.time()
    model.train()
    hidden_states = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    ).hidden_states
    pooled = []
    for layer, norm in zip(layers, norms):
        mode_features = masked_dct_pool(
            hidden_states[layer].float(),
            attention_mask,
            mask_first_token=True,
            n_modes=2,
        )
        pooled.extend([norm(mode_features[:, 0]), norm(mode_features[:, 1])])
    z = torch.cat(pooled, dim=-1)
    loss = F.cross_entropy(plain(z), target) + F.cross_entropy(centered(center_triplet(z)), target)
    loss.backward()
    optimizer.step()
    payload["single_triplet_forward_backward_step_seconds"] = time.time() - step_started
    payload["single_triplet_loss"] = float(loss.detach())
    payload["peak_rss_after_backward_gib"] = _peak_rss_gib()
    payload["backward_step_completed"] = True
    payload["total_probe_seconds"] = time.time() - started
    payload["final_disk"] = _disk_summary(work_dir)

    validate_probe_payload(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_probe(args.work_dir)
    except Exception as exc:
        payload = {
            "candidate": CANDIDATE,
            "scope": SCOPE,
            "official_rows_scored": 0,
            "competition_test_rows_used": 0,
            "public_lb_tuning": False,
            "holdout_predictions_computed": False,
            "model_loaded": False,
            "backward_step_completed": False,
            "probe_error_type": type(exc).__name__,
            "probe_error": str(exc),
            "peak_rss_gib": _peak_rss_gib(),
            "disk": _disk_summary(args.work_dir),
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
