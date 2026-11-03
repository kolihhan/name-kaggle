from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
EXPECTED_HOLDOUT_SOURCES = ["alperen", "hailey", "willrice"]
EXPECTED_TRAIN_ROWS = 14502
EXPECTED_HELDOUT_ROWS = 14506
EXPECTED_ELIGIBLE_LABELS = 17
REPRESENTATIONS = {"mean", "dct"}


def scientific_scope() -> dict[str, object]:
    return {
        "scope": "cpu_shadow_external_transfer_training_only",
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "kaggle_submission_authorized": False,
        "frozen_cuda_v3_gate_authorized": False,
    }


def initial_progress() -> dict[str, int | float]:
    return {
        "group_index": 0,
        "microbatch_index": 0,
        "optimizer_steps": 0,
        "loss_sum": 0.0,
        "loss_count": 0,
    }


def validate_progress(
    progress: dict[str, Any], *, group_count: int, current_group_size: int | None
) -> None:
    group_index = int(progress.get("group_index", -1))
    microbatch_index = int(progress.get("microbatch_index", -1))
    optimizer_steps = int(progress.get("optimizer_steps", -1))
    if group_index < 0 or group_index > group_count:
        raise RuntimeError(f"CPU shadow group_index out of range: {group_index}")
    if optimizer_steps != group_index:
        raise RuntimeError(
            f"CPU shadow optimizer/group cursor mismatch: {optimizer_steps} != {group_index}"
        )
    if group_index == group_count:
        if microbatch_index != 0:
            raise RuntimeError("completed CPU shadow cursor must have microbatch_index=0")
        return
    if current_group_size is None or current_group_size <= 0:
        raise RuntimeError("current_group_size required for incomplete CPU shadow progress")
    if microbatch_index < 0 or microbatch_index >= current_group_size:
        raise RuntimeError(
            f"CPU shadow microbatch_index out of range: {microbatch_index} / {current_group_size}"
        )


def build_lineage_fingerprint(
    *,
    representation: str,
    model_revision: str,
    training_spec: dict[str, Any],
    holdout_sources: Sequence[str],
    labels: Sequence[str],
    triplets: Sequence[Sequence[int]],
) -> str:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unsupported CPU shadow representation: {representation}")
    payload = {
        "candidate": "dct-v3-cpu-shadow",
        "representation": representation,
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "training_spec": training_spec,
        "holdout_sources": list(holdout_sources),
        "labels": list(labels),
        "triplets": [[int(v) for v in rows] for rows in triplets],
        "backend": "cpu_float16_no_autocast_no_gradscaler",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed_all(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _representation(
    *,
    representation: str,
    model: Any,
    norms: Any,
    layers: list[int],
    encoding: dict[str, Any],
    torch: Any,
):
    from scripts.dct_temporal_pool import masked_dct_pool
    from scripts.run_two_stage_lora24_gate_a import masked_mean_pool

    hidden_states = model(
        **encoding,
        output_hidden_states=True,
        use_cache=False,
    ).hidden_states
    if representation == "mean":
        pooled = [
            norm(
                masked_mean_pool(
                    hidden_states[layer].float(),
                    encoding["attention_mask"],
                    mask_first_token=True,
                )
            )
            for layer, norm in zip(layers, norms)
        ]
    elif representation == "dct":
        pooled = []
        for layer, norm in zip(layers, norms):
            mode_features = masked_dct_pool(
                hidden_states[layer].float(),
                encoding["attention_mask"],
                mask_first_token=True,
                n_modes=2,
            )
            pooled.extend([norm(mode_features[:, 0]), norm(mode_features[:, 1])])
    else:  # pragma: no cover
        raise ValueError(representation)
    return torch.cat(pooled, dim=-1)


def run_cpu_shadow_chunk(
    *,
    representation: str,
    safe_path: Path,
    snapshot_path: Path,
    output_checkpoint: Path,
    result_json: Path,
    resume_checkpoint: Path | None = None,
    max_microbatches: int = 1,
) -> dict[str, Any]:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unsupported CPU shadow representation: {representation}")
    if max_microbatches <= 0:
        raise ValueError("max_microbatches must be positive")

    import pandas as pd
    import torch
    from torch import nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from scripts.cpu_shadow_resume import load_shadow_checkpoint, save_shadow_checkpoint
    from scripts.dct_v3_transfer_split import build_transfer_split
    from scripts.run_two_stage_lora24_gate_a import (
        build_prompt_triplets,
        center_triplet,
        compute_class_weights,
        gate_a_training_spec,
        group_triplets_for_accumulation,
        select_eligible_labels,
    )

    started = time.time()
    spec = gate_a_training_spec()
    seed = int(spec["seed"])
    frame = pd.read_parquet(safe_path).reset_index(drop=True)
    split = build_transfer_split(frame)
    train_idx = np.asarray(split["train_idx"], dtype=int)
    heldout_idx = np.asarray(split["heldout_idx"], dtype=int)
    train_frame = frame.iloc[train_idx]
    heldout_frame = frame.iloc[heldout_idx]
    eligible_labels = select_eligible_labels(train_frame, heldout_frame)
    labels = sorted(frame["model"].astype(str).unique().tolist())

    checks = {
        "holdout_sources": EXPECTED_HOLDOUT_SOURCES,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "heldout_rows": EXPECTED_HELDOUT_ROWS,
        "eligible_label_count": EXPECTED_ELIGIBLE_LABELS,
        "prompt_id_overlap": 0,
        "prompt_key_overlap": 0,
    }
    for key, expected in checks.items():
        observed = split.get(key)
        if key == "eligible_label_count":
            observed = len(eligible_labels)
        if observed != expected:
            raise RuntimeError(
                f"CPU shadow frozen split drift for {key}: {observed!r} != {expected!r}"
            )
    if len(labels) != 18:
        raise RuntimeError(f"CPU shadow expected 18 labels, got {len(labels)}")

    class_weights_map = compute_class_weights(
        train_frame["model"].astype(str).to_numpy(), classes=labels
    )
    class_weights = torch.tensor(
        [class_weights_map[label] for label in labels], dtype=torch.float32
    )
    label_to_idx = {label: index for index, label in enumerate(labels)}
    y = frame["model"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64)

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise RuntimeError("CPU shadow tokenizer has no EOS token")
    token_lists = tokenizer(
        frame["text"].fillna("").astype(str).tolist(),
        add_special_tokens=False,
        truncation=True,
        max_length=int(spec["max_length_before_eos"]),
    )["input_ids"]
    tokens = [ids + [int(tokenizer.eos_token_id)] for ids in token_lists]

    config = AutoConfig.from_pretrained(str(snapshot_path), local_files_only=True)
    if config.model_type != "qwen3" or int(config.hidden_size) != 4096:
        raise RuntimeError("unexpected Qwen3 config for CPU shadow")
    config.num_hidden_layers = int(spec["keep_layers"])
    config.use_cache = False
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = list(config.layer_types[: int(spec["keep_layers"])])

    base = AutoModel.from_pretrained(
        str(snapshot_path),
        config=config,
        torch_dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.enable_input_require_grads()

    _seed_all(seed, torch)
    layers = [int(value) for value in spec["pooled_layers"]]
    hidden_size = int(config.hidden_size)
    representation_dim = hidden_size * len(layers) * (2 if representation == "dct" else 1)
    norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers]).float()
    plain = nn.Linear(representation_dim, len(labels)).float()
    centered = nn.Linear(representation_dim, len(labels)).float()
    dropout = nn.Dropout(float(spec["head_dropout"]))

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

    adapter_parameters = [p for p in model.parameters() if p.requires_grad]
    head_parameters = (
        list(norms.parameters()) + list(plain.parameters()) + list(centered.parameters())
    )
    if not adapter_parameters:
        raise RuntimeError("CPU shadow has no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": float(spec["adapter_lr"])},
            {"params": head_parameters, "lr": float(spec["head_lr"])},
        ],
        weight_decay=float(spec["weight_decay"]),
    )

    triplets = build_prompt_triplets(frame, train_idx, seed=seed)
    accumulation = int(spec["gradient_accumulation_triplets"])
    step_groups = group_triplets_for_accumulation(triplets, accumulation=accumulation)
    total_steps = len(step_groups)
    warmup_steps = max(1, int(float(spec["warmup_fraction"]) * total_steps))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress_value = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress_value))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    lineage = build_lineage_fingerprint(
        representation=representation,
        model_revision=MODEL_REVISION,
        training_spec=spec,
        holdout_sources=list(split["holdout_sources"]),
        labels=labels,
        triplets=triplets,
    )
    modules = {
        "model": model,
        "norms": norms,
        "plain": plain,
        "centered": centered,
    }

    if resume_checkpoint is not None:
        progress = load_shadow_checkpoint(
            resume_checkpoint,
            expected_lineage_fingerprint=lineage,
            modules=modules,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    else:
        progress = initial_progress()
        _seed_all(seed, torch)

    group_index = int(progress["group_index"])
    current_group_size = len(step_groups[group_index]) if group_index < total_steps else None
    validate_progress(
        progress,
        group_count=total_steps,
        current_group_size=current_group_size,
    )

    model.train()
    norms.train()
    plain.train()
    centered.train()
    dropout.train()

    def make_encoding(rows: np.ndarray) -> dict[str, Any]:
        return tokenizer.pad(
            {"input_ids": [tokens[int(row)] for row in rows]},
            return_tensors="pt",
        )

    processed = 0
    while int(progress["group_index"]) < total_steps and processed < max_microbatches:
        group_index = int(progress["group_index"])
        rows_group = step_groups[group_index]
        microbatch_index = int(progress["microbatch_index"])
        validate_progress(
            progress,
            group_count=total_steps,
            current_group_size=len(rows_group),
        )
        if microbatch_index == 0:
            optimizer.zero_grad(set_to_none=True)

        rows = rows_group[microbatch_index]
        encoding = make_encoding(rows)
        z = _representation(
            representation=representation,
            model=model,
            norms=norms,
            layers=layers,
            encoding=encoding,
            torch=torch,
        )
        target = torch.as_tensor(y[rows], dtype=torch.long)
        plain_logits = plain(dropout(z))
        centered_logits = centered(dropout(center_triplet(z)))
        loss = F.cross_entropy(
            plain_logits, target, weight=class_weights
        ) + F.cross_entropy(
            centered_logits, target, weight=class_weights
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite CPU shadow loss: {float(loss.detach())}")
        (loss / len(rows_group)).backward()
        progress["loss_sum"] = float(progress["loss_sum"]) + float(loss.detach())
        progress["loss_count"] = int(progress["loss_count"]) + 1
        progress["microbatch_index"] = microbatch_index + 1
        processed += 1

        if int(progress["microbatch_index"]) == len(rows_group):
            torch.nn.utils.clip_grad_norm_(
                adapter_parameters + head_parameters,
                float(spec["gradient_clip"]),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress["group_index"] = group_index + 1
            progress["microbatch_index"] = 0
            progress["optimizer_steps"] = int(progress["optimizer_steps"]) + 1

    save_shadow_checkpoint(
        output_checkpoint,
        lineage_fingerprint=lineage,
        modules=modules,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
    )
    complete = int(progress["group_index"]) == total_steps
    result = scientific_scope()
    result.update(
        {
            "representation": representation,
            "backend": "cpu_float16_no_autocast_no_gradscaler",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "holdout_sources": list(split["holdout_sources"]),
            "train_rows": int(len(train_idx)),
            "triplet_microbatches": int(len(triplets)),
            "optimizer_steps_total": int(total_steps),
            "lineage_fingerprint": lineage,
            "progress": dict(progress),
            "microbatches_processed_this_chunk": int(processed),
            "training_complete": bool(complete),
            "mean_training_loss_so_far": float(progress["loss_sum"])
            / max(1, int(progress["loss_count"])),
            "seconds": float(time.time() - started),
            "checkpoint": str(output_checkpoint),
        }
    )
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one resumable free-CPU shadow chunk for v1-mean or v3-DCT Stage-1"
    )
    parser.add_argument(
        "--representation", choices=sorted(REPRESENTATIONS), required=True
    )
    parser.add_argument("--safe-path", type=Path, required=True)
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--max-microbatches", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cpu_shadow_chunk(
        representation=args.representation,
        safe_path=args.safe_path,
        snapshot_path=args.snapshot_path,
        output_checkpoint=args.output_checkpoint,
        result_json=args.result_json,
        resume_checkpoint=args.resume_checkpoint,
        max_microbatches=args.max_microbatches,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
