from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_two_stage_lora24_gate_a import (
    MIN_ELIGIBLE_HELDOUT_ROWS,
    MIN_ELIGIBLE_LABELS,
    build_prompt_triplets,
    build_stage1_split,
    center_triplet,
    compute_class_weights,
    gate_a_condition_specs,
    gate_a_decision,
    gate_a_training_spec,
    group_triplets_for_accumulation,
    masked_mean_pool,
    score_predictions,
    select_eligible_labels,
    truncate_qwen_layers,
)
from scripts.run_phase14_public_sparse import (
    EXPECTED_EXTERNAL_ROWS,
    EXPECTED_RAW_ROWS,
    EXPECTED_RAW_SHA256,
    EXPECTED_SAFE_PROMPT_IDS,
    build_conservative_subset,
)


MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
TRANSFORMERS_VERSION = "4.57.1"
PEFT_VERSION = "0.17.1"
CONDITIONS = ("A0", "A1")
CANDIDATE = "two-stage-lora24-v1"
RESULT_FILENAME = "two_stage_lora24_gate_a.json"
CHECKPOINT_DIRNAME = "two_stage_lora24_gate_a_checkpoint"


def execution_scope() -> dict[str, object]:
    return {
        "scope": "external_transfer_gate_a_only_no_official_oof",
        "official_rows_scored": 0,
        "official_oof_runs": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "kaggle_submission_authorized": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _find_pool_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    candidates = sorted(Path("/kaggle/input").glob("**/pool_v4.parquet"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one pool_v4.parquet under /kaggle/input, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _prepare_stage1(pool_path: Path, safe_path: Path) -> dict[str, Any]:
    raw_sha = _sha256_file(pool_path)
    if raw_sha != EXPECTED_RAW_SHA256:
        raise RuntimeError(f"raw pool SHA mismatch: {raw_sha} != {EXPECTED_RAW_SHA256}")

    pool = pd.read_parquet(pool_path)
    if len(pool) != EXPECTED_RAW_ROWS:
        raise RuntimeError(f"raw pool row mismatch: {len(pool)} != {EXPECTED_RAW_ROWS}")

    safe = build_conservative_subset(pool, validate_full_contract=True).reset_index(drop=True)
    if len(safe) != EXPECTED_EXTERNAL_ROWS:
        raise RuntimeError(f"Phase14 row mismatch: {len(safe)} != {EXPECTED_EXTERNAL_ROWS}")
    if int(safe["prompt_id"].nunique()) != EXPECTED_SAFE_PROMPT_IDS:
        raise RuntimeError(
            f"Phase14 prompt mismatch: {safe['prompt_id'].nunique()} != {EXPECTED_SAFE_PROMPT_IDS}"
        )
    labels = sorted(safe["model"].astype(str).unique().tolist())
    if len(labels) != 18:
        raise RuntimeError(f"expected 18 labels in Phase14 subset, found {len(labels)}")

    split = build_stage1_split(safe)
    train_idx = np.asarray(split["train_idx"], dtype=int)
    heldout_idx = np.asarray(split["heldout_idx"], dtype=int)
    train = safe.iloc[train_idx]
    heldout = safe.iloc[heldout_idx]
    eligible_labels = select_eligible_labels(train, heldout)
    eligible_mask = heldout["model"].astype(str).isin(eligible_labels)
    eligible_heldout_rows = int(eligible_mask.sum())

    if split["prompt_id_overlap"] != 0 or split["prompt_key_overlap"] != 0:
        raise RuntimeError(f"prompt overlap survived source holdout: {split}")

    source_counts = {
        source: int((heldout["source"].astype(str) == source).sum())
        for source in split["holdout_sources"]
    }
    source_eligible_counts = {
        source: int(
            (
                (heldout["source"].astype(str) == source)
                & heldout["model"].astype(str).isin(eligible_labels)
            ).sum()
        )
        for source in split["holdout_sources"]
    }

    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe.to_parquet(safe_path, index=False)

    return {
        "raw_pool_path": str(pool_path),
        "raw_pool_sha256": raw_sha,
        "raw_rows": int(len(pool)),
        "phase14_rows": int(len(safe)),
        "phase14_prompt_ids": int(safe["prompt_id"].nunique()),
        "phase14_labels": labels,
        "holdout_sources": list(split["holdout_sources"]),
        "train_rows": int(len(train_idx)),
        "heldout_rows": int(len(heldout_idx)),
        "eligible_labels": eligible_labels,
        "eligible_label_count": int(len(eligible_labels)),
        "eligible_heldout_rows": eligible_heldout_rows,
        "source_rows": source_counts,
        "source_eligible_rows": source_eligible_counts,
        "prompt_id_overlap": int(split["prompt_id_overlap"]),
        "prompt_key_overlap": int(split["prompt_key_overlap"]),
    }


def _seed_all(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_environment(gpu_index: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    package_root = str(Path(__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = package_root if not existing else package_root + os.pathsep + existing
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _condition_command(
    *,
    condition: str,
    safe_path: Path,
    snapshot_path: Path,
    worker_output: Path,
    checkpoint_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.run_two_stage_lora24_gate_a_gpu",
        "--worker-condition",
        condition,
        "--safe-path",
        str(safe_path),
        "--snapshot-path",
        str(snapshot_path),
        "--worker-output",
        str(worker_output),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]


def _run_workers(
    *,
    safe_path: Path,
    snapshot_path: Path,
    runtime_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Gate A requires CUDA")
    gpu_count = int(torch.cuda.device_count())
    if gpu_count < 1:
        raise RuntimeError("Gate A found no CUDA devices")

    worker_outputs = {condition: runtime_dir / f"{condition}.json" for condition in CONDITIONS}
    logs = {condition: runtime_dir / f"{condition}.log" for condition in CONDITIONS}

    def launch(condition: str, gpu_index: int) -> subprocess.Popen[Any]:
        handle = logs[condition].open("w", encoding="utf-8")
        proc = subprocess.Popen(
            _condition_command(
                condition=condition,
                safe_path=safe_path,
                snapshot_path=snapshot_path,
                worker_output=worker_outputs[condition],
                checkpoint_dir=checkpoint_dir,
            ),
            env=_worker_environment(gpu_index),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        proc._gate_a_log_handle = handle  # type: ignore[attr-defined]
        return proc

    processes: dict[str, subprocess.Popen[Any]] = {}
    if gpu_count >= 2:
        processes["A0"] = launch("A0", 0)
        processes["A1"] = launch("A1", 1)
        failed = False
        for condition in CONDITIONS:
            rc = processes[condition].wait()
            getattr(processes[condition], "_gate_a_log_handle").close()
            print(f"===== {condition} worker rc={rc} =====", flush=True)
            print(logs[condition].read_text(encoding="utf-8", errors="replace")[-12000:], flush=True)
            if rc != 0:
                failed = True
        if failed:
            raise RuntimeError("one or more Gate A condition workers failed")
    else:
        for condition in CONDITIONS:
            proc = launch(condition, 0)
            rc = proc.wait()
            getattr(proc, "_gate_a_log_handle").close()
            print(f"===== {condition} worker rc={rc} =====", flush=True)
            print(logs[condition].read_text(encoding="utf-8", errors="replace")[-12000:], flush=True)
            if rc != 0:
                raise RuntimeError(f"Gate A condition worker failed: {condition}")

    return {
        condition: json.loads(worker_outputs[condition].read_text(encoding="utf-8"))
        for condition in CONDITIONS
    }


def _score_condition(
    frame: pd.DataFrame,
    heldout_idx: np.ndarray,
    predictions: list[str],
    eligible_labels: list[str],
    holdout_sources: list[str],
) -> dict[str, Any]:
    eval_frame = frame.iloc[heldout_idx][["source", "model"]].copy().reset_index(drop=True)
    if len(eval_frame) != len(predictions):
        raise RuntimeError("prediction count does not match heldout rows")
    eval_frame["prediction"] = predictions

    pooled_frame = eval_frame[eval_frame["model"].astype(str).isin(eligible_labels)]
    pooled = score_predictions(
        pooled_frame["model"].astype(str).tolist(),
        pooled_frame["prediction"].astype(str).tolist(),
        labels=eligible_labels,
    )

    sources: dict[str, Any] = {}
    for source in holdout_sources:
        source_frame = pooled_frame[pooled_frame["source"].astype(str) == source]
        source_labels = [
            label
            for label in eligible_labels
            if bool((source_frame["model"].astype(str) == label).any())
        ]
        if not source_labels:
            raise RuntimeError(f"heldout source has zero eligible evaluation labels: {source}")
        sources[source] = score_predictions(
            source_frame["model"].astype(str).tolist(),
            source_frame["prediction"].astype(str).tolist(),
            labels=source_labels,
        )
        sources[source]["eligible_labels"] = source_labels
        sources[source]["rows"] = int(len(source_frame))

    return {"pooled": pooled, "sources": sources}


def _run_condition_worker(
    *,
    condition: str,
    safe_path: Path,
    snapshot_path: Path,
    worker_output: Path,
    checkpoint_dir: Path,
) -> None:
    if condition not in CONDITIONS:
        raise ValueError(condition)

    # Kaggle images historically shipped torchao variants that PEFT probes even
    # though this recipe does not use torchao. Match the readable public LoRA24
    # training kernel by hiding that optional package from PEFT.
    import importlib.util as importlib_util

    original_find_spec = importlib_util.find_spec
    importlib_util.find_spec = (
        lambda name, *args, **kwargs: None
        if name.split(".")[0] == "torchao"
        else original_find_spec(name, *args, **kwargs)
    )

    import torch
    from torch import nn
    import torch.nn.functional as F
    import peft
    from peft import LoraConfig, get_peft_model
    import peft.import_utils as peft_import_utils
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    peft_import_utils.is_torchao_available = lambda *args, **kwargs: False

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers version drift: {transformers.__version__} != {TRANSFORMERS_VERSION}"
        )
    if peft.__version__ != PEFT_VERSION:
        raise RuntimeError(f"peft version drift: {peft.__version__} != {PEFT_VERSION}")
    if not torch.cuda.is_available():
        raise RuntimeError("condition worker requires CUDA")

    started = time.time()
    spec = gate_a_training_spec()
    condition_spec = gate_a_condition_specs()[condition]
    train_lora = bool(condition_spec["train_lora"])
    seed = int(spec["seed"])
    device = torch.device("cuda:0")

    frame = pd.read_parquet(safe_path).reset_index(drop=True)
    split = build_stage1_split(frame)
    train_idx = np.asarray(split["train_idx"], dtype=int)
    heldout_idx = np.asarray(split["heldout_idx"], dtype=int)
    train_frame = frame.iloc[train_idx]
    heldout_frame = frame.iloc[heldout_idx]
    eligible_labels = select_eligible_labels(train_frame, heldout_frame)
    labels = sorted(frame["model"].astype(str).unique().tolist())
    if len(labels) != 18:
        raise RuntimeError(f"condition worker expected 18 labels, got {len(labels)}")

    class_weights_map = compute_class_weights(
        train_frame["model"].astype(str).to_numpy(), classes=labels
    )
    class_weights = torch.tensor(
        [class_weights_map[label] for label in labels], dtype=torch.float32, device=device
    )
    label_to_idx = {label: index for index, label in enumerate(labels)}
    y = frame["model"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64)

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Qwen tokenizer has no EOS token")
    token_lists = tokenizer(
        frame["text"].fillna("").astype(str).tolist(),
        add_special_tokens=False,
        truncation=True,
        max_length=int(spec["max_length_before_eos"]),
    )["input_ids"]
    tokens = [ids + [int(tokenizer.eos_token_id)] for ids in token_lists]

    config = AutoConfig.from_pretrained(str(snapshot_path), local_files_only=True)
    if config.model_type != "qwen3" or int(config.hidden_size) != 4096:
        raise RuntimeError(
            f"unexpected model config: model_type={config.model_type} hidden_size={config.hidden_size}"
        )
    if int(config.num_hidden_layers) < int(spec["keep_layers"]):
        raise RuntimeError("base model has fewer layers than frozen keep_layers")
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
    ).to(device)
    truncate_qwen_layers(base, keep_layers=int(spec["keep_layers"]))
    if not hasattr(base, "norm"):
        raise RuntimeError("Qwen3 AutoModel is missing final norm")

    if train_lora:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base.enable_input_require_grads()
    else:
        for parameter in base.parameters():
            parameter.requires_grad = False
        base.eval()

    # Reset immediately before readout creation so A0/A1 norms and heads have
    # exactly the same initialization. LoRA initialization happens afterwards.
    _seed_all(seed, torch)
    layers = [int(value) for value in spec["pooled_layers"]]
    hidden_size = int(config.hidden_size)
    norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers]).to(device).float()
    plain = nn.Linear(hidden_size * len(layers), len(labels)).to(device).float()
    centered = nn.Linear(hidden_size * len(layers), len(labels)).to(device).float()
    dropout = nn.Dropout(float(spec["head_dropout"]))

    model: Any = base
    if train_lora:
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
    head_parameters = list(norms.parameters()) + list(plain.parameters()) + list(centered.parameters())
    if train_lora and not adapter_parameters:
        raise RuntimeError("A1 has no trainable LoRA parameters")
    if not train_lora and adapter_parameters:
        raise RuntimeError("A0 unexpectedly has trainable backbone parameters")

    param_groups: list[dict[str, Any]] = []
    if train_lora:
        param_groups.append(
            {"params": adapter_parameters, "lr": float(spec["adapter_lr"])}
        )
    param_groups.append({"params": head_parameters, "lr": float(spec["head_lr"])})
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=float(spec["weight_decay"])
    )

    triplets = build_prompt_triplets(frame, train_idx, seed=seed)
    accumulation = int(spec["gradient_accumulation_triplets"])
    step_groups = group_triplets_for_accumulation(
        triplets,
        accumulation=accumulation,
    )
    total_steps = len(step_groups)
    warmup_steps = max(1, int(float(spec["warmup_fraction"]) * total_steps))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)

    def make_encoding(rows: np.ndarray) -> dict[str, Any]:
        batch = tokenizer.pad(
            {"input_ids": [tokens[int(row)] for row in rows]},
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in batch.items()}

    def representation(enc: dict[str, Any]) -> Any:
        if train_lora:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden_states = model(
                    **enc, output_hidden_states=True, use_cache=False
                ).hidden_states
        else:
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                hidden_states = model(
                    **enc, output_hidden_states=True, use_cache=False
                ).hidden_states
        pooled = [
            norm(
                masked_mean_pool(
                    hidden_states[layer].float(),
                    enc["attention_mask"],
                    mask_first_token=True,
                )
            )
            for layer, norm in zip(layers, norms)
        ]
        return torch.cat(pooled, dim=-1)

    _seed_all(seed, torch)
    if train_lora:
        model.train()
    else:
        model.eval()
    norms.train()
    plain.train()
    centered.train()
    dropout.train()

    loss_sum = 0.0
    loss_count = 0
    skipped_steps = 0
    optimizer_steps = 0
    for rows_group in step_groups:
        optimizer.zero_grad(set_to_none=True)
        group_size = len(rows_group)
        if group_size <= 0:
            raise RuntimeError("empty optimizer-step group")

        for rows in rows_group:
            enc = make_encoding(rows)
            z = representation(enc)
            target = torch.as_tensor(y[rows], dtype=torch.long, device=device)
            plain_logits = plain(dropout(z))
            centered_logits = centered(dropout(center_triplet(z)))
            loss = F.cross_entropy(
                plain_logits, target, weight=class_weights
            ) + F.cross_entropy(
                centered_logits, target, weight=class_weights
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"non-finite training loss in {condition}: {float(loss.detach())}"
                )
            scaler.scale(loss / group_size).backward()
            loss_sum += float(loss.detach().cpu())
            loss_count += 1

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            adapter_parameters + head_parameters,
            float(spec["gradient_clip"]),
        )
        previous_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < previous_scale:
            skipped_steps += 1
        scheduler.step()
        optimizer_steps += 1
        if optimizer_steps % 50 == 0 or optimizer_steps == total_steps:
            print(
                f"{condition} step {optimizer_steps}/{total_steps} "
                f"mean_loss={loss_sum / max(1, loss_count):.6f} skipped={skipped_steps} "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )

    if optimizer_steps != total_steps:
        raise RuntimeError(
            f"optimizer step mismatch for {condition}: {optimizer_steps} != {total_steps}"
        )
    if loss_count != len(triplets):
        raise RuntimeError(
            f"exact-one-epoch mismatch for {condition}: {loss_count} != {len(triplets)}"
        )

    if condition == "A1":
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = checkpoint_dir / "adapter"
        tokenizer_dir = checkpoint_dir / "tokenizer"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(tokenizer_dir)
        state: dict[str, Any] = {}
        for name, tensor in norms.state_dict().items():
            state[f"norms.{name}"] = tensor.detach().cpu().contiguous()
        for prefix, module in (("plain", plain), ("centered", centered)):
            for name, tensor in module.state_dict().items():
                state[f"{prefix}.{name}"] = tensor.detach().cpu().contiguous()
        save_file(state, str(checkpoint_dir / "heads.safetensors"))
        _write_json(
            checkpoint_dir / "meta.json",
            {
                "candidate": CANDIDATE,
                "condition": condition,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "labels": labels,
                "training_spec": spec,
                "implementation_invariants": {
                    "stock_qwen_final_norm_retained": True,
                    "triplet_padding": False,
                    "exact_one_epoch": True,
                },
                "holdout_sources": list(split["holdout_sources"]),
                "prompt_id_overlap": int(split["prompt_id_overlap"]),
                "prompt_key_overlap": int(split["prompt_key_overlap"]),
            },
        )

    model.eval()
    norms.eval()
    plain.eval()
    centered.eval()
    dropout.eval()

    order = np.argsort(
        [len(tokens[int(row)]) for row in heldout_idx], kind="stable"
    )
    prediction_indices = np.empty(len(heldout_idx), dtype=np.int64)
    eval_batch_size = 6
    with torch.inference_mode():
        for start in range(0, len(order), eval_batch_size):
            positions = order[start : start + eval_batch_size]
            rows = heldout_idx[positions]
            enc = make_encoding(rows)
            z = representation(enc)
            logits = plain(z)
            prediction_indices[positions] = logits.argmax(1).detach().cpu().numpy()
    predictions = [labels[int(index)] for index in prediction_indices]
    scores = _score_condition(
        frame,
        heldout_idx,
        predictions,
        eligible_labels,
        list(split["holdout_sources"]),
    )

    result = {
        "condition": condition,
        "condition_spec": condition_spec,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "libraries": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "peft": str(peft.__version__),
        },
        "implementation_invariants": {
            "stock_qwen_final_norm_retained": True,
            "triplet_padding": False,
            "exact_one_epoch": True,
        },
        "train": {
            "rows": int(len(train_idx)),
            "triplet_microbatches": int(len(triplets)),
            "optimizer_steps": int(optimizer_steps),
            "skipped_scaler_steps": int(skipped_steps),
            "mean_training_loss": float(loss_sum / max(1, loss_count)),
            "seconds": float(time.time() - started),
        },
        "scores": scores,
    }
    _write_json(worker_output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    del model, base, norms, plain, centered, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def _aggregate_report(
    *,
    data: dict[str, Any],
    conditions: dict[str, dict[str, Any]],
    checkpoint_dir: Path,
) -> dict[str, Any]:
    a0 = conditions["A0"]
    a1 = conditions["A1"]
    holdout_sources = list(data["holdout_sources"])
    pooled_delta = float(
        a1["scores"]["pooled"]["balanced_accuracy"]
        - a0["scores"]["pooled"]["balanced_accuracy"]
    )
    source_deltas = [
        float(
            a1["scores"]["sources"][source]["balanced_accuracy"]
            - a0["scores"]["sources"][source]["balanced_accuracy"]
        )
        for source in holdout_sources
    ]
    decision = gate_a_decision(
        eligible_label_count=int(data["eligible_label_count"]),
        eligible_heldout_rows=int(data["eligible_heldout_rows"]),
        pooled_delta=pooled_delta,
        source_deltas=source_deltas,
    )

    report: dict[str, Any] = execution_scope()
    report.update(
        {
            "candidate": CANDIDATE,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "transformers": TRANSFORMERS_VERSION,
                "peft": PEFT_VERSION,
            },
            "training_spec": gate_a_training_spec(),
            "implementation_invariants": {
                "stock_qwen_final_norm_retained": True,
                "triplet_padding": False,
                "exact_one_epoch": True,
            },
            "data": data,
            "conditions": conditions,
            "checkpoint": {
                "condition": "A1",
                "path": str(checkpoint_dir),
                "required_for_gate_b_if_authorized": True,
            },
            "gate_a": {
                "pooled_delta_a1_minus_a0": pooled_delta,
                "source_deltas_a1_minus_a0": {
                    source: delta for source, delta in zip(holdout_sources, source_deltas)
                },
                "mean_source_delta": float(np.mean(source_deltas)),
                "negative_source_count": int(np.sum(np.asarray(source_deltas) < 0.0)),
                "decision": decision,
            },
        }
    )
    return report


def _inconclusive_report(data: dict[str, Any], reason: str) -> dict[str, Any]:
    report: dict[str, Any] = execution_scope()
    report.update(
        {
            "candidate": CANDIDATE,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "transformers": TRANSFORMERS_VERSION,
                "peft": PEFT_VERSION,
            },
            "training_spec": gate_a_training_spec(),
            "implementation_invariants": {
                "stock_qwen_final_norm_retained": True,
                "triplet_padding": False,
                "exact_one_epoch": True,
            },
            "data": data,
            "conditions": {},
            "checkpoint": None,
            "gate_a": {
                "decision": {
                    "status": "inconclusive",
                    "official_oof_authorized": False,
                    "reason": reason,
                }
            },
        }
    )
    return report


def run_orchestrator(
    *,
    pool: str | None,
    output: Path,
    checkpoint_dir: Path,
) -> None:
    started = time.time()
    runtime_dir = Path("/tmp/two-stage-lora24-gate-a")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    safe_path = runtime_dir / "phase14_safe.parquet"

    pool_path = _find_pool_path(pool)
    data = _prepare_stage1(pool_path, safe_path)
    if (
        int(data["eligible_label_count"]) < MIN_ELIGIBLE_LABELS
        or int(data["eligible_heldout_rows"]) < MIN_ELIGIBLE_HELDOUT_ROWS
    ):
        report = _inconclusive_report(
            data, "insufficient_eligible_labels_or_rows_before_model_load"
        )
        report["seconds"] = float(time.time() - started)
        _write_json(output, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return
    if any(int(value) <= 0 for value in data["source_eligible_rows"].values()):
        report = _inconclusive_report(
            data, "one_or_more_holdout_sources_have_zero_eligible_rows"
        )
        report["seconds"] = float(time.time() - started)
        _write_json(output, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return

    from huggingface_hub import snapshot_download

    print(
        f"downloading pinned model {MODEL_ID}@{MODEL_REVISION}",
        flush=True,
    )
    snapshot_path = Path(
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    )
    conditions = _run_workers(
        safe_path=safe_path,
        snapshot_path=snapshot_path,
        runtime_dir=runtime_dir,
        checkpoint_dir=checkpoint_dir,
    )
    report = _aggregate_report(
        data=data, conditions=conditions, checkpoint_dir=checkpoint_dir
    )
    report["seconds"] = float(time.time() - started)
    _write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working") / RESULT_FILENAME,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/kaggle/working") / CHECKPOINT_DIRNAME,
    )
    parser.add_argument("--worker-condition", choices=CONDITIONS, default=None)
    parser.add_argument("--safe-path", type=Path, default=None)
    parser.add_argument("--snapshot-path", type=Path, default=None)
    parser.add_argument("--worker-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_condition is not None:
        required = {
            "safe_path": args.safe_path,
            "snapshot_path": args.snapshot_path,
            "worker_output": args.worker_output,
            "checkpoint_dir": args.checkpoint_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"worker mode missing arguments: {missing}")
        _run_condition_worker(
            condition=args.worker_condition,
            safe_path=args.safe_path,
            snapshot_path=args.snapshot_path,
            worker_output=args.worker_output,
            checkpoint_dir=args.checkpoint_dir,
        )
        return

    run_orchestrator(
        pool=args.pool,
        output=args.output,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
