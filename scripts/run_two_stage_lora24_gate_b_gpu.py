from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from scripts.run_strong_sparse_anchor import load_train
from scripts.run_two_stage_lora24_gate_a import (
    build_prompt_triplets,
    center_triplet,
    compute_class_weights,
    group_triplets_for_accumulation,
    masked_mean_pool,
    truncate_qwen_layers,
)
from scripts.run_two_stage_lora24_gate_b import (
    gate_b_training_spec,
    split_official_fold,
    validate_gate_a_checkpoint_meta,
)


CANDIDATE = "two-stage-lora24-v1"
MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
TRANSFORMERS_VERSION = "4.57.1"
PEFT_VERSION = "0.17.1"
CHECKPOINT_DATASET_REF = "kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint"
CHECKPOINT_DATASET_VERSION = 1
CHECKPOINT_MANIFEST_SHA256 = "c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0"
SOURCE_ARTIFACT_ID = 13094069155
SOURCE_ARTIFACT_DIGEST = "sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66"


def execution_scope() -> dict[str, object]:
    return {
        "scope": "gate_b_official_oof_fold_worker",
        "public_stage1_rows_mixed_into_stage2": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "kaggle_submission_authorized": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_all(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_manifest_sha256(checkpoint_dir: Path) -> str:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    files = []
    for path in sorted(p for p in checkpoint_dir.rglob("*") if p.is_file()):
        files.append(
            {
                "path": path.relative_to(checkpoint_dir).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        )
    if len(files) != 12:
        raise RuntimeError(f"expected 12 locked checkpoint files, found {len(files)}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stage2_training_accounting(
    *,
    triplet_microbatches_per_epoch: int,
    accumulation: int,
    epochs: int,
) -> dict[str, int]:
    if triplet_microbatches_per_epoch <= 0:
        raise ValueError("triplet_microbatches_per_epoch must be positive")
    if accumulation <= 0:
        raise ValueError("accumulation must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    steps_per_epoch = math.ceil(triplet_microbatches_per_epoch / accumulation)
    final_group_size = triplet_microbatches_per_epoch - accumulation * (steps_per_epoch - 1)
    return {
        "epochs": int(epochs),
        "triplet_microbatches_per_epoch": int(triplet_microbatches_per_epoch),
        "triplet_microbatches_total": int(triplet_microbatches_per_epoch * epochs),
        "optimizer_steps_per_epoch": int(steps_per_epoch),
        "optimizer_steps_total": int(steps_per_epoch * epochs),
        "final_partial_accumulation_group_size": int(final_group_size),
    }


def validate_fold_predictions(
    valid: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    labels: Sequence[str],
    fold: int,
) -> None:
    required = {
        "id",
        "prompt_id",
        "model",
        "fold_id",
        "plain_prediction",
        "centered_prediction",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise RuntimeError(f"fold prediction columns missing: {missing}")
    if len(predictions) != len(valid) or predictions["id"].duplicated().any():
        raise RuntimeError("each validation id must appear exactly once")
    if set(predictions["id"].tolist()) != set(valid["id"].tolist()):
        raise RuntimeError("each validation id must appear exactly once")
    if set(predictions["fold_id"].astype(int).tolist()) != {fold}:
        raise RuntimeError(f"prediction rows do not belong exclusively to fold {fold}")

    keys = ["id", "prompt_id", "model", "fold_id"]
    expected = valid[keys].sort_values("id").reset_index(drop=True)
    observed = predictions[keys].sort_values("id").reset_index(drop=True)
    if not expected.equals(observed):
        raise RuntimeError("validation prediction row alignment mismatch")

    allowed = set(str(label) for label in labels)
    for column in ("plain_prediction", "centered_prediction"):
        if predictions[column].isna().any():
            raise RuntimeError(f"{column} contains null predictions")
        outside = sorted(set(predictions[column].astype(str)) - allowed)
        if outside:
            raise RuntimeError(f"{column} contains predictions outside label set: {outside}")


def load_exact_stage1_checkpoint(
    *,
    checkpoint_dir: Path,
    snapshot_path: Path,
    official_labels: Sequence[str],
    device: Any,
) -> dict[str, Any]:
    """Load the locked Gate-A A1 adapter/readouts/tokenizer as trainable Stage-2 state."""
    import importlib.util as importlib_util

    original_find_spec = importlib_util.find_spec
    importlib_util.find_spec = (
        lambda name, *args, **kwargs: None
        if name.split(".")[0] == "torchao"
        else original_find_spec(name, *args, **kwargs)
    )

    import torch
    from torch import nn
    import peft
    from peft import PeftModel
    import peft.import_utils as peft_import_utils
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    peft_import_utils.is_torchao_available = lambda *args, **kwargs: False

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers version drift: {transformers.__version__} != {TRANSFORMERS_VERSION}"
        )
    if peft.__version__ != PEFT_VERSION:
        raise RuntimeError(f"peft version drift: {peft.__version__} != {PEFT_VERSION}")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate B fold worker requires CUDA")

    required_paths = [
        checkpoint_dir / "adapter" / "adapter_model.safetensors",
        checkpoint_dir / "adapter" / "adapter_config.json",
        checkpoint_dir / "heads.safetensors",
        checkpoint_dir / "meta.json",
        checkpoint_dir / "tokenizer",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    observed_manifest = checkpoint_manifest_sha256(checkpoint_dir)
    if observed_manifest != CHECKPOINT_MANIFEST_SHA256:
        raise RuntimeError(
            f"checkpoint manifest drift: {observed_manifest} != {CHECKPOINT_MANIFEST_SHA256}"
        )

    meta = json.loads((checkpoint_dir / "meta.json").read_text(encoding="utf-8"))
    validate_gate_a_checkpoint_meta(meta)
    labels = [str(value) for value in meta["labels"]]
    official_labels = [str(value) for value in official_labels]
    if labels != official_labels:
        raise RuntimeError(
            f"checkpoint label order differs from official labels: {labels} != {official_labels}"
        )

    spec = gate_b_training_spec()
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir / "tokenizer"), local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise RuntimeError("locked Qwen tokenizer has no EOS token")

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
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.enable_input_require_grads()

    model = PeftModel.from_pretrained(
        base,
        str(checkpoint_dir / "adapter"),
        is_trainable=True,
    )
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or any("lora_" not in name for name in trainable_names):
        raise RuntimeError(f"unexpected trainable backbone parameters: {trainable_names[:10]}")
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    layers = [int(value) for value in spec["pooled_layers"]]
    hidden_size = int(config.hidden_size)
    norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers]).to(device).float()
    plain = nn.Linear(hidden_size * len(layers), len(labels)).to(device).float()
    centered = nn.Linear(hidden_size * len(layers), len(labels)).to(device).float()
    dropout = nn.Dropout(float(spec["head_dropout"]))

    saved = load_file(str(checkpoint_dir / "heads.safetensors"))
    norm_state = {
        key.removeprefix("norms."): value
        for key, value in saved.items()
        if key.startswith("norms.")
    }
    plain_state = {
        key.removeprefix("plain."): value
        for key, value in saved.items()
        if key.startswith("plain.")
    }
    centered_state = {
        key.removeprefix("centered."): value
        for key, value in saved.items()
        if key.startswith("centered.")
    }
    norms.load_state_dict(norm_state, strict=True)
    plain.load_state_dict(plain_state, strict=True)
    centered.load_state_dict(centered_state, strict=True)

    return {
        "torch": torch,
        "model": model,
        "tokenizer": tokenizer,
        "norms": norms,
        "plain": plain,
        "centered": centered,
        "dropout": dropout,
        "labels": labels,
        "layers": layers,
        "manifest_sha256": observed_manifest,
        "meta": meta,
    }


def _tokenize_official(frame: pd.DataFrame, tokenizer: Any, *, max_length_before_eos: int) -> list[list[int]]:
    token_lists = tokenizer(
        frame["text"].fillna("").astype(str).tolist(),
        add_special_tokens=False,
        truncation=True,
        max_length=max_length_before_eos,
    )["input_ids"]
    eos = int(tokenizer.eos_token_id)
    return [list(ids) + [eos] for ids in token_lists]


def _make_encoding(
    tokenizer: Any,
    tokens: list[list[int]],
    rows: np.ndarray,
    device: Any,
) -> dict[str, Any]:
    batch = tokenizer.pad(
        {"input_ids": [tokens[int(row)] for row in rows]},
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in batch.items()}


def _representation(
    *,
    model: Any,
    norms: Any,
    layers: Sequence[int],
    encoding: dict[str, Any],
    torch: Any,
) -> Any:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        hidden_states = model(
            **encoding,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
    pooled = [
        norm(
            masked_mean_pool(
                hidden_states[int(layer)].float(),
                encoding["attention_mask"],
                mask_first_token=True,
            )
        )
        for layer, norm in zip(layers, norms)
    ]
    return torch.cat(pooled, dim=-1)


def train_two_stage2_epochs(
    *,
    frame: pd.DataFrame,
    fit_idx: np.ndarray,
    triplets: list[np.ndarray],
    tokens: list[list[int]],
    y: np.ndarray,
    class_weights: Any,
    model: Any,
    tokenizer: Any,
    norms: Any,
    plain: Any,
    centered: Any,
    dropout: Any,
    layers: Sequence[int],
    device: Any,
    torch: Any,
    F: Any,
) -> dict[str, Any]:
    """Replay one frozen fit-triplet ordering for exactly two Stage-2 epochs."""
    if len(fit_idx) == 0 or len(triplets) == 0:
        raise RuntimeError("Stage-2 fit partition/triplets must be non-empty")
    spec = gate_b_training_spec()
    accumulation = int(spec["gradient_accumulation_triplets"])
    step_groups = group_triplets_for_accumulation(triplets, accumulation=accumulation)
    accounting = stage2_training_accounting(
        triplet_microbatches_per_epoch=len(triplets),
        accumulation=accumulation,
        epochs=int(spec["epochs"]),
    )
    total_steps = int(accounting["optimizer_steps_total"])
    warmup_steps = max(1, int(float(spec["warmup_fraction"]) * total_steps))

    adapter_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    head_parameters = list(norms.parameters()) + list(plain.parameters()) + list(centered.parameters())
    if not adapter_parameters:
        raise RuntimeError("Stage-2 has no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": float(spec["adapter_lr"])},
            {"params": head_parameters, "lr": float(spec["head_lr"])},
        ],
        weight_decay=float(spec["weight_decay"]),
    )

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)

    model.train()
    norms.train()
    plain.train()
    centered.train()
    dropout.train()

    started = time.time()
    loss_sum = 0.0
    loss_count = 0
    optimizer_steps = 0
    skipped_steps = 0
    per_epoch_loss: list[float] = []

    for epoch in range(int(spec["epochs"])):
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for rows_group in step_groups:
            optimizer.zero_grad(set_to_none=True)
            group_size = len(rows_group)
            if group_size <= 0:
                raise RuntimeError("empty Stage-2 optimizer-step group")
            for rows in rows_group:
                encoding = _make_encoding(tokenizer, tokens, rows, device)
                z = _representation(
                    model=model,
                    norms=norms,
                    layers=layers,
                    encoding=encoding,
                    torch=torch,
                )
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
                        f"non-finite Stage-2 loss at epoch={epoch}: {float(loss.detach())}"
                    )
                scaler.scale(loss / group_size).backward()
                observed_loss = float(loss.detach().cpu())
                loss_sum += observed_loss
                loss_count += 1
                epoch_loss_sum += observed_loss
                epoch_loss_count += 1

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

        if epoch_loss_count != len(triplets):
            raise RuntimeError(
                f"Stage-2 epoch {epoch} consumed {epoch_loss_count} triplets; expected {len(triplets)}"
            )
        per_epoch_loss.append(epoch_loss_sum / epoch_loss_count)
        print(
            f"stage2 epoch={epoch + 1}/{int(spec['epochs'])} "
            f"mean_loss={per_epoch_loss[-1]:.6f} optimizer_steps={optimizer_steps}/{total_steps}",
            flush=True,
        )

    if loss_count != int(accounting["triplet_microbatches_total"]):
        raise RuntimeError(
            f"Stage-2 two-epoch triplet mismatch: {loss_count} != {accounting['triplet_microbatches_total']}"
        )
    if optimizer_steps != total_steps:
        raise RuntimeError(f"Stage-2 optimizer step mismatch: {optimizer_steps} != {total_steps}")

    return {
        **accounting,
        "warmup_steps": int(warmup_steps),
        "optimizer_steps": int(optimizer_steps),
        "skipped_scaler_steps": int(skipped_steps),
        "mean_training_loss": float(loss_sum / loss_count),
        "mean_loss_by_epoch": [float(value) for value in per_epoch_loss],
        "seconds": float(time.time() - started),
    }


def _infer_validation(
    *,
    frame: pd.DataFrame,
    valid_idx: np.ndarray,
    tokens: list[list[int]],
    model: Any,
    tokenizer: Any,
    norms: Any,
    plain: Any,
    centered: Any,
    dropout: Any,
    layers: Sequence[int],
    labels: Sequence[str],
    device: Any,
    torch: Any,
    seed: int,
) -> pd.DataFrame:
    valid_triplets = build_prompt_triplets(frame, valid_idx, seed=seed)
    plain_indices = np.full(len(frame), -1, dtype=np.int64)
    centered_indices = np.full(len(frame), -1, dtype=np.int64)

    model.eval()
    norms.eval()
    plain.eval()
    centered.eval()
    dropout.eval()
    with torch.inference_mode():
        for rows in valid_triplets:
            if np.any(plain_indices[rows] >= 0) or np.any(centered_indices[rows] >= 0):
                raise RuntimeError("validation row encountered more than once during inference")
            encoding = _make_encoding(tokenizer, tokens, rows, device)
            z = _representation(
                model=model,
                norms=norms,
                layers=layers,
                encoding=encoding,
                torch=torch,
            )
            plain_indices[rows] = plain(z).argmax(1).detach().cpu().numpy()
            centered_indices[rows] = centered(center_triplet(z)).argmax(1).detach().cpu().numpy()

    if np.any(plain_indices[valid_idx] < 0) or np.any(centered_indices[valid_idx] < 0):
        raise RuntimeError("one or more validation rows were not predicted exactly once")

    result = frame.iloc[valid_idx][["id", "prompt_id", "model", "fold_id"]].copy()
    result["plain_prediction"] = [labels[int(index)] for index in plain_indices[valid_idx]]
    result["centered_prediction"] = [labels[int(index)] for index in centered_indices[valid_idx]]
    return result.reset_index(drop=True)


def run_fold_worker(
    *,
    fold: int,
    train_path: Path,
    folds_path: Path,
    checkpoint_dir: Path,
    snapshot_path: Path,
    output_dir: Path,
) -> None:
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("Gate B fold worker requires CUDA")
    spec = gate_b_training_spec()
    seed = int(spec["seed"])
    _seed_all(seed, torch)
    device = torch.device("cuda:0")

    train = load_train(train_path, folds_path).reset_index(drop=True)
    labels = sorted(train["model"].astype(str).unique().tolist())
    if len(labels) != 18:
        raise RuntimeError(f"expected 18 official labels, found {len(labels)}")
    fit_idx, valid_idx = split_official_fold(train, fold)
    fit_frame = train.iloc[fit_idx]
    valid_frame = train.iloc[valid_idx]

    scope = execution_scope()
    if int(scope["public_stage1_rows_mixed_into_stage2"]) != 0:
        raise RuntimeError("Gate B Stage-2 must not mix public Stage-1 rows")

    class_weights_map = compute_class_weights(
        fit_frame["model"].astype(str).to_numpy(), classes=labels
    )
    class_weights = torch.tensor(
        [class_weights_map[label] for label in labels],
        dtype=torch.float32,
        device=device,
    )

    checkpoint = load_exact_stage1_checkpoint(
        checkpoint_dir=checkpoint_dir,
        snapshot_path=snapshot_path,
        official_labels=labels,
        device=device,
    )
    model = checkpoint["model"]
    tokenizer = checkpoint["tokenizer"]
    norms = checkpoint["norms"]
    plain = checkpoint["plain"]
    centered = checkpoint["centered"]
    dropout = checkpoint["dropout"]
    layers = checkpoint["layers"]

    tokens = _tokenize_official(
        train,
        tokenizer,
        max_length_before_eos=int(spec["max_length_before_eos"]),
    )
    label_to_idx = {label: index for index, label in enumerate(labels)}
    y = train["model"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64)
    triplets = build_prompt_triplets(train, fit_idx, seed=seed)

    train_meta = train_two_stage2_epochs(
        frame=train,
        fit_idx=fit_idx,
        triplets=triplets,
        tokens=tokens,
        y=y,
        class_weights=class_weights,
        model=model,
        tokenizer=tokenizer,
        norms=norms,
        plain=plain,
        centered=centered,
        dropout=dropout,
        layers=layers,
        device=device,
        torch=torch,
        F=F,
    )

    predictions = _infer_validation(
        frame=train,
        valid_idx=valid_idx,
        tokens=tokens,
        model=model,
        tokenizer=tokenizer,
        norms=norms,
        plain=plain,
        centered=centered,
        dropout=dropout,
        layers=layers,
        labels=labels,
        device=device,
        torch=torch,
        seed=seed,
    )
    validate_fold_predictions(valid_frame, predictions, labels=labels, fold=fold)

    plain_ba = float(
        balanced_accuracy_score(predictions["model"], predictions["plain_prediction"])
    )
    centered_ba = float(
        balanced_accuracy_score(predictions["model"], predictions["centered_prediction"])
    )
    fit_prompts = set(fit_frame["prompt_id"].astype(str))
    valid_prompts = set(valid_frame["prompt_id"].astype(str))
    prompt_overlap = sorted(fit_prompts & valid_prompts)
    if prompt_overlap:
        raise RuntimeError(f"prompt leakage survived fold split: {prompt_overlap[:5]}")

    report: dict[str, Any] = execution_scope()
    report.update(
        {
            "candidate": CANDIDATE,
            "fold": int(fold),
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "transformers": TRANSFORMERS_VERSION,
                "peft": PEFT_VERSION,
            },
            "checkpoint": {
                "dataset_ref": CHECKPOINT_DATASET_REF,
                "dataset_version": CHECKPOINT_DATASET_VERSION,
                "manifest_sha256": checkpoint["manifest_sha256"],
                "source_artifact_id": SOURCE_ARTIFACT_ID,
                "source_artifact_digest": SOURCE_ARTIFACT_DIGEST,
            },
            "training_spec": spec,
            "data": {
                "official_rows": int(len(train)),
                "fit_rows": int(len(fit_idx)),
                "validation_rows": int(len(valid_idx)),
                "fit_prompts": int(len(fit_prompts)),
                "validation_prompts": int(len(valid_prompts)),
                "prompt_overlap": 0,
                "labels": labels,
            },
            "train": train_meta,
            "predictions": {
                "rows": int(len(predictions)),
                "plain_rows": int(predictions["plain_prediction"].notna().sum()),
                "centered_rows": int(predictions["centered_prediction"].notna().sum()),
                "plain_fold_balanced_accuracy": plain_ba,
                "centered_fold_balanced_accuracy": centered_ba,
            },
            "invariants": {
                "fresh_stage1_checkpoint_loaded_for_worker": True,
                "fit_only_class_weights": True,
                "same_triplet_order_replayed_each_epoch": True,
                "validation_used_for_training": False,
                "post_stage2_checkpoint_saved": False,
            },
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / f"fold_{fold}_predictions.csv", index=False)
    _write_json(output_dir / f"fold_{fold}.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    del model, norms, plain, centered
    gc.collect()
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen two-stage-lora24-v1 Gate B fold worker")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_fold_worker(
        fold=args.fold,
        train_path=args.train,
        folds_path=args.folds,
        checkpoint_dir=args.checkpoint_dir,
        snapshot_path=args.snapshot_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
