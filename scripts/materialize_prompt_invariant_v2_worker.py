from __future__ import annotations

import argparse
from pathlib import Path


def _replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"v1 worker drift at {name}: expected one match, found {count}")
    return text.replace(old, new, 1)


def materialize(source: str) -> str:
    text = source
    text = _replace_once(
        text,
        '''from scripts.run_phase14_public_sparse import (\n    EXPECTED_EXTERNAL_ROWS,\n    EXPECTED_RAW_ROWS,\n    EXPECTED_RAW_SHA256,\n    EXPECTED_SAFE_PROMPT_IDS,\n    build_conservative_subset,\n)\n''',
        '''from scripts.run_phase14_public_sparse import (\n    EXPECTED_EXTERNAL_ROWS,\n    EXPECTED_RAW_ROWS,\n    EXPECTED_RAW_SHA256,\n    EXPECTED_SAFE_PROMPT_IDS,\n    build_conservative_subset,\n)\nfrom scripts.cross_prompt_prototype import CrossPromptPrototypeBank\nfrom scripts.prototype_training_integration import (\n    CONTRASTIVE_TAU,\n    CONTRASTIVE_WEIGHT,\n    PROMPT_BUCKET_SEED,\n    PrototypeStepBuffer,\n    commit_prototypes_after_scaler_step,\n)\n''',
        "imports",
    )
    text = _replace_once(
        text,
        'CANDIDATE = "two-stage-lora24-v1"',
        'CANDIDATE = "prompt-invariant-contrastive-v2"',
        "candidate",
    )
    text = _replace_once(
        text,
        '''    layers = [int(value) for value in spec["pooled_layers"]]\n    hidden_size = int(config.hidden_size)\n    norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers]).to(device).float()\n    plain = nn.Linear(hidden_size * len(layers), len(labels)).to(device).float()\n    centered = nn.Linear(hidden_size * len(layers), len(labels)).to(device).float()\n    dropout = nn.Dropout(float(spec["head_dropout"]))\n''',
        '''    layers = [int(value) for value in spec["pooled_layers"]]\n    hidden_size = int(config.hidden_size)\n    representation_dim = hidden_size * len(layers)\n    norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers]).to(device).float()\n    plain = nn.Linear(representation_dim, len(labels)).to(device).float()\n    centered = nn.Linear(representation_dim, len(labels)).to(device).float()\n    dropout = nn.Dropout(float(spec["head_dropout"]))\n    prototype_bank = (\n        CrossPromptPrototypeBank(len(labels), representation_dim, device=device)\n        if condition == "A1"\n        else None\n    )\n''',
        "prototype bank",
    )
    text = _replace_once(
        text,
        '''    loss_sum = 0.0\n    loss_count = 0\n    skipped_steps = 0\n''',
        '''    loss_sum = 0.0\n    loss_count = 0\n    contrastive_loss_sum = 0.0\n    contrastive_eligible_anchors = 0\n    contrastive_total_anchors = 0\n    skipped_steps = 0\n''',
        "loss counters",
    )
    text = _replace_once(
        text,
        '''    for rows_group in step_groups:\n        optimizer.zero_grad(set_to_none=True)\n        group_size = len(rows_group)\n''',
        '''    for rows_group in step_groups:\n        optimizer.zero_grad(set_to_none=True)\n        prototype_step_buffer = PrototypeStepBuffer()\n        group_size = len(rows_group)\n''',
        "step buffer",
    )
    text = _replace_once(
        text,
        '''            target = torch.as_tensor(y[rows], dtype=torch.long, device=device)\n            plain_logits = plain(dropout(z))\n            centered_logits = centered(dropout(center_triplet(z)))\n            loss = F.cross_entropy(\n                plain_logits, target, weight=class_weights\n            ) + F.cross_entropy(\n                centered_logits, target, weight=class_weights\n            )\n''',
        '''            target = torch.as_tensor(y[rows], dtype=torch.long, device=device)\n            prompt_ids = frame.iloc[rows]["prompt_id"].astype(str).tolist()\n            plain_logits = plain(dropout(z))\n            centered_logits = centered(dropout(center_triplet(z)))\n            plain_loss = F.cross_entropy(plain_logits, target, weight=class_weights)\n            centered_loss = F.cross_entropy(centered_logits, target, weight=class_weights)\n            if prototype_bank is None:\n                prototype_loss = z.sum() * 0.0\n            else:\n                prototype_result = prototype_bank.loss(\n                    z, target, prompt_ids, tau=CONTRASTIVE_TAU, seed=PROMPT_BUCKET_SEED\n                )\n                prototype_loss = prototype_result.loss\n                prototype_step_buffer.append(z, target, prompt_ids)\n                contrastive_loss_sum += float(prototype_loss.detach().cpu())\n                contrastive_eligible_anchors += int(prototype_result.eligible_anchors)\n                contrastive_total_anchors += int(prototype_result.total_anchors)\n            loss = plain_loss + centered_loss + CONTRASTIVE_WEIGHT * prototype_loss\n''',
        "training loss",
    )
    text = _replace_once(
        text,
        '''        previous_scale = float(scaler.get_scale())\n        scaler.step(optimizer)\n        scaler.update()\n        if float(scaler.get_scale()) < previous_scale:\n            skipped_steps += 1\n        scheduler.step()\n''',
        '''        previous_scale = float(scaler.get_scale())\n        scaler.step(optimizer)\n        scaler.update()\n        current_scale = float(scaler.get_scale())\n        if current_scale < previous_scale:\n            skipped_steps += 1\n        if prototype_bank is not None:\n            commit_prototypes_after_scaler_step(\n                prototype_bank,\n                prototype_step_buffer,\n                previous_scale=previous_scale,\n                current_scale=current_scale,\n                seed=PROMPT_BUCKET_SEED,\n            )\n        scheduler.step()\n''',
        "post optimizer step",
    )
    text = _replace_once(
        text,
        '''            "mean_training_loss": float(loss_sum / max(1, loss_count)),\n            "seconds": float(time.time() - started),\n''',
        '''            "mean_training_loss": float(loss_sum / max(1, loss_count)),\n            "prompt_invariant_contrastive": {\n                "weight": CONTRASTIVE_WEIGHT,\n                "temperature": CONTRASTIVE_TAU,\n                "prompt_bucket_seed": PROMPT_BUCKET_SEED,\n                "prototype_buckets": 2,\n                "eligible_anchors": int(contrastive_eligible_anchors),\n                "total_anchors": int(contrastive_total_anchors),\n                "coverage": float(contrastive_eligible_anchors / max(1, contrastive_total_anchors)),\n                "mean_loss_per_triplet": float(contrastive_loss_sum / max(1, loss_count)),\n                "updates_only_after_successful_optimizer_step": True,\n                "extra_backbone_forwards": 0,\n            },\n            "seconds": float(time.time() - started),\n''',
        "result metadata",
    )
    if "train_two_stage2_epochs" in text or "stage2" in text.lower():
        raise RuntimeError("unexpected Stage-2 code in Stage-1 worker materialization")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rendered = materialize(args.source.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"materialized {args.output} from exact v1 worker with frozen v2 change")


if __name__ == "__main__":
    main()
