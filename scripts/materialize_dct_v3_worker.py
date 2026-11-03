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
        '''from scripts.run_phase14_public_sparse import (\n    EXPECTED_EXTERNAL_ROWS,\n    EXPECTED_RAW_ROWS,\n    EXPECTED_RAW_SHA256,\n    EXPECTED_SAFE_PROMPT_IDS,\n    build_conservative_subset,\n)\nfrom scripts.dct_temporal_pool import masked_dct_pool\n''',
        "imports",
    )
    text = _replace_once(
        text,
        'CANDIDATE = "two-stage-lora24-v1"',
        'CANDIDATE = "dct-temporal-fingerprint-v3"',
        "candidate",
    )
    text = _replace_once(
        text,
        '''    layers = [int(value) for value in spec["pooled_layers"]]\n    hidden_size = int(config.hidden_size)\n    norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers]).to(device).float()\n    plain = nn.Linear(hidden_size * len(layers), len(labels)).to(device).float()\n    centered = nn.Linear(hidden_size * len(layers), len(labels)).to(device).float()\n    dropout = nn.Dropout(float(spec["head_dropout"]))\n''',
        '''    layers = [int(value) for value in spec["pooled_layers"]]\n    hidden_size = int(config.hidden_size)\n    representation_dim = hidden_size * len(layers) * 2\n    norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers]).to(device).float()\n    plain = nn.Linear(representation_dim, len(labels)).to(device).float()\n    centered = nn.Linear(representation_dim, len(labels)).to(device).float()\n    dropout = nn.Dropout(float(spec["head_dropout"]))\n''',
        "head width",
    )
    text = _replace_once(
        text,
        '''        pooled = [\n            norm(\n                masked_mean_pool(\n                    hidden_states[layer].float(),\n                    enc["attention_mask"],\n                    mask_first_token=True,\n                )\n            )\n            for layer, norm in zip(layers, norms)\n        ]\n        return torch.cat(pooled, dim=-1)\n''',
        '''        pooled = []\n        for layer, norm in zip(layers, norms):\n            mode_features = masked_dct_pool(\n                hidden_states[layer].float(),\n                enc["attention_mask"],\n                mask_first_token=True,\n                n_modes=2,\n            )\n            pooled.extend([\n                norm(mode_features[:, 0]),\n                norm(mode_features[:, 1]),\n            ])\n        return torch.cat(pooled, dim=-1)\n''',
        "representation pooling",
    )
    if "CrossPromptPrototypeBank" in text or "CONTRASTIVE_WEIGHT" in text:
        raise RuntimeError("rejected v2 contrastive code leaked into v3 worker")
    if "train_two_stage2_epochs" in text:
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
    print(f"materialized {args.output} from exact v1 worker with frozen DCT v3 change")


if __name__ == "__main__":
    main()
