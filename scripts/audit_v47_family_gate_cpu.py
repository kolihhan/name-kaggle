"""Source-pinned CPU-only mechanism check of Lightning V47 family gates.

No competition files, checkpoint weights, inference, training, or submissions.
Usage: python scripts/audit_v47_family_gate_cpu.py --notebook /tmp/v47.ipynb
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


EXPECTED_SHA256 = "be996d4921795156b2bc27e78171c70de02c9a43fdcfcaf7bc949c72e7dce32d"
FUNCTIONS = {"_np_logsoftmax", "_apply_conditional_family_gate"}
FAMILY_TO_KEY = [
    ("claude", "claude", "claude_style", False),
    ("claude_lora", "claude", "claude_lora", True),
    ("gpt", "gpt", "family", False),
    ("gemini", "gemini", "family", False),
    ("glm", "glm", "family", False),
]


def _parse_original(path: Path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"V47 notebook SHA drift: {digest}")
    notebook = json.loads(raw)
    matches = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/legacy/comp_cached.py"
        in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exact complementary code cell, found {len(matches)}")
    source = "\n".join(matches[0].splitlines()[1:])
    tree = ast.parse(source)
    defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if set(node.name for node in defs) != FUNCTIONS:
        raise ValueError("V47 family gate functions missing")
    constants = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "LABELS":
                if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name) or node.value.func.id != "sorted":
                    raise ValueError("unexpected V47 label declaration")
                constants["labels"] = sorted(ast.literal_eval(node.value.args[0]))
            if isinstance(target, ast.Name) and target.id == "GATE_LAM":
                constants["gate_lam"] = ast.literal_eval(node.value)
    if len(constants.get("labels", [])) != 18 or set(constants.get("gate_lam", {})) != {"claude_style", "claude_lora", "family"}:
        raise ValueError("V47 label or gate-weight contract drift")
    module = compile(ast.Module(body=defs, type_ignores=[]), filename="<EXACT LIGHTNING V47 FUNCTIONS>", mode="exec")
    return digest, constants, module


def _bind(module, labels, logits, gates):
    idx = {
        family: np.asarray([i for i, label in enumerate(labels) if label.startswith(family)], dtype=int)
        for family in ("claude", "gpt", "gemini", "glm")
    }
    ns = {
        "np": np,
        "TE": [""] * len(logits),
        "LOGIT": logits.copy(),
        "GATES": gates,
        "FAMILY_IDX": idx,
    }
    exec(module, ns)
    return ns, idx


def audit(notebook_path: Path) -> dict:
    digest, contract, code = _parse_original(notebook_path)
    labels, lam = contract["labels"], contract["gate_lam"]
    rng = np.random.default_rng(20261105)
    n = 384
    z = rng.normal(0, 1.3, size=(n, len(labels)))
    idx = {
        family: np.asarray([i for i, label in enumerate(labels) if label.startswith(family)], dtype=int)
        for family in ("claude", "gpt", "gemini", "glm")
    }
    for j in idx.values():
        z[:64, j] = rng.normal(0, 0.03, size=(64, len(j)))
        z[64:128, j] = -8.0
        z[64:128, j[0]] = 8.0
    gates = {
        name: rng.normal(0, 2, size=(n, len(idx[family])))
        for name, family, _, _ in FAMILY_TO_KEY
    }
    gates["claude_lora"] -= logsumexp(gates["claude_lora"], axis=1, keepdims=True)
    ns, idx = _bind(code, labels, z, gates)
    results = {}
    for name, family, weight_key, logprob in FAMILY_TO_KEY:
        pre = ns["LOGIT"].copy()
        j = idx[family]
        untouched = np.asarray([i for i in range(len(labels)) if i not in set(j)])
        with contextlib.redirect_stdout(io.StringIO()):
            ns["_apply_conditional_family_gate"](
                name, family, lam[weight_key], scores_are_logprob=logprob
            )
        post = ns["LOGIT"]
        family_err = float(np.max(np.abs(logsumexp(pre[:, j], axis=1) - logsumexp(post[:, j], axis=1))))
        outside_err = float(np.max(np.abs(pre[:, untouched] - post[:, untouched])))
        delta = np.max(np.abs(pre[:, j] - post[:, j]), axis=1)
        assert np.isfinite(post).all()
        assert family_err < 1e-12 and outside_err == 0.0
        assert np.all(delta[64:128] < 1e-10), f"{name} changed high-confidence family"
        assert np.any(delta[:64] > 1e-9), f"{name} gate never modified ambiguous rows"
        results[name] = {
            "family_log_mass_max_error": family_err,
            "outside_logits_max_error": outside_err,
            "changed_rows": int((delta > 1e-10).sum()),
            "high_confidence_rows_unchanged": int((delta[64:128] < 1e-10).sum()),
            "tested_rows": n,
        }

    # Construct an example where preserving total family probability does NOT
    # prevent the highest-scoring individual label from switching families.
    ex = np.full((1, len(labels)), -6.0)
    claude = idx["claude"]
    ex[0, claude] = [-0.01, -0.03, -6.0, -6.0, -6.0]
    ex[0, idx["gpt"][0]] = 0.03
    eg = {"claude": np.array([[5.0, -5.0, -5.0, -5.0, -5.0]])}
    ns2, _ = _bind(code, labels, ex, eg)
    with contextlib.redirect_stdout(io.StringIO()):
        ns2["_apply_conditional_family_gate"]("claude", "claude", lam["claude_style"])
    old_label = labels[int(ex[0].argmax())]
    new_label = labels[int(ns2["LOGIT"][0].argmax())]
    assert old_label.startswith("gpt") and new_label.startswith("claude"), (old_label, new_label)
    return {
        "source_sha256": digest,
        "scope": "exact published function, synthetic logits only, zero official data, no accuracy inference",
        "gate_tests": results,
        "crafted_counterexample": {
            "top_label_before": old_label,
            "top_label_after": new_label,
            "family_switch_despite_mass_preservation": True,
        },
        "verdict": "PASS_SYNTHETIC_MECHANISM_ONLY",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = audit(args.notebook)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
