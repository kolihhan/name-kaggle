#!/usr/bin/env python3
"""Frozen pretrained source-geometry study. Raw rows/caches stay outside Git."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_phase14_public_sparse import build_conservative_subset, sha256_file
from scripts.run_strong_sparse_anchor import load_train
from scripts.run_two_stage_lora24_gate_a import build_stage1_split

SEED = 20261108
TAU = 0.07
SPECS = {
    "simcse": ("princeton-nlp__unsup-simcse-roberta-base", "pytorch_model.bin",
                "25ff6c2ce68194d2b5636bc19f4f5556318fbd9636ff72f332e714e976deabcf"),
    "detective": ("heyongxin233__DeTeCtive", "Deepfake_best.pth",
                  "31a47928ea17f1fc872312e9896a0952c1dae48dba5660b9594ee5e05f2487e1"),
    "lusr": ("rrivera1849__LUSR", "model.safetensors",
             "9dc936bd31f349d874a79880549a5a9a9ca0a7ead64bebb83904cac2666cd5b9"),
}
FINGERPRINTS = {
    "official": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
    "folds": "94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d",
    "pool": "37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec",
}


def masked_mean(hidden, mask):
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("empty attention mask")
    return (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / mask.sum(dim=1, keepdim=True)


def compressed_pool(hidden, mask, logical_length=512):
    """Restore duplicate absolute-position padding queries without computing each."""
    import torch
    counts = mask.to(torch.long).sum(dim=1)
    width = hidden.shape[1]
    if (counts == 0).any() or (counts > logical_length).any():
        raise ValueError("empty or oversized token sequence")
    if not torch.equal(mask.bool(), torch.arange(width)[None, :] < counts[:, None]):
        raise ValueError("padding compression requires right padding")
    if ((counts < logical_length) & (counts >= width)).any():
        raise ValueError("missing representative padding query")
    total = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1)
    pad = hidden[torch.arange(len(hidden)), counts.clamp(max=width - 1)]
    return {"published": (total + pad * (logical_length - counts)[:, None]) / logical_length,
            "masked": total / counts[:, None]}


def text_fingerprint(texts) -> str:
    return hashlib.sha256(json.dumps(list(texts), ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def compatible_state(state, position_count):
    """Old Transformers persisted this deterministic buffer, not a learned weight."""
    import torch
    result = dict(state)
    positions = result.pop("embeddings.position_ids", None)
    if positions is not None:
        expected = torch.arange(position_count, dtype=positions.dtype).unsqueeze(0)
        if not torch.equal(positions, expected):
            raise ValueError("INVALID: noncanonical legacy position buffer")
    return result


def tokenizer_fingerprints(config_dir):
    files = ["vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]
    if (config_dir / "tokenizer.json").exists():
        files.append("tokenizer.json")
    return {fn: sha256_file(config_dir / fn) for fn in files}


def validate_cache(name, texts, arrays, meta, assets=None):
    expected = {"model": name, "weight_sha256": SPECS[name][2],
                "input_sha256": text_fingerprint(texts), "max_length": 512,
                "pooling": "fixed512-published-and-valid-masked-v1", "dtype": "float32",
                "compute_method": "duplicate-absolute-position-padding-compression-v1",
                "torch": "2.6.0+cpu", "transformers": "4.48.3", "rows": len(texts)}
    if assets is not None:
        config_dir = assets / SPECS["simcse" if name == "detective" else name][0]
        expected["config_sha256"] = sha256_file(config_dir / "config.json")
        expected["tokenizer_files"] = tokenizer_fingerprints(config_dir)
    if any(meta.get(k) != value for k, value in expected.items()):
        raise ValueError("INVALID: scoring cache provenance mismatch")
    for field in ("canonical_full512_max_delta", "batch_invariance_max_delta"):
        checks = meta.get(field, {})
        if set(checks) != {"published", "masked"} or any(
            not np.isfinite(v) or v < 0 or v > 1e-5 for v in checks.values()
        ):
            raise ValueError("INVALID: missing or failed real-checkpoint equivalence proof")
    for view in ("published", "masked"):
        x = arrays[view]
        if (x.shape != (len(texts), 768) or x.dtype != np.float32 or not np.isfinite(x).all()
                or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)):
            raise ValueError("INVALID: scoring cache shape/finiteness/unit-norm failure")


def kernel_class_scores(query, reference, y, labels, mode, tau=TAU):
    if mode not in {"density", "centroid"} or tau <= 0:
        raise ValueError("invalid density mode or temperature")
    similarity = np.asarray(query, dtype=np.float64) @ np.asarray(reference, dtype=np.float64).T / tau
    scores = np.empty((len(query), len(labels)))
    for j, label in enumerate(labels):
        mask = np.asarray(y) == label
        if not mask.any():
            raise ValueError(f"missing reference label: {label}")
        values = similarity[:, mask]
        scores[:, j] = (logsumexp(values, axis=1) - np.log(mask.sum())
                        if mode == "density" else values.mean(axis=1))
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite class evidence")
    return scores


def fold_reference_indices(frame, fold):
    if (frame.groupby("prompt_id").fold_id.nunique() != 1).any():
        raise ValueError("prompt spans reference/validation folds")
    train = np.flatnonzero(frame.fold_id.to_numpy() != fold)
    valid = np.flatnonzero(frame.fold_id.to_numpy() == fold)
    if not len(train) or not len(valid):
        raise ValueError("empty reference/validation partition")
    return train, valid


def align_anchor(frame, anchor):
    if anchor.id.duplicated().any() or set(anchor.id) != set(frame.id):
        raise ValueError("anchor id coverage mismatch")
    joined = frame[["id", "prompt_id", "model", "fold_id"]].merge(
        anchor, on="id", validate="one_to_one", how="left", suffixes=("", "_anchor"))
    for col in ("prompt_id", "model", "fold_id"):
        if not np.array_equal(joined[col].astype(str), joined[col + "_anchor"].astype(str)):
            raise ValueError(f"anchor {col} mismatch")
    return joined.plain_prediction.to_numpy(dtype=str)


def paired_prompt_interval(frame, candidate, anchor, replicates=5000):
    labels = sorted(frame.model.unique())
    groups, idx = np.unique(frame.prompt_id.astype(str), return_inverse=True)
    numerator = np.zeros((len(groups), len(labels)))
    denominator = np.zeros_like(numerator)
    y = frame.model.to_numpy()
    delta = (candidate == y).astype(float) - (anchor == y).astype(float)
    for j, label in enumerate(labels):
        mask = y == label
        np.add.at(numerator[:, j], idx[mask], delta[mask])
        np.add.at(denominator[:, j], idx[mask], 1)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(replicates):
        counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        den = counts @ denominator
        if np.any(den == 0):
            continue
        values.append(float(((counts @ numerator) / den).mean()))
    return {"prompt_groups": len(groups), "requested": replicates, "retained": len(values),
            "quantiles": [0.0125, 0.9875],
            "ci975": np.quantile(values, [0.0125, 0.9875]).tolist() if values else None}


def _normalized(text):
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", str(text)).lower().split())


def prepare_data(root, pool):
    for key, path in (("official", root / "data/train.csv"),
                      ("folds", root / "artifacts/phase1/folds.csv"), ("pool", pool)):
        if sha256_file(path) != FINGERPRINTS[key]:
            raise ValueError(f"{key} input fingerprint drift")
    official = load_train(root / "data/train.csv", root / "artifacts/phase1/folds.csv")
    safe = build_conservative_subset(pd.read_parquet(pool))
    split = build_stage1_split(safe)
    if set(split["holdout_sources"]) != {"abdullah", "semenoff", "zakir"}:
        raise ValueError("source selection drift")
    external = safe.iloc[split["heldout_idx"]].copy()
    prompt_ids = set(official.prompt_id.astype(str))
    answers = set(official.text.map(_normalized))
    keep = ~external.prompt_id.astype(str).isin(prompt_ids)
    keep &= ~external.prompt_key.astype(str).isin(prompt_ids)
    keep &= ~external.text.map(_normalized).isin(answers)
    excluded = int((~keep).sum())
    external = external.loc[keep].reset_index(drop=True)
    if len(official) != 2699 or official.model.nunique() != 18:
        raise ValueError("official label/row contract drift")
    for fold in range(5):
        fold_reference_indices(official, fold)
    return official, external, {"external_rows_before": int(len(keep)),
                               "excluded_exact_overlap": excluded,
                               "official_prompt_key_metadata": "unavailable; compare both external keys to official prompt_id",
                               "source_protocol_causal_control": False}


def encode(name, texts, assets, cache):
    import torch
    from transformers import AutoTokenizer, RobertaConfig, RobertaModel, __version__ as transformers_version
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    if torch.version.cuda is not None:
        raise RuntimeError("INVALID: frozen study requires CPU-only torch build")
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    directory, filename, digest = SPECS[name]
    weight_path = assets / directory / filename
    if sha256_file(weight_path) != digest:
        raise ValueError("INVALID: checkpoint weight fingerprint drift")
    config_dir = assets / (SPECS["simcse"][0] if name == "detective" else directory)
    meta = {"model": name, "weight_sha256": digest, "input_sha256": text_fingerprint(texts),
            "config_sha256": sha256_file(config_dir / "config.json"),
            "pooling": "fixed512-published-and-valid-masked-v1", "max_length": 512,
            "dtype": "float32", "torch": torch.__version__, "transformers": transformers_version,
            "compute_method": "duplicate-absolute-position-padding-compression-v1",
            "tokenizer_files": tokenizer_fingerprints(config_dir)}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            recorded = json.loads(str(z["meta"]))
            if any(recorded.get(k) != value for k, value in meta.items()):
                raise ValueError("INVALID: embedding cache provenance mismatch")
        print(f"Verified cache {name}", flush=True)
        return
    config = RobertaConfig.from_pretrained(config_dir, local_files_only=True)
    model = RobertaModel(config).to("cpu").eval()
    state = load_file(str(weight_path), device="cpu") if filename.endswith("safetensors") else torch.load(
        weight_path, map_location="cpu", weights_only=True)
    prefix = "model." if name == "detective" else ("roberta." if any(k.startswith("roberta.") for k in state) else "")
    mapped = {k[len(prefix):]: value for k, value in state.items() if not prefix or k.startswith(prefix)}
    mapped = compatible_state(mapped, config.max_position_embeddings)
    status = model.load_state_dict(mapped, strict=False)
    if status.missing_keys or status.unexpected_keys:
        raise ValueError(f"INVALID: checkpoint coverage {status}")
    ignored = sorted(set(state) - {prefix + k for k in mapped})
    if any(not k.startswith("mlp.") and k != prefix + "embeddings.position_ids" for k in ignored):
        raise ValueError("INVALID: unexpected checkpoint modules")
    tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    if (config.position_embedding_type != "absolute" or tokenizer.padding_side != "right"
            or tokenizer.pad_token_id != config.pad_token_id):
        raise ValueError("INVALID: absolute-position/right-padding compression assumptions fail")
    tokenized = tokenizer(texts, padding=False, truncation=True, max_length=512)
    lengths = np.array([len(row) for row in tokenized["input_ids"]])
    order = np.argsort(lengths, kind="stable")
    features = {k: np.empty((len(texts), config.hidden_size), dtype=np.float32)
                for k in ("published", "masked")}
    start, capped = time.monotonic(), 0
    with torch.inference_mode():
        for lo in range(0, len(texts), 16):
            rows = order[lo:lo + 16]
            examples = []
            for row in rows:
                example = {key: list(values[row]) for key, values in tokenized.items()}
                if lengths[row] < 512:
                    for key in example:
                        example[key].append(tokenizer.pad_token_id if key == "input_ids" else 0)
                examples.append(example)
            batch = tokenizer.pad(examples, padding=True, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            pooled = compressed_pool(hidden, batch["attention_mask"])
            for key, value in pooled.items():
                features[key][rows] = torch.nn.functional.normalize(value, dim=-1).numpy()
            capped += int((batch["attention_mask"].sum(dim=1) == 512).sum())
            if lo == 0:
                canonical_batch = tokenizer([texts[row] for row in rows], padding="max_length",
                                            truncation=True, max_length=512, return_tensors="pt")
                canonical_hidden = model(**canonical_batch).last_hidden_state
                canonical_pooled = {"published": canonical_hidden.mean(dim=1),
                                    "masked": masked_mean(canonical_hidden, canonical_batch["attention_mask"])}
                canonical_diff = {k: float(np.max(np.abs(
                    torch.nn.functional.normalize(v, dim=-1).numpy() - features[k][rows])))
                    for k, v in canonical_pooled.items()}
                if max(canonical_diff.values()) > 1e-5:
                    raise ValueError("INVALID: compressed representation differs from real full-512 model")
                meta["canonical_full512_max_delta"] = canonical_diff
                single = {k: v[:1] for k, v in canonical_batch.items()}
                single_hidden = model(**single).last_hidden_state
                single_pooled = {"published": single_hidden.mean(dim=1),
                                 "masked": masked_mean(single_hidden, single["attention_mask"])}
                batch_diffs = {k: float(np.max(np.abs(
                    torch.nn.functional.normalize(v, dim=-1).numpy() - features[k][rows[:1]])))
                    for k, v in single_pooled.items()}
                if max(batch_diffs.values()) > 1e-5:
                    raise ValueError("INVALID: real-checkpoint fixed-padding batch invariance failed")
                meta["batch_invariance_max_delta"] = batch_diffs
            if lo % 256 == 0 or lo + 16 >= len(texts):
                print(f"ENCODE {name} {min(lo + 16, len(texts))}/{len(texts)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    for key, array in features.items():
        if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5):
            raise ValueError("INVALID: nonfinite or nonunit embedding")
    meta.update({"elapsed_seconds": time.monotonic() - start, "capped_at_512": capped,
                 "ignored_non_encoder_modules": ignored, "rows": len(texts)})
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **features, meta=json.dumps(meta, sort_keys=True))
    print(f"ENCODE_COMPLETE {name} {meta['elapsed_seconds']:.1f}s", flush=True)
    del model, state, mapped
    gc.collect()


def _metrics(y, guess, labels, anchor=None):
    true_family = np.array([str(x).split("-", 1)[0] for x in y])
    pred_family = np.array([str(x).split("-", 1)[0] for x in guess])
    recall = recall_score(y, guess, labels=labels, average=None, zero_division=0)
    same_provider = true_family == pred_family
    wrong = y != guess
    sibling, cross = [], []
    for label in labels:
        mask = y == label
        if not mask.any():
            raise ValueError("INVALID: missing reported label support")
        sibling.append(float((wrong[mask] & same_provider[mask]).mean()))
        cross.append(float((wrong[mask] & ~same_provider[mask]).mean()))
    recalls = dict(zip(labels, recall.tolist()))
    families = sorted(set(true_family))
    result = {"ba": float(recall.mean()), "per_label_recall": recalls,
              "rows": len(y), "label_count": len(labels),
              "provider_ba": float(balanced_accuracy_score(true_family, pred_family)),
              "macro_sibling_error": float(np.mean(sibling)),
              "macro_cross_provider_error": float(np.mean(cross)),
              "provider_exact_macro_recall": {f: float(np.mean([r for label, r in recalls.items()
                                              if str(label).split("-", 1)[0] == f])) for f in families}}
    if anchor is not None:
        result.update({"rescues": int(((guess == y) & (anchor != y)).sum()),
                       "regressions": int(((guess != y) & (anchor == y)).sum()),
                       "disagreements": int((guess != anchor).sum())})
    return result


def _linear_scores(fit, y, query, labels):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced",
                          max_iter=2000, solver="lbfgs", random_state=SEED))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(fit, y)
    if list(model[-1].classes_) != labels:
        raise ValueError("INVALID: linear readout class order drift")
    return model.decision_function(query)


def evaluate(stage, official, external, isolation, anchor_path, cache_dir, output, predictions, assets):
    labels = sorted(official.model.unique())
    y = official.model.to_numpy(dtype=str)
    anchor = align_anchor(official, pd.read_csv(anchor_path))
    anchor_metrics = _metrics(y, anchor, labels)
    if abs(anchor_metrics["ba"] - 0.5467611235396471) > 1e-12:
        raise ValueError("INVALID: original LoRA24 OOF replay drift")
    models = ["simcse", "detective"] if stage == "hm" else ["simcse", "lusr"]
    embeddings, lineage = {}, {}
    expected_texts = official.text.astype(str).tolist() + external.text.astype(str).tolist()
    for name in models:
        with np.load(cache_dir / f"{name}.npz", allow_pickle=False) as z:
            lineage[name] = json.loads(str(z["meta"]))
            embeddings[name] = {view: z[view].copy() for view in ("published", "masked")}
            validate_cache(name, expected_texts, embeddings[name], lineage[name], assets)
    arms = ({f"{tag}_{view}": (name, mode, view) for tag, name, mode in
             [("B0", "simcse", "centroid"), ("B1", "simcse", "density"),
              ("M0", "detective", "centroid"), ("M1", "detective", "density")]
             for view in ["published", "masked"]} if stage == "hm" else
            {"R0": ("simcse", "linear", "masked"), "R1": ("lusr", "linear", "masked")})
    report = {"stage": stage, "official_rows": len(official), "external_rows": len(external),
              "isolation": isolation, "model_lineage": lineage, "anchor": anchor_metrics,
              "official": {}, "external": {}, "contract_sha256": sha256_file(
                  Path(__file__).resolve().parents[1] / "artifacts/candidates/pretrained-source-geometry-v1/CONTRACT.md")}
    saved, source_support = {}, {}
    for source, group in external.groupby("source"):
        counts = group.model.value_counts()
        eligible = sorted(k for k, n in counts.items() if n >= 20 and k in labels)
        if len(eligible) < 8 or int(group.model.isin(eligible).sum()) < 300:
            raise ValueError("INVALID: predeclared external source support unavailable")
        source_support[source] = eligible
        report["external"][source] = {"eligible_labels": eligible, "arms": {}}
    for arm, (name, mode, view) in arms.items():
        x = embeddings[name][view][:len(official)]
        ext = embeddings[name][view][len(official):]
        oof = np.full((len(official), 18), np.nan)
        for fold in range(5):
            fit, valid = fold_reference_indices(official, fold)
            oof[valid] = (_linear_scores(x[fit], y[fit], x[valid], labels) if mode == "linear"
                          else kernel_class_scores(x[valid], x[fit], y[fit], labels, mode))
        ext_scores = (_linear_scores(x, y, ext, labels) if mode == "linear" else
                      kernel_class_scores(ext, x, y, labels, mode))
        if not np.isfinite(oof).all() or not np.isfinite(ext_scores).all():
            raise ValueError("INVALID: incomplete 18-class scores")
        guess, ext_guess = np.array(labels)[oof.argmax(axis=1)], np.array(labels)[ext_scores.argmax(axis=1)]
        report["official"][arm] = _metrics(y, guess, labels, anchor)
        saved[f"pred_{arm}"], saved[f"scores_{arm}"] = guess, oof.astype(np.float32)
        saved[f"external_pred_{arm}"] = ext_guess
        for source, eligible in source_support.items():
            mask = (external.source.to_numpy() == source) & external.model.isin(eligible).to_numpy()
            report["external"][source]["arms"][arm] = _metrics(external.model.to_numpy()[mask], ext_guess[mask], eligible)
        print(f"SCORE {stage} {arm} OOF={report['official'][arm]['ba']:.6f}", flush=True)
    candidate, control = ("M1_published", "B1_published") if stage == "hm" else ("R1", "R0")
    cand = report["official"][candidate]
    interval = paired_prompt_interval(official, saved[f"pred_{candidate}"], anchor)
    deltas = {s: d["arms"][candidate]["ba"] - d["arms"][control]["ba"] for s, d in report["external"].items()}
    gates = {"beats_own_lora_by_one_point": cand["ba"] >= 0.5567611235396471,
             "paired_ci975_above_zero": bool(interval["ci975"] and interval["ci975"][0] > 0),
             "matched_control_two_points": cand["ba"] - report["official"][control]["ba"] >= .020,
             "provider_guard": cand["provider_ba"] >= anchor_metrics["provider_ba"] - .010,
             "label_guard": all(cand["per_label_recall"][label] >= anchor_metrics["per_label_recall"][label] - .10 for label in labels),
             "source_mean_two_points": float(np.mean(list(deltas.values()))) >= .020,
             "two_positive_sources": sum(v > 0 for v in deltas.values()) >= 2}
    if stage == "hm":
        gates["density_beats_centroid_half_point"] = cand["ba"] - report["official"]["M0_published"]["ba"] >= .005
        gates["masked_oof_gain_positive"] = report["official"]["M1_masked"]["ba"] > report["official"]["B1_masked"]["ba"]
        gates["masked_source_gain_positive"] = np.mean([d["arms"]["M1_masked"]["ba"] - d["arms"]["B1_masked"]["ba"] for d in report["external"].values()]) > 0
    report.update({"candidate": candidate, "control": control, "gates": {k: bool(v) for k, v in gates.items()},
                   "paired_interval": interval, "source_deltas": deltas,
                   "decision": "PROMOTE_PENDING_PREFLIGHT" if all(gates.values()) else "REJECT_CANDIDATE"})
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions, **saved, y=y, labels=np.array(labels), anchor=anchor,
                        external_y=external.model.to_numpy(dtype=str),
                        external_source=external.source.to_numpy(dtype=str))
    report["private_prediction_archive_sha256"] = sha256_file(predictions)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": stage, "candidate_ba": cand["ba"], "gates": report["gates"],
                      "decision": report["decision"], "ci975": interval["ci975"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["encode", "evaluate"])
    parser.add_argument("--model", choices=list(SPECS))
    parser.add_argument("--stage", choices=["hm", "hr"], default="hm")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    official, external, isolation = prepare_data(root, args.pool)
    if args.action == "encode":
        if args.model is None:
            parser.error("encode requires --model")
        encode(args.model, official.text.astype(str).tolist() + external.text.astype(str).tolist(),
               args.assets, args.cache / f"{args.model}.npz")
    else:
        if any(v is None for v in (args.anchor, args.output, args.predictions)):
            parser.error("evaluate requires --anchor, --output and --predictions")
        evaluate(args.stage, official, external, isolation, args.anchor, args.cache,
                 args.output, args.predictions, args.assets)


if __name__ == "__main__":
    main()
