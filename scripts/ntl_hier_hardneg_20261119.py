"""Preregistered prompt-disjoint hierarchical and sibling hard-negative diagnostic.

NO public LB / Oleg / external source or inference-stage training.
This script uses the historically reused official TRAIN group folds and therefore
does NOT measure the prefit Apex 0.702 incumbent's out-of-fold generalization.
Only aggregate results; do not write row predictions or original passages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import encode_fold, load_snapshot

SEED = 20261119
STEPS = 120
PROMPTS_PER_STEP = 12
LR = 0.003
WEIGHT_DECAY = 0.01
CONTRAST_WEIGHT = 0.15
TEMP = 0.2
BOOTSTRAPS = 2000
ARMS = ("flat", "hier", "hardneg")
FAMILIES = ("claude", "gemini", "glm", "gpt", "kimi", "qwen")


def family(label: str) -> str:
    for fam in FAMILIES:
        if label.startswith(fam + "-") or label.startswith(fam):
            return fam
    raise ValueError("Unrecognized exact competition label: " + label)


def define_network(input_dim, label_families):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Attribution(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.GELU(), nn.LayerNorm(128)
            )
            self.flat = nn.Linear(128, len(label_families))
            self.family = nn.Linear(128, len(FAMILIES))
            self.sibling = nn.Linear(128, len(label_families))
            self.register_buffer("label_families", torch.tensor(label_families, dtype=torch.long))

        def forward(self, x, arm):
            z = self.encoder(x)
            if arm == "flat":
                return self.flat(z), z
            fam = F.log_softmax(self.family(z), dim=1)
            child = self.sibling(z)
            sibling_logp = torch.zeros_like(child)
            for fi in range(len(FAMILIES)):
                col = torch.nonzero(self.label_families == fi, as_tuple=False).flatten()
                sibling_logp[:, col] = F.log_softmax(child[:, col], dim=1)
            return fam[:, self.label_families] + sibling_logp, z

    return Attribution()


def contrastive_loss(z, labels, label_families, prompt_codes):
    """Positive: same true model across prompts.
    Negatives: all other models; sibling x2, same-prompt sibling x3.
    Uses the train-only prompt codes; no prompt metadata at test time.
    """
    import torch
    import torch.nn.functional as F

    n = len(labels)
    if n < 2:
        raise ValueError("Degenerate batch")
    z = F.normalize(z, p=2, dim=1)
    sim = (z @ z.T) / TEMP
    diag = torch.eye(n, dtype=torch.bool, device=z.device)
    same_label = labels[:, None].eq(labels[None, :])
    same_family = label_families[:, None].eq(label_families[None, :])
    same_prompt = prompt_codes[:, None].eq(prompt_codes[None, :])
    positive = same_label & ~same_prompt & ~diag
    sibling = same_family & ~same_label
    weights = torch.where(sibling & same_prompt, 3.0,
                          torch.where(sibling, 2.0, 1.0)).to(sim.dtype)
    logits = (sim + torch.log(weights)).masked_fill(diag, -1e6)
    numerator = sim.masked_fill(~positive, -1e6).logsumexp(dim=1)
    denominator = logits.logsumexp(dim=1)
    available = positive.any(dim=1)
    if not bool(available.any()):
        raise RuntimeError("No cross-prompt same-model positives in batch")
    return (denominator - numerator)[available].mean()


def fold_batches(prompt_ids, seed):
    rng = np.random.default_rng(seed)
    groups = np.unique(prompt_ids)
    lookup = {group: np.flatnonzero(prompt_ids == group) for group in groups}
    if len(groups) < PROMPTS_PER_STEP:
        raise ValueError("Too few train prompt groups")
    for _ in range(STEPS):
        selected = rng.choice(groups, size=PROMPTS_PER_STEP, replace=False)
        yield np.concatenate([lookup[gid] for gid in selected])


def train_arm(x, y, class_to_family, prompts, batches, arm, seed):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    model = define_network(x.shape[1], class_to_family)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    features = torch.from_numpy(x.astype(np.float32, copy=False))
    target = torch.from_numpy(y.astype(np.int64, copy=False))
    group_names, group_ids = np.unique(prompts, return_inverse=True)
    group_t = torch.from_numpy(group_ids.astype(np.int64))
    fam_t = torch.tensor(class_to_family, dtype=torch.long)
    final = None
    for step, ids in enumerate(batches, 1):
        xb = features[ids]
        yb = target[ids]
        scores, z = model(xb, arm)
        primary = F.nll_loss(scores, yb) if arm != "flat" else F.cross_entropy(scores, yb)
        loss = primary
        if arm == "hardneg":
            loss = loss + CONTRAST_WEIGHT * contrastive_loss(
                z, yb, fam_t[yb], group_t[ids]
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        final = float(loss.detach())
    if step != STEPS:
        raise AssertionError("Optimizer steps drifted")
    return model, {"optimizer_steps":step, "last_train_loss":round(final, 5),
                   "train_prompt_groups":len(group_names),
                   "train_rows":len(x)}


def paired_bootstrap(frame, labels, predictions):
    """Re-sample entire heldout prompt groups, stratified by the five folds."""
    rng = np.random.default_rng(SEED)
    prompts = np.unique(frame.prompt_id.to_numpy())
    nclasses = len(np.unique(labels))
    groups = {p: np.flatnonzero(frame.prompt_id.to_numpy() == p) for p in prompts}
    totals = np.zeros((len(prompts), nclasses), dtype=np.int64)
    changes = {arm: np.zeros_like(totals) for arm in ("flat", "hier")}
    for j, prompt in enumerate(prompts):
        ids = groups[prompt]
        for cl in range(nclasses):
            mask = labels[ids] == cl
            totals[j, cl] = int(mask.sum())
            for baseline in changes:
                delta = ((predictions["hardneg"][ids] == labels[ids]).astype(int) -
                         (predictions[baseline][ids] == labels[ids]).astype(int))
                changes[baseline][j, cl] = int(delta[mask].sum())
    fold_for_prompt = np.array([int(frame.fold_id.iloc[groups[p][0]]) for p in prompts])
    chunks = [np.flatnonzero(fold_for_prompt == fold) for fold in range(5)]
    if any(len(chunk) != 30 for chunk in chunks):
        raise ValueError("Expected 30 distinct prompts for each fold")
    drawn = np.concatenate(
        [rng.choice(chunk, size=(BOOTSTRAPS, 30), replace=True) for chunk in chunks],
        axis=1
    )
    denominator = totals[drawn].sum(axis=1)
    if not (denominator > 0).all():
        raise ValueError("Bootstrap omitted true class in a replicate")
    out = {}
    for arm, delta in changes.items():
        replicated = (delta[drawn].sum(axis=1) / denominator).mean(axis=1)
        out[arm] = np.quantile(replicated, [.025, .975]).round(6).tolist()
    return out


def family_error_breakdown(y, pred, class_to_family):
    bad = pred != y
    family_correct = class_to_family[pred] == class_to_family[y]
    return {
        "wrong_total": int(bad.sum()),
        "wrong_cross_family": int((bad & ~family_correct).sum()),
        "wrong_within_family": int((bad & family_correct).sum()),
    }


def summary(frame, classes, target, predictions, train_logs):
    family_id = np.array([FAMILIES.index(family(name)) for name in classes])
    ba = {key: float(balanced_accuracy_score(target, val)) for key, val in predictions.items()}
    intervals = paired_bootstrap(frame, target, predictions)
    comparisons = {}
    for before in ("flat", "hier"):
        old_correct = predictions[before] == target
        new_correct = predictions["hardneg"] == target
        old_family_good = family_id[predictions[before]] == family_id[target]
        new_family_good = family_id[predictions["hardneg"]] == family_id[target]
        comparisons["hardneg_minus_" + before] = {
            "delta_ba": round(ba["hardneg"] - ba[before], 6),
            "prompt_cluster_ci95": intervals[before],
            "rescues": int((~old_correct & new_correct).sum()),
            "regressions": int((old_correct & ~new_correct).sum()),
            "previously_within_family_rescues": int((~old_correct & old_family_good & new_correct).sum()),
            "new_within_family_regressions": int((old_correct & ~new_correct & new_family_good).sum()),
            "family_rescues": int((~old_family_good & new_family_good).sum()),
            "family_regressions": int((old_family_good & ~new_family_good).sum()),
        }
    family_error = {arm:family_error_breakdown(target, pred, family_id)
                    for arm, pred in predictions.items()}
    primary = comparisons["hardneg_minus_hier"]
    secondary = comparisons["hardneg_minus_flat"]
    within_net = (primary["previously_within_family_rescues"] -
                  primary["new_within_family_regressions"])
    pass_gate = (primary["delta_ba"] >= .03 and
                 primary["prompt_cluster_ci95"][0] > 0 and
                 family_error["hardneg"]["wrong_cross_family"] <=
                     family_error["hier"]["wrong_cross_family"] and
                 within_net > 0 and ba["hier"] >= ba["flat"])
    folds = []
    for i in range(5):
        mask = frame.fold_id.to_numpy() == i
        folds.append({
            "fold":i, "n":int(mask.sum()),
            "n_prompt_groups":int(frame.prompt_id[mask].nunique()),
            "ba":{arm:round(float(balanced_accuracy_score(target[mask], p[mask])),6)
                  for arm,p in predictions.items()}
        })
    perclass = {}
    for i, label in enumerate(classes):
        mask=target==i
        perclass[str(label)]={
            "n":int(mask.sum()),
            "recall":{arm:round(float((p[mask]==target[mask]).mean()),6)
                      for arm,p in predictions.items()},
        }
    return {
        "experiment":"NTL-20261119-HIER-HARDNEG-CPU-V0",
        "scientific_scope":"reused_official_train_prompt_OOF_not_independent_transfer",
        "incumbent_official_public_ba":0.702,
        "incumbent_OOF":"UNAVAILABLE_and_NOT_invented",
        "rows":len(frame), "prompts":int(frame.prompt_id.nunique()),
        "classes":len(classes),
        "arms":list(ARMS),
        "fixed_training": {
            "optimizer_steps_per_fit":STEPS, "prompt_groups_per_step":PROMPTS_PER_STEP,
            "learning_rate":LR,"weight_decay":WEIGHT_DECAY,
            "contrastive_weight":CONTRAST_WEIGHT,"contrastive_temperature":TEMP,
            "same_architecture_hier_vs_hardneg":True,
            "feature_encoder":"train-fold-only char TFIDF 2-5 -> SVD256 (fixed) -> trainable MLP128",
            "no_external_data":True,"no_gpu":True,"no_official_submission":True,
            "no_promotion_from_reused_folds":True
        },
        "BA":{arm:round(score,6) for arm,score in ba.items()},
        "comparisons":comparisons,
        "family_errors":family_error,
        "by_fold":folds,"by_true_class":perclass,
        "train_log":train_logs,
        "decision":"MECHANISM_GATE_PASS_DEV_ONLY" if pass_gate else "REJECT_FIXED_HARDNEG_OBJECTIVE",
        "promotion_allowed":False,
        "independent_external_transfer":False,
    }


def run(train_path: Path, fold_path: Path):
    import torch
    torch.set_num_threads(2)
    frame = load_snapshot(train_path, fold_path)
    classes, y = np.unique(frame.model.to_numpy(), return_inverse=True)
    if len(classes) != 18:
        raise AssertionError("Unexpected labels")
    labels_family = np.array([FAMILIES.index(family(label)) for label in classes])
    pred = {arm: np.full(len(frame), -1, dtype=np.int64) for arm in ARMS}
    training = []
    for fold in range(5):
        test_mask = frame.fold_id.to_numpy() == fold
        train_mask = ~test_mask
        with threadpool_limits(limits=2):
            x_train, x_val = encode_fold(
                frame.text[train_mask].astype(str),
                frame.text[test_mask].astype(str)
            )
        y_train = y[train_mask]
        prompt_train = frame.prompt_id.to_numpy()[train_mask].astype(str)
        batches = list(fold_batches(prompt_train, SEED + fold))
        current = {"fold":fold}
        for arm in ARMS:
            model, tr = train_arm(
                x_train,y_train,labels_family,prompt_train,batches,arm,SEED+fold
            )
            model.eval()
            with torch.no_grad():
                scores,_ = model(torch.from_numpy(x_val.astype(np.float32,copy=False)),arm)
                pred[arm][test_mask] = scores.argmax(dim=1).cpu().numpy()
            current[arm] = tr
        training.append(current)
        print("NTL_HIER_HARDNEG_FOLD_COMPLETE=" + json.dumps({
            "fold":fold,"n_eval":int(test_mask.sum()),"n_train":int(train_mask.sum()),
            "n_prompts_eval":int(frame.prompt_id[test_mask].nunique())
        }),flush=True)
    if any((p < 0).any() for p in pred.values()):
        raise RuntimeError("Missing holdout predictions")
    return summary(frame,classes,y,pred,training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",type=Path,default=Path("data/train.csv"))
    parser.add_argument("--folds",type=Path,default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = run(args.train,args.folds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("NTL_HIER_HARDNEG_FINAL_RESULT="+json.dumps({
        "BA":result["BA"], "comparisons":result["comparisons"],
        "family_errors":result["family_errors"],
        "decision":result["decision"],
        "promotion_allowed":False
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
