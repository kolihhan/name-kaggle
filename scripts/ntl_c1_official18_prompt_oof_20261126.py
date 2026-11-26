"""One-shot 18-class prompt-grouped C1 mechanism test on official TRAIN only.

Frozen E5-large-v2 335M encoder + equal-capacity CE control vs CE+prompt-
matched contrastive projector. Results are TRAIN DEV, not Kaggle Public.
"""
from __future__ import annotations
import hashlib, io, json, os, subprocess, tempfile, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from scripts.ntl_c1_prompt_contrastive_20261126 import prompt_group_splits, train_pair
from scripts.ntl_kimi_e5_frozen_transfer_20261126 import embed, MODEL, REVISION

SEED = 20261126
FOLDS = 3
EPOCHS = 8
PROJ_DIM = 256
ALPHA = 0.2
TARGET_BA = 0.5
MIN_DELTA_BA = 0.025


def log(name, **data):
    print(name + "=" + json.dumps(data, sort_keys=True, default=str), flush=True)


def load_official():
    assert os.getenv("KAGGLE_API_TOKEN")
    with tempfile.TemporaryDirectory(prefix="ntl-c1-data-") as td:
        proc = subprocess.run(
            ["kaggle", "competitions", "download", "name-that-llm", "-p", td],
            capture_output=True, text=True, timeout=180
        )
        if proc.returncode:
            raise RuntimeError("Authenticated competition train download unavailable")
        files = list(Path(td).glob("*.zip"))
        assert len(files) == 1
        with zipfile.ZipFile(files[0]) as z:
            df = pd.read_csv(io.BytesIO(z.read("train.csv")))
    assert len(df) == 2699 and df.prompt_id.nunique() == 150 and df.model.nunique() == 18
    assert {"prompt_id", "text", "model"}.issubset(df.columns)
    assert df.text.notna().all() and df.model.notna().all()
    return df


def run():
    df = load_official()
    label_names = sorted(df.model.unique().tolist())
    class_id = {value: i for i, value in enumerate(label_names)}
    y = df.model.map(class_id).to_numpy(dtype=np.int64)
    prompt = pd.factorize(df.prompt_id, sort=True)[0].astype(np.int64)
    counts = df.groupby("prompt_id").model.nunique()
    assert int(counts.min()) == 17 and int(counts.max()) == 18
    folds = prompt_group_splits(prompt, y, n_splits=FOLDS)
    log(
        "NTL_C1_FROZEN_PROTOCOL",
        n=int(len(df)), classes=len(label_names),
        prompt_groups=int(len(np.unique(prompt))),
        e5_model=MODEL, e5_revision=REVISION, encoder_trainable=False,
        trainable_projection_head=True, fold_count=FOLDS, epochs=EPOCHS,
        projection_dim=PROJ_DIM, contrastive_weight=ALPHA,
        loss="CE + 0.2 cross-prompt model-identity InfoNCE with same-prompt negatives x2",
        official_train_only=True, source_disjoint_transfer_proven=False,
        independent_validation=False,
        primary_metric="18-class balanced accuracy prompt-group OOF",
        advance_minimum_absolute_ba=TARGET_BA,
        advance_minimum_delta_vs_same_capacity_CE=MIN_DELTA_BA,
        github_cpu=True, no_kaggle_gpu=True, no_kaggle_submission=True,
    )
    t0 = time.monotonic()
    z = embed(df.text.astype(str).tolist())
    assert z.shape == (len(df), 1024) and np.isfinite(z).all()
    log(
        "NTL_C1_EMBEDDING_READY",
        shape=list(z.shape),
        float32_sha=hashlib.sha256(z.tobytes()).hexdigest(),
        seconds=round(time.monotonic() - t0, 1),
    )
    ce = np.full(len(y), -1, dtype=np.int64)
    c1 = np.full(len(y), -1, dtype=np.int64)
    for k, (train, valid) in enumerate(folds):
        assert not (set(prompt[train]) & set(prompt[valid]))
        assert len(set(y[train])) == 18 and len(set(y[valid])) == 18
        a, b = train_pair(
            z, y, prompt, train, valid,
            epochs=EPOCHS, dim=PROJ_DIM, seed=SEED + k, alpha=ALPHA,
        )
        assert a.shape == b.shape == (len(valid), 18)
        ce[valid] = a.argmax(1)
        c1[valid] = b.argmax(1)
        log(
            "NTL_C1_FOLD_RESULT", fold=k, train_rows=len(train),
            heldout_rows=len(valid),
            heldout_groups=int(np.unique(prompt[valid]).size),
            ce_ba=float(balanced_accuracy_score(y[valid], ce[valid])),
            contrast_ba=float(balanced_accuracy_score(y[valid], c1[valid])),
        )
    assert np.all(ce >= 0) and np.all(c1 >= 0)
    ba0 = float(balanced_accuracy_score(y, ce))
    ba1 = float(balanced_accuracy_score(y, c1))
    rescues = int(np.sum((ce != y) & (c1 == y)))
    regressions = int(np.sum((ce == y) & (c1 != y)))
    rng = np.random.default_rng(SEED)
    unique_prompts = np.unique(prompt)
    indexes = {value: np.flatnonzero(prompt == value) for value in unique_prompts}
    boots = []
    for _ in range(1000):
        select = np.concatenate([
            indexes[g] for g in rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
        ])
        boots.append(
            float(balanced_accuracy_score(y[select], c1[select]) -
                  balanced_accuracy_score(y[select], ce[select]))
        )
    lower, upper = np.quantile(boots, [.025, .975])
    labels = {
        lab: {
            "n": int(np.sum(y == i)),
            "CE_recall": round(float(np.mean(ce[y == i] == i)), 4),
            "C1_recall": round(float(np.mean(c1[y == i] == i)), 4),
        }
        for i, lab in enumerate(label_names)
    }
    passes = (
        ba1 >= TARGET_BA and ba1 - ba0 >= MIN_DELTA_BA and lower > 0
        and rescues > regressions
    )
    log(
        "NTL_C1_OFFICIAL_DEV_RESULT",
        ce_only_ba=ba0, contrastive_ba=ba1, delta_ba=ba1 - ba0,
        paired_rescues=rescues, paired_regressions=regressions,
        prompt_bootstrap95_delta=[float(lower), float(upper)],
        per_class=labels, gate_pass=bool(passes),
        gate_reason="CONTINUE_TO_STRONG_ENCODER_ONLY_IF_TRUE",
        kaggle_public_score=None, not_a_new_submission=True,
        total_elapsed_seconds=round(time.monotonic() - t0, 1),
    )


if __name__ == "__main__":
    run()
