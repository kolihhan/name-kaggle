"""Bounded REAL-data Colab T4 DCT-v3 LoRA training PILOT.

Not a scientific v3 transfer gate, not a Kaggle submission, not eligible
to update frozen v3 evaluation state. Trains only on the canonical frozen
external-source training split; heldout rows are never encoded or scored.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path("/content/name-kaggle-real-t4")
PINNED_COMMIT = "b86e8cb68631a18a2a4d1342658ecdca740a3f2b"
DATA = Path("/content/conservative_public_subset.parquet")
OUT = Path("/content/dct_v3_t4_real_train")
MAX_STEPS = 2  # Two six-triplet gradient-accumulation groups. NEVER a full epoch.


def check_no_official_data(root: Path):
    for path in ("data/train.csv", "data/test.csv", "data/sample_submission.parquet"):
        if (root / path).exists():
            raise RuntimeError(f"Official competition data unexpectedly checked out: {path}")


def setup():
    if not DATA.is_file():
        raise FileNotFoundError("Missing external Phase14 safe subset. Upload it with colab upload first.")
    if not (ROOT / ".git").is_dir():
        subprocess.run(["git", "init", str(ROOT)], check=True)
        subprocess.run(["git", "-C", str(ROOT), "remote", "add", "origin",
                        "https://github.com/kolihhan/name-kaggle.git"], check=True)
        subprocess.run(["git", "-C", str(ROOT), "sparse-checkout", "init", "--cone"], check=True)
        subprocess.run(["git", "-C", str(ROOT), "sparse-checkout", "set", "scripts"], check=True)
        subprocess.run(["git", "-C", str(ROOT), "fetch", "--depth=1", "--filter=blob:none",
                        "origin", PINNED_COMMIT], check=True)
        subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Source drift: {commit} != {PINNED_COMMIT}")
    check_no_official_data(ROOT)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==4.57.1", "peft==0.17.1", "tokenizers==0.22.1",
                    "huggingface-hub==0.36.0", "numpy", "pandas", "pyarrow",
                    "scipy", "scikit-learn", "safetensors", "accelerate"], check=True)
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def run():
    import random
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    # Frozen-worker compatibility; we don't use torchao.
    import importlib.util as iu
    old_find_spec = iu.find_spec
    iu.find_spec = lambda name, *a, **k: None if name.split(".")[0] == "torchao" else old_find_spec(name, *a, **k)
    from peft import LoraConfig, get_peft_model
    import peft.import_utils as pi
    pi.is_torchao_available = lambda *a, **k: False
    import transformers, peft

    from scripts.dct_v3_cpu_feasibility import _download_truncated_snapshot
    from scripts.dct_v3_transfer_split import build_transfer_split
    from scripts.run_two_stage_lora24_gate_a import (
        gate_a_training_spec, build_prompt_triplets, compute_class_weights, center_triplet
    )
    from scripts.dct_temporal_pool import masked_dct_pool

    if transformers.__version__ != "4.57.1" or peft.__version__ != "0.17.1":
        raise RuntimeError("Scientific source/runtime dependency drift")
    if not torch.cuda.is_available():
        raise RuntimeError("Colab did not provide CUDA")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram < 14:
        raise RuntimeError("T4 capacity below probed memory")
    device = torch.device("cuda:0")
    OUT.mkdir(exist_ok=True)
    started = time.time()

    safe = pd.read_parquet(DATA)
    split = build_transfer_split(safe)
    assert split["holdout_sources"] == ["alperen", "hailey", "willrice"]
    assert int(split["train_rows"]) == 14502
    assert int(split["heldout_rows"]) == 14506
    assert int(split["eligible_label_count"]) == 17
    assert split["prompt_id_overlap"] == 0 and split["prompt_key_overlap"] == 0
    assert split["selection_digest"] == "03edcd6d665d5d7992e777b1a83d88845260b06727bc8c7760e73355c0b0af29"
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    heldout_idx = set(int(x) for x in split["heldout_idx"])
    spec = gate_a_training_spec()
    assert int(spec["keep_layers"]) == 24 and int(spec["gradient_accumulation_triplets"]) == 6
    triplets = build_prompt_triplets(safe, train_idx, seed=int(spec["seed"]))
    groups = [triplets[i : i+6] for i in range(0, len(triplets), 6)]
    groups = groups[:MAX_STEPS]
    if len(groups) != MAX_STEPS or any(len(group) != 6 for group in groups):
        raise RuntimeError("Incorrect fixed optimizer group size")
    selected = sorted(set(int(r) for group in groups for batch in group for r in batch))
    if set(selected) & heldout_idx:
        raise RuntimeError("Heldout source leakage into training pilot")
    print(f"TRAINING_REAL_TEXT rows={len(selected)} microbatches={sum(map(len, groups))} steps={len(groups)}", flush=True)

    root, download = _download_truncated_snapshot(Path("/content/qwen3-dct-v3-real-training"))
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer EOS absent")
    tokens = {}
    for idx in selected:
        text = str(safe.iloc[idx]["text"])
        ids = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=int(spec["max_length_before_eos"]))["input_ids"]
        tokens[idx] = ids + [int(tokenizer.eos_token_id)]

    cfg = AutoConfig.from_pretrained(str(root), local_files_only=True)
    if cfg.model_type != "qwen3" or int(cfg.hidden_size) != 4096 or int(cfg.num_hidden_layers) != 36:
        raise RuntimeError("Unexpected Qwen3 config")
    cfg.num_hidden_layers = 24
    cfg.use_cache = False
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = list(cfg.layer_types[:24])

    def seed_all():
        seed=int(spec["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_all()
    base=AutoModel.from_pretrained(str(root), config=cfg, torch_dtype=torch.float16,
                                   local_files_only=True,low_cpu_mem_usage=True).to(device)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    base.enable_input_require_grads()
    seed_all()  # Frozen worker: head init is seeded before LoRA initialization.
    layers=[int(v) for v in spec["pooled_layers"]]
    norms=nn.ModuleList([nn.LayerNorm(4096) for _ in layers]).to(device).float()
    plain=nn.Linear(4096*len(layers)*2,18).to(device).float()
    centered=nn.Linear(4096*len(layers)*2,18).to(device).float()
    dropout=nn.Dropout(float(spec["head_dropout"]))
    lora=spec["lora"]
    model=get_peft_model(base,LoraConfig(
        r=int(lora["r"]),lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),bias=str(lora["bias"]),
        target_modules=list(lora["target_modules"])))
    for param in model.parameters():
        if param.requires_grad:
            param.data=param.data.float()
    adapter_params=[p for p in model.parameters() if p.requires_grad]
    head_params=list(norms.parameters())+list(plain.parameters())+list(centered.parameters())
    optimizer=torch.optim.AdamW([
        {"params":adapter_params,"lr":float(spec["adapter_lr"])},
        {"params":head_params,"lr":float(spec["head_lr"])},
    ],weight_decay=float(spec["weight_decay"]))
    total_steps=(len(triplets)+5)//6
    warmup=max(1,int(float(spec["warmup_fraction"])*total_steps))
    def lr_mult(step):
        if step < warmup:
            return (step + 1) / warmup
        import math
        return 0.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_mult)
    scaler=torch.amp.GradScaler("cuda",init_scale=256.0)

    labels=sorted(safe["model"].astype(str).unique().tolist())
    if len(labels)!=18:
        raise RuntimeError("Expected 18 source labels")
    train_labels=safe.iloc[train_idx]["model"].astype(str).to_numpy()
    weights=compute_class_weights(train_labels,classes=labels)
    weight_tensor=torch.tensor([weights[label] for label in labels],device=device,dtype=torch.float32)
    label_lookup={v:i for i,v in enumerate(labels)}

    def encode(batch):
        ids=[tokens[int(i)] for i in batch]
        batch_enc=tokenizer.pad({"input_ids":ids},return_tensors="pt")
        return {k:v.to(device) for k,v in batch_enc.items()}

    def representation(enc):
        with torch.autocast("cuda",dtype=torch.float16):
            hs=model(**enc,output_hidden_states=True,use_cache=False).hidden_states
        pooled=[]
        for l,norm in zip(layers,norms):
            features=masked_dct_pool(hs[l].float(),enc["attention_mask"],
                                     mask_first_token=True,n_modes=2)
            pooled.extend([norm(features[:,0]),norm(features[:,1])])
        return torch.cat(pooled,dim=-1)

    # Capture actual parameter difference to verify optimizer really updated.
    before=plain.weight.detach().clone()
    losses=[]
    completed_updates=0
    seed_all()
    model.train(); norms.train(); plain.train(); centered.train(); dropout.train()
    torch.cuda.reset_peak_memory_stats()
    for step, group in enumerate(groups):
        optimizer.zero_grad(set_to_none=True)
        count=len(group)
        for j, rows in enumerate(group):
            enc=encode(rows)
            z=representation(enc)
            targets=torch.tensor(
                [label_lookup[str(safe.iloc[int(i)]["model"])] for i in rows],
                device=device,dtype=torch.long)
            loss=F.cross_entropy(plain(dropout(z)),targets,weight=weight_tensor) + \
                 F.cross_entropy(centered(dropout(center_triplet(z))),targets,weight=weight_tensor)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite loss in real training")
            scaler.scale(loss/count).backward()
            val=float(loss.detach().cpu())
            losses.append(val)
            print(f"real_data step={step+1}/{MAX_STEPS} microbatch={j+1}/6 loss={val:.5f}",flush=True)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(adapter_params+head_params,float(spec["gradient_clip"]))
        prev=float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) >= prev:
            completed_updates+=1
        scheduler.step()
        print(f"optimizer_update={completed_updates} / requested={step+1}",flush=True)

    delta=float(torch.linalg.vector_norm(plain.weight.detach()-before).cpu())
    if completed_updates != MAX_STEPS or not delta>0:
        raise RuntimeError(f"Training did not update parameters: {completed_updates=}, {delta=}")
    checkpoint=OUT/"checkpoint"
    adapter_dir=checkpoint/"adapter"
    adapter_dir.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(adapter_dir,safe_serialization=True)
    head_state={}
    for name,mod in [("norms",norms),("plain",plain),("centered",centered)]:
        for key,v in mod.state_dict().items():
            head_state[f"{name}.{key}"]=v.detach().cpu().contiguous()
    save_file(head_state,str(checkpoint/"heads.safetensors"))
    result={
        "scope":"real_external_train_only_t4_lora_pilot",
        "scientific_decision_authorized":False,
        "kaggle_submission_authorized":False,
        "paired_v1_vs_v3_transfer_evaluated":False,
        "official_rows_scored":0,
        "competition_test_rows_used":0,
        "public_lb_tuning":False,
        "heldout_predictions_computed":False,
        "source_commit":PINNED_COMMIT,
        "backbone":"Qwen/Qwen3-8B-Base",
        "keep_layers":24,
        "dct_modes":[0,1],
        "trainable":"LoRA r16 plus norms/heads",
        "holdout_sources":split["holdout_sources"],
        "source_train_rows":int(split["train_rows"]),
        "unique_training_rows_seen":len(selected),
        "triplets_processed":len(losses),
        "optimizer_steps_completed":completed_updates,
        "loss_first":losses[0],
        "loss_last":losses[-1],
        "loss_mean":float(np.mean(losses)),
        "head_parameter_delta_l2":delta,
        "gpu":torch.cuda.get_device_name(0),
        "peak_reserved_gib":round(torch.cuda.max_memory_reserved()/1024**3,3),
        "elapsed_seconds":round(time.time()-started,2),
        "checkpoint_scope":"trainable-only weights, not exact-resumable optimizer trajectory",
        "download_shards":int(download["selected_shard_count"]),
    }
    (OUT/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    with zipfile.ZipFile("/content/dct_v3_t4_real_train_checkpoint.zip","w",
                         compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in checkpoint.rglob("*"):
            if p.is_file():
                z.write(p,p.relative_to(OUT))
        z.write(OUT/"result.json","result.json")
    print("TRAINING_PILOT_SUCCESS",json.dumps(result,sort_keys=True),flush=True)


if __name__ == "__main__":
    setup()
    run()
