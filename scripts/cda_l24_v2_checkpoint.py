"""Load SHA-pinned CDA-L24 v2 source-trained Stage-A checkpoint for official Stage-B.

Separate from the immutable v1 loader so historical weight pinning stays strict.
No source head is loaded at inference. Stage-A provenance and integrity are
validated by the caller BEFORE loading trainable model weights.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.cda_l24_v2_stage_a import (
    CANDIDATE, EXPECTED_POOL_SHA256, RESERVED_SOURCE,
)
from scripts.run_two_stage_lora24_gate_b_gpu import (
    CHECKPOINT_MANIFEST_SHA256, MODEL_ID, MODEL_REVISION,
    PEFT_VERSION, TRANSFORMERS_VERSION,
)
from scripts.run_two_stage_lora24_gate_a import truncate_qwen_layers
from scripts.run_two_stage_lora24_gate_b import gate_b_training_spec


def directory_manifest_digest(root: Path) -> str:
    """All files hashed; no trusting editable in-checkpoint manifest hashes."""
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = []
    for p in sorted(p for p in root.rglob("*") if p.is_file()):
        name=p.relative_to(root).as_posix()
        if name.endswith("manifest_digest.json"):
            raise ValueError("self-referential checkpoint digest not permitted")
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        entries.append({"path":name,"sha256":digest,"size":p.stat().st_size})
    if not entries:
        raise ValueError("empty checkpoint")
    return hashlib.sha256(json.dumps(
        entries,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def validate_cda_v2_checkpoint(
    root: Path, *, labels: Sequence[str], expected_sha256: str, arm: str,
    expected_source_manifest_sha256: str,
) -> dict[str, Any]:
    if len(expected_sha256)!=64 or len(expected_source_manifest_sha256)!=64:
        raise ValueError("full SHA256 checkpoint and clearance identity required")
    actual=directory_manifest_digest(root)
    if actual!=expected_sha256:
        raise RuntimeError("CDA Stage A checkpoint integrity mismatch")
    required=[root/"adapter/adapter_model.safetensors",
              root/"adapter/adapter_config.json",
              root/"heads.safetensors",root/"meta.json",
              root/"tokenizer"]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)
    meta=json.loads((root/"meta.json").read_text())
    required_fields={
        "candidate":CANDIDATE,"arm":arm,
        "source_pool_sha256":EXPECTED_POOL_SHA256,
        "base_stage1_sha256":CHECKPOINT_MANIFEST_SHA256,
        "source_transfer_reserved":RESERVED_SOURCE,
        "source_head_persisted":False,
        "provenance_manifest_sha256":expected_source_manifest_sha256,
    }
    for k,v in required_fields.items():
        if meta.get(k)!=v:
            raise RuntimeError(f"checkpoint origin mismatch: {k}")
    if meta.get("labels")!=list(labels) or len(labels)!=18:
        raise RuntimeError("checkpoint/official model vocabulary mismatch")
    if sorted(meta.get("source_ids",[]))!=sorted(
        ["arxitekton/ntl-claude-gen-v1","trnhip/ntl-api-gen"]):
        raise RuntimeError("v2 training source identity drift")
    stats=meta.get("run_stats",{})
    if (int(stats.get("crossed_labels",0))<8 or
        int(stats.get("transfer_labels_present",0))!=18 or
        stats.get("transfer_source")!=RESERVED_SOURCE):
        raise RuntimeError("v2 crossed label / independent original-prompt coverage failure")
    sha_map=meta.get("train_stats",{}).get("dataset_sha256",{})
    if set(sha_map)!=set(["arxitekton/ntl-claude-gen-v1",
                          "trnhip/ntl-api-gen",
                          "kilupy/ntl-18-model-corpus"]):
        raise RuntimeError("v2 original dataset fingerprint set mismatch")
    actual=hashlib.sha256(json.dumps(sha_map,sort_keys=True).encode()).hexdigest()
    if actual!=expected_source_manifest_sha256:
        raise RuntimeError("v2 original dataset source manifest digest mismatch")
    return meta


def load_cda_v2_stage1_checkpoint(
    *, checkpoint_dir:Path, snapshot_path:Path,
    official_labels:Sequence[str], device:Any, arm:str,
    expected_sha256:str, source_manifest_sha256:str,
) -> dict[str,Any]:
    """Load a *new* already trained C0/C1 checkpoint. CUDA-only science."""
    import importlib.util as importlib_util
    original_find=importlib_util.find_spec
    importlib_util.find_spec=lambda name,*args,**kwargs: (
       None if name.split(".")[0]=="torchao"
       else original_find(name,*args,**kwargs))
    import torch
    import peft
    import peft.import_utils as peft_import_utils
    import transformers
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    from torch import nn
    peft_import_utils.is_torchao_available=lambda *a,**kw:False

    if not torch.cuda.is_available():
        raise RuntimeError("CDA Stage B requires CUDA")
    if transformers.__version__!=TRANSFORMERS_VERSION:
        raise RuntimeError("transformers version drift")
    if peft.__version__!=PEFT_VERSION:
        raise RuntimeError("peft version drift")
    meta=validate_cda_v2_checkpoint(checkpoint_dir,labels=official_labels,
         expected_sha256=expected_sha256,arm=arm,
         expected_source_manifest_sha256=source_manifest_sha256)
    tokenizer=AutoTokenizer.from_pretrained(
         str(checkpoint_dir/"tokenizer"),local_files_only=True)
    tokenizer.padding_side="right"
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise RuntimeError("CDA tokenizer missing EOS")
    cfg=AutoConfig.from_pretrained(str(snapshot_path),local_files_only=True)
    spec=gate_b_training_spec()
    if cfg.model_type!="qwen3" or int(cfg.hidden_size)!=4096:
        raise RuntimeError("unexpected model config")
    if cfg.num_hidden_layers<int(spec["keep_layers"]):
        raise RuntimeError("model too short")
    cfg.num_hidden_layers=int(spec["keep_layers"])
    cfg.use_cache=False
    if getattr(cfg,"layer_types",None) is not None:
        cfg.layer_types=list(cfg.layer_types[:int(spec["keep_layers"])])
    base=AutoModel.from_pretrained(
         str(snapshot_path),config=cfg,
         torch_dtype=torch.float16,local_files_only=True,
         low_cpu_mem_usage=True).to(device)
    truncate_qwen_layers(base,keep_layers=int(spec["keep_layers"]))
    base.gradient_checkpointing_enable(
         gradient_checkpointing_kwargs={"use_reentrant":False})
    base.enable_input_require_grads()
    model=PeftModel.from_pretrained(base,str(checkpoint_dir/"adapter"),
                                   is_trainable=True)
    for p in model.parameters():
        if p.requires_grad:p.data=p.data.float()
    layers=[int(x) for x in spec["pooled_layers"]]
    hidden=int(cfg.hidden_size)
    norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in layers]).to(device).float()
    plain=nn.Linear(hidden*len(layers),len(official_labels)).to(device).float()
    centered=nn.Linear(hidden*len(layers),len(official_labels)).to(device).float()
    dropout=nn.Dropout(float(spec["head_dropout"]))
    saved=load_file(str(checkpoint_dir/"heads.safetensors"))
    for name,module in (("norms",norms),("plain",plain),("centered",centered)):
        state={k.removeprefix(name+"."):v for k,v in saved.items()
               if k.startswith(name+".")}
        if not state:
            raise RuntimeError(f"missing Stage A {name} weights")
        module.load_state_dict(state,strict=True)
    return dict(torch=torch,model=model,tokenizer=tokenizer,
                norms=norms,plain=plain,centered=centered,
                dropout=dropout,layers=layers,labels=list(official_labels),
                meta=meta,manifest_sha256=expected_sha256)
