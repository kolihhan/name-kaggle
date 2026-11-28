"""Private stage trace for a future authorized v36 diagnostic.
No scorer changes, training, submission, or public artifact export.
Call only inside a private diagnostic run, never on a public notebook.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile
import numpy as np

STAGES = ("native", "post_v1", "apex_refined", "final")
PIN = "5988d64922650bf94490810effb7278907b301c07f581efbc6d4865d84e63d52"

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def axis(a, name):
    a = np.asarray(a)
    if a.ndim != 1 or a.dtype.kind not in "US" or not len(a):
        raise ValueError("invalid " + name)
    a = a.astype(str)
    if len(set(a)) != len(a) or any(not s.strip() for s in a):
        raise ValueError("duplicate/empty " + name)
    return a

def hexsha(s):
    if not isinstance(s, str) or not re.fullmatch("[0-9a-f]{64}", s):
        raise ValueError("invalid SHA256")
    return s

class Trace:
    def __init__(self, root, ids, labels, *, source_sha, cohort_sha, asset_manifest_sha):
        if os.environ.get("NTL_PRIVATE_TRACE_AUTHORIZED") != "1":
            raise PermissionError("private diagnostic authorization required")
        # This location gate is not an attestation of notebook visibility.
        self.root = Path(root).resolve()
        self.ids, self.labels = axis(ids, "IDs"), axis(labels, "labels")
        if len(self.labels) != 18 or source_sha != PIN:
            raise ValueError("scorer/class contract mismatch")
        self.manifest = {"schema": "ntl-private-stage-v1", "source_sha": source_sha,
            "cohort_sha": hexsha(cohort_sha), "asset_manifest_sha": hexsha(asset_manifest_sha),
            "rows": len(self.ids), "labels": len(self.labels), "stages": {},
            "private_only": True, "promotion": False}
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)

    def write(self, name, **arrays):
        if name in self.manifest["stages"] or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("duplicate/invalid trace stage")
        p = self.root / (name + ".npz")
        fd, tmp = tempfile.mkstemp(suffix=".npz", dir=self.root)
        os.close(fd)
        try:
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, p)
        finally:
            Path(tmp).unlink(missing_ok=True)
        self.manifest["stages"][name] = {"file": p.name, "sha256": digest(p),
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in arrays.items()}}

    def scores(self, name, ids, labels, logits, *, semantics):
        if name not in ("native", "post_v1", "apex_refined") and not name.startswith("expert_"):
            raise ValueError("unknown scores stage")
        if semantics not in ("logits", "log_probability", "weighted_log_score"):
            raise ValueError("unknown score semantics")
        ids, labels = axis(ids, "IDs"), axis(labels, "labels")
        z = np.asarray(logits)
        if set(ids) != set(self.ids) or set(labels) != set(self.labels):
            raise ValueError("full stage axes mismatch")
        if z.shape != (len(ids), 18) or z.dtype.kind != "f" or not np.isfinite(z).all():
            raise ValueError("invalid scores")
        ri = {v:i for i,v in enumerate(ids)}; ci = {v:i for i,v in enumerate(labels)}
        z = z[np.ix_([ri[v] for v in self.ids], [ci[v] for v in self.labels])]
        self.write(name, ids=self.ids, labels=self.labels, logits=z, semantics=np.array(semantics))

    def capture_npz(self, name, path, *, semantics):
        with np.load(path, allow_pickle=False) as a:
            self.scores(name, a["ids"], a["labels"], a["logits"], semantics=semantics)

    def mask(self, name, ids, values):
        if name not in ("v1_selected", "apex_protection", "typography_changed"):
            raise ValueError("unknown mask")
        ids = axis(ids, "IDs"); v = np.asarray(values)
        if set(ids) != set(self.ids) or v.shape != (len(ids),) or v.dtype.kind != "b":
            raise ValueError("mask axis mismatch")
        ix = {x:i for i,x in enumerate(ids)}
        self.write(name, ids=self.ids, mask=v[[ix[x] for x in self.ids]])

    def final(self, ids, predictions):
        ids = axis(ids, "IDs"); p = np.asarray(predictions)
        if set(ids) != set(self.ids) or p.shape != ids.shape or p.dtype.kind not in "US":
            raise ValueError("final axis mismatch")
        if not set(p.astype(str)) <= set(self.labels):
            raise ValueError("unknown final class")
        ix = {x:i for i,x in enumerate(ids)}
        self.write("final", ids=self.ids, predictions=p[[ix[x] for x in self.ids]].astype(str))

    def seal(self, *, v1_status, runtime, independent_replay_verified=False):
        if v1_status not in ("applied", "skipped", "failed"):
            raise ValueError("unknown V1 outcome")
        s = self.manifest["stages"]
        required = {"native", "apex_refined", "final"}
        if v1_status == "applied":
            required |= {"post_v1", "v1_selected", "apex_protection"}
        if not required <= set(s):
            raise ValueError("incomplete trace: " + ",".join(sorted(required-set(s))))
        if set(runtime) != {"seconds", "peak_memory_bytes", "fallback_count"} or any(
            not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in runtime.values()):
            raise ValueError("invalid runtime")
        for a in s.values():
            if digest(self.root / a["file"]) != a["sha256"]:
                raise ValueError("trace changed")
        self.manifest.update(v1_status=v1_status, runtime=runtime,
                             independent_replay_verified=bool(independent_replay_verified))
        p = self.root / "manifest.json"
        with p.open("x") as f:
            json.dump(self.manifest, f, sort_keys=True)
        # This is only a local receipt, not independent validation or a promotion decision.
        return {"stage_count": len(s), "rows": len(self.ids), "v1_status": v1_status,
                "replay_verified": bool(independent_replay_verified), "promotion": False}
