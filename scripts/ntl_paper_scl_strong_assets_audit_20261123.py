"""Audit whether 0.715 strong frozen embeddings are reusable for a paper SCL readout.

Kaggle source only (not weights or data); print aggregates and relevant
declared dataset/kernel mounts. No API money, inference, training or submission.
"""
import hashlib,json,re,subprocess,tempfile
from pathlib import Path

REF="kolihhan/ntl-estopadilla-glm53-0716-repro-20261122"
KEYS=("embed","featur","cache","hidden","train","fold","logit","checkpoint","oof")
with tempfile.TemporaryDirectory(prefix="ntl-supcon-asset-") as tmp:
    root=Path(tmp)
    p=subprocess.run(["kaggle","kernels","pull",REF,"-p",str(root),"-m"],capture_output=True,text=True,timeout=130)
    if p.returncode:raise RuntimeError("owner strong code unavailable "+str(p.returncode))
    md=json.loads((root/"kernel-metadata.json").read_text())
    nb=json.loads((root/md["code_file"]).read_text())
    cells=["".join(c.get("source",[])) if isinstance(c.get("source"),list)
           else str(c.get("source","")) for c in nb["cells"] if c.get("cell_type")=="code"]
    sha=hashlib.sha256("\n".join(cells).encode()).hexdigest()
    if sha!="202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8":
        raise RuntimeError("owner 0.715 source SHA drift")
    files={}
    for cell in cells:
        if not cell.startswith("%%writefile "):continue
        f=cell.splitlines()[0].split()[-1]
        src=cell.split("\n",1)[1]
        files[f]=src
    print("NTL_SUPCON_STRONG_ASSET_MANIFEST="+json.dumps({
      "score_source_sha":sha,
      "model_sources":md.get("model_sources"),
      "dataset_sources":md.get("dataset_sources"),
      "kernel_sources":md.get("kernel_sources"),
      "embedded_writefiles":sorted(files),
      "no_scoring_or_training":True
    },sort_keys=True),flush=True)
    for path,source in files.items():
        matches=[]
        for i,line in enumerate(source.splitlines(),1):
            lo=line.lower()
            if any(t in lo for t in ("savez","save(", "load(", "embedding", "hidden_states",
                "npz", "oof", "kaggle/input", "train.csv","folds.csv","l24_features","features(","features[","cached")):
                matches.append({"n":i,"code":line.strip()[:210]})
        if matches:
            print("NTL_SUPCON_STRONG_EMBED_DECLARATIONS="+json.dumps({
                "path":path,"matches":matches[:55],
                "truncated":len(matches)>55
            },sort_keys=True),flush=True)
    # Cells executing framework often inline calls, not %%writefile.
    for i,cell in enumerate(cells):
        matches=[]
        for ln,line in enumerate(cell.splitlines(),1):
            ll=line.lower()
            if any(q in ll for q in ("final-logits","savez", "train.csv", "train_final",
                 "l24_features", "encoder", "writefile", "np.save", "npz")):
                matches.append({"n":ln,"code":line[:150]})
        if matches:
            print("NTL_SUPCON_NOTEBOOK_GUARD="+json.dumps({
                "cell":i,"matches":matches[:28],
                "truncated":len(matches)>28},sort_keys=True),flush=True)
