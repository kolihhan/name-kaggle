"""Second source-only pass: inspect high-code decision routing instead of title claims.

No fit, inference, model weights, datasets, prompts, or submission. Print bounded
public source excerpts only for threshold/guard evidence (not full notebook code).
"""
from __future__ import annotations
import ast
import difflib
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

REFS = {
    "dennis": "dennis6886/glm-user-prefix-guarded-fable-return",
    "sanpier_v3": "sanpier/0-709-320-token-glm-likelihood-qwen-ensemble/3",
    "sanpier_latest": "sanpier/0-709-320-token-glm-likelihood-qwen-ensemble",
    "areeb_0719": "areebbb/0-719-preserve-glm-to-fable-corrections",
    "hcluo_0719": "hcluo8888/ntl-0719-glm-prefix-fable-gate",
    "ranjeet": "ranjeet258/name-that-llm-sibling-odds",
}
MARK = re.compile(r"fable_return|glm|likelihood|prefix|guard|protect|deadline|native|comp_stack|score|margin|confidence|threshold",re.I)
EXCLUDE = re.compile(r"^(?:#|import |from |print\s*\(|plt\.|ax\.|fig\.|colors|palette)",re.I)
def event(tag,o):print("NTL_DECISIONCODE_"+tag+"="+json.dumps(o,sort_keys=True,ensure_ascii=False),flush=True)

def pull(ref,dest):
    dest.mkdir()
    try:
        r=subprocess.run(["kaggle","kernels","pull",ref,"-m","-p",str(dest)],
            capture_output=True,text=True,timeout=100)
    except Exception as e:return {"failed":type(e).__name__}
    if r.returncode:return {"failed":"CLI_RETURN_NONZERO","code":r.returncode}
    md=json.loads((dest/"kernel-metadata.json").read_text())
    nb=json.loads((dest/md["code_file"]).read_text())
    cs=[''.join(c.get("source",[])) if isinstance(c.get("source",[]),list)
        else str(c.get("source","")) for c in nb["cells"] if c.get("cell_type")=="code"]
    parts={}
    for i,c in enumerate(cs):
        lines=c.splitlines()
        m=re.match(r"\s*%%writefile\s+(?:-a\s+)?([a-zA-Z0-9_./-]+)",lines[0]) if lines else None
        k=m.group(1) if m else f"CELL_{i:02d}"
        if k in parts:k=f"{k}_COPY_{i}"
        parts[k]='\n'.join(lines[1:] if m else lines)
    return {"parts":parts,"sha":hashlib.sha256('\n'.join(cs).encode()).hexdigest(),
            "files":len(cs),"mounts":{"model":len(md.get("model_sources") or []),
             "kernel":len(md.get("kernel_sources") or []),
             "data":len(md.get("dataset_sources") or [])}}

def main():
    with tempfile.TemporaryDirectory(prefix="ntl-highcode-decision-") as root:
        found={}
        for name,ref in REFS.items():
            v=pull(ref,Path(root)/name)
            if "failed" in v:
                event("PULL_FAILED",{"key":name,"requested":ref,"reason":v})
                continue
            found[name]=v
            event("SOURCE",{"key":name,"sha":v["sha"],"cells":v["files"],
                  "part_count":len(v["parts"]),"mounts":v["mounts"]})
            priority = sorted(v["parts"],key=lambda k:(
                 0 if "prefix_decisions" in k else
                 1 if "apex_postprocess" in k else
                 2 if "prefix_variant" in k else
                 3 if "/v1_stage.py" in k else
                 4 if "infer.py" in k else
                 5 if "CELL_2" in k else 10,k))
            printed=0
            for part in priority:
                if printed>=9:break
                src=v["parts"][part]
                if not MARK.search(src) or not (
                   "prefix" in part or "apex" in part or "v1_stage" in part
                   or "CELL_2" in part or "CELL_3" in part):continue
                ls=src.splitlines()
                matched=[]
                for i,line in enumerate(ls):
                    if (len(line)>300 or not MARK.search(line)
                      or EXCLUDE.search(line.strip())):continue
                    if not any(x in line.lower() for x in ("if ","return","fable","glm","prefix","deadline","guard","margin","budget","save","threshold")):
                        continue
                    matched.append({"line":i+1,"code":line.strip()[:215]})
                    if len(matched)>=14:break
                if matched:
                    functions=re.findall(r"(?m)^\s*(?:def|class)\s+([A-Za-z_][A-Za-z_0-9]*)",src)
                    event("MODULE",{"source":name,"path":part,"lines":len(ls),
                          "functions":functions[:30],"routing_lines":matched})
                    printed+=1
        if "sanpier_v3" in found and "sanpier_latest" in found:
            a,b=found["sanpier_v3"]["parts"],found["sanpier_latest"]["parts"]
            changed=sorted(k for k in set(a)&set(b) if a[k]!=b[k])
            ranked=[k for k in changed if not k.startswith("CELL_")]
            print_scopes=[]
            for k in ranked[:10]:
                aa=a[k].splitlines();bb=b[k].splitlines()
                ops=[op for op in difflib.SequenceMatcher(a=aa,b=bb,autojunk=False).get_opcodes() if op[0]!="equal"]
                print_scopes.append({"part":k,"change_hunks":len(ops),
                     "changed_loc_count":sum((x[2]-x[1])+(x[4]-x[3]) for x in ops)})
            event("SANPIER_V3_VS_LATEST",{
                "changed_parts":changed,"functional_module_changes":print_scopes,
                "score_metadata_v3_0717_not_proven_for_latest":True})
    event("END",{"complete":True,"trained":False,"submitted":False,
                  "source_payloads_ephemeral":True})

if __name__=="__main__": main()
