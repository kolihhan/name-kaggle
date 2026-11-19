"""Read-only pagination of official Kaggle competition discussion listings and Code listings."""
import subprocess,csv,io,re
for comp in ("gemma-4-developer-agent","name-that-llm"):
    for pg in range(1,7):
        cmd=["kaggle","competitions","topics","list",comp,"-s","recent","-v"]
        if pg>1:cmd+=["--page-token",str(pg)]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
        if p.returncode:
            print("FORUM_PAGE_ERROR",comp,pg,p.stderr[:180],flush=True)
            break
        raw=p.stdout
        lines=[l for l in raw.splitlines() if l and not l.startswith("Warning:") and not l.startswith("Next Page Token")]
        rows=list(csv.DictReader(io.StringIO("\n".join(lines))))
        print("FORUM_PAGE",comp,pg,"count",len(rows),"columns",list(rows[0].keys()) if rows else [],flush=True)
        for r in rows:
            title=r.get("title") or ""
            print("FORUM_TOPIC",comp,pg,r.get("id"),r.get("postDate"),r.get("commentCount"),
                  r.get("votes"),repr(title[:145]),flush=True)
        if "Next Page Token" not in raw:break
    for pg in range(1,7):
        p=subprocess.run(["kaggle","kernels","list","--competition",comp,"--page",str(pg),"--page-size","20"],
                         capture_output=True,text=True,timeout=35)
        if p.returncode:
            print("KERNEL_PAGE_ERROR",comp,pg,p.stderr[:180],flush=True)
            break
        a=[x.strip() for x in p.stdout.splitlines() if x.strip()]
        data=[x for x in a if "/" in x and not x.startswith("---")]
        print("KERNEL_PAGE",comp,pg,"rows",len(data),flush=True)
        for x in data:print("KERNEL_ITEM",comp,pg,re.sub(r"\s+"," ",x)[:185],flush=True)
        if len(data)<12:break
print("KAGGLE_FORUM_AND_CODE_PAGINATION_AUDIT=PASS",flush=True)
