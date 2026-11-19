"""Read ONLY publicly accessible competition forum discussions. No scoring."""
import re,subprocess
TOPICS={
"gemma-4-developer-agent":[760126,760309,759514,760201,760014,751596,746250,750974,747168,750713,749191,753973,748323,747519,751487,750350],
"name-that-llm":[759187,749447,750462,753177,751802,752766,753779,760286,760276,752197,757199,751722,759103,755314,754860],
}
for competition,ids in TOPICS.items():
    for topic in ids:
        print(f"NTL_GEMMA_THREAD_START={competition}/{topic}",flush=True)
        try:
            r=subprocess.run(["kaggle","competitions","topics","show",f"{competition}/{topic}"],text=True,
                             capture_output=True,timeout=35,check=False)
        except subprocess.TimeoutExpired:
            print("NTL_GEMMA_TOPIC_TIMEOUT",competition,topic,flush=True)
            continue
        if r.returncode:
            print("NTL_GEMMA_TOPIC_FAILURE",competition,topic,"returncode",r.returncode,
                  "stderr_category",r.stderr.splitlines()[:2],flush=True)
            continue
        text=re.sub(r"(?i)(?:api[_-]?key|token|password)\s*[=:]\s*\S+","<redacted>",r.stdout)
        print("NTL_GEMMA_TOPIC_LENGTH",competition,topic,len(text),flush=True)
        lines=text.splitlines()
        # Public discussion intro + replies. Keep enough to catch counter-evidence.
        for i,line in enumerate(lines[:90]):
            if not line.strip():continue
            print("NTL_GEMMA_DISCUSSION_EVIDENCE",competition,topic,i,line[:235],flush=True)
        print(f"NTL_GEMMA_THREAD_END={competition}/{topic}",flush=True)
print("NTL_GEMMA_TOPIC_AUDIT_FINISHED=TRUE",flush=True)
