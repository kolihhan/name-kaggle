#!/usr/bin/env python3
"""Source-only audit: test actual matched sibling requests without exposing texts.

Read a separately licensed public *external* corpus (never official Kaggle train)
and count only exact same-prompt + same request + same requested length pairs.
No training/inference, external text export, or model selection.
"""
import argparse, collections, csv, hashlib, itertools, json
from pathlib import Path

PAIR_FAMILIES = {
    "glm": ("glm-5.3", "glm-5.3-flash"),
    "gpt56": ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"),
}
NEEDED = {
    "source_tag", "model", "provider", "served_model", "prompt_id",
    "prompt", "prompt_exact", "target_words", "generator", "text", "row_id"
}

def norm(x):
    return str(x or "").strip()

def truthy(x):
    return norm(x).lower() in {"true", "1", "yes"}

def audit(csv_path, sha_expected=None):
    p = Path(csv_path)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if sha_expected and h != sha_expected:
        raise AssertionError(f"source hash drift: {h}")
    with p.open(encoding="utf-8-sig", newline="") as stream:
        rd = csv.DictReader(stream)
        cols = set(rd.fieldnames or [])
        if missing := NEEDED - cols:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        rows = list(rd)
    # Include only source identities named in the source author’s documented
    # exact/pinned provider provenance.
    rules = {"zaiglm": {"provider": "Z.AI", "family": "glm"},
             "oa56": {"provider": "OpenAI", "family": "gpt56"}}
    result = {"source_sha256": h, "rows_total": len(rows), "source_tags": {},
              "target_families": {}, "notes": [
                "Cross-source provenance and immutable served-version claims are not inferred from slugs alone.",
                "Prompt-only pairing is insufficient if model-specific target lengths or request bodies differ.",
                "Same API defaults are not explicit proof of identical backend decoding or version snapshots.",
                "No full 2 x 2 protocol contrast: this corpus does not record independent temperature variants.",
              ]}
    for tag, rule in rules.items():
        group = [x for x in rows if norm(x["source_tag"])==tag
                 and norm(x["model"]) in PAIR_FAMILIES[rule["family"]]]
        good = [x for x in group if norm(x["provider"])==rule["provider"] and
                truthy(x["prompt_exact"]) and norm(x["served_model"]) and
                norm(x["prompt"]) and norm(x["target_words"])]
        keys_prompt = collections.defaultdict(set)
        keys_length = collections.defaultdict(set)
        keys_request = collections.defaultdict(set)
        lengths = collections.defaultdict(set)
        for r in good:
            m = norm(r["model"])
            pkey=(norm(r["prompt_id"]),norm(r["prompt"]))
            target=norm(r["target_words"])
            # No raw prompts in output. Here template equality is determined by
            # generator/source_tag plus exact prompt and word target; API call text
            # wasn't available in an explicit "request" column in this corpus.
            keys_prompt[pkey].add(m)
            lengths[pkey].add(target)
            keys_length[(pkey,target)].add(m)
            # If an explicit request column is supplied, require equality as well.
            if "request" in cols and norm(r.get("request")):
                keys_request[(pkey,target,norm(r["request"]))].add(m)
        def pair_stats(mapping):
            ct=collections.Counter()
            for xs in mapping.values():
                for a,b in itertools.combinations(sorted(xs),2):
                    ct[(a,b)]+=1
            return {"total_pair_prompt_keys": sum(ct.values()),
                    "pairs": [{"models":[*p],"keys":n} for p,n in ct.most_common()]}
        prompt_stats=pair_stats(keys_prompt)
        length_stats=pair_stats(keys_length)
        request_stats=pair_stats(keys_request)
        result["target_families"][tag]={
            "family":rule["family"],"rows_tag":len(group),"rows_verified_for_pairing":len(good),
            "prompt_only":prompt_stats,
            "prompt_plus_requested_words":length_stats,
            "exact_request_available": bool(keys_request),
            "prompt_plus_requested_words_plus_exact_request":request_stats if keys_request else None,
            "shared_model_prompt_keys_with_multiple_requested_lengths":
                sum(len(z)>1 and len(keys_prompt[k])>=2 for k,z in lengths.items()),
            "same_provider_documented":rule["provider"],
            "frozen_support_gate_20_length_matched_prompt_pairs":
                length_stats["total_pair_prompt_keys"]>=20
        }
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--expected-sha256")
    args=ap.parse_args()
    print("SIBLING_PAIR_AUDIT_RESULT", json.dumps(
        audit(args.csv,args.expected_sha256),sort_keys=True))
