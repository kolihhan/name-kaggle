"""Source-transfer falsifier for a source-agnostic sibling length direction.

Inputs are *previously verified aggregate counts*, NOT new rows or a fresh
holdout. Uses an exact two-sided Fisher test and Wilson descriptive intervals.
No supervised fitting, competition inference, or prediction export.
"""
from __future__ import annotations
import json,math
from dataclasses import dataclass

@dataclass(frozen=True)
class Cohort:
    name:str
    supports_sol_longer:int
    n:int
    evidence:str

COHORTS=(
    Cohort("official_train_same_prompt",75,150,"ntl-official-sibling-length-negative-control-20261112.md"),
    Cohort("sweeper_agentic_snippets",195,196,"generator-vs-channel-identification-decision-20261106.md"),
    Cohort("proxy_primary_prompt_rlow_t1",20,47,"generator-vs-channel-identification-decision-20261106.md"),
)

def wilson(success:int,n:int,z:float=1.96):
    if n<1 or not 0<=success<=n:raise ValueError("invalid counts")
    p=success/n
    d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(center-half,5),round(center+half,5)]

def fishertwo(a:int,n1:int,b:int,n2:int)->float:
    if not (n1>0 and n2>0 and 0<=a<=n1 and 0<=b<=n2):
        raise ValueError("invalid contingency")
    # Hypergeometric distribution conditioned on both margins.
    total_yes=a+b
    population=n1+n2
    bottom=math.lgamma(population+1)-math.lgamma(n1+1)-math.lgamma(n2+1)
    lo=max(0,n1-(population-total_yes));hi=min(n1,total_yes)
    def logp(x):
        # choose(K,x)*choose(N-K,n1-x)/choose(N,n1)
        return (math.lgamma(total_yes+1)-math.lgamma(x+1)-math.lgamma(total_yes-x+1)
          + math.lgamma(population-total_yes+1)-math.lgamma(n1-x+1)
          - math.lgamma(population-total_yes-n1+x+1)-bottom)
    obs=logp(a)
    values=[logp(x) for x in range(lo,hi+1) if logp(x)<=obs+1e-10]
    if not values:return 0.
    top=max(values)
    return min(1.,math.exp(top)*sum(math.exp(x-top) for x in values))

def evaluate()->dict:
    if not (len(COHORTS)==3 and [x.n for x in COHORTS]==[150,196,47]):
        raise ValueError("frozen_source_inventory_drift")
    rows=[{
        "source":x.name,"sol_longer":x.supports_sol_longer,"total_prompt_pairs":x.n,
        "rate":round(x.supports_sol_longer/x.n,5),
        "wilson95_descriptive":wilson(x.supports_sol_longer,x.n),
        "provenance":x.evidence,
    } for x in COHORTS]
    tests=[]
    for a,b in ((COHORTS[1],COHORTS[0]),(COHORTS[1],COHORTS[2])):
        p=fishertwo(a.supports_sol_longer,a.n,b.supports_sol_longer,b.n)
        tests.append({
            "contrast":a.name+" vs "+b.name,
            "rate_difference":round(a.supports_sol_longer/a.n-b.supports_sol_longer/b.n,5),
            "two_sided_fisher_exact_p_posthoc":p,
        })
    return {
        "question":"Is Sol-longer-than-Luna a source-invariant within-prompt direction?",
        "evidence_level":"POSTHOC_PREVIOUSLY_USED_AGGREGATE_DESCRIPTIVE",
        "hypothesis_1":"stable_length_direction_intrinsic_to_generator",
        "hypothesis_2":"source_and_request_protocol_confounded_length",
        "cohorts":rows,"heterogeneity_tests":tests,
        "proxy_target_words_equal_in_primary_stratum":0,
        "proxy_length_order_matches_requested_target_in_primary_stratum":47,
        "interpretation":"Reject universal source-agnostic Sol>Luna length ordering as a candidate rule. Cannot identify intrinsic model behavior: source, task, target-length policy and serving context differ. The proxy 47/47 alignment is protocol confounding, not independently scored signal.",
        "holdout_fresh":False,"balanced_accuracy_measured":False,
        "new_training":False,"new_gpu":False,"official_submission":False,
    }

if __name__=="__main__":
    print("NTL_PRO_SOURCE_HETEROGENEITY="+json.dumps(evaluate(),sort_keys=True),flush=True)
