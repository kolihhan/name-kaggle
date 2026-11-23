"""Independent source-conditional token-transition LM for NTL, CPU research only.

See docs/research/ntl-generative-transition-v1-contract-20261123.md.
No hidden data, external Champion predictions, GPU, public-LB tuning or ensemble.
The real and shuffled LM share exactly the same per-class unigram counts.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot
from scripts.run_style_baseline import FUNCTION_WORDS
from scripts.run_strong_sparse_anchor import build_fold_features, make_svc
from scripts.ntl_mpcd_v1_20261123 import grouped_ci

SEED = 20261123
K = 18
ALPHA_U = 20.
ALPHA_BG = 16.
ALPHA_GLOBAL_BG = 12.
ALPHA_GLOBAL_U = .5
FWORDS = frozenset(FUNCTION_WORDS)
PUNCT = frozenset('. , ; : ! ? — – - … " \' ’ “ ” ( ) [ ] { } * # / % & + = < > ~'.split())
TOKENS = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[0-9]+|[^\w\s]|\w+", flags=re.UNICODE)
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?\Z")
ARMS = ("unigram", "real_context", "shuffled_context", "strong_sparse")


def shape_token(token: str) -> str:
    if token.isdigit():
        return "<NUM>"
    if WORD.fullmatch(token):
        lower = token.lower()
        if lower in FWORDS:
            return lower
        n = len(token)
        length = str(n) if n <= 8 else "9-12" if n <= 12 else "13+"
        case = "UPPER" if token.isupper() else "TITLE" if token.istitle() else "LOWER"
        return f"<W{length}:{case}>"
    if token in PUNCT:
        return token
    if token.isalpha():
        return "<OTHER_WORD>"
    return "<SYM>"


def encode(text: str) -> tuple[str,...]:
    words=TOKENS.findall(text)
    if not words:
        raise ValueError("empty tokenized response")
    return tuple(shape_token(tok) for tok in words)


def shuffled(tokens: tuple[str,...], *, rowid: str, fold: int) -> tuple[str,...]:
    payload=f"{SEED}|{fold}|{rowid}".encode("utf-8")
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8],"little"))
    order=rng.permutation(len(tokens))
    result=tuple(tokens[int(i)] for i in order)
    if Counter(tokens)!=Counter(result):
        raise AssertionError("negative control must preserve ALL token marginals")
    return result


def sequence_counts(token_rows: list[tuple[str,...]]):
    unigrams=Counter()
    bigrams=Counter()
    context=Counter()
    n=0
    for seq in token_rows:
        unigrams.update(seq)
        prev="<BOS>"
        for tok in seq:
            context[prev]+=1
            bigrams[(prev,tok)]+=1
            prev=tok
            n+=1
    return unigrams,bigrams,context,n


class ConditionalProxy:
    """Empirical Bayes 18-way style-token Markov generative model."""
    def __init__(self, train_tokens, classes, *, permuted=None):
        y=np.asarray(classes,dtype=np.int16)
        if len(y)!=len(train_tokens) or set(y.tolist()) != set(range(K)):
            raise ValueError("class labels must contain exactly 18 classes")
        if permuted is None:permuted=train_tokens
        if len(permuted)!=len(train_tokens):
            raise ValueError("mismatched shuffled training size")
        self.vocabulary=frozenset(tok for seq in train_tokens for tok in seq) | {"<UNK>"}
        self.size=len(self.vocabulary)
        self.global_u,self.global_b,self.global_h,self.global_N=sequence_counts(train_tokens)
        self.class_u=[]
        self.class_b=[]
        self.class_h=[]
        self.class_N=[]
        for k in range(K):
            original=[train_tokens[i] for i in np.flatnonzero(y==k)]
            alternate=[permuted[i] for i in np.flatnonzero(y==k)]
            cu,_,_,nu=sequence_counts(original)
            u,b,h,na=sequence_counts(alternate)
            if cu!=u or nu!=na:
                raise AssertionError("context-destroying control changed true unigram model")
            self.class_u.append(cu)
            self.class_b.append(b)
            self.class_h.append(h)
            self.class_N.append(nu)
        if min(self.class_N)<=0:
            raise AssertionError("missing source class token evidence")
        if not np.isclose(sum(self.p0_u(tok) for tok in self.vocabulary),1.0,atol=1e-9):
            raise ArithmeticError("global unigram probability not normalized")
        # exact analytic probability normalization for one observed context:
        ctx="<BOS>"
        for j in (0,K-1):
            val=sum(self.p_cond(j,ctx,t) for t in self.vocabulary)
            if not np.isclose(val,1.,atol=1e-9):
                raise ArithmeticError(f"class conditional sum not one: {j} {val}")
        self._uni_cache={}
        self._bg_cache={}

    def p0_u(self, tok):
        return (self.global_u[tok]+ALPHA_GLOBAL_U) / (
            self.global_N+ALPHA_GLOBAL_U*self.size
        )

    def p0_cond(self,prev,tok):
        return (self.global_b[(prev,tok)]+ALPHA_GLOBAL_BG*self.p0_u(tok)) / (
            self.global_h[prev]+ALPHA_GLOBAL_BG
        )

    def p_uni(self,k,tok):
        return (self.class_u[k][tok]+ALPHA_U*self.p0_u(tok)) / (
            self.class_N[k]+ALPHA_U
        )

    def p_cond(self,k,prev,tok):
        return (self.class_b[k][(prev,tok)]+ALPHA_BG*self.p0_cond(prev,tok)) / (
            self.class_h[k][prev]+ALPHA_BG
        )

    def project(self,seq):
        return tuple(t if t in self.vocabulary else "<UNK>" for t in seq)

    def unigram_score(self,seq):
        seq=self.project(seq)
        counts=Counter(seq)
        result=np.zeros(K,dtype=np.float64)
        for tok,n in counts.items():
            if tok not in self._uni_cache:
                self._uni_cache[tok]=np.log(np.asarray([
                    self.p_uni(k,tok) for k in range(K)],dtype=np.float64))
            result+=n*self._uni_cache[tok]
        return result / len(seq)

    def conditional_score(self,seq):
        seq=self.project(seq)
        pairs=Counter()
        prev="<BOS>"
        for tok in seq:
            pairs[(prev,tok)]+=1
            prev=tok
        result=np.zeros(K,dtype=np.float64)
        for pair,n in pairs.items():
            if pair not in self._bg_cache:
                prev,tok=pair
                self._bg_cache[pair]=np.log(np.asarray([
                    self.p_cond(k,prev,tok) for k in range(K)],dtype=np.float64))
            result+=n*self._bg_cache[pair]
        return result/len(seq)


def quick_tests():
    assert encode("The Cat — HELLO 123!") == (
        "the", "<W3:TITLE>", "—", "<W5:UPPER>", "<NUM>", "!"
    )
    t=("the","and","<W4:LOWER>",".")
    assert Counter(shuffled(t,rowid="toy",fold=0))==Counter(t)
    x=[t,tuple(reversed(t))]*18
    y=np.tile(np.arange(K,dtype=np.int16),2)
    model=ConditionalProxy(x,y)
    assert np.isfinite(model.unigram_score(t)).all()
    assert np.isfinite(model.conditional_score(t)).all()
    z=model.conditional_score(t)
    # Subtracting a shared LM log-likelihood has NO class-decoding effect.
    assert int(np.argmax(z))==int(np.argmax(z-3.1415926535))
    print("NTL_GTF_V1_FORMAL_IDENTITY_AND_LM_NORMALIZATION_SELFTEST=PASS",flush=True)


def paired_metrics(y, base, alt, prompts):
    b=balanced_accuracy_score(y,base)
    a=balanced_accuracy_score(y,alt)
    return {
        "BA_delta":round(float(a-b),7),
        "prompt_bootstrap_CI95":grouped_ci(y,base,alt,prompts,seed=SEED,draws=1000),
        "rescues":int(((base!=y)&(alt==y)).sum()),
        "regressions":int(((base==y)&(alt!=y)).sum()),
        "prediction_changes":int((base!=alt).sum())
    }


def main():
    quick_tests()
    d=load_snapshot(Path("data/train.csv"),Path("artifacts/phase1/folds.csv"))
    labels,y=np.unique(d.model.astype(str).to_numpy(),return_inverse=True)
    if len(labels)!=K or len(d)!=2699:
        raise AssertionError("full official historical 18-class contract violated")
    folds=d.fold_id.to_numpy()
    prompt=d.prompt_id.astype(str).to_numpy()
    ids=d.id.astype(str).to_numpy()
    start=time.perf_counter()
    preds={arm:np.full(len(d),-1,dtype=np.int16) for arm in ARMS}
    fold_results=[]
    rows=[encode(str(t)) for t in d.text]
    with threadpool_limits(limits=2):
        for fold in range(5):
            train=np.flatnonzero(folds != fold)
            val=np.flatnonzero(folds == fold)
            if set(prompt[train])&set(prompt[val]):
                raise AssertionError("train/heldout prompt overlap")
            tr=[rows[i] for i in train]
            validation=[rows[i] for i in val]
            tr_y=y[train]
            altered=[shuffled(seq,rowid=ids[i],fold=fold) for i,seq in zip(train,tr)]
            changed=sum(seq!=fake for seq,fake in zip(tr,altered))
            if changed/len(tr)<.95:
                raise AssertionError("destroying context did not modify >=95% training passages")
            real=ConditionalProxy(tr,tr_y)
            fake=ConditionalProxy(tr,tr_y,permuted=altered)
            if real.class_u!=fake.class_u or real.global_u!=fake.global_u:
                raise AssertionError("class-specific unigram distributions changed")
            for name,model,method in (
                ("unigram",real,"unigram_score"),
                ("real_context",real,"conditional_score"),
                ("shuffled_context",fake,"conditional_score")
            ):
                p=np.asarray([int(np.argmax(getattr(model,method)(seq)))
                              for seq in validation],dtype=np.int16)
                preds[name][val]=p
            # Same-row competitive direct single-model control; no blend.
            texts_train=d.text.iloc[train].astype(str).tolist()
            texts_val=d.text.iloc[val].astype(str).tolist()
            x,z,metadata=build_fold_features(texts_train,texts_val)
            svc=make_svc()
            svc.fit(x,tr_y)
            pred=svc.predict(z).astype(np.int16)
            preds["strong_sparse"][val]=pred
            scores={arm:round(float(balanced_accuracy_score(y[val],preds[arm][val])),7)
                    for arm in ARMS}
            fold_results.append({"fold":fold,"train_rows":len(train),
                "heldout_rows":len(val),"train_prompts":len(set(prompt[train])),
                "heldout_prompts":len(set(prompt[val])),
                "token_vocabulary":real.size,"changed_shuffled_train_rows":changed,
                "supervised_sparse_width":metadata["total_features"],"BA18":scores})
            print("NTL_GTF_V1_FOLD="+json.dumps(fold_results[-1],sort_keys=True),flush=True)
    if any(np.any(v<0) for v in preds.values()):
        raise AssertionError("some OOF predictions absent")
    scores={key:float(balanced_accuracy_score(y,p)) for key,p in preds.items()}
    recall={key:recall_score(y,p,labels=np.arange(K),average=None,zero_division=0)
            for key,p in preds.items()}
    comparisons={
        "B_minus_A":paired_metrics(y,preds["unigram"],preds["real_context"],prompt),
        "B_minus_C":paired_metrics(y,preds["shuffled_context"],preds["real_context"],prompt),
        "B_minus_D":paired_metrics(y,preds["strong_sparse"],preds["real_context"],prompt)
    }
    wins=sum(z["BA18"]["real_context"]>z["BA18"]["shuffled_context"]
             for z in fold_results)
    recall_fall=(recall["unigram"]-recall["real_context"]).max()
    gate=bool(
        comparisons["B_minus_A"]["BA_delta"]>=.01 and
        comparisons["B_minus_C"]["BA_delta"]>=.01 and
        comparisons["B_minus_A"]["prompt_bootstrap_CI95"][0]>0 and
        comparisons["B_minus_C"]["prompt_bootstrap_CI95"][0]>0 and
        wins>=3 and recall_fall<=.15
    )
    providers={}
    for i,label in enumerate(labels):
        fam=label.split("-")[0]
        providers.setdefault(fam,[]).append(i)
    report={
        "candidate":"generator-token-transition-fingerprint-v1",
        "scope":"historical 2699 official train only, frozen 150 prompt groups, 18 classes",
        "not_external_holdout":True,"not_public_score":True,
        "true_target_model_likelihood":False,
        "causal_proxy":"class-conditional bigram counts over abstract function/style tokens",
        "arms":{k:{"OOF_BA18":round(float(v),7),
                   "class_recall":{label:round(float(recall[k][i]),5)
                                   for i,label in enumerate(labels)},
                   "provider_mean_exact_recall":{
                       fam:round(float(np.mean(recall[k][indices])),5)
                       for fam,indices in providers.items()}}
                for k,v in scores.items()},
        "comparisons":comparisons,
        "real_vs_shuffled_fold_wins":wins,
        "max_recall_drop_B_vs_A":round(float(recall_fall),7),
        "predeclared_mechanism_gate":("PASS" if gate else "FAIL"),
        "competitive_vs_same_row_sparse":scores["real_context"]>=scores["strong_sparse"],
        "folds":fold_results,
        "total_wall_seconds":round(time.perf_counter()-start,1),
        "shared_LM_residual_argmax_math":"common subtraction cancels; tested",
        "use_0715_champion":False,"model_ensembling":False,
        "gpu":False,"official_submission":False,
    }
    print("NTL_GTF_V1_RESULT="+json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
