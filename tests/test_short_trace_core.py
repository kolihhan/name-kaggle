"""Short-TRACE: deterministic null control + safety invariants (CPU only)."""
import numpy as np
import pandas as pd
import pytest

from scripts.short_trace_core import bins_from_ranks, make_features, N_BINS
from scripts.run_short_trace_8b_pilot import sample_rows, balanced_accuracy, prompt_bootstrap


def test_rank_bins_edges():
    assert bins_from_ranks(np.array([0,1,2,3,4,7,8,15,16,31,32,63,64,127,128,10000]))\
        .tolist() == [0,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8]
    with pytest.raises(ValueError):
        bins_from_ranks(np.array([0,-1]))
    with pytest.raises(ValueError):
        bins_from_ranks(np.array([[0,1]]))


def test_real_shuffle_and_histogram_invariants():
    ranks=np.asarray(([0]*12+[2]*12+[15]*12+[1024]*12)*3,dtype=np.int64)
    first=make_features(ranks, "SOME TEST TEXT")
    second=make_features(ranks, "SOME TEST TEXT")
    assert first["H"].shape==(N_BINS,)
    assert first["S"].shape==first["T"].shape==(N_BINS+N_BINS*N_BINS,)
    assert all(np.array_equal(first[k],second[k]) for k in first)
    assert np.isclose(first["H"].sum(),1)
    assert np.isclose(first["S"][N_BINS:].sum(),1)
    assert np.isclose(first["T"][N_BINS:].sum(),1)
    assert np.array_equal(first["S"][:N_BINS], first["H"])
    assert np.array_equal(first["T"][:N_BINS], first["H"])
    assert not np.array_equal(first["S"],first["T"])
    assert np.isclose(first["S"][N_BINS:].reshape(N_BINS,N_BINS).sum(),1)


def test_empty_and_one_token_do_not_cross_documents():
    e=make_features(np.array([],dtype=np.int64),"")
    o=make_features(np.array([0],dtype=np.int64),"a")
    assert np.array_equal(e["T"],np.zeros(90,dtype=np.float32))
    assert o["H"][0]==1 and o["T"][N_BINS:].sum()==0
    assert o["S"][N_BINS:].sum()==0


def test_sample_one_per_label_prompt_and_deterministic():
    rows=[]
    for label in ("a","b"):
        for p in range(12):
            for repeat in range(3):
                rows.append({"model":label,"prompt_id":f"p{p}",
                             "source":"thirdparty","text":f"{label}-{p}-{repeat}"})
    f=pd.DataFrame(rows)
    a=sample_rows(f,["a","b"],5)
    b=sample_rows(f.sample(frac=1,random_state=42),["a","b"],5)
    assert a[["model","prompt_id","text"]].equals(b[["model","prompt_id","text"]])
    assert len(a)==10
    assert a.groupby("model")["prompt_id"].nunique().to_dict()=={"a":5,"b":5}
    with pytest.raises(ValueError):
        sample_rows(f,["a","b"],13)


def test_bootstrap_and_macro_balanced_accuracy():
    y=np.array(["a","b","a","b","a","b"],dtype=str)
    p=np.array(["a","b","a","b","a","b"],dtype=str)
    q=np.array(["b","a","b","a","b","a"],dtype=str)
    g=np.array(["p0","p0","p1","p1","p2","p2"],dtype=str)
    assert balanced_accuracy(y,p)==1
    assert balanced_accuracy(y,q)==0
    ci=prompt_bootstrap(y,p,q,g,iterations=20)
    assert np.allclose(ci,[1,1])
