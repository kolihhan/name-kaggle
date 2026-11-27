import numpy as np
from ntl_effective_delta_merge import merge_delta, factor_mean

def test_equivalent_gauges():
    rng=np.random.default_rng(12)
    a=rng.normal(size=(4,20));b=rng.normal(size=(30,4))
    aa=[a,-a];bb=[b,-b];weights=[.5,.5]
    m,n,res=merge_delta(aa,bb,weights,4)
    assert np.allclose(m@n,b@a,rtol=1e-10,atol=1e-10)
    assert res<1e-12
    assert np.allclose(factor_mean(aa,bb,weights),0)

def test_low_rank_error_matches_optimum():
    rng=np.random.default_rng(3)
    aa=[rng.normal(size=(3,21)),rng.normal(size=(2,21))]
    bb=[rng.normal(size=(18,3)),rng.normal(size=(18,2))]
    w=[.25,.75]
    target=sum(v*(b@a) for a,b,v in zip(aa,bb,w))
    b,a,err=merge_delta(aa,bb,w,3)
    expected=np.linalg.svd(target,compute_uv=False)[3:]
    assert np.isclose(np.linalg.norm(target-b@a),np.linalg.norm(expected))
    assert np.isclose(err,np.linalg.norm(target-b@a)/np.linalg.norm(target))

def test_zero():
    a=np.ones((2,4));b=np.ones((6,2))
    x,y,e=merge_delta([a,a],[b,-b],[.5,.5],2)
    assert np.linalg.norm(x@y)<1e-12
    assert e<1e-12
