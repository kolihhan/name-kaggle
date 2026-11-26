import numpy as np
from cgf import contrast, predict, sha_ok, permute_within_prompt

def test_contrast():
    x=np.array([1.,2.,4.]); z=contrast(x,1)
    assert np.allclose(z,[-4/3,-1/3,5/3,-.5])
    assert np.allclose(z,contrast(x+100,1))
def test_predict():
    r=np.array([[1.,0.],[2.,0.],[0.,2.],[0.,1.]])
    assert list(predict([r],[np.array([[3.,0.],[0.,3.]])],np.array(['a','a','b','b'])))==['a','b']
def test_sha():
    assert not sha_ok(b'hello','0'*60)
    assert sha_ok(b'hello','2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824')
def test_placebo():
    p=np.array(['p','q','p','q']); x=np.arange(4)[:,None]; z=permute_within_prompt(x,p,123)
    for k in ['p','q']: assert set(z[p==k,0])==set(x[p==k,0])
