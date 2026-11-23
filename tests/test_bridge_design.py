import numpy as np
import pandas as pd


def test_nested_bridge_provenance_reverse_exclusion_and_training_only_additive_recovery():
    import bridge_design as c

    table=c.subjects();meta=table.set_index('FILE_ID');plans=c.plans();assert len(plans)==1200
    for r in range(50):
        for k in [2,4]:
            group=[p for p in plans if p['repeat']==r and p['k']==k and p['method']=='C2'];previous={'oneway':set(),'twoway':set()}
            for b in [2,5,10]:
                one=next(p for p in group if p['b']==b and p['direction']=='oneway');two=next(p for p in group if p['b']==b and p['direction']=='twoway')
                assert set(one['bridge'])==set(s for s in two['bridge'] if meta.loc[s,'label']==0)
                for p in [one,two]:
                    bridge=meta.loc[p['bridge']];original=meta.loc[p['pure_train']].groupby('site').label.first()
                    assert bridge.label.eq(bridge.site.map(original).rsub(1)).all()
                    assert set(bridge.site).isdisjoint(meta.loc[p['test'],'site'])
                    assert set(p['bridge']).issubset(p['original_reverse']) and set(p['bridge']).isdisjoint(p['pure_train'])
                    assert set(p['reverse'])==set(p['original_reverse'])-set(p['bridge'])
                    assert previous[p['direction']].issubset(p['bridge']);previous[p['direction']]=set(p['bridge'])
                    assert len(p['bridge'])==k*b*(1 if p['direction']=='oneway' else 2)
    assert all(p['b']>0 for p in plans if p['method']=='C2')  # b0仅引用B0，不进入C2调用路径。

    effects=np.array([[3,-1,.5],[-3,1,-.5],[1,2,2],[-1,-2,-2]],float);beta=np.array([.8,-.6,1.2]);intercept=np.array([10,-4,6])
    t=pd.DataFrame({'site':np.repeat(['a','b','c','d','unseen'],2),'label':[0,1]*5})
    x=np.vstack([intercept+beta*y+(effects[i] if i<4 else np.array([100,200,-100])) for i in range(5) for y in [0,1]])
    train=np.arange(8);fit=c.additive_fit(x,t,train)
    np.testing.assert_allclose(fit['effects']*fit['scale'].scale_,effects,atol=1e-12)
    np.testing.assert_allclose(fit['diagnosis']*fit['scale'].scale_,beta,atol=1e-12)
    np.testing.assert_allclose(fit['intercept']*fit['scale'].scale_+fit['scale'].mean_,intercept,atol=1e-12)
    np.testing.assert_allclose(fit['effects'].sum(axis=0),0,atol=1e-14)
    changed=x.copy();changed[8:]=1e6;other=c.additive_fit(changed,t,train)
    np.testing.assert_array_equal(fit['effects'],other['effects']);np.testing.assert_array_equal(fit['scale'].mean_,other['scale'].mean_)
    corrected=c.additive_transform(x,t,np.arange(10),fit)
    np.testing.assert_allclose(corrected[:8],np.tile(corrected[:2],(4,1)),atol=1e-12)
    np.testing.assert_allclose(corrected[8:],fit['scale'].transform(x[8:]),atol=1e-12)
