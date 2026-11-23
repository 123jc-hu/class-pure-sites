import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def test_within_site_pairs_translation_invariance_and_fit_isolation():
    import within_site_pair_model as m

    rng=np.random.default_rng(13);sites=np.repeat(['a','b','c'],40);y=np.tile([0,1],60)
    x=rng.normal(size=(144,7));x[:120,0]+=y;table=pd.DataFrame({'site':np.r_[sites,np.repeat('test',24)],'label':np.r_[y,np.tile([0,1],12)]})
    train=np.arange(120);test=np.arange(120,144);z=StandardScaler().fit_transform(x[train]);a,t=m.pairs(y,sites)
    assert len(a)==1200 and np.all(sites[a]==sites[t]) and np.all(y[a]==1) and np.all(y[t]==0)
    shift=np.repeat(rng.normal(size=(3,7))*5,40,axis=0)
    for model in ['L1','L3']:
        first=m.fit_weight(z,y,sites,model,1.);second=m.fit_weight(z+shift,y,sites,model,1.)
        assert np.linalg.norm(first['w'])>.01
        np.testing.assert_allclose(first['w'],second['w'],atol=1e-6,rtol=1e-6)
    changed=x.copy();changed[test]=rng.normal(size=(24,7))*1e3
    for model in ['L0','L1','L2','L3']:
        first=m.fit_model(x,table,train,model,.1);second=m.fit_model(changed,table,train,model,.1)
        np.testing.assert_array_equal(first['scale'].mean_,second['scale'].mean_)
        np.testing.assert_allclose(first['w'],second['w'],atol=1e-10,rtol=1e-10)
    shuffled=table.copy();shuffled['label']=rng.permutation(shuffled.label.to_numpy())
    first=m.fit_model(x,table,train,'L2',.1);second=m.fit_model(x,shuffled,train,'L2',.1)
    np.testing.assert_array_equal(m.transform(first,x,table,np.r_[train,test]),m.transform(second,x,shuffled,np.r_[train,test]))
    np.testing.assert_allclose(m.transform(first,x,table,test).mean(axis=0),0,atol=1e-14)

    actual=m.subjects();lookup=pd.Index(actual.FILE_ID);primary,secondary=m.source_jobs()
    for j in primary:
        tr=lookup.get_indexer(j['train']);te=set(lookup.get_indexer(j['test']))
        for a,b in m.inner_folds(actual,tr,j['seed']):
            assert set(a).isdisjoint(b) and set(a)|set(b)==set(tr) and not (set(a)|set(b))&te
            for indices in [a,b]:assert pd.crosstab(actual.loc[indices,'site'],actual.loc[indices,'label']).min(axis=1).gt(0).all()
    assert len(secondary)==100
    for j in secondary:
        for tr,va in m.inner_folds(actual,lookup.get_indexer(j['train']),j['seed']):
            a,b=m.pairs(actual.loc[va,'label'].to_numpy(),actual.loc[va,'site'].to_numpy())
            assert len(a)==(8 if j['direction']=='oneway' else 16)
            assert set(tr).isdisjoint(va) and not (set(tr)|set(va))&set(lookup.get_indexer(j['test']+j['reverse']))
    assert m.pair_auc(np.array([0,1,0,1]),np.array(['a','a','b','b']),np.array([0.,0.,1.,2.]))==.75
