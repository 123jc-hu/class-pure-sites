import numpy as np


def test_connectivity_pure_mixed_reverse_and_training_only_scaling():
    import abide_class_pure_sites as a

    # 错一位的上三角映射、混入测试站点或全体拟合Scaler都会改变这些行为。
    x=np.random.default_rng(4).normal(size=(4,200))
    x[:,0]=[-1,-1,1,1];x[:,1]=[-1,1,-1,1];x[:,2]=x[:,0]+x[:,1];x[:,199]=5
    v,zero=a.fisher_features(x)
    assert v.shape==(19900,) and zero==1
    np.testing.assert_allclose(v[[0,1,199]],[0,np.arctanh(1/np.sqrt(2)),np.arctanh(1/np.sqrt(2))],atol=1e-14)
    assert v[198]==v[-1]==0

    table=a.subjects();meta=table.set_index('FILE_ID');example=None
    for repeat in range(50):
        jobs=a.repeat_design(table,repeat)
        for job in jobs:
            train=meta.loc[job['train']];test=meta.loc[job['test']]
            assert not set(train.index)&set(test.index)
            assert not set(train.site)&set(test.site)
            if job['arm']!='P':continue
            assert train.groupby('site').label.nunique().eq(1).all()
            assert set(job['asd_sites']).isdisjoint(job['td_sites'])
            assert train.label.value_counts().to_dict()=={0:job['n'],1:job['n']}
            mixed=next(q for q in jobs if q['arm']=='M' and q['k']==job['k'] and q['n']==job['n'])
            m=meta.loc[mixed['train']]
            assert set(m.site)==set(train.site)==set(job['asd_sites']+job['td_sites'])
            assert m.label.value_counts().to_dict()=={0:job['n'],1:job['n']}
            reverse=meta.loc[job['reverse']]
            assert set(reverse.index).isdisjoint(train.index)
            assert reverse.label.eq(reverse.site.map(train.groupby('site').label.first()).rsub(1)).all()
            if example is None:example=job

    features=np.random.default_rng(12).normal(size=(len(table),3));indices=table.index[table.FILE_ID.isin(example['train'])].to_numpy()
    fitted=a.select_model(features,table.label.to_numpy(),indices,0)
    np.testing.assert_allclose(fitted.best_estimator_.named_steps['scale'].mean_,features[indices].mean(axis=0),atol=1e-14)
    changed=features.copy();changed[~np.isin(np.arange(len(table)),indices)]=1e6
    other=a.select_model(changed,table.label.to_numpy(),indices,0)
    np.testing.assert_array_equal(fitted.best_estimator_.named_steps['scale'].mean_,other.best_estimator_.named_steps['scale'].mean_)
    np.testing.assert_array_equal(fitted.best_estimator_.named_steps['lr'].coef_,other.best_estimator_.named_steps['lr'].coef_)
