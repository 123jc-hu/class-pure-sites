import numpy as np
import pandas as pd


def test_lspo_isolation_training_only_equal_site_nap_and_label_free_scaling():
    import site_pair_validation as b

    # 不等站点人数使按被试加权的类中心与所需等权站点中心不同。
    x=np.array([[0,0,0],[4,0,0],[4,0,0],[4,0,0],[0,0,2],[0,0,2],[0,4,2],[100,-50,9],[-30,80,-9]],float)
    t=pd.DataFrame({'site':['a1','a2','a2','a2','b1','b1','b2','unseen_a','unseen_b'],'label':[1,1,1,1,0,0,0,1,0]})
    train=np.arange(7);held=np.array([7,8]);folds=b.site_pairs(t,train);seen=set()
    for tr,va in folds:
        assert set(t.loc[tr,'site']).isdisjoint(t.loc[va,'site'])
        v=t.loc[va].drop_duplicates('site');assert sorted(v.label)==[0,1] and len(v)==2
        seen.add(tuple(sorted(v.site)))
        scale,u=b.nap_fit(x,t,tr);changed=x.copy();changed[np.r_[va,held]]=1e6
        scale2,u2=b.nap_fit(changed,t,tr)
        np.testing.assert_array_equal(scale.mean_,scale2.mean_);np.testing.assert_array_equal(u,u2)
        z=b.nap_transform(x[tr],scale,u);np.testing.assert_allclose(z@u,0,atol=1e-12)
        assert u.shape==(3,0)  # 内层每类一个站点，没有可估计的同类站点差异。
    assert seen=={('a1','b1'),('a1','b2'),('a2','b1'),('a2','b2')}
    np.testing.assert_array_equal(b.site_offsets(x,t,train),[[-2,0,0],[2,0,0],[0,-2,0],[0,2,0]])
    scale,u=b.nap_fit(x,t,train);assert u.shape==(3,2)
    np.testing.assert_allclose(b.nap_transform(x[train],scale,u)@u,0,atol=1e-12)
    changed=x.copy();changed[held]=1e9;_,other=b.nap_fit(changed,t,train);np.testing.assert_array_equal(u,other)
    before=b.site_standardize(x,t,train,np.arange(len(t)))
    changed_labels=t.copy();changed_labels['label']=np.random.default_rng(3).permutation(t.label)
    np.testing.assert_array_equal(before,b.site_standardize(x,changed_labels,train,np.arange(len(t))))
