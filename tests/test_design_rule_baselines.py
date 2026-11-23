import numpy as np
import pandas as pd


def test_budget_common_reverse_combat_isolation_l1_and_pure_leakage(monkeypatch):
    import design_rule_baselines as d

    table=d.m.subjects();meta=table.set_index('FILE_ID');lookup=pd.Index(table.FILE_ID)
    original={(j['feature'],j['k'],j['direction'],j['repeat']):j for j in d.b.plans() if j['method']=='L3' and j['b']==10}
    for j in d.designs():
        if j['allocation']!='concentrated':continue
        ten=original[j['feature'],j['k'],j['direction'],j['repeat']]
        assert set(j['bridge']).issubset(ten['bridge']) and len(j['bridge'])==5*j['k']*(1+(j['direction']=='twoway'))
        assert set(meta.loc[j['bridge'],'site'])==set(j['selected_asd_sites']+(j['selected_td_sites'] if j['direction']=='twoway' else []))
        assert not set(j['common_reverse'])&(set(j['bridge'])|set(j['equal_bridge']))
        assert set(j['common_reverse'])==set(j['original_reverse'])-set(j['bridge'])-set(j['equal_bridge'])
        train=lookup.get_indexer(j['train']);keep=d.l1_train(table,train)
        counts=pd.crosstab(table.loc[train,'site'],table.loc[train,'label']);mixed=set(counts.index[counts.min(axis=1).gt(0)])
        assert set(table.loc[keep,'site'])==mixed
    for k in [2,4]:
        pure=pd.DataFrame({'site':np.repeat(np.arange(2*k),20),'label':np.repeat(np.r_[np.ones(k),np.zeros(k)],20)})
        np.testing.assert_allclose(d.leakage(pure,np.arange(len(pure))),2/k,atol=1e-15)

    rng=np.random.default_rng(14);t=pd.DataFrame({'site':np.repeat(['a','b','test'],20),'label':np.tile([0,1],30)})
    x=rng.normal(size=(60,40))+np.repeat(rng.normal(size=(3,40)),20,axis=0);before=x.copy();train=np.arange(40);target=np.arange(40,60);seen=[];real=d.neuroCombat
    def recording(**kwargs):
        seen.append(kwargs);return real(**kwargs)
    monkeypatch.setattr(d,'neuroCombat',recording)
    fitted=d.combat_lr(x,t,train,target,0)
    assert len(seen)==1 and len(seen[0]['covars'])==len(train)
    np.testing.assert_array_equal(seen[0]['dat'],before[train].T)
    np.testing.assert_array_equal(seen[0]['covars']['site'],t.loc[train,'site'])
    assert seen[0]['categorical_cols']==['label'] and seen[0]['parametric'] and seen[0]['eb']
    np.testing.assert_array_equal(x,before)
    np.testing.assert_array_equal(fitted['probability'],fitted['model'].predict_proba(before[target])[:,1])
    changed=before.copy();changed[target]*=100
    again=d.combat_lr(changed,t,train,target,0)
    np.testing.assert_array_equal(fitted['model'].best_estimator_.named_steps['scale'].mean_,again['model'].best_estimator_.named_steps['scale'].mean_)
    np.testing.assert_array_equal(fitted['model'].best_estimator_.named_steps['lr'].coef_,again['model'].best_estimator_.named_steps['lr'].coef_)

    d.combat_lr(x,t,train,target,0,eb=False)
    assert seen[-1]["eb"] is False
