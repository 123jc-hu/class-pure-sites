import numpy as np
import pandas as pd


def test_site_groups_bridge_membership_and_identical_isolated_inner_folds(monkeypatch):
    import group_dro as g

    table=pd.read_csv('outputs/T12A/subjects.csv');meta=table.set_index('FILE_ID');jobs=g.plans()
    assert len(jobs)==500
    for (_,_,_),work in pd.DataFrame(jobs).groupby(['design','k','repeat']):
        reference=work.iloc[0].to_dict();expected=g.inner_folds(table,reference)
        for j in work.to_dict('records'):
            assert j['train']==reference['train'] and j['test']==reference['test'] and j['reverse']==reference['reverse']
            assert set(meta.loc[j['train'],'site']).isdisjoint(meta.loc[j['test'],'site'])
            assert set(j['train']).isdisjoint(j['test']+j['reverse'])
            lookup=pd.Index(table.FILE_ID);tr=lookup.get_indexer(j['train']);names,groups=g.site_groups(table,tr)
            np.testing.assert_array_equal(names[groups],meta.loc[j['train'],'site'])
            if j['design']=='bridge_b5':
                for sid in j['bridge']:
                    pos=j['train'].index(sid);assert names[groups[pos]]==meta.loc[sid,'site']
            for (train,validation),(wanted_train,wanted_validation) in zip(g.inner_folds(table,j),expected):
                np.testing.assert_array_equal(train,wanted_train);np.testing.assert_array_equal(validation,wanted_validation)
                assert set(train).isdisjoint(validation) and set(train)|set(validation)==set(tr)
                assert set(table.loc[np.r_[train,validation],'FILE_ID']).isdisjoint(j['test']+j['reverse'])
                names,groups=g.site_groups(table,train)
                np.testing.assert_array_equal(names[groups],table.loc[train,'site'])
    import torch
    observed=[];step=torch.optim.Adam.step
    def record_step(optimizer,*args,**kwargs):
        observed.append(optimizer.param_groups[0]['lr'])
        return step(optimizer,*args,**kwargs)
    monkeypatch.setattr(torch.optim.Adam,'step',record_step)
    x=np.array([[-1.,.5],[1.,-.5],[-.5,-1.],[.5,1.]])
    for method in ['GD_ERM','GroupDRO']:
        observed.clear();g.optimize(x,np.array([0,1,0,1]),np.array([0,0,1,1]),method,.01,[.001])
        assert len(observed)==2000
        np.testing.assert_allclose(observed,.01*(1-np.arange(2000)/2000),atol=1e-15,rtol=0)
