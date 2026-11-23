from collections import Counter

import numpy as np


def test_adhd200_natural_isolation_and_balanced_construction(monkeypatch):
    import adhd200_replication as d

    table = d.table()
    meta = table.set_index('FILE_ID')
    assert table.loc[table.site.eq('Peking'),'label'].value_counts().to_dict()=={0:141,1:99}
    primary,cross,control,feasibility = d.construction_plans(table)
    pure = [j for j in primary if j['arm']=='P']
    assert list(Counter((j['td_sites'][0],j['asd_sites'][0]) for j in pure).values())==[3]*20
    for j in primary+cross+control:
        train,test = meta.loc[j['train']],meta.loc[j['test']]
        assert set(train.index).isdisjoint(test.index)
        assert set(train.site).isdisjoint(test.site)
        assert len(set(test.site))==3
        assert not set(meta.loc[j['train']+j['test']+j['reverse'],'site'])&{'WashU','Pittsburgh'}
    for j in pure:
        train,reverse = meta.loc[j['train']],meta.loc[j['reverse']]
        labels = train.groupby('site').label.first()
        assert train.groupby('site').label.nunique().eq(1).all()
        assert labels[j['td_sites'][0]]==0 and labels[j['asd_sites'][0]]==1
        assert reverse.label.eq(1-reverse.site.map(labels)).all()
        assert set(reverse.index).isdisjoint(train.index)
    by_key = {j['job_id']:j for j in cross}
    for j in cross+control:
        reference = by_key[j['C0_job_id']] if j['method']=='S0' else j
        labels = meta.loc[j['pure_train']].groupby('site').label.first()
        added = meta.loc[j['bridge']]
        expected = added.site.map(labels)
        assert added.label.eq(expected if j['method']=='S0' else 1-expected).all()
        assert len(added)==j['b']*(1+(j['direction']=='twoway'))
        assert set(added.index).isdisjoint(j['pure_train']+j['test']+j['reverse'])
        assert j['reverse']==[i for i in j['original_reverse'] if i not in reference['bridge']]
    assert sum(f['feasible'] for f in feasibility if f['b']==5 and f['direction']=='twoway')==48

    natural = d.natural_plans(table)
    for w in natural:
        j = w['job']
        assert set(j['train']).isdisjoint(j['test'])
        assert j['standardizer_fit']==j['train']
        if j['evaluation']=='LOSO':
            site_column = 'SITE_ID' if j['held_site'].startswith('Peking_') else 'site'
            assert set(meta.loc[j['test'],site_column])=={j['held_site']}
            assert j['held_site'] not in set(meta.loc[j['train'],site_column])
    chosen = next(w for w in natural if w['job']['held_site']=='WashU' and w['job']['pool']=='main')
    train = table.index[table.FILE_ID.isin(chosen['job']['train'])].to_numpy()
    x = np.random.default_rng(11).normal(size=(len(table),3))
    x[table.site.eq('WashU')] += 1e6
    select = d.a.select_model

    def audit(x,y,indices,seed):
        np.testing.assert_array_equal(indices,train)
        fitted = select(x,y,indices,seed)
        np.testing.assert_allclose(fitted.best_estimator_.named_steps['scale'].mean_,x[train].mean(0),atol=1e-14)
        return fitted

    monkeypatch.setattr(d.a,'select_model',audit)
    output = d.fit_task(chosen,(table,x))
    for fold in output['splits']:
        assert set(fold['train']+fold['validation'])==set(chosen['job']['train'])
        assert fold['standardizer_fit']==fold['train']
