import pandas as pd


def test_same_class_additions_nested_isolated_and_count_matched():
    import same_class_control as s

    table=s.c.subjects();meta=table.set_index('FILE_ID');jobs,feasibility=s.plans()
    source={j['job_id']:j for j in s.source_plans()}
    assert len(feasibility)==400 and len(jobs)==sum(q['feasible'] for q in feasibility)
    lookup={(j['repeat'],j['k'],j['b'],j['direction']):j for j in jobs}
    for j in jobs:
        old=source[j['C0_job_id']];added=meta.loc[j['bridge']]
        original=meta.loc[j['pure_train']].groupby('site').label.first()
        relevant=j['td_sites'] if j['direction']=='oneway' else j['asd_sites']+j['td_sites']
        assert set(added.site)==set(relevant)
        assert added.label.eq(added.site.map(original)).all()
        assert added.groupby('site').size().eq(j['b']).all()
        assert len(j['bridge'])==len(set(j['bridge']))
        assert set(j['bridge']).isdisjoint(j['pure_train']+j['test']+j['original_reverse'])
        assert set(added.site).isdisjoint(meta.loc[j['test'],'site'])
        assert j['test']==old['test'] and j['reverse']==old['reverse']
        assert j['train']==j['pure_train']+j['bridge']
        pd.testing.assert_series_equal(added.label.value_counts().sort_index(),meta.loc[old['bridge'],'label'].value_counts().sort_index())
        pd.testing.assert_series_equal(meta.loc[j['train'],'label'].value_counts().sort_index(),meta.loc[old['train'],'label'].value_counts().sort_index())
        if j['b']==5:
            small=lookup[j['repeat'],j['k'],2,j['direction']]
            for site in relevant:
                assert added.loc[added.site.eq(site)].index.tolist()[:2]==meta.loc[small['bridge']].query('site==@site').index.tolist()
