
def test_abide2_merged_sites_pure_mixed_reverse_and_additions():
    import abide2_replication as r
    table=r.table();meta=table.set_index('FILE_ID');pool=table.loc[table.construction_eligible]
    assert pool.site.nunique()==13 and pool.site.eq('NYU').sum()==73
    assert meta.loc[pool.loc[pool.site.eq('NYU'),'FILE_ID'],'label'].value_counts().to_dict()=={1:51,0:22}
    assert not pool.SITE_ID.isin(['ABIDEII-KUL_3','ABIDEII-UCLA_1']).any()
    primary,cross,control,feasible=r.plans();assert len(primary)==350 and len(cross)==400
    for j in primary:
        train=meta.loc[j['train']];test=meta.loc[j['test']]
        assert len(j['test_sites'])==4 and set(test.site)==set(j['test_sites'])
        assert not set(j['train'])&set(j['test']) and set(train.site).isdisjoint(test.site)
        assert not (set(train.site)|set(test.site))&{'KUL','UCLA'}
        assert set(j['test_sites']).isdisjoint({'NYU','OHSU','KKI'})
        if j['arm']!='P':continue
        assert train.groupby('site').label.nunique().eq(1).all()
        mixed=next(q for q in primary if q['repeat']==j['repeat'] and q['k']==j['k'] and q['arm']=='M');m=meta.loc[mixed['train']]
        assert set(m.site)==set(train.site) and m.label.value_counts().to_dict()==train.label.value_counts().to_dict()=={0:20,1:20}
        rv=meta.loc[j['reverse']];original=train.groupby('site').label.first()
        assert set(rv.index).isdisjoint(train.index) and rv.label.eq(rv.site.map(original).rsub(1)).all()
    for jobs,opposite in [(cross,True),(control,False)]:
        for j in jobs:
            pure=meta.loc[j['pure_train']];added=meta.loc[j['bridge']];original=pure.groupby('site').label.first();expected=added.site.map(original)
            assert added.label.eq(1-expected if opposite else expected).all()
            assert set(added.site).isdisjoint(meta.loc[j['test'],'site']) and set(added.index).isdisjoint(j['test']+j['reverse']+j['pure_train'])
            assert added.groupby('site').size().eq(j['b']).all() and len(added)==j['k']*j['b']*(1+(j['direction']=='twoway'))
            assert j['n_train']==len(j['train']) and j['n_pure']==40
    assert len(feasible)==400 and len(control)==sum(x['feasible'] for x in feasible)
