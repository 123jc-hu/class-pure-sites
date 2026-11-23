def test_real_rosters_and_training_scope():
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import deep_pair_pilot as pilot

    table = pilot.table()
    jobs = pilot.plans(table)
    refs = pilot.references(table, jobs)
    assert len(jobs) == 10
    assert refs.groupby('method').size().to_dict() == {
        'deep_ERM': 3551, 'linear_C0': 3551, 'linear_L3': 3551}
    features = np.random.default_rng(0).normal(size=(len(table), 8))
    for j in jobs:
        train, target, ext = pilot.indices(table, j)
        train_ids, target_ids, ext_ids = (set(table.loc[i, 'FILE_ID']) for i in (train, target, ext))
        assert train_ids == set(j['train']) == set(j['A_train'])
        assert ext_ids == set(j['external']) and ext_ids | train_ids == set(j['A_ext'])
        assert train_ids.isdisjoint(target_ids) and ext_ids.isdisjoint(train_ids | target_ids)
        assert set(table.loc[ext, 'site']).isdisjoint(table.loc[np.r_[train,target], 'site'])
        sites = table.loc[train, 'site'].to_numpy()
        y = table.loc[train, 'label'].to_numpy()
        left, right = pilot.pair_indices(table, train)
        assert len(left) == 400 and set(np.r_[left,right]) == set(range(100))
        assert np.all((sites[left]==sites[right]) & (y[left]==1) & (y[right]==0))

    # Perturb every external and held-out row to catch accidental scaler fitting on those rows.
    j = jobs[0]
    train, target, _ = pilot.indices(table, j)
    altered = features.copy()
    altered[np.setdiff1d(np.arange(len(table)), train)] += 1e5
    expected = StandardScaler().fit(features[train]).mean_
    for head in ['L3', 'CE']:
        original = pilot.fit_head(features, table, train, target, j['seed'], head)
        changed = pilot.fit_head(altered, table, train, target, j['seed'], head)
        assert original['C'] == changed['C'] and original['inner'] == changed['inner']
        np.testing.assert_array_equal(changed['scaler_mean'], expected)
        for f in changed['inner']:
            tr, va = set(f['train']), set(f['validation'])
            assert tr.isdisjoint(va) and tr | va == set(j['train'])
            assert f['standardizer_fit'] == f['train']
