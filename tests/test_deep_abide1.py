import numpy as np
from sklearn.preprocessing import StandardScaler


def test_training_membership_groups_and_reference_alignment(monkeypatch,tmp_path):
    import pandas as pd
    import torch
    import deep_abide1 as task

    table = task.pilot.table()
    jobs = task.plans(table)
    references = task.references(table, jobs)
    assert len(jobs) == 200
    assert references.groupby('method').ngroups == 6
    for j in jobs:
        train, target, pool = task.indices(table, j)
        assert set(train).issubset(pool) and not set(target)&set(pool)
        assert set(table.loc[pool,'FILE_ID']) == set(j['train']+j['external'])
        names, groups = task.group_dro.site_groups(table, train)
        np.testing.assert_array_equal(names[groups], table.loc[train,'site'])
        names, domain = task.group_dro.site_groups(table, pool)
        np.testing.assert_array_equal(names[domain], table.loc[pool,'site'])
        sampled = task.unlabeled_batch(pool, np.random.default_rng(1))
        assert len(sampled)==len(set(sampled))==14 and set(sampled).issubset(pool)
    j = jobs[0]
    train, target, _ = task.indices(table, j)
    x = np.random.default_rng(0).normal(size=(len(table),8))
    altered = x.copy()
    altered[np.setdiff1d(np.arange(len(table)), train)] += 1e5
    expected = StandardScaler().fit(x[train]).mean_
    for method in ['E-CE','E-L3','E-GDRO']:
        result = task.fit_head(x,table,j,train,target,method)
        changed = task.fit_head(altered,table,j,train,target,method)
        assert result['C']==changed['C'] and result['inner']==changed['inner']
        np.testing.assert_array_equal(changed['scaler_mean'],expected)
        for fold in changed['inner']:
            tr, va = set(fold['train']), set(fold['validation'])
            assert tr.isdisjoint(va) and tr|va==set(j['train'])
            assert fold['standardizer_fit']==fold['train']

    # 归一化只进入站点分支。诊断分支仍看到原 FC128 特征。
    features = torch.arange(256, dtype=torch.float32).reshape(2,128)
    monkeypatch.setattr(task.pilot.Encoder,'forward',lambda model,x:features)
    model = task.DANN(3)
    received = []
    with model.domain.register_forward_pre_hook(lambda module,args:received.append(args[0])):
        diagnosis, _ = model(None,.5)
    torch.testing.assert_close(diagnosis,model(None))
    torch.testing.assert_close(received[0],torch.nn.functional.layer_norm(features,(128,)))

    probe_table = pd.DataFrame({'FILE_ID':np.arange(12),'site':['a']*6+['b']*6})
    probe_pool = np.r_[0:5,6:11]
    site_features = torch.tensor([[-1.]]*6+[[1.]]*6)
    monkeypatch.setattr(task.pilot,'cuda_matrices',lambda x,scale,ids:site_features[ids])
    monkeypatch.setattr(task.pilot.Encoder,'forward',lambda model,x:x)
    monkeypatch.setattr(task.pilot,'OUT',tmp_path)
    probe = task.site_probe(None,probe_table,probe_pool,model,None,'D-DANN')
    scores = pd.read_csv(tmp_path/'site_probe_predictions.csv')
    np.testing.assert_array_equal(scores.FILE_ID,probe_pool)
    assert probe['probe_accuracy']==1. and probe['probe_chance']==.5
