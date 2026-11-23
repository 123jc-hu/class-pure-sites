import numpy as np
import pandas as pd


def test_natural_collections_labels_outer_inner_and_scaler_isolation():
    import natural_pure_collections as n

    t=n.table();meta=t.set_index('FILE_ID');jobs=n.plans()
    assert t.loc[t.pool_main].shape[0]==547 and t.loc[t.pool_secondary].shape[0]==567
    assert t.loc[t.collection.eq('ABIDEII-KUL_3'),'label'].eq(1).all()
    assert t.loc[t.collection.eq('ABIDEII-NYU_1'),'site'].eq('NYU_1').all()
    assert t.loc[t.collection.eq('ABIDEII-NYU_2'),'site'].eq('NYU_2').all()
    assert not t.loc[t.pool_main,'collection'].eq('ABIDEII-NYU_2').any()
    assert len(jobs)==543
    for j in jobs:
        train=set(j['train']);test=set(j['test']);assert train.isdisjoint(test)
        if j['pool']=='main':assert not meta.loc[j['train']+j['test'],'collection'].eq('ABIDEII-NYU_2').any()
        if j['evaluation']!='subject':
            if j['site_definition']=='merged':assert not meta.loc[j['train'],'collection'].isin(['ABIDEII-NYU_1','ABIDEII-NYU_2']).any()
            else:assert set(meta.loc[j['train'],'site']).isdisjoint(meta.loc[j['test'],'site'])
        for a,b in n.inner_folds(t,j):
            tr=set(t.loc[a,'FILE_ID']);va=set(t.loc[b,'FILE_ID']);assert tr.isdisjoint(va) and tr|va==train and not (tr|va)&test
        assert j['standardizer_fit']==j['train']
    primary=[j for j in jobs if j['pool']=='main' and j['evaluation']=='subject']
    for repeat in range(50):
        test=[sid for j in primary if j['repeat']==repeat for sid in j['test']]
        assert len(test)==len(set(test))==547 and set(test)==set(t.loc[t.pool_main,'FILE_ID'])
