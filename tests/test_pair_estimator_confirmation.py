import numpy as np
import pandas as pd


def test_pairs_bridge_folds_and_shared_atlas_rosters():
    import pair_estimator_confirmation as b
    import within_site_pair_model as m

    table=m.subjects();lookup=pd.Index(table.FILE_ID);jobs=b.plans()
    cc={j['source_job_id']:j for j in jobs if j['feature']=='CC200'}
    ho={j['source_job_id']:j for j in jobs if j['feature']=='HO' and j['method']=='L3'}
    assert len(cc)==len(ho)==600
    for key,j in cc.items():
        for field in ['train','test','reverse','bridge']:assert j[field]==ho[key][field]
        train=lookup.get_indexer(j['train']);s=table.loc[train,'site'].to_numpy();y=table.loc[train,'label'].to_numpy();a,t=m.pairs(y,s)
        mixed=pd.crosstab(s,y).min(axis=1).gt(0)
        assert np.all(s[a]==s[t]) and np.all(y[a]==1) and np.all(y[t]==0)
        assert set(s[a])==set(mixed.index[mixed])
        folds=b.folds(table,train,j)
        assert len(folds)==min(5,j['b'])
        for tr,va in folds:
            v=table.loc[va];aa,tt=m.pairs(v.label.to_numpy(),v.site.to_numpy());assert len(aa)>0
            if j['b']==2:
                counts=pd.crosstab(v.site,v.label).reindex(mixed.index[mixed])
                assert counts.min(axis=1).eq(1).all() and set(v.site.to_numpy()[aa])==set(mixed.index[mixed])
            assert set(tr).isdisjoint(va) and set(tr)|set(va)==set(train)
            assert not (set(table.loc[tr,'FILE_ID'])|set(table.loc[va,'FILE_ID']))&set(j['test']+j['reverse'])
