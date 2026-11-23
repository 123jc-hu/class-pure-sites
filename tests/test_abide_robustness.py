import numpy as np


def test_ho_mapping_qc_design_and_training_only_cnn_edge_scaling():
    import abide_robustness as d

    x=np.random.default_rng(4).normal(size=(4,111));x[:,0]=[-1,-1,1,1];x[:,1]=[-1,1,-1,1]
    x[:,2]=x[:,83]=x[:,0]+x[:,1];x[:,82]=7;x[:,110]=5
    vector,zero=d.ho_features(x)
    assert vector.shape==(5995,) and zero==1
    np.testing.assert_allclose(vector[[0,1,81,109]],[0,*([np.arctanh(1/np.sqrt(2))]*3)],atol=1e-14)
    assert vector[108]==vector[-1]==0
    header=next((d.a.ROOT/'rois_ho').glob('*.1D')).read_text().splitlines()[0].split()
    assert header[82]=='#3455'

    table=d.qc_table();assert len(table)==873;meta=table.set_index('FILE_ID');jobs=d.curve_plans(table)
    for j in jobs:
        assert set(j['train']+j['test']+j['reverse']).issubset(meta.index)
        tr=meta.loc[j['train']];te=meta.loc[j['test']]
        assert set(tr.site).isdisjoint(te.site)
        if j['arm']!='P':continue
        assert tr.groupby('site').label.nunique().eq(1).all()
        m=next(q for q in jobs if q['repeat']==j['repeat'] and q['k']==j['k'] and q['arm']=='M')
        assert set(meta.loc[m['train'],'site'])==set(tr.site)
        assert tr.label.value_counts().to_dict()==meta.loc[m['train'],'label'].value_counts().to_dict()=={0:40,1:40}
        rv=meta.loc[j['reverse']];assert set(rv.index).isdisjoint(tr.index)
        assert rv.label.eq(rv.site.map(tr.groupby('site').label.first()).rsub(1)).all()
    bridges,controls,_=d.bridge_plans(table,jobs)
    by={j['job_id']:j for j in bridges}
    for j in bridges+controls:
        added=meta.loc[j['bridge']];labels=meta.loc[j['pure_train']].groupby('site').label.first()
        want=added.site.map(labels) if j['method']=='S0' else 1-added.site.map(labels)
        assert added.label.eq(want).all() and added.groupby('site').size().eq(5).all()
        assert set(j['bridge']).isdisjoint(j['pure_train']+j['test']+j['reverse'])
        if j['method']=='S0':assert added.label.value_counts().to_dict()==meta.loc[by[j['C0_job_id']]['bridge'],'label'].value_counts().to_dict()

    edges=np.random.default_rng(12).normal(size=(5,19900));train=np.array([0,1,2]);fit=d.edge_scaler(edges,train)
    np.testing.assert_allclose(fit.mean_,edges[train].mean(axis=0),atol=1e-14)
    edges[3:]=1e6;other=d.edge_scaler(edges,train)
    np.testing.assert_array_equal(fit.mean_,other.mean_);np.testing.assert_array_equal(fit.scale_,other.scale_)
    matrix=d.edge_matrices(edges,fit,train)
    np.testing.assert_allclose(matrix[:,0,d.a.PAIRS[0],d.a.PAIRS[1]],fit.transform(edges[train]),rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(matrix[:,0].diagonal(axis1=1,axis2=2),0)
    np.testing.assert_array_equal(matrix[:,0],matrix[:,0].transpose(0,2,1))
