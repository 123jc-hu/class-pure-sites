def test_published_methods_split_and_training_labels(monkeypatch,tmp_path):
    import numpy as np
    import pandas as pd
    import published_methods as t
    from published_methods_fidelity import kunda_default_phenotypes
    from published_methods_norev import kunda_norev, parisot_norev, reverse_kernel_scores
    from published_methods_report import read_predictions
    from published_methods_run import work

    sample = tmp_path/'predictions.csv'
    sample.write_text('prediction,score\nTrue,0.9\nFalse,0.1\n1,0.8\n0,0.2\n')
    assert read_predictions(sample).prediction.tolist()==[1,0,1,0]

    phenotypes = pd.DataFrame(dict(SEX=[1,2,1,2],EYE_STATUS_AT_SCAN=[2,1,1,2],
        HANDEDNESS_CATEGORY=['','L','Mixed','L->R'],AGE_AT_SCAN=[9,10,11,12],FIQ=[-9999,97,np.nan,83]))
    np.testing.assert_array_equal(kunda_default_phenotypes(phenotypes),[
        [1,0,9,100,0,0,1,0,1], [0,1,10,97,0,1,0,1,0],
        [1,0,11,100,1,0,0,1,0], [0,1,12,83,1,0,0,0,1]])

    table = t.a.subjects()
    meta = table.set_index('FILE_ID')
    jobs = t.main_jobs()
    assert len(jobs)==300
    for j in jobs:
        train,test,reverse = (set(j[k]) for k in ('train','test','reverse'))
        assert len(train|test|reverse)==len(train)+len(test)+len(reverse)
        assert set(meta.loc[j['train'],'site']).isdisjoint(j['test_sites'])
        assert set(meta.loc[j['test'],'site'])==set(j['test_sites'])

    first = work(table,1)
    assert len(first)==33
    b0 = next(j for j in jobs if (j['repeat'],j['method'],j['k'])==(0,'B0',2))
    for model in ('Parisot-HO','Parisot-CC200','Kunda'):
        folds = [j for j in first if j['model']==model and j['fold']>=0]
        assert sorted(i for j in folds for i in j['test'])==sorted(b0['train'])
        for j in folds:
            assert set(j['train']).isdisjoint(j['test']) and len(j['train'])==64
            assert set(j['train'])|set(j['test'])==set(b0['train']) and not j['reverse']

    rng = np.random.default_rng(123)
    x = rng.normal(1,.5,(80,12))
    train = np.arange(50)
    labels = np.tile([0,1],40)
    site = np.repeat(['one','two'],40)
    sex = labels.copy()
    originals = t.RFE.fit

    def check_rfe(self,features,y):
        np.testing.assert_array_equal(features,x[train])
        np.testing.assert_array_equal(y,labels[train])
        return originals(self,features,y)

    monkeypatch.setattr(t.RFE,'fit',check_rfe)
    assert t.parisot_design(x,site,sex,train,labels[train],n_features=8)[0].shape==(80,8)
    phenotypes = np.column_stack([sex,60-sex])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        for predict in (lambda y:t.parisot_fit(x,site,sex,train,y,n_features=8),
                        lambda y:t.kunda_ridge(x,site,phenotypes,train,y,dimensions=(50,))):
            baseline = predict(labels[train])[0]
            shuffled = labels.copy()
            shuffled[50:] = rng.permutation(shuffled[50:])
            np.testing.assert_allclose(predict(shuffled[train])[0],baseline,rtol=0,atol=1e-12)
        old,_ = t.parisot_fit(x,site,sex,train,labels[train],n_features=8)
        for query in (x[:4],rng.normal(size=(4,12))):
            plain,reverse = parisot_norev(x,site,sex,train,labels[train],query,site[:4],sex[:4],n_features=8)
            np.testing.assert_array_equal(plain,old)
            assert reverse.shape==(4,) and np.isfinite(reverse).all()
        values = rng.normal(size=(88,8,8))
        corr = values@values.transpose(0,2,1)+np.eye(8)
        baseline,_,_ = t.kunda_ridge(t.tpe(corr[:80],train),site,phenotypes,train,labels[train],dimensions=(50,))
        for query in (corr[80:84],corr[84:]):
            plain,reverse,_,grid = kunda_norev(corr[:80],site,phenotypes,train,labels[train],query,phenotypes[:4])
            np.testing.assert_array_equal(plain,baseline)
            assert grid==[50] and reverse.shape==(4,) and np.isfinite(reverse).all()

    from sklearn.linear_model import RidgeClassifier
    kernel = rng.random((12,12))
    cross = rng.random((4,12))
    w = rng.normal(size=(12,3))
    pheno = rng.normal(size=(12,2))
    ridge = RidgeClassifier().fit(np.c_[kernel@w,pheno],labels[:12])
    score = reverse_kernel_scores(ridge,cross,w,kernel.mean(axis=0),pheno[:4])
    np.testing.assert_allclose(score,ridge.decision_function(np.c_[cross@w,pheno[:4]]),rtol=0,atol=1e-12)
