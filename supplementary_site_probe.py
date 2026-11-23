"""T18 A1 的嵌套多分类站点探针。"""
from pathlib import Path
import sys
from warnings import catch_warnings, simplefilter

sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from supplementary_analyses import a, INSTITUTIONS

MAX_ITER = 2000


def populations():
    from abide2_replication import table as abide2_table, CACHE as abide2_cache
    from adhd200_replication import table as adhd_table, CACHE as adhd_cache

    assert sys.flags.isolated
    groups = {}
    for dataset,t,cache in [('ABIDE-I',a.subjects(),a.CACHE),('ABIDE-II',abide2_table(),abide2_cache),
                            ('ADHD-200',adhd_table(),adhd_cache)]:
        id_column = {'ABIDE-I':'FILE_ID','ABIDE-II':'DCAN_ID','ADHD-200':'ScanDir ID'}[dataset]
        order = pd.read_csv(cache/'row_order.csv',dtype=str)[id_column]
        x = np.load(cache/'fisher_z.npy',mmap_mode='r',allow_pickle=False)
        assert order.tolist()==t.FILE_ID.tolist() and len(x)==len(t)
        eligible = np.ones(len(t),dtype=bool) if dataset=='ABIDE-I' else t.construction_eligible if dataset=='ABIDE-II' else t.site.ne('Pittsburgh')
        assert int(eligible.sum())=={'ABIDE-I':1035,'ABIDE-II':546,'ADHD-200':769}[dataset]
        for scope,mask in [('all',eligible),('TD',eligible&t.label.eq(0)),
                           ('case',eligible&t.label.eq(1))]:
            groups[dataset+'_'+scope] = (x,t,t.site.to_numpy(),t.index[mask].to_numpy(),False)
        if dataset=='ABIDE-I':
            for name in sorted(INSTITUTIONS):
                groups[name] = (x,t,t.SITE_ID.to_numpy(),t.index[t.site.eq(name)].to_numpy(),True)
    assert len(np.unique(groups['ABIDE-II_all'][2][groups['ABIDE-II_all'][3]]))==13
    return groups


def folds(labels,indices,seed):
    return [(indices[train],indices[test]) for train,test in
            StratifiedKFold(5,shuffle=True,random_state=seed).split(indices,labels[indices])]


def probe_fold(x,y,train,test,seed,binary=False,fixed_c=None):
    with catch_warnings(record=True) as caught:
        simplefilter('always')
        result = _fit_probe(x,y,train,test,seed,binary,fixed_c)
    result['convergence_warnings'] = sum(issubclass(w.category,ConvergenceWarning) for w in caught)
    result['other_warnings'] = sorted({str(w.message) for w in caught if not issubclass(w.category,ConvergenceWarning)})
    return result


def _fit_probe(x,y,train,test,seed,binary,fixed_c):
    assert not set(train)&set(test)
    metric = roc_auc_score if binary else balanced_accuracy_score
    inner = []
    c = fixed_c
    if c is None:
        for tr,va in folds(y,train,seed):
            scale = StandardScaler().fit(x[tr])
            z,v = scale.transform(x[tr]),scale.transform(x[va])
            values = []
            for candidate in a.GRID:
                fit = LogisticRegression(C=candidate,max_iter=MAX_ITER).fit(z,y[tr])
                pred = fit.predict_proba(v)[:,1] if binary else fit.predict(v)
                values.append(metric(y[va],pred))
            inner.append(dict(train=tr.tolist(),validation=va.tolist(),standardizer_fit=tr.tolist(),
                              validation_metric=values))
        c = a.GRID[int(np.argmax(np.mean([f['validation_metric'] for f in inner],axis=0)))]
    scale = StandardScaler().fit(x[train])
    fit = LogisticRegression(C=c,max_iter=MAX_ITER).fit(scale.transform(x[train]),y[train])
    z = scale.transform(x[test])
    return dict(C=c,score=fit.predict_proba(z)[:,1] if binary else fit.predict(z),inner=inner)


def permuted_labels(y,idx,group_id,repeat):
    copy = y.copy()
    copy[idx] = np.random.default_rng([2026,group_id,repeat]).permutation(y[idx])
    return copy
