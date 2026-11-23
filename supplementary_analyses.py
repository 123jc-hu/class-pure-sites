"""T18 的固定训练、评估和校准名单。"""
from itertools import combinations
from functools import cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import abide_class_pure_sites as a
import bridge_design as bridge
import within_site_pair_model as pair

INSTITUTIONS = {'LEUVEN','UCLA','UM'}
OUT = ROOT/'outputs/T18'


def anchors():
    pure = [j|dict(method='B0',b=0,direction='none') for j in bridge.original_jobs()]
    linked = [j for j in bridge.read_plans() if j['method']=='C0' and j['b']==5]
    assert len(pure)==100 and len(linked)==200
    return pure+linked


def same_diagnosis(t,j):
    candidates = t.loc[~t.FILE_ID.isin(j['train'])]
    return {label:{role:candidates.loc[candidates.label.eq(label)&candidates.site.isin(j[role]),'FILE_ID'].tolist()
                    for role in ('asd_sites','td_sites')} for label in (0,1)}


def normative_plan(t,j):
    meta = t.set_index('FILE_ID')
    rng = np.random.default_rng(j['repeat'])
    held = meta.loc[j['test']]
    reference = []
    for _,group in held.groupby('site'):
        td = group.index[group.label.eq(0)].to_numpy()
        reference.extend(rng.choice(td,len(td)//2,replace=False).tolist())
    reference_set = set(reference)
    train_td = [i for i in j['train'] if meta.at[i,'label']==0]
    return dict(train_td=train_td,calibration=reference,test=[i for i in j['test'] if i not in reference_set],
                reverse=j['reverse'])


def same_design(t,repeat):
    rng = np.random.default_rng(repeat)
    sites = set(t.site)
    small = sorted(sites-a.LARGE-INSTITUTIONS)
    test_sites = sorted(rng.choice(small,5,replace=False).tolist())
    test = t.loc[t.site.isin(test_sites),'FILE_ID'].tolist()
    same_sites = [(f'{site}_1',f'{site}_2') for site in sorted(INSTITUTIONS)]
    assignments = {sub:int(label) for pair in same_sites for sub,label in
                   zip(pair,rng.permutation([0,1]))}
    rates = pd.crosstab(t.site,t.label)
    pool = sorted(sites-INSTITUTIONS-set(test_sites))
    asd = [site for site in pool if rates.loc[site,1]>=13]
    td = [site for site in pool if rates.loc[site,0]>=13]
    options = [(case,control) for case in combinations(asd,3)
               for control in combinations([site for site in td if site not in case],3)]
    case,control = options[int(rng.integers(len(options)))]
    cross = dict.fromkeys(case,1)|dict.fromkeys(control,0)
    output = []
    for method,roles,column in [('P_same',assignments,'SITE_ID'),('P_cross',cross,'site')]:
        train,reverse = [],[]
        for site,label in sorted(roles.items()):
            cohort = t.loc[t[column].eq(site)]
            ids = cohort.loc[cohort.label.eq(label),'FILE_ID'].to_numpy()
            train.extend(rng.choice(ids,13,replace=False).tolist())
            reverse.extend(cohort.loc[cohort.label.ne(label),'FILE_ID'].tolist())
        output.append(dict(method=method,repeat=repeat,seed=repeat,train=train,test=test,reverse=reverse,
                           test_sites=test_sites,site_labels=roles))
    mixed = []
    for j,k in same_sites:
        for site,n in [(j,7-repeat%2),(k,6+repeat%2)]:
            for label in (0,1):
                ids = t.loc[t.SITE_ID.eq(site)&t.label.eq(label),'FILE_ID'].to_numpy()
                mixed.extend(rng.choice(ids,n,replace=False).tolist())
    output.append(dict(method='M_same',repeat=repeat,seed=repeat,train=mixed,test=test,reverse=[],
                       test_sites=test_sites))
    return output


@cache
def selected_c():
    f = pd.read_csv(ROOT/'outputs/T12A/completed.csv').set_index('job_id').C.to_dict()
    f.update(pd.read_csv(ROOT/'outputs/T12C/completed.csv').set_index('job_id').C.to_dict())
    l3 = pd.read_csv(ROOT/'outputs/T13B/selected_C.csv').query('feature == "CC200" and method == "L3"')
    return f,l3.set_index(['k','b','direction','repeat'],verify_integrity=True).C.to_dict()


def a2_predict(x,t,j,method,groups):
    all_ids = pd.Index(t.FILE_ID)
    train = all_ids.get_indexer(j['train'])
    ids,roles = [],[]
    for label,categories in groups.items():
        n = min(map(len,categories.values()))
        if n>=5:
            for role,selection in categories.items():
                ids.extend(selection)
                roles.extend([role]*len(selection))
    target = all_ids.get_indexer(ids)
    assert set(train).isdisjoint(target)
    choices,l3 = selected_c()
    c = l3[j['k'],j['b'],j['direction'],j['repeat']] if method=='L3' else choices[j['job_id']]
    if method=='L3':
        fit = pair.fit_model(x,t,train,'L3',c)
        values = pair.decision(fit,x,t,target)
    else:
        fitted = make_pipeline(StandardScaler(),LogisticRegression(C=c)).fit(x[train],t.loc[train,'label'])
        values = fitted.predict_proba(x[target])[:,1]
    return t.loc[target,['FILE_ID','site','label']].assign(score=values,site_role=roles,C=c,
               method=method,source_job_id=j['job_id'],k=j['k'],b=j['b'],direction=j['direction'],repeat=j['repeat'])


def distance(z,mu,precision):
    residual = z-mu
    return np.sqrt(np.maximum(0.,np.einsum('ij,jk,ik->i',residual,precision,residual)))


def equal_site_reference(table, repeat, train_td):
    rng = np.random.default_rng(repeat)
    return [i for _,group in table.set_index('FILE_ID').loc[train_td].groupby('site')
            for i in rng.choice(group.index,5,replace=False)]


def normative_predict(x,t,j,split):
    all_ids = pd.Index(t.FILE_ID)
    train,reference,test,reverse = [all_ids.get_indexer(split[name])
                                    for name in ['train_td','calibration','test','reverse']]
    target = np.r_[test,reverse]
    assert t.loc[train,'label'].eq(0).all() and t.loc[reference,'label'].eq(0).all()
    assert set(train).isdisjoint(target) and set(reference).isdisjoint(target)
    data = x[train]
    pca = PCA(n_components=30,svd_solver='full').fit(data)
    train_pc = pca.transform(data)
    cov = LedoitWolf().fit(train_pc)
    out_pc = pca.transform(x[target])
    scores,search = {},[]
    if 'equal_site_td' not in split:
        scores['O1'] = distance(out_pc,cov.location_,cov.precision_)
        grid = [.05,.1,.2]
        cv = list(KFold(5,shuffle=True,random_state=j['repeat']).split(data))
        means = [np.mean([-OneClassSVM(nu=nu).fit(data[tr]).decision_function(data[va]).mean()
                          for tr,va in cv]) for nu in grid]
        nu = grid[int(np.argmin(means))]
        scores['O2'] = -OneClassSVM(nu=nu).fit(data).decision_function(x[target])
        search = [dict(source_job_id=j['job_id'],nu=n,mean_held_TD_score=v,selected=n==nu) for n,v in zip(grid,means)]
    if j['b']==5:
        locations_train = train
        locations_pc = train_pc
        method = 'O3'
        if 'equal_site_td' in split:
            locations_train = all_ids.get_indexer(split['equal_site_td'])
            assert set(locations_train)<=set(train)
            locations_pc = train_pc[pd.Index(train).get_indexer(locations_train)]
            method = 'O3-eq'
        ref_pc = pca.transform(x[reference])
        locations = pd.DataFrame(np.r_[locations_pc,ref_pc]).assign(
            site=t.loc[np.r_[locations_train,reference],'site'].to_numpy()).groupby('site').mean()
        mu = locations.loc[t.loc[target,'site']].to_numpy()
        scores[method] = distance(out_pc,mu,cov.precision_)
    frame = t.loc[target,['FILE_ID','site','label']].assign(source_job_id=j['job_id'],k=j['k'],b=j['b'],
                  direction=j['direction'],repeat=j['repeat'],reverse=np.arange(len(target))>=len(test))
    values = pd.concat([frame.assign(method=method,score=score) for method,score in scores.items()],ignore_index=True)
    return values,search


def split_predict(x,t,j):
    method = j['method']
    key = f"{method}_r{j['repeat']}"
    result = a.fit_job(j|dict(job_id=key,no_zero_roi=False,arm=method,k=3,n=39),data=(t,x))
    frame = result['pred'][['FILE_ID','SITE_ID','site','label','probability','prediction','reverse']].copy()
    frame = frame.assign(method=method,repeat=j['repeat'],score=frame.probability,job_id=key)
    return frame,result['done'],result['splits']
