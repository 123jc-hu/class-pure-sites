"""T12-A：固定ABIDE站点构造，普通被试内嵌套LR。"""
from itertools import combinations
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path.home()/'datasets_cache/ABIDE_PCP'
CACHE=Path.home()/'datasets_cache/ABIDE_derived/T12A'
OUT=Path('outputs/T12A')
GRID=[.001,.01,.1,1.]
SETTINGS=[(1,40),(2,40),(4,40),(6,40),(2,20),(2,80)]
LARGE={'NYU','UM','UCLA','USM'}
PAIRS=np.triu_indices(200,1)
META=['FILE_ID','SITE_ID','site','DX_GROUP','SEX','AGE_AT_SCAN','func_mean_fd','label']


def subjects():
    f=pd.read_csv(ROOT/'Phenotypic_V1_0b_preprocessed1.csv',usecols=META[:2]+['DX_GROUP','SEX','AGE_AT_SCAN','func_mean_fd'])
    f=f.loc[f.FILE_ID.ne('no_filename')].reset_index(drop=True)
    f['site']=f.SITE_ID.str.replace(r'_[12]$','',regex=True);f['label']=f.DX_GROUP.eq(1).astype(int)
    assert len(f)==1035 and f.FILE_ID.is_unique and f.site.nunique()==17
    return f[META]


def fisher_features(x):
    zero=np.ptp(x,axis=0)==0;keep=~zero;corr=np.zeros((x.shape[1],x.shape[1]))
    corr[np.ix_(keep,keep)]=np.corrcoef(x[:,keep],rowvar=False)
    return np.arctanh(np.clip(corr[np.triu_indices(x.shape[1],1)],-.999999,.999999)),int(zero.sum())


def repeat_design(table,repeat,site_curve_only=False,large_sites=None,test_site_count=5,settings=None,*,site_pair=None):
    large_sites=LARGE if large_sites is None else large_sites;settings=SETTINGS if settings is None else settings
    rng=np.random.default_rng(repeat);sites=sorted(table.site.unique());small=sorted(set(sites)-large_sites)
    test_sites=sorted(rng.choice(small,test_site_count,replace=False).tolist() if site_pair is None else set(sites)-set(site_pair));pool=table.loc[~table.site.isin(test_sites)]
    counts=pd.crosstab(pool.site,pool.label);jobs=[]
    test=table.loc[table.site.isin(test_sites),'FILE_ID'].tolist()
    for k,n in settings[:4] if site_curve_only else settings:
        p,m=ceil(n/k),ceil(n/(2*k));both=counts.index[(counts[0]>=m)&(counts[1]>=m)].tolist()
        asd=[s for s in both if counts.loc[s,1]>=p];td=[s for s in both if counts.loc[s,0]>=p]
        feasible=[(list(a),list(t)) for a in combinations(asd,k) for t in combinations([s for s in td if s not in a],k)]
        if not feasible:continue  # T12-D的QC池按任务规定跳过不可行格子，不消耗抽签随机数。
        a,t=feasible[int(rng.integers(len(feasible)))] if site_pair is None else ([site_pair[1]],[site_pair[0]]);all_sites=sorted(a+t)
        context=dict(repeat=repeat,seed=repeat,k=k,n=n,test_sites=test_sites,asd_sites=a,td_sites=t,test=test)
        for arm in ['P','M']:
            train=[]
            for label,selected in [(1,a),(0,t)] if arm=='P' else [(1,all_sites),(0,all_sites)]:
                candidates=[]
                for site in selected:
                    ids=pool.loc[pool.site.eq(site)&pool.label.eq(label),'FILE_ID'].to_numpy()
                    candidates.extend(rng.choice(ids,p if arm=='P' else m,replace=False).tolist())
                train.extend(rng.choice(candidates,n,replace=False).tolist())
            reverse=pool.loc[(pool.site.isin(a)&pool.label.eq(0))|(pool.site.isin(t)&pool.label.eq(1)),'FILE_ID'].tolist() if arm=='P' else []
            assert not set(train)&set(test) and set(pool.set_index('FILE_ID').loc[train,'site'])==set(all_sites)
            jobs.append(context|dict(arm=arm,train=train,reverse=reverse,feasible_assignments=len(feasible)))
    context=dict(repeat=repeat,seed=repeat,k=0,n=0,test_sites=test_sites,asd_sites=[],td_sites=[],test=test,reverse=[],train=pool.FILE_ID.tolist())
    jobs.extend(context|dict(arm=arm) for arm in ['F','FD'])
    return jobs


def select_model(x,y,indices,seed):
    pipe=Pipeline([('scale',StandardScaler()),('lr',LogisticRegression())])
    cv=StratifiedKFold(5,shuffle=True,random_state=seed)
    return GridSearchCV(pipe,{'lr__C':GRID},scoring='roc_auc',cv=cv,n_jobs=1,error_score='raise').fit(x[indices],y[indices])


# 缺失头动只在每个训练部分内估计中位数，连接特征没有缺失填补。
def motion_model(x,y,indices,seed):
    from sklearn.impute import SimpleImputer
    pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('lr',LogisticRegression())])
    cv=StratifiedKFold(5,shuffle=True,random_state=seed)
    return GridSearchCV(pipe,{'lr__C':GRID},scoring='roc_auc',cv=cv,n_jobs=1,error_score='raise').fit(x[indices],y[indices])


def dump(name,obj):
    import json
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))


def append(name,frame):
    with (OUT/name).open('a') as f:frame.to_csv(f,index=False,header=f.tell()==0);f.flush()


def prepare():
    import hashlib
    import json
    from time import perf_counter
    from importlib.metadata import version
    OUT.mkdir();CACHE.mkdir(parents=True,exist_ok=True);table=subjects();values=[];rows=[];hashes={};start=perf_counter()
    print('START T12A feature extraction',flush=True)
    for row in table.itertuples():
        path=ROOT/'rois_cc200'/(row.FILE_ID+'_rois_cc200.1D');x=np.loadtxt(path,skiprows=1)
        v,zero=fisher_features(x);values.append(v);rows.append(dict(FILE_ID=row.FILE_ID,n_zero_roi=zero,n_timepoints=len(x)))
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    np.save(CACHE/'fisher_z.npy',np.array(values));table[['FILE_ID']].to_csv(CACHE/'row_order.csv',index=False)
    counts=pd.DataFrame(rows);table=table.merge(counts,on='FILE_ID',validate='one_to_one');assert table.n_zero_roi.gt(0).sum()==46
    table.to_csv(OUT/'subjects.csv',index=False);table[['FILE_ID','site','n_zero_roi','n_timepoints']].to_csv(OUT/'zero_variance_ROI.csv',index=False)
    jobs=[]
    for sens in [False,True]:
        f=table.loc[table.n_zero_roi.eq(0)].copy() if sens else table
        assert f.site.nunique()==17
        for repeat in range(50):
            design=repeat_design(f,repeat,site_curve_only=sens)
            for job in design:
                if sens and job['arm'] in ['F','FD']:continue
                job.update(no_zero_roi=sens,job_id=f'nz{int(sens)}_r{repeat}_{job["arm"]}_k{job["k"]}_n{job["n"]}')
                jobs.append(job)
    assert len(jobs)==1100
    meta=table.set_index('FILE_ID');audit=[]
    for j in jobs:
        tr=meta.loc[j['train']];te=meta.loc[j['test']]
        assert set(tr.site).isdisjoint(te.site) and set(tr.index).isdisjoint(te.index)
        if j['arm'] in ['P','M']:
            assert tr.label.value_counts().to_dict()=={0:j['n'],1:j['n']}
            assert set(tr.site)==set(j['asd_sites']+j['td_sites'])
            assert tr.groupby('site').label.nunique().eq(1 if j['arm']=='P' else 2).all()
        if j['reverse']:
            rv=meta.loc[j['reverse']];assert set(rv.index).isdisjoint(tr.index) and rv.label.eq(rv.site.map(tr.groupby('site').label.first()).rsub(1)).all()
        audit.append(dict(job_id=j['job_id'],n_train=len(tr),n_test=len(te),n_reverse=len(j['reverse']),n_train_sites=tr.site.nunique(),n_test_sites=te.site.nunique()))
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs));pd.DataFrame(audit).to_csv(OUT/'split_counts.csv',index=False)
    pd.crosstab(table.site,table.DX_GROUP).to_csv(OUT/'site_counts.csv')
    inputs=[ROOT/'Phenotypic_V1_0b_preprocessed1.csv',ROOT/'README_local.md',ROOT/'file_ids.txt']
    hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    dump('input_sha256.json',hashes)
    dump('config.json',dict(task='T12A',stage='prepared',seeds=list(range(50)),CPU_process_limit=6,worker_processes=5,BLAS_threads_per_process=1,
        feature='CC200 Pearson,Fisher z,clip +/-0.999999,upper triangle row-major19900,float64;zero-variance ROI correlations=0',
        site_merge='SITE_ID regex _[12]$ removed',settings=SETTINGS,main_jobs=700,sensitivity_jobs=400,C=GRID,
        LR=dict(penalty='L2',solver='lbfgs',max_iter=100,tol=.0001,class_weight=None),inner='5-fold subject-stratified AUC,shuffle,random_state=repeat;StandardScaler fit inside every fold',
        missing_fd='training-part median inside pipeline per inner fold and final fit,user approved',
        assignment='uniform over all feasible labeled disjoint ASD/TD site subsets jointly feasible for P/M;subject draws without replacement then random trim per class',
        repeat_stream='numpy default_rng(repeat);test sites once then settings in fixed order;P followed by M;new identical seed stream for sensitivity',
        sensitivity='remove exactly46 zero-variance-ROI people;only k curve,P/M;test sites same seeds as main,subject/feasible assignment draws regenerated',
        intervals='2.5/97.5 percentiles of 50 repeat metrics or paired differences;variability range,NOT independent-sample confidence interval',
        extraction_seconds=perf_counter()-start,versions={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','joblib']},
        cache=str(CACHE),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('tests/test_abide_class_pure_sites.py'),Path('docs/tasks/T12A_abide_class_pure_sites.md'),Path('AGENTS.md')]},
        cache_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CACHE/'fisher_z.npy',CACHE/'row_order.csv']},
        splits_sha256=hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()))
    print('FEATURES_DONE subjects=1035,zero_ROI_subjects=46,outer_jobs=1100',flush=True)


def fit_job(job,data=None):
    import warnings
    from collections import Counter
    from time import perf_counter
    tick=perf_counter();table=pd.read_csv(OUT/'subjects.csv') if data is None else data[0];lookup=pd.Index(table.FILE_ID);train=lookup.get_indexer(job['train']);test=lookup.get_indexer(job['test']);reverse=lookup.get_indexer(job['reverse'])
    assert (train>=0).all() and (test>=0).all() and (reverse>=0).all()
    x=(table[['func_mean_fd']].to_numpy() if job['arm']=='FD' else np.load(CACHE/'fisher_z.npy',mmap_mode='r')) if data is None else data[1]
    if data is None:assert pd.read_csv(CACHE/'row_order.csv').FILE_ID.tolist()==table.FILE_ID.tolist()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');fitted=(motion_model if job['arm']=='FD' else select_model)(x,table.label.to_numpy(),train,job['seed'])
    context={k:job[k] for k in ['job_id','no_zero_roi','repeat','seed','arm','k','n']};parts=[]
    for is_reverse,idx in [(False,test)]+([(True,reverse)] if len(reverse) else []):
        frame=table.iloc[idx].copy()
        for key,value in context.items():frame[key]=value
        frame['reverse']=is_reverse;frame['probability']=fitted.predict_proba(x[idx])[:,1];frame['prediction']=frame.probability.ge(.5);parts.append(frame)
    cv=fitted.cv_results_;diagnostics=[]
    for i,c in enumerate(GRID):diagnostics.append(context|dict(C=c,inner_mean_AUC=cv['mean_test_score'][i],inner_sd_AUC=cv['std_test_score'][i],**{f'fold{fold}_AUC':cv[f'split{fold}_test_score'][i] for fold in range(5)}))
    splits=[]
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=job['seed']).split(train,table.label.to_numpy()[train])):
        splits.append(context|dict(inner_fold=fold,train=table.iloc[train[a]].FILE_ID.tolist(),validation=table.iloc[train[b]].FILE_ID.tolist()))
    warning_rows=[context|dict(category=category,message=message,count=count) for (category,message),count in Counter((w.category.__name__,str(w.message)) for w in caught).items()]
    done=context|dict(C=float(fitted.best_params_['lr__C']),best_inner_AUC=float(fitted.best_score_),n_train=len(train),n_test=len(test),n_reverse=len(reverse),seconds=perf_counter()-tick,final_n_iter=int(fitted.best_estimator_.named_steps['lr'].n_iter_.max()),warning_count=len(caught))
    return dict(pred=pd.concat(parts,ignore_index=True),done=done,diagnostics=pd.DataFrame(diagnostics),splits=splits,warnings=pd.DataFrame(warning_rows))


def save_result(result):
    import json
    for name,key in [('predictions.csv','pred'),('fit_diagnostics.csv','diagnostics')]:append(name,result[key])
    if len(result['warnings']):append('warnings.csv',result['warnings'])
    with (OUT/'inner_splits.jsonl').open('a') as f:
        for s in result['splits']:f.write(json.dumps(s)+'\n')
    append('completed.csv',pd.DataFrame([result['done']]))


def read_jobs():
    import json
    return [json.loads(s) for s in (OUT/'splits.jsonl').read_text().splitlines()]


def timing():
    import json
    from time import perf_counter
    from joblib import Parallel,delayed,parallel_config
    jobs=read_jobs();selected=[next(j for j in jobs if j['repeat']==0 and not j['no_zero_roi'] and j['arm']==arm and j['k']==k and j['n']==n) for arm,k,n in [('P',1,40),('M',1,40),('P',2,80),('M',2,80),('F',0,0),('FD',0,0)]]
    print('START T12A timing;6 full nested outer jobs',flush=True);start=perf_counter();rows=[]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=5,return_as='generator_unordered')(delayed(fit_job)(j) for j in selected)
        for result in results:
            save_result(result);rows.append(result['done']);print('TIMING_JOB_DONE '+json.dumps(result['done']),flush=True)
    elapsed=perf_counter()-start;rates=pd.DataFrame(rows);conn=rates.loc[rates.arm.isin(['P','M'])];fit=conn.seconds/conn.n_train
    # 计入主/敏感性每个实际训练人数，F和FD分别按完整测量估计；5工作进程。
    f_rate=float(rates.loc[rates.arm.eq('F'),'seconds'].iloc[0]/rates.loc[rates.arm.eq('F'),'n_train'].iloc[0]);fd=float(rates.loc[rates.arm.eq('FD'),'seconds'].iloc[0]);worker_seconds=0.
    for j in jobs:
        worker_seconds+=fd if j['arm']=='FD' else f_rate*len(j['train']) if j['arm']=='F' else float(fit.mean())*len(j['train'])
    point=worker_seconds/5+elapsed+120
    result=dict(point_hours=point/3600,with_25percent_margin_hours=point*1.25/3600,worker_seconds=worker_seconds,pilot_wall_seconds=elapsed,pilot_jobs=rows,postprocess_reserved_seconds=120,all_outer_jobs=1100,remaining_outer_jobs=1094,pilot_predictions_reused=True)
    dump('timing.json',result);cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='timed',timing=result);dump('config.json',cfg)
    print('TIMING_COMPLETE '+json.dumps({k:result[k] for k in ['point_hours','with_25percent_margin_hours','pilot_wall_seconds']}),flush=True)


def run():
    import json
    from datetime import datetime,timezone,timedelta
    from joblib import Parallel,delayed,parallel_config
    cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='running',started_at=datetime.now(timezone(timedelta(hours=8))).isoformat());dump('config.json',cfg)
    done=set(pd.read_csv(OUT/'completed.csv').job_id);jobs=[j for j in read_jobs() if j['job_id'] not in done]
    print('START T12A formal remaining_outer_jobs='+str(len(jobs)),flush=True)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for result in Parallel(n_jobs=5,return_as='generator_unordered')(delayed(fit_job)(j) for j in jobs):
            save_result(result);print('TRAIN_DONE '+json.dumps(result['done']),flush=True)
    cfg.update(stage='training_complete',finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat());dump('config.json',cfg);print('ALL_TRAINING_COMPLETE outer_jobs=1100',flush=True)


if __name__=='__main__':
    import sys
    {'prepare':prepare,'timing':timing,'run':run}[sys.argv[1]]()
