"""T12-B：沿用T12-A纯类名单，嵌套B0估计与固定站点结构方法。"""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import abide_class_pure_sites as a

OUT=Path('outputs/T12B');KEY=['method','repeat','k'];CONTEXT=['job_id','method','repeat','seed','k','n','T12A_job_id'];METHODS=['B0','B1','B2','B3','E_subj','E_LSPO']


def dump(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))


def append(name,frame):
    with (OUT/name).open('a') as f:frame.to_csv(f,index=False,header=f.tell()==0);f.flush()


def site_pairs(table,indices):
    indices=np.asarray(indices);t=table.loc[indices];assert t.groupby('site').label.nunique().eq(1).all()
    asd=sorted(t.loc[t.label.eq(1),'site'].unique());td=sorted(t.loc[t.label.eq(0),'site'].unique());folds=[]
    for s in asd:
        for u in td:
            held=t.site.isin([s,u]).to_numpy();folds.append((indices[~held],indices[held]))
    return folds


def site_offsets(x,table,indices):
    t=table.loc[indices];sites=sorted(t.site.unique());means=np.array([x[t.index[t.site.eq(s)]].mean(axis=0) for s in sites])
    labels=np.array([t.loc[t.site.eq(s),'label'].iloc[0] for s in sites]);delta=means.copy()
    for label in [0,1]:delta[labels==label]-=means[labels==label].mean(axis=0)
    return delta


def nap_fit(x,table,indices):
    scale=StandardScaler().fit(x[indices]);z=scale.transform(x[indices]);t=table.loc[indices].reset_index(drop=True)
    delta=site_offsets(z,t,np.arange(len(t)));_,s,vt=np.linalg.svd(delta,full_matrices=False)
    rank=int((s>max(delta.shape)*np.finfo(float).eps*s[0]).sum());u=vt[:rank].T
    assert rank<=t.site.nunique()-2
    return scale,u


def nap_transform(x,scale,u):
    z=scale.transform(x);return z-(z@u)@u.T


def site_standardize(x,table,train,apply):
    train=np.asarray(train);apply=np.asarray(apply);result=np.empty((len(apply),x.shape[1]))
    for site in sorted(table.loc[apply,'site'].unique()):
        tr=train[table.loc[train,'site'].eq(site).to_numpy()];mask=table.loc[apply,'site'].eq(site).to_numpy()
        fit=tr if len(tr) else apply[mask]
        result[mask]=StandardScaler().fit(x[fit]).transform(x[apply[mask]])
    return result


def jobs():
    return [j for j in a.read_jobs() if not j['no_zero_roi'] and j['arm']=='P' and j['n']==40 and j['k'] in [2,4,6]]


def data():
    table=pd.read_csv(a.OUT/'subjects.csv');assert table.FILE_ID.tolist()==pd.read_csv(a.CACHE/'row_order.csv').FILE_ID.tolist()
    return table,np.load(a.CACHE/'fisher_z.npy',mmap_mode='r')


def subject_folds(table,train,seed):
    return [(train[tr],train[va]) for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(train,table.loc[train,'label'])]


def transformed(x,table,train,test,method):
    if method=='B2':
        scale,u=nap_fit(x,table,train);return nap_transform(x[train],scale,u),nap_transform(x[test],scale,u),u.shape[1]
    if method=='B3':return site_standardize(x,table,train,train),site_standardize(x,table,train,test),0
    scale=StandardScaler().fit(x[train]);return scale.transform(x[train]),scale.transform(x[test]),0


def selected_method(x,table,train,test,method,seed,context):
    folds=subject_folds(table,train,seed) if method=='B3' else site_pairs(table,train);diagnostics=[];splits=[];scores=[]
    for fold,(tr,va) in enumerate(folds):
        z,v,rank=transformed(x,table,tr,va,method);values=[]
        for c in a.GRID:
            net=LogisticRegression(C=c).fit(z,table.loc[tr,'label']);auc=roc_auc_score(table.loc[va,'label'],net.predict_proba(v)[:,1]);values.append(auc)
            diagnostics.append(context|dict(inner_fold=fold,C=c,AUC=auc,U_rank=rank,n_train=len(tr),n_validation=len(va)))
        scores.append(values);splits.append(context|dict(role='selection',fold=fold,train=table.loc[tr,'FILE_ID'].tolist(),validation=table.loc[va,'FILE_ID'].tolist(),U_fit=table.loc[tr,'FILE_ID'].tolist() if method=='B2' else []))
    mean=np.mean(scores,axis=0);c=a.GRID[int(np.argmax(mean))];z,v,rank=transformed(x,table,train,test,method)
    net=LogisticRegression(C=c).fit(z,table.loc[train,'label'])
    return net.predict_proba(v)[:,1],dict(C=c,U_rank=rank,selection_AUC=float(mean.max()),final_n_iter=int(net.n_iter_.max())),diagnostics,splits


def b0_fit(x,table,train,test,seed,context,outer_fold):
    fitted=a.select_model(x,table.label.to_numpy(),train,seed);cv=fitted.cv_results_;diagnostics=[];splits=[]
    for fold,(tr,va) in enumerate(subject_folds(table,train,seed)):
        splits.append(context|dict(role='B0_inner',fold=fold,outer_fold=outer_fold,train=table.loc[tr,'FILE_ID'].tolist(),validation=table.loc[va,'FILE_ID'].tolist(),U_fit=[]))
        for i,c in enumerate(a.GRID):diagnostics.append(context|dict(inner_fold=fold,outer_fold=outer_fold,C=c,AUC=float(cv[f'split{fold}_test_score'][i]),U_rank=0,n_train=len(tr),n_validation=len(va)))
    details=dict(C=float(fitted.best_params_['lr__C']),U_rank=0,selection_AUC=float(fitted.best_score_),final_n_iter=int(fitted.best_estimator_.named_steps['lr'].n_iter_.max()))
    return fitted.predict_proba(x[test])[:,1],details,diagnostics,splits


def fit_job(task,data_override=None):
    job,method=task;tick=perf_counter();table,x=data() if data_override is None else data_override;lookup=pd.Index(table.FILE_ID);train=lookup.get_indexer(job['train']);test=lookup.get_indexer(job['test']);reverse=lookup.get_indexer(job['reverse'])
    context=dict(job_id=f'{method}_r{job["repeat"]}_k{job["k"]}',method=method,repeat=job['repeat'],seed=job['seed'],k=job['k'],n=job['n'],T12A_job_id=job['job_id'])
    diagnostics=[];splits=[];folds=[];parts=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if method in ['E_subj','E_LSPO']:
            outer=subject_folds(table,train,job['seed']) if method=='E_subj' else site_pairs(table,train)
            for fold,(tr,va) in enumerate(outer):
                probability,details,d,s=b0_fit(x,table,tr,va,job['seed'],context,fold);diagnostics.extend(d);splits.extend(s)
                auc=roc_auc_score(table.loc[va,'label'],probability);folds.append(context|dict(outer_fold=fold,AUC=auc,n_train=len(tr),n_validation=len(va),**details))
                splits.append(context|dict(role='estimate_outer',fold=fold,train=table.loc[tr,'FILE_ID'].tolist(),validation=table.loc[va,'FILE_ID'].tolist(),U_fit=[]))
                p=table.loc[va].copy()
                for key,value in context.items():p[key]=value
                p['outer_fold']=fold;p['probability']=probability;p['prediction']=p.probability.ge(.5);parts.append(p)
            pred=pd.concat(parts,ignore_index=True)
            estimate=roc_auc_score(pred.label,pred.probability) if method=='E_subj' else float(np.mean([q['AUC'] for q in folds]));details=dict(estimate=estimate)
        else:
            apply=np.r_[test,reverse]
            if method=='B0':probability,details,diagnostics,splits=b0_fit(x,table,train,apply,job['seed'],context,-1)
            else:probability,details,diagnostics,splits=selected_method(x,table,train,apply,method,job['seed'],context)
            pred=table.loc[apply].copy()
            for key,value in context.items():pred[key]=value
            pred['reverse']=np.r_[np.zeros(len(test),bool),np.ones(len(reverse),bool)];pred['probability']=probability;pred['prediction']=pred.probability.ge(.5)
    warn=[context|dict(category=c,message=m,count=n) for (c,m),n in Counter((w.category.__name__,str(w.message)) for w in caught).items()]
    return dict(pred=pred,done=context|details|dict(seconds=perf_counter()-tick,warning_count=len(caught)),diagnostics=pd.DataFrame(diagnostics),splits=splits,folds=pd.DataFrame(folds),warnings=pd.DataFrame(warn))


def save_result(result):
    name='estimation_predictions.csv' if result['done']['method'].startswith('E_') else 'predictions.csv'
    append(name,result['pred']);append('selection_fits.csv',result['diagnostics'].reindex(columns=CONTEXT+['inner_fold','outer_fold','C','AUC','U_rank','n_train','n_validation']))
    if len(result['folds']):append('estimation_folds.csv',result['folds'])
    if len(result['warnings']):append('warnings.csv',result['warnings'])
    with (OUT/'inner_splits.jsonl').open('a') as f:
        for s in result['splits']:f.write(json.dumps(s)+'\n')
    append('completed.csv',pd.DataFrame([result['done']]).reindex(columns=CONTEXT+['C','U_rank','selection_AUC','final_n_iter','estimate','seconds','warning_count']))


def prepare():
    import shutil
    OUT.mkdir();table,_=data();table.to_csv(OUT/'subjects.csv',index=False);work=jobs();assert len(work)==150
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in work))
    sources=['site_pair_validation.py','tests/test_site_pair_validation.py','abide_class_pure_sites.py','docs/tasks/T12B_site_pair_validation.md','AGENTS.md']
    inputs=[str(a.CACHE/'fisher_z.npy'),str(a.CACHE/'row_order.csv')]+['outputs/T12A/'+f for f in ['splits.jsonl','subjects.csv','predictions.csv','main_table.csv','repeat_metrics.csv','completed.csv']]
    dump('config.json',dict(task='T12B',stage='prepared',seeds=list(range(50)),k=[2,4,6],n=40,CPU_process_limit=6,worker_processes=5,BLAS_threads_per_process=1,methods=METHODS,C=a.GRID,
        LR='T12A defaults:L2,lbfgs,max_iter100,tol1e-4,no class weight,no warm start',
        B0='exact T12A select_model;subject StratifiedKFold5,shuffle,seed=repeat',
        LSPO='sorted ASD sites x sorted TD sites;equal mean pair AUC;not pooled over pairs',
        estimates='outer E_subj pooled5fold OOF AUC;outer E_LSPO mean pair AUC;inner both B0 subject5fold seed=repeat',
        B2='train StandardScaler;site means minus equal-site mean within diagnostic class;SVD all nonzero directions;repeat per LSPO fold',
        rank_cutoff='s>max(delta.shape)*eps_float64*s[0];k2 inner one site per class gives empty U',
        B3='user confirmed:each inner training-site scaler fit training subjects;final P-site scalers apply reverse;new test-site scaler fit all that test site unlabeled;no second global scaling',
        correlation='user confirmed Pearson r across50 repeats per k/estimator/truth;no correlation p test',
        intervals='repeat2.5/97.5 percentiles;variability,not independent-sample CI',BNU_watch_not_applicable=True,
        expected_method_jobs=600,expected_estimation_jobs=300,expected_LR_fits=103550,
        source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},input_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in inputs}))
    for src,name in [('/tmp/T12B_pytest_red.log','pytest_red.log'),('/tmp/T12B_pytest_green.log','pytest_task.log')]:shutil.copyfile(src,OUT/name)


def parallel(tasks,label):
    from joblib import Parallel,delayed,parallel_config
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for result in Parallel(n_jobs=5,return_as='generator_unordered')(delayed(fit_job)(t) for t in tasks):
            save_result(result);print(label+' '+json.dumps(result['done']),flush=True)


def reproduce():
    prepare();print('START T12B B0 reproduction:150 jobs',flush=True);parallel([(j,'B0') for j in jobs()],'B0_DONE')
    old=pd.read_csv(a.OUT/'predictions.csv',float_precision='round_trip');old=old.loc[~old.no_zero_roi&old.arm.eq('P')&old.n.eq(40)&old.k.isin([2,4,6])]
    new=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');q=new.merge(old,on=['FILE_ID','repeat','k','reverse'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(q)==len(new)==len(old);maximum=float(abs(q.probability_new-q.probability_old).max());assert maximum==0
    assert np.array_equal(q.prediction_new,q.prediction_old)
    result=dict(jobs=150,predictions=len(q),max_probability_difference=maximum,predictions_identical=True)
    dump('B0_reproduction.json',result);cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='B0_reproduced',B0_reproduction=result);dump('config.json',cfg);print('B0_REPRODUCTION_PASS '+json.dumps(result),flush=True)


def timing():
    print('START T12B timing',flush=True);tick=perf_counter();selected=[(next(j for j in jobs() if j['repeat']==0 and j['k']==k),method) for k in [2,6] for method in ['E_subj','E_LSPO','B1','B2','B3']]
    parallel(selected,'TIMING_DONE');elapsed=perf_counter()-tick;done=pd.read_csv(OUT/'completed.csv');pilot=done.loc[done.method.ne('B0')];lookup={(q.method,q.k):q.seconds for q in pilot.itertuples()};worker_seconds=0.
    for method in METHODS[1:]:
        for k in [2,4,6]:
            rate=lookup[method,k] if k in [2,6] else (lookup[method,2]+lookup[method,6])/2
            if method in ['B1','B2','E_LSPO'] and k==4:rate=lookup[method,2]+(lookup[method,6]-lookup[method,2])*(16-4)/(36-4)
            worker_seconds+=rate*50
    b0_worker_seconds=float(done.loc[done.method.eq('B0'),'seconds'].sum());point=(worker_seconds+b0_worker_seconds)/5+120;result=dict(point_hours=point/3600,with_25percent_margin_hours=point*1.25/3600,worker_seconds=worker_seconds,B0_worker_seconds=b0_worker_seconds,pilot_wall_seconds=elapsed,pilot_jobs=10,all_jobs=900,completed_jobs=160,remaining_jobs=740,pilot_results_reused=True)
    dump('timing.json',result);cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='timed',timing=result);dump('config.json',cfg);print('TIMING_COMPLETE '+json.dumps(result),flush=True)


def run():
    from datetime import datetime,timezone,timedelta
    cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='running',started_at=datetime.now(timezone(timedelta(hours=8))).isoformat());dump('config.json',cfg)
    done=set(pd.read_csv(OUT/'completed.csv').job_id);tasks=[(j,m) for j in jobs() for m in METHODS if f'{m}_r{j["repeat"]}_k{j["k"]}' not in done]
    print('START T12B remaining_jobs='+str(len(tasks)),flush=True);parallel(tasks,'TRAIN_DONE');cfg.update(stage='training_complete',finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat());dump('config.json',cfg);print('ALL_TRAINING_COMPLETE jobs=900',flush=True)


if __name__=='__main__':
    import sys
    {'reproduce':reproduce,'timing':timing,'run':run}[sys.argv[1]]()
