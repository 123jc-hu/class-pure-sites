"""T13-A：固定站点条件下的LR与站点内配对目标。"""
from pathlib import Path
import json,hashlib,warnings
from collections import Counter
from time import perf_counter
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import abide_class_pure_sites as a

OUT=Path('outputs/T13A');GRID=[.001,.01,.1,1.,10.]


def subjects():return pd.read_csv('outputs/T12A/subjects.csv')


def source_jobs():
    primary=[j for j in map(json.loads,Path('outputs/T12A/splits.jsonl').read_text().splitlines()) if not j['no_zero_roi'] and j['arm']=='F']
    secondary=[j for j in map(json.loads,Path('outputs/T12C/splits.jsonl').read_text().splitlines()) if j['method']=='C0' and j['k']==2 and j['b']==5]
    return primary,secondary


def inner_folds(table,train,seed,n_splits=5):
    t=table.loc[train];strata=t.site+':'+t.label.astype(str)
    return [(train[tr],train[va]) for tr,va in StratifiedKFold(n_splits,shuffle=True,random_state=seed).split(train,strata)]


def pairs(y,sites):
    left=[];right=[]
    for site in np.unique(sites):
        a=np.flatnonzero((sites==site)&(y==1));t=np.flatnonzero((sites==site)&(y==0));left.append(np.repeat(a,len(t)));right.append(np.tile(t,len(a)))
    return np.concatenate(left),np.concatenate(right)


def pair_auc(y,sites,score):
    a,t=pairs(y,sites);assert len(a)>0
    return float(np.mean((score[a]>score[t])+.5*(score[a]==score[t])))


def site_means(z,sites):
    keys,inverse=np.unique(sites,return_inverse=True);means=np.array([z[inverse==i].mean(axis=0) for i in range(len(keys))])
    return keys,inverse,means


def fit_weight(z,y,sites,model,C):
    n,p=z.shape;keys,inverse,means=site_means(z,sites);x=z-means[inverse];penalty=1/(C*n)
    # 中心化仅重参数化自由站点截距/配对差值，保持原目标与w的惩罚不变。
    if model=='L1':
        def objective(theta):
            w=theta[:p];eta=x@w+theta[p:][inverse];r=(expit(eta)-y)/n
            value=np.mean(np.logaddexp(0,eta)-y*eta)+penalty*(w@w)/2
            return value,np.r_[x.T@r+penalty*w,np.bincount(inverse,weights=r,minlength=len(keys))]
        initial=np.zeros(p+len(keys))
    else:
        a,t=pairs(y,sites);assert len(a)>0
        def objective(w):
            score=x@w;delta=score[a]-score[t];r=expit(-delta)/len(a);g=np.bincount(t,weights=r,minlength=n)-np.bincount(a,weights=r,minlength=n)
            return np.logaddexp(0,-delta).mean()+penalty*(w@w)/2,x.T@g+penalty*w
        initial=np.zeros(p)
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=5000,gtol=1e-8,ftol=1e-12,maxls=50))
    assert result.success,result.message
    w=result.x[:p];alpha=result.x[p:]-means@w if model=='L1' else np.empty(0)
    return dict(w=w,alpha=alpha,sites=keys,n_iter=int(result.nit),n_fun=int(result.nfev),objective=float(result.fun),gradient_max=float(abs(result.jac).max()),success=bool(result.success),message=str(result.message))


def fit_model(x,table,train,model,C):
    scale=StandardScaler().fit(x[train]);z=scale.transform(x[train]);t=table.loc[train];sites=t.site.to_numpy();y=t.label.to_numpy();keys,inverse,means=site_means(z,sites)
    if model in ['L0','L2']:
        net=LogisticRegression(C=C).fit(z-means[inverse] if model=='L2' else z,y)
        result=dict(w=net.coef_[0],b=float(net.intercept_[0]),n_iter=int(net.n_iter_.max()),n_fun=None,objective=None,gradient_max=None,success=True,message='sklearn defaults;convergence warnings recorded')
    else:result=fit_weight(z,y,sites,model,C)|dict(b=0.)
    return result|dict(scale=scale,model=model,sites=keys,means=means)


def transform(fit,x,table,indices):
    z=fit['scale'].transform(x[indices])
    if fit['model']=='L2':
        sites=table.loc[indices,'site'].to_numpy()
        for site in np.unique(sites):
            mask=sites==site
            mean=fit['means'][np.flatnonzero(fit['sites']==site)[0]] if site in fit['sites'] else z[mask].mean(axis=0)
            z[mask]-=mean
    return z


def decision(fit,x,table,indices):return transform(fit,x,table,indices)@fit['w']+fit['b']


def validation_auc(table,indices,score,setting):
    t=table.loc[indices]
    if setting=='secondary':return pair_auc(t.label.to_numpy(),t.site.to_numpy(),score)
    values=[]
    for site in sorted(t.site.unique()):
        mask=t.site.eq(site).to_numpy();y=t.label.to_numpy()[mask]
        if min(np.sum(y==0),np.sum(y==1))>=2:values.append(roc_auc_score(y,score[mask]))
    return float(np.mean(values))


def plans():
    primary,secondary=source_jobs();work=[]
    for setting,source,models in [('main',primary,['L0','L1','L2','L3']),('secondary',secondary,['L3'])]:
        for j in source:
            for model in models:
                direction='main' if setting=='main' else j['direction']
                work.append(j|dict(job_id=f'{setting}_{model}_r{j["repeat"]}_{direction}',setting=setting,model=model,method=model,k=0 if setting=='main' else 2,b=0 if setting=='main' else 5,direction=direction,n_pure=len(j['train']) if setting=='main' else 80,n_added=0 if setting=='main' else len(j['bridge']),n_train=len(j['train']),T12A_job_id=j['job_id'] if setting=='main' else j['T12A_job_id'],source_job_id=j['job_id']))
    return work


def fit_job(job,cache=a.CACHE,n_splits=5,data=None):
    tick=perf_counter();table=subjects() if data is None else data[0];lookup=pd.Index(table.FILE_ID);x=np.load(cache/'fisher_z.npy',mmap_mode='r') if data is None else data[1];train=lookup.get_indexer(job['train']);target=lookup.get_indexer(job['test']+job['reverse'])
    assert (train>=0).all() and (target>=0).all() and set(train).isdisjoint(target)
    keys=['job_id','setting','model','method','k','b','direction','repeat','seed','n_pure','n_added','n_train','T12A_job_id'];context={key:job[key] for key in keys};diagnostics=[];splits=[];scores=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for fold,(tr,va) in enumerate(inner_folds(table,train,job['seed'],n_splits)):
            assert set(tr).isdisjoint(va) and set(tr).union(va).isdisjoint(target)
            pair_a,_=pairs(table.loc[va,'label'].to_numpy(),table.loc[va,'site'].to_numpy());assert len(pair_a)>0
            values=[]
            for C in GRID:
                fit=fit_model(x,table,tr,job['model'],C);score=decision(fit,x,table,va);value=validation_auc(table,va,score,job['setting']);values.append(value)
                diagnostics.append(context|dict(role='selection',inner_fold=fold,C=C,AUC=value,validation_pairs=len(pair_a),n_actual_train=len(tr),n_iter=fit['n_iter'],n_fun=fit['n_fun'],objective=fit['objective'],gradient_max=fit['gradient_max'],success=fit['success'],message=fit['message']))
            scores.append(values);splits.append(context|dict(inner_fold=fold,train=table.loc[tr,'FILE_ID'].tolist(),validation=table.loc[va,'FILE_ID'].tolist(),standardizer_fit=table.loc[tr,'FILE_ID'].tolist(),site_mean_fit=table.loc[tr,'FILE_ID'].tolist(),validation_target_mean_used=False))
        mean=np.mean(scores,axis=0);C=GRID[int(np.argmax(mean))];fit=fit_model(x,table,train,job['model'],C);score=decision(fit,x,table,target)
    diagnostics.append(context|dict(role='final',inner_fold=-1,C=C,AUC=None,validation_pairs=None,n_actual_train=len(train),n_iter=fit['n_iter'],n_fun=fit['n_fun'],objective=fit['objective'],gradient_max=fit['gradient_max'],success=fit['success'],message=fit['message']))
    pred=table.loc[target].copy()
    for key,value in context.items():pred[key]=value
    pred['reverse']=np.r_[np.zeros(len(job['test']),bool),np.ones(len(job['reverse']),bool)];pred['score']=score;pred['probability']=expit(score);pred['prediction']=score>=0
    warn=[context|dict(category=category,message=message,count=count) for (category,message),count in Counter((w.category.__name__,str(w.message)) for w in caught).items()]
    return dict(pred=pred,done=context|dict(C=C,selection_AUC=float(mean.max()),seconds=perf_counter()-tick,final_n_iter=fit['n_iter'],warning_count=len(caught)),diagnostics=pd.DataFrame(diagnostics),splits=splits,warnings=pd.DataFrame(warn),provenance=context|dict(train=job['train'],test=job['test'],reverse=job['reverse'],standardizer_fit=job['train'],site_mean_fit=job['train'],unlabelled_test_mean_ids=job['test'] if job['model']=='L2' else []))


def append(name,frame):
    with (OUT/name).open('a') as f:frame.to_csv(f,index=False,header=f.tell()==0);f.flush()


def dump(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def save(result):
    append('predictions.csv',result['pred']);append('completed.csv',pd.DataFrame([result['done']]));append('solver_fits.csv',result['diagnostics'])
    if len(result['warnings']):append('warnings.csv',result['warnings'])
    with (OUT/'inner_splits.jsonl').open('a') as f:
        for row in result['splits']:f.write(json.dumps(row)+'\n')
    with (OUT/'fit_provenance.jsonl').open('a') as f:f.write(json.dumps(result['provenance'])+'\n')


def parallel(work):
    from joblib import Parallel,delayed,parallel_config
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for result in Parallel(n_jobs=5,return_as='generator_unordered')(delayed(fit_job)(j) for j in work):
            save(result);print('TRAIN_DONE '+json.dumps(result['done']),flush=True)


def prepare():
    import shutil
    from importlib.metadata import version
    OUT.mkdir();table=subjects();assert table.FILE_ID.tolist()==pd.read_csv(a.CACHE/'row_order.csv').FILE_ID.tolist();table.to_csv(OUT/'subjects.csv',index=False);work=plans();assert len(work)==300
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in work));shutil.copyfile('/tmp/T13A_secondary_preflight.csv',OUT/'secondary_preflight.csv')
    sources=['within_site_pair_model.py','within_site_pair_report.py','tests/test_within_site_pair_model.py','docs/tasks/T13A_within_site_pair_model.md','AGENTS.md']
    inputs=[a.CACHE/'fisher_z.npy',a.CACHE/'row_order.csv']+[Path('outputs')/task/name for task,names in [('T12A',['splits.jsonl','subjects.csv','predictions.csv']),('T12C',['splits.jsonl','predictions.csv'])] for name in names]
    dump('config.json',dict(task='T13A',stage='prepared',created_at=datetime.now().astimezone().isoformat(),seeds=list(range(50)),models=['L0','L1','L2','L3'],main_jobs=200,secondary_jobs=100,expected_fits=7800,C=GRID,CPU_process_limit=6,workers=5,BLAS_threads=1,LR='sklearn defaults:L2,lbfgs,max_iter100,tol1e-4,no class weights',custom_solver=dict(method='L-BFGS-B',maxiter=5000,gtol=1e-8,ftol=1e-12,maxls=50,initial='zero,no warm starts'),objective='L1 subject-mean logistic;L3 mean all same-site ASD-TD pairs;both +||w||2/(2*C*N);N=all actual training subjects,including unpaired sites',L1='one free intercept per training site,no separate global intercept;site intercepts omitted for all evaluation;within-training-site centering only exact intercept reparameterization',L2='train-only StandardScaler first,no second scaling;known validation sites use inner-training means;new test sites use their own unlabelled cohort means,transductive;no labels in means',L3='no intercept;within-training-site centering leaves every pair difference identical;score remains w dot original standardized evaluation x',inner='both settings site-by-diagnosis StratifiedKFold5 shuffle seed=repeat',main_selection='equal mean validation-site AUC;both classes at least2',secondary_selection='all same-site validation pairs pooled;correct proportion,tie.5;both classes at least1;no repeat/fold skipped',selection_aggregation='equal mean5fold values;argmax firstC on ties',evaluation='raw decision scores for new AUCs;probability=expit(score);prediction=score>=0;frozen reference probabilities used without reconstructing logits',N2='mean absolute per-repeat L3-L1 within-site AUC difference,not absolute difference of means',intervals='2.5/97.5 repeat variability percentiles,not CI;positive fraction uses>1e-12 per prior user decision;criteria means unchanged',BNU_watch_not_applicable=True,references='read-only T12A F/B0 and T12C C0/C2,no refits;C2 reverse AUC known artifact per T12C review,report only',versions={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','joblib']},source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},splits_sha256=hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()))
    for src,name in [('/tmp/T13A_pytest_red.log','pytest_red.log'),('/tmp/T13A_pytest_green.log','pytest_task.log')]:shutil.copyfile(src,OUT/name)


def timing():
    prepare();print('START T13A timing',flush=True);tick=perf_counter();work=plans();selected=[next(j for j in work if j['setting']=='main' and j['model']==model and j['repeat']==0) for model in ['L0','L1','L2','L3']]+[next(j for j in work if j['setting']=='secondary' and j['direction']==direction and j['repeat']==0) for direction in ['oneway','twoway']]
    parallel(selected);done=pd.read_csv(OUT/'completed.csv');rates={(q.setting,q.model,q.direction):q.seconds/q.n_train for q in done.itertuples()};worker_seconds=sum(rates[j['setting'],j['model'],j['direction']]*len(j['train']) for j in work);point=worker_seconds/5+120
    result=dict(point_hours=point/3600,with_25percent_margin_hours=point*1.25/3600,worker_seconds=worker_seconds,pilot_wall_seconds=perf_counter()-tick,pilot_jobs=6,remaining_jobs=294,pilot_results_reused=True,formula='pilot seconds/trainN by model+setting+direction;sum all300 actualN /5 +120s')
    dump('timing.json',result);cfg=json.loads((OUT/'config.json').read_text());cfg['stage']='timed';dump('config.json',cfg);print('TIMING_COMPLETE '+json.dumps(result),flush=True)


def run():
    cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='running',started_at=datetime.now().astimezone().isoformat());dump('config.json',cfg);done=set(pd.read_csv(OUT/'completed.csv').job_id);work=[j for j in plans() if j['job_id'] not in done]
    print('START T13A remaining_jobs='+str(len(work)),flush=True);tick=perf_counter();parallel(work);cfg.update(stage='training_complete',formal_seconds=perf_counter()-tick,finished_at=datetime.now().astimezone().isoformat());dump('config.json',cfg);print('ALL_TRAINING_COMPLETE jobs=300',flush=True)


if __name__=='__main__':
    import sys
    {'timing':timing,'run':run}[sys.argv[1]]()
