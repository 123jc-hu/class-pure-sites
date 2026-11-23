"""T12-C：固定嵌套桥接名单，普通LR或诊断保护的加性站点校正。"""
from pathlib import Path
import hashlib,json,warnings
from collections import Counter
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import abide_class_pure_sites as a

OUT=Path('outputs/T12C');CONTEXT=['job_id','method','k','b','direction','repeat','seed','n_pure','n_added','n_train','T12A_job_id']


def dump(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def append(name,frame):
    with (OUT/name).open('a') as f:frame.to_csv(f,index=False,header=f.tell()==0);f.flush()


def subjects():return pd.read_csv(a.OUT/'subjects.csv')


def original_jobs():
    return [j for j in a.read_jobs() if not j['no_zero_roi'] and j['arm']=='P' and j['n']==40 and j['k'] in [2,4]]


def plans(table=None,work=None,budgets=(2,5,10)):
    table=subjects() if table is None else table;meta=table.set_index('FILE_ID',drop=False);result=[]
    for j in original_jobs() if work is None else work:
        rng=np.random.default_rng([j['repeat'],j['k'],3]);pool=meta.loc[j['reverse']]
        orders={s:rng.permutation(pool.loc[pool.site.eq(s),'FILE_ID'].to_numpy()).tolist() for s in sorted(j['asd_sites']+j['td_sites'])}
        for b in budgets:
            for direction in ['oneway','twoway']:
                sites=sorted(j['asd_sites']+(j['td_sites'] if direction=='twoway' else []));bridge=[v for s in sites for v in orders[s][:b]]
                reverse=[s for s in j['reverse'] if s not in bridge]
                for method in ['C0','C2']:
                    result.append(dict(job_id=f'{method}_r{j["repeat"]}_k{j["k"]}_b{b}_{direction}',method=method,k=j['k'],b=b,direction=direction,repeat=j['repeat'],seed=j['seed'],bridge_seed=[j['repeat'],j['k'],3],n_pure=len(j['train']),n_added=len(bridge),n_train=len(j['train'])+len(bridge),T12A_job_id=j['job_id'],pure_train=j['train'],bridge=bridge,train=j['train']+bridge,test=j['test'],reverse=reverse,original_reverse=j['reverse'],asd_sites=j['asd_sites'],td_sites=j['td_sites'],test_sites=j['test_sites']))
    return result


def additive_fit(x,table,indices):
    scale=StandardScaler().fit(x[indices]);z=scale.transform(x[indices]);t=table.loc[indices];sites=sorted(t.site.unique());incidence=np.column_stack([t.site.eq(s) for s in sites]).astype(float)
    design=np.column_stack([np.ones(len(t)),t.label.to_numpy(),incidence[:,:-1]-incidence[:,[-1]]])
    coef,_,rank,_=np.linalg.lstsq(design,z,rcond=None);assert rank==design.shape[1]
    effects=np.vstack([coef[2:],-coef[2:].sum(axis=0)])
    return dict(scale=scale,sites=sites,effects=effects,diagnosis=coef[1],intercept=coef[0],rank=int(rank))


def additive_transform(x,table,indices,fit):
    z=fit['scale'].transform(x[indices]);site_index=pd.Index(fit['sites']).get_indexer(table.loc[indices,'site']);known=site_index>=0
    z[known]-=fit['effects'][site_index[known]]
    return z


def inner_folds(table,train,seed):
    return [(train[tr],train[va]) for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(train,table.loc[train,'label'])]


def fit_job(job,data=None):
    tick=perf_counter();table=subjects() if data is None else data[0]
    if data is None:assert table.FILE_ID.tolist()==pd.read_csv(a.CACHE/'row_order.csv').FILE_ID.tolist()
    lookup=pd.Index(table.FILE_ID)
    x=np.load(a.CACHE/'fisher_z.npy',mmap_mode='r') if data is None else data[1];train=lookup.get_indexer(job['train']);test=lookup.get_indexer(job['test']);reverse=lookup.get_indexer(job['reverse']);target=np.r_[test,reverse]
    assert (train>=0).all() and (target>=0).all() and not set(train)&set(target)
    ctx={k:job[k] for k in CONTEXT};diagnostics=[];splits=[];ranks=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if job['method']=='C0':
            fitted=a.select_model(x,table.label.to_numpy(),train,job['seed']);cv=fitted.cv_results_;c=float(fitted.best_params_['lr__C']);probability=fitted.predict_proba(x[target])[:,1];selection=float(fitted.best_score_);n_iter=int(fitted.best_estimator_.named_steps['lr'].n_iter_.max());final_rank=0;effect_sum=0.
        else:
            assert job['b']>0
            scores=[]
            for fold,(tr,va) in enumerate(inner_folds(table,train,job['seed'])):
                fit=additive_fit(x,table,tr);z=additive_transform(x,table,tr,fit);v=additive_transform(x,table,va,fit);ranks.append(fit['rank']);values=[]
                for c in a.GRID:
                    net=LogisticRegression(C=c).fit(z,table.loc[tr,'label']);values.append(roc_auc_score(table.loc[va,'label'],net.predict_proba(v)[:,1]))
                scores.append(values)
            means=np.mean(scores,axis=0);c=a.GRID[int(np.argmax(means))];selection=float(means.max());fit=additive_fit(x,table,train);final_rank=fit['rank'];effect_sum=float(abs(fit['effects'].sum(axis=0)).max())
            net=LogisticRegression(C=c).fit(additive_transform(x,table,train,fit),table.loc[train,'label']);probability=net.predict_proba(additive_transform(x,table,target,fit))[:,1];n_iter=int(net.n_iter_.max())
    for fold,(tr,va) in enumerate(inner_folds(table,train,job['seed'])):
        for i,candidate in enumerate(a.GRID):
            value=float(cv[f'split{fold}_test_score'][i]) if job['method']=='C0' else scores[fold][i]
            diagnostics.append(ctx|dict(inner_fold=fold,C=candidate,AUC=value,additive_rank=0 if job['method']=='C0' else ranks[fold],n_inner_train=len(tr),n_validation=len(va)))
        splits.append(ctx|dict(inner_fold=fold,train=table.loc[tr,'FILE_ID'].tolist(),validation=table.loc[va,'FILE_ID'].tolist(),standardizer_fit=table.loc[tr,'FILE_ID'].tolist(),additive_fit=table.loc[tr,'FILE_ID'].tolist() if job['method']=='C2' else [],train_bridge=[s for s in table.loc[tr,'FILE_ID'] if s in job['bridge']],validation_bridge=[s for s in table.loc[va,'FILE_ID'] if s in job['bridge']]))
    pred=table.loc[target].copy()
    for key,value in ctx.items():pred[key]=value
    pred['is_bridge']=pred.FILE_ID.isin(job['bridge']);pred['reverse']=np.r_[np.zeros(len(test),bool),np.ones(len(reverse),bool)];pred['probability']=probability;pred['prediction']=pred.probability.ge(.5)
    warn=[ctx|dict(category=category,message=message,count=count) for (category,message),count in Counter((w.category.__name__,str(w.message)) for w in caught).items()]
    return dict(pred=pred,done=ctx|dict(C=c,selection_AUC=selection,final_n_iter=n_iter,additive_rank=final_rank,site_effect_sum_max_abs=effect_sum,seconds=perf_counter()-tick,n_test=len(test),n_reverse=len(reverse),warning_count=len(caught)),diagnostics=pd.DataFrame(diagnostics),splits=splits,warnings=pd.DataFrame(warn))


def save_result(result):
    append('predictions.csv',result['pred']);append('completed.csv',pd.DataFrame([result['done']]));append('selection_fits.csv',result['diagnostics'])
    if len(result['warnings']):append('warnings.csv',result['warnings'])
    with (OUT/'inner_splits.jsonl').open('a') as f:
        for s in result['splits']:f.write(json.dumps(s)+'\n')


def read_plans():return [json.loads(s) for s in (OUT/'splits.jsonl').read_text().splitlines()]


def prepare():
    import shutil
    OUT.mkdir();table=subjects();table.to_csv(OUT/'subjects.csv',index=False);work=plans();assert len(work)==1200
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in work));meta=table.set_index('FILE_ID',drop=False);bridge_rows=[];ranks=[]
    for j in work:
        tr=meta.loc[j['train']];te=meta.loc[j['test']];assert set(tr.site).isdisjoint(te.site) and set(j['train']).isdisjoint(j['reverse']) and set(j['bridge']).isdisjoint(j['reverse'])
        if j['method']=='C0':
            f=meta.loc[j['bridge']].copy()
            for key in ['k','b','direction','repeat','seed']:f[key]=j[key]
            f['is_bridge']=True;bridge_rows.append(f)
        else:
            for fold,(train_idx,validation_idx) in enumerate(inner_folds(table,pd.Index(table.FILE_ID).get_indexer(j['train']),j['seed'])):
                t=table.loc[train_idx];sites=sorted(t.site.unique());h=np.column_stack([t.site.eq(s) for s in sites]).astype(float);d=np.column_stack([np.ones(len(t)),t.label,h[:,:-1]-h[:,[-1]]]);rank=np.linalg.matrix_rank(d);assert rank==d.shape[1];ranks.append(dict(job_id=j['job_id'],inner_fold=fold,rank=int(rank),columns=d.shape[1]))
    pd.concat(bridge_rows,ignore_index=True).to_csv(OUT/'bridge_subjects.csv',index=False);pd.DataFrame(ranks).to_csv(OUT/'design_rank_preflight.csv',index=False)
    sources=['bridge_design.py','tests/test_bridge_design.py','abide_class_pure_sites.py','docs/tasks/T12C_bridge_design.md','AGENTS.md'];inputs=[str(a.CACHE/'fisher_z.npy'),str(a.CACHE/'row_order.csv')]+['outputs/T12A/'+s for s in ['splits.jsonl','subjects.csv','predictions.csv','repeat_metrics.csv','completed.csv']]+['outputs/T12B/B0_reproduction.json']
    dump('config.json',dict(task='T12C',stage='prepared',seeds=list(range(50)),bridge_seed='numpy default_rng([repeat,k,3]);all training sites sorted;each site opposite-class IDs retain T12A reverse-list order before permutation',k=[2,4],b=[2,5,10],directions=['oneway','twoway'],methods=['C0','C2'],C=a.GRID,worker_processes=5,CPU_process_limit=6,BLAS_threads=1,
        additive='train StandardScaler;OLS columns[intercept,diagnosis,S-1 sum-zero station contrasts];all features share design;subtract gamma only;unknown test sites no gamma',
        least_squares_rcond='numpy default None;rank must equal S+1',inner='subject-stratified5fold,shuffle,seed=repeat;each inner scaler/OLS refit;C0/C2 identical folds',LR='T12A defaults:L2,lbfgs,max_iter100,tol1e-4,no class weights',
        reverse_evaluation='original reverse minus current bridges;B0 existing probabilities filtered to same remaining people for paired reverse comparisons',bridge_prediction_flag='bridges used only for training;test/reverse prediction is_bridge=False;bridge_subjects.csv is_bridge=True',
        recovery='mean(method-B0 within each repeat)/mean(M-B0 within each repeat),not mean repeat ratios;no clipping',positive_fraction_tolerance=1e-12,positive_rule='carry forward user-confirmed T12B numerical tie convention;diff>1e-12 only for positive proportion,not means/decisions',
        intervals='50repeat2.5/97.5percentile variability ranges,not independent-sample confidence intervals',BNU_watch_not_applicable=True,expected_jobs=1200,expected_LR_fits=25200,
        source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},input_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in inputs},splits_sha256=hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()))
    for src,target in [('/tmp/T12C_pytest_red.log','pytest_red.log'),('/tmp/T12C_pytest_green.log','pytest_task.log')]:shutil.copyfile(src,OUT/target)


def parallel(work,label):
    from joblib import Parallel,delayed,parallel_config
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for result in Parallel(n_jobs=5,return_as='generator_unordered')(delayed(fit_job)(j) for j in work):save_result(result);print(label+' '+json.dumps(result['done']),flush=True)


def timing():
    prepare();print('START T12C timing',flush=True);start=perf_counter();work=read_plans();selected=[next(j for j in work if j['repeat']==0 and j['method']==method and j['k']==k and j['b']==b and j['direction']=='twoway') for method in ['C0','C2'] for k,b in [(2,2),(2,5),(4,10)]]
    parallel(selected,'TIMING_DONE');done=pd.read_csv(OUT/'completed.csv');seconds=0.
    for j in work:
        rates=done.loc[done.method.eq(j['method'])];seconds+=float((rates.seconds/rates.n_train).mean())*j['n_train']
    point=seconds/5+120;result=dict(point_hours=point/3600,with_25percent_margin_hours=point*1.25/3600,pilot_wall_seconds=perf_counter()-start,worker_seconds=seconds,all_jobs=1200,completed_jobs=6,remaining_jobs=1194,pilot_results_reused=True)
    dump('timing.json',result);cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='timed',timing=result);dump('config.json',cfg);print('TIMING_COMPLETE '+json.dumps(result),flush=True)


def run():
    from datetime import datetime,timezone,timedelta
    cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='running',started_at=datetime.now(timezone(timedelta(hours=8))).isoformat());dump('config.json',cfg);done=set(pd.read_csv(OUT/'completed.csv').job_id);work=[j for j in read_plans() if j['job_id'] not in done]
    print('START T12C remaining_jobs='+str(len(work)),flush=True);parallel(work,'TRAIN_DONE');cfg.update(stage='training_complete',finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat());dump('config.json',cfg);print('ALL_TRAINING_COMPLETE jobs=1200',flush=True)


if __name__=='__main__':
    import sys
    {'timing':timing,'run':run}[sys.argv[1]]()
