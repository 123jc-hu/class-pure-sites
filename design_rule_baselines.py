"""T14-A：固定预算设计、泄漏指数与既有估计器的两项对照。"""
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import json

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

import pair_estimator_confirmation as b
import within_site_pair_model as m
import bridge_design as c

OUT=Path('outputs/T14A')


def designs():
    source=[j for j in b.plans() if j['method']=='L3'];ten={(j['feature'],j['k'],j['direction'],j['repeat']):j for j in source if j['b']==10};meta=m.subjects().set_index('FILE_ID');result=[]
    for j in source:
        if j['b']!=5:continue
        rng=np.random.default_rng([j['repeat'],j['k'],5]);asd=rng.choice(sorted(j['asd_sites']),j['k']//2,replace=False).tolist();td=rng.choice(sorted(j['td_sites']),j['k']//2,replace=False).tolist();selected=asd+(td if j['direction']=='twoway' else [])
        pool=ten[j['feature'],j['k'],j['direction'],j['repeat']]['bridge'];bridge=[v for v in pool if meta.loc[v,'site'] in selected];common=[v for v in j['original_reverse'] if v not in set(j['bridge']+bridge)]
        for allocation,added in [('equal',j['bridge']),('concentrated',bridge)]:
            result.append(j|dict(job_id=f'{j["feature"]}_r{j["repeat"]}_k{j["k"]}_{j["direction"]}_{allocation}',allocation=allocation,bridge=added,equal_bridge=j['bridge'],concentrated_bridge=bridge,train=j['pure_train']+added,reverse=[v for v in j['original_reverse'] if v not in added],common_reverse=common,selected_asd_sites=asd,selected_td_sites=td if j['direction']=='twoway' else [],allocation_seed=[j['repeat'],j['k'],5],n_train=len(j['pure_train'])+len(added),n_added=len(added)))
    return result


def leakage(table,train):
    counts=pd.crosstab(table.loc[train,'site'],table.loc[train,'label']);difference=counts[1]/counts[1].sum()-counts[0]/counts[0].sum()
    return float((difference*difference).sum())


def l1_train(table,train):
    t=table.loc[train];counts=pd.crosstab(t.site,t.label);mixed=counts.index[counts.min(axis=1).gt(0)]
    return train[t.site.isin(mixed).to_numpy()]


def combat_lr(x,table,train,target,seed,eb=True):
    covars=table.loc[train,['site','label']].reset_index(drop=True);messages=StringIO()
    with redirect_stdout(messages):harmonized=neuroCombat(dat=x[train].T,covars=covars,batch_col='site',categorical_cols=['label'],eb=eb,parametric=True,mean_only=False,ref_batch=None)['data'].T
    fitted=c.a.select_model(harmonized,table.loc[train,'label'].to_numpy(),np.arange(len(train)),seed)
    return dict(model=fitted,probability=fitted.predict_proba(x[target])[:,1],messages=messages.getvalue())


def write_designs():
    from importlib.metadata import version
    import hashlib,inspect,shutil
    from datetime import datetime
    OUT.mkdir();table=m.subjects();lookup=pd.Index(table.FILE_ID);table.to_csv(OUT/'subjects.csv',index=False);work=designs();(OUT/'designs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in work));rows=[]
    old=[j|dict(allocation='equal',method='C0') for j in b.plans() if j['method']=='L3']
    pure=[j|dict(feature=feature,allocation='none',method='B0',b=0,direction='none') for feature in b.CACHES for j in c.original_jobs()]
    for j in pure+old+[j for j in work if j['allocation']=='concentrated']:
        indices=lookup.get_indexer(j['train']);assert (indices>=0).all()
        rows.append(dict(feature=j['feature'],method='B0' if j['allocation']=='none' else 'C0',allocation=j['allocation'],k=j['k'],b=j['b'],direction=j['direction'],repeat=j['repeat'],n_train=len(indices),n_added=len(j.get('bridge',[])),L=leakage(table,indices)))
    raw=pd.DataFrame(rows);assert len(raw)==1800;raw.to_csv(OUT/'all_L_repeats.csv',index=False)
    summary=raw.groupby(['feature','method','allocation','k','b','direction']).agg(mean_L=('L','mean'),minimum_L=('L','min'),maximum_L=('L','max'),n_added=('n_added','first'),repeats=('repeat','nunique')).reset_index();assert len(summary)==36 and summary.repeats.eq(50).all();summary.to_csv(OUT/'all_L.csv',index=False)
    predicted=summary.loc[summary.allocation.eq('concentrated')].merge(summary.loc[summary.allocation.eq('equal')&summary.b.eq(5),['feature','k','direction','mean_L']].rename(columns={'mean_L':'equal_L'}),on=['feature','k','direction']);predicted['L_difference']=predicted.mean_L-predicted.equal_L;predicted.to_csv(OUT/'predicted_L.csv',index=False)
    sources=['design_rule_baselines.py','tests/test_design_rule_baselines.py','bridge_design.py','within_site_pair_model.py','pair_estimator_confirmation.py','docs/tasks/T14A_design_rule_baselines.md','docs/THEORY_A.md','AGENTS.md']
    inputs=[cache/name for cache in b.CACHES.values() for name in ['fisher_z.npy','row_order.csv']]+[Path('outputs/T12A')/name for name in ['subjects.csv','splits.jsonl']]+[Path('outputs/T12C/splits.jsonl')]
    package=Path(inspect.getfile(neuroCombat)).parent
    cfg=dict(task='T14A',stage='predicted_L_saved_awaiting_protocol_details',created_at=datetime.now().astimezone().isoformat(),seeds=list(range(50)),neuroCombat_version=version('neuroCombat'),only_new_dependency='neuroCombat;pip --no-deps --no-build-isolation',CPU_process_limit=6,workers=5,BLAS_threads=1,allocation_seed='numpy default_rng([repeat,k,5]);choice sorted ASD sites first,then sorted TD sites;shared ASD selection across directions and atlases',bridge_order='T12C b10 prefix at selected sites;no new subject permutation',common_reverse='original reverse minus union(equal and concentrated bridges)',L='sum_j (a_j/N_A-t_j/N_T)^2;actual subjects,computed before outcomes',BNU_watch_not_applicable=True,source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources},input_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},neuroCombat_source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in package.glob('*.py')},predicted_L_sha256=hashlib.sha256((OUT/'predicted_L.csv').read_bytes()).hexdigest(),pending=['null-noise dimension/seed/per-direction gate','G2 native versus common reverse cohort'])
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    for source,name in [('/tmp/T14A_pytest_red.log','pytest_red.log'),('/tmp/T14A_pytest_green.log','pytest_task.log'),('/tmp/T14A_neuroCombat_install.log','neuroCombat_install.log')]:shutil.copyfile(source,OUT/name)
    print('PREDICTED_L_SAVED '+json.dumps(dict(cells=len(predicted),all_L_cells=len(summary),designs=len(work)),ensure_ascii=False),flush=True)


def null_job(j,eb=True):
    from time import perf_counter
    import warnings
    from collections import Counter
    from sklearn.metrics import roc_auc_score
    tick=perf_counter();table=m.subjects();lookup=pd.Index(table.FILE_ID);p={'CC200':19900,'HO':5995}[j['feature']];rng=np.random.default_rng([2026,j['repeat']]);sites,inverse=np.unique(table.site.to_numpy(),return_inverse=True)
    x=rng.standard_normal((len(table),p));x+=rng.standard_normal((len(sites),p))[inverse]
    train=lookup.get_indexer(j['train']);target=lookup.get_indexer(j['reverse']);assert set(train).isdisjoint(target)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');fitted=combat_lr(x,table,train,target,j['seed'],eb=eb)
    context={key:j[key] for key in ['feature','k','b','direction','repeat','seed']};probability=fitted['probability'];result=table.loc[target,m.a.META].copy()
    for key,value in context.items():result[key]=value
    result['probability']=probability;result['prediction']=probability>=.5
    row=context|dict(noise_seed=[2026,j['repeat']],dimensions=p,n_train=len(train),n_reverse=len(target),C=float(fitted['model'].best_params_['lr__C']),reverse_AUC=roc_auc_score(table.loc[target,'label'],probability),seconds=perf_counter()-tick,warning_count=len(caught))
    warn=[context|dict(category=cat,message=msg,count=n) for (cat,msg),n in Counter((w.category.__name__,str(w.message)) for w in caught).items()]
    return row,result,fitted['messages'],warn


def null_test(eb=False):
    import hashlib
    from datetime import datetime
    from joblib import Parallel,delayed,parallel_config
    stem='null' if eb else 'null_noEB';cfg=json.loads((OUT/'config.json').read_text());(OUT/'pre_noEB_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    cfg.update(stage=stem+'_running',revision='2026-11-20:EB test AUC only;noEB reverse only if null passes;G1/G2/G4 continue',null_noEB_started=True,real_data_training_started=False)
    for name in cfg['source_sha256']:cfg['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));work=[j for j in designs() if j['allocation']=='equal' and j['k']==2 and j['repeat']<5];assert len(work)==20
    (OUT/(stem+'_splits.jsonl')).write_text(''.join(json.dumps(j)+'\n' for j in work));print('START T14A ComBat '+stem+'20',flush=True);rows=[];predictions=[];warning=[]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for row,pred,messages,warn in Parallel(n_jobs=5,return_as='generator_unordered')(delayed(null_job)(j,eb) for j in work):
            rows.append(row);predictions.append(pred);warning.extend(warn)
            with (OUT/(stem+'_combat_messages.log')).open('a') as f:f.write(json.dumps({key:row[key] for key in ['feature','direction','repeat']})+'\n'+messages)
            print('NULL_DONE '+json.dumps(row),flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(OUT/(stem+'_metrics.csv'),index=False);pd.concat(predictions,ignore_index=True).to_csv(OUT/(stem+'_predictions.csv'),index=False)
    summary=raw.groupby(['feature','direction']).agg(mean_reverse_AUC=('reverse_AUC','mean'),repeats=('repeat','nunique')).reset_index();assert len(summary)==4 and summary.repeats.eq(5).all();summary['passed']=summary.mean_reverse_AUC.between(.4,.6);summary.to_csv(OUT/(stem+'_summary.csv'),index=False)
    if warning:pd.DataFrame(warning).to_csv(OUT/(stem+'_warnings.csv'),index=False)
    passed=bool(summary.passed.all());decision=dict(eb=eb,passed=passed,gate='all4 means in[.40,.60]',groups=summary.to_dict('records'),warnings=int(raw.warning_count.sum()),formal_training_allowed=True,report_reverse=passed)
    (OUT/(stem+'_gate.json')).write_text(json.dumps(decision,ensure_ascii=False,indent=2));cfg.update(stage='null_noEB_complete',null_noEB_passed=passed,report_noEB_reverse=passed,null_noEB_finished_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('NULL_COMPLETE '+json.dumps(decision,ensure_ascii=False),flush=True)


KEYS=b.KEYS+['allocation']


def jobs():
    table=m.subjects();lookup=pd.Index(table.FILE_ID);result=[]
    for j in designs():
        for method in (['C0','L3'] if j['allocation']=='concentrated' else ['L1','ComBat-EB','ComBat-noEB']):
            train=j['train'];dropped=[]
            if method=='L1':
                keep=l1_train(table,lookup.get_indexer(train));train=table.loc[keep,'FILE_ID'].tolist();dropped=[v for v in j['train'] if v not in set(train)]
            result.append(j|dict(job_id=j['job_id']+'_'+method,method=method,model=method,train=train,dropped_pure=dropped,n_train=len(train),n_pure=len(train)-j['n_added']))
    return result


def fit_job(j):
    from time import perf_counter
    import warnings
    from collections import Counter
    tick=perf_counter()
    if j['method'] in ['C0','L3']:result=b.fit_job(j)
    elif j['method']=='L1':result=m.fit_job(j,cache=b.CACHES[j['feature']],n_splits=5)
    else:
        table=m.subjects();lookup=pd.Index(table.FILE_ID);x=np.load(b.CACHES[j['feature']]/'fisher_z.npy',mmap_mode='r');tr=lookup.get_indexer(j['train']);report_reverse=j['method']=='ComBat-noEB' and json.loads((OUT/'null_noEB_gate.json').read_text())['passed'];ids=j['test']+(j['reverse'] if report_reverse else []);target=lookup.get_indexer(ids)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');fitted=combat_lr(x,table,tr,target,j['seed'],eb=j['method']=='ComBat-EB')
        net=fitted['model'];cv=net.cv_results_;diagnostics=[];splits=[]
        for fold,(a,v) in enumerate(c.inner_folds(table,tr,j['seed'])):
            for i,C in enumerate(c.a.GRID):diagnostics.append(dict(inner_fold=fold,C=C,AUC=float(cv[f'split{fold}_test_score'][i])))
            splits.append(dict(inner_fold=fold,train=table.loc[a,'FILE_ID'].tolist(),validation=table.loc[v,'FILE_ID'].tolist(),standardizer_fit=table.loc[a,'FILE_ID'].tolist(),ComBat_fit=j['train']))
        pred=table.loc[target].copy();pred['reverse']=pred.FILE_ID.isin(j['reverse']);pred['probability']=fitted['probability'];pred['score']=pred.probability;pred['prediction']=pred.probability>=.5
        warn=[dict(category=cat,message=msg,count=n) for (cat,msg),n in Counter((w.category.__name__,str(w.message)) for w in caught).items()]
        result=dict(pred=pred,done=dict(C=float(net.best_params_['lr__C']),selection_AUC=float(net.best_score_),final_n_iter=int(net.best_estimator_.named_steps['lr'].n_iter_.max()),warning_count=len(caught)),diagnostics=pd.DataFrame(diagnostics),splits=splits,warnings=pd.DataFrame(warn),combat_messages=fitted['messages'])
    context={key:j[key] for key in KEYS}
    for name in ['pred','diagnostics','warnings']:
        for key,value in context.items():result[name][key]=value
    result['done'].update(context);result['done'].update(seconds=perf_counter()-tick,fit_count=26 if j['method'] in ['L1','L3'] else 21)
    for s in result['splits']:s.update(context)
    result['provenance']=context|dict(standardizer_fit=j['train'],ComBat_fit=j['train'] if j['method'].startswith('ComBat') else [],dropped_pure=j['dropped_pure'],test=j['test'],native_reverse=j['reverse'],predicted_reverse=j['reverse'] if not j['method'].startswith('ComBat') or (j['method']=='ComBat-noEB' and json.loads((OUT/'null_noEB_gate.json').read_text())['passed']) else [])
    return result


def append(name,frame):
    with (OUT/name).open('a') as f:frame.to_csv(f,index=False,header=f.tell()==0)


def save(result):
    append('predictions.csv',result['pred'][m.a.META+KEYS+['reverse','score','probability','prediction']]);q=result['done'];append('completed.csv',pd.DataFrame([{key:q[key] for key in KEYS+['C','selection_AUC','seconds','final_n_iter','warning_count','fit_count']}]))
    append('selection_fits.csv',result['diagnostics'].reindex(columns=KEYS+['role','inner_fold','C','AUC','validation_pairs','n_actual_train','n_iter','n_fun','objective','gradient_max','success','message','n_inner_train','n_validation']))
    if len(result['warnings']):append('warnings.csv',result['warnings'].reindex(columns=KEYS+['category','message','count']))
    for name,rows in [('inner_splits.jsonl',result['splits']),('fit_provenance.jsonl',[result['provenance']])]:
        with (OUT/name).open('a') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    if 'combat_messages' in result:
        with (OUT/'combat_messages.log').open('a') as f:f.write(q['job_id']+'\n'+result['combat_messages'])


def parallel(work):
    from joblib import Parallel,delayed,parallel_config
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for result in Parallel(n_jobs=5,return_as='generator_unordered')(delayed(fit_job)(j) for j in work):save(result);print('TRAIN_DONE '+json.dumps({key:result['done'][key] for key in ['job_id','seconds','C','warning_count']}),flush=True)


def prepare_run():
    import hashlib
    table=m.subjects();lookup=pd.Index(table.FILE_ID);work=jobs();assert len(work)==2000;(OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in work));pre=[]
    for j in work:
        tr=lookup.get_indexer(j['train']);assert set(j['train']).isdisjoint(j['test']+j['reverse'])
        if j['method'] not in ['L1','L3']:continue
        for fold,(a,v) in enumerate(b.folds(table,tr,j)):
            t=table.loc[v];p,_=m.pairs(t.label.to_numpy(),t.site.to_numpy());assert len(p)>0
            pre.append(dict(job_id=j['job_id'],inner_fold=fold,pairs=len(p)))
    assert len(pre)==4000;pd.DataFrame(pre).to_csv(OUT/'pair_preflight.csv',index=False)
    cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='prepared',real_data_training_started=False,new_jobs=2000,new_fits=46000,report_noEB_reverse=json.loads((OUT/'null_noEB_gate.json').read_text())['passed'],report_EB_reverse=False,report_ComBat_metrics='within_site_AUC only when null fails;no reverse predictions generated',L1='drop all pure training sites before scaler/inner folds;N=actual remaining;direct T13A L1,5-fold site-by-diagnosis,pair validation AUC',G4='mean of absolute paired per-repeat L1-L3 within-site AUC difference<.005 in each atlas/k twoway b5 cell',ComBat_inner='outer-train ComBat once;scaler and LR refit in diagnosis-stratified5fold4C inner;inner ComBat not refit',stop_reason=None)
    cfg['source_sha256']['design_rule_baselines_report.py']=hashlib.sha256(Path('design_rule_baselines_report.py').read_bytes()).hexdigest()
    for name in cfg['source_sha256']:cfg['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    for task,names in [('T13B',['predictions.csv','reference_predictions.csv','selected_C.csv']),('T12A',['predictions.csv','completed.csv']),('T12D/V-HO',['predictions.csv','completed.csv'])]:
        for name in names:
            path=Path('outputs')/task/name;cfg['input_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    cfg['splits_sha256']=hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest();(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))


def timing():
    from time import perf_counter
    prepare_run();print('START T14A timing',flush=True);tick=perf_counter();work=jobs();pilot=[j for j in work if j['repeat'] in [0,16]];parallel(pilot);done=pd.read_csv(OUT/'completed.csv');rates=done.groupby(['feature','method','k','direction']).seconds.mean();remaining=[j for j in work if j['job_id'] not in set(done.job_id)];point=perf_counter()-tick+sum(rates[j['feature'],j['method'],j['k'],j['direction']] for j in remaining)/5+120
    result=dict(point_hours=point/3600,with_25percent_margin_hours=point*1.25/3600,pilot_wall_seconds=perf_counter()-tick,pilot_jobs=len(pilot),remaining_jobs=len(remaining),pilot_reused=True,formula='actual mean repeats0+16 per feature/method/k/direction;remaining sum/5+pilot wall+120s');(OUT/'timing.json').write_text(json.dumps(result,indent=2));cfg=json.loads((OUT/'config.json').read_text());cfg['stage']='timed';(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('TIMING_COMPLETE '+json.dumps(result),flush=True)


def run():
    from time import perf_counter
    from datetime import datetime
    cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='running',real_data_training_started=True,started_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));done=set(pd.read_csv(OUT/'completed.csv').job_id);work=[j for j in jobs() if j['job_id'] not in done];print('START T14A remaining_jobs='+str(len(work)),flush=True);tick=perf_counter();parallel(work);cfg.update(stage='training_complete',formal_seconds=perf_counter()-tick,finished_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('ALL_TRAINING_COMPLETE jobs=2000',flush=True)


if __name__=='__main__':
    import sys
    {'null_noEB':null_test,'timing':timing,'run':run}[sys.argv[1]]()
