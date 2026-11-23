"""T13-B：只切换图谱和L3折数，直接调用既有估计器。"""
from pathlib import Path
from datetime import datetime
from time import perf_counter
import hashlib,json,shutil

import numpy as np
import pandas as pd

import within_site_pair_model as m
import bridge_design as c

OUT=Path('outputs/T13B')
CACHES={'CC200':Path.home()/'datasets_cache/ABIDE_derived/T12A','HO':Path.home()/'datasets_cache/ABIDE_derived/T12D'}
KEYS=['feature','job_id','source_job_id','method','model','setting','k','b','direction','repeat','seed','n_pure','n_added','n_train','T12A_job_id']


def plans():
    source=[json.loads(line) for line in Path('outputs/T12C/splits.jsonl').read_text().splitlines() if json.loads(line)['method']=='C0'];work=[]
    for feature,methods in [('CC200',['L3']),('HO',['L3','C0'])]:
        for j in source:
            for method in methods:
                reused=feature=='CC200' and j['k']==2 and j['b']==5
                work.append(j|dict(job_id=f'{feature}_{method}_r{j["repeat"]}_k{j["k"]}_b{j["b"]}_{j["direction"]}',source_job_id=j['job_id'],feature=feature,method=method,model=method,setting='secondary',reuse_T13A=reused))
    return work


def folds(table,train,job):return m.inner_folds(table,train,job['seed'],min(5,job['b']))


def preflight():
    table=m.subjects();lookup=pd.Index(table.FILE_ID);rows=[]
    for feature,cache in CACHES.items():assert pd.read_csv(cache/'row_order.csv').FILE_ID.tolist()==table.FILE_ID.tolist(),feature
    for j in plans():
        if j['method']!='L3':continue
        train=lookup.get_indexer(j['train']);t=table.loc[train];a,b=m.pairs(t.label.to_numpy(),t.site.to_numpy())
        for fold,(tr,va) in enumerate(folds(table,train,j)):
            v=table.loc[va];aa,bb=m.pairs(v.label.to_numpy(),v.site.to_numpy());assert len(aa)>0,j['job_id']
            counts=pd.crosstab(v.site,v.label);mixed=counts.loc[counts.min(axis=1).gt(0)]
            assert len(mixed)==j['k']*(1 if j['direction']=='oneway' else 2)
            rows.append({key:j[key] for key in ['feature','job_id','k','b','direction','repeat']}|dict(inner_fold=fold,n_splits=min(5,j['b']),training_pairs=len(a),inner_training_pairs=len(m.pairs(table.loc[tr,'label'].to_numpy(),table.loc[tr,'site'].to_numpy())[0]),validation_pairs=len(aa),mixed_validation_sites=len(mixed),minority_min=int(mixed.min(axis=1).min())))
    result=pd.DataFrame(rows);assert len(result)==4800 and result.validation_pairs.gt(0).all();return result


def fit_job(j):
    if j['method']=='L3':result=m.fit_job(j,cache=CACHES[j['feature']],n_splits=min(5,j['b']))
    else:
        c.a.CACHE=CACHES[j['feature']];result=c.fit_job(j)
        result['pred']['score']=result['pred'].probability;result['done']['final_n_fun']=None
    context={key:j[key] for key in KEYS}
    for name in ['pred','diagnostics','warnings']:
        for key,value in context.items():result[name][key]=value
    result['done'].update(context);result['done']['fit_count']=min(5,j['b'])*5+1 if j['method']=='L3' else 21
    for s in result['splits']:s.update(context)
    result['provenance']=context|dict(standardizer_fit=j['train'],test=j['test'],reverse=j['reverse'],reference=False,feature_cache=str(CACHES[j['feature']]))
    return result


def append(name,f):
    with (OUT/name).open('a') as target:f.to_csv(target,index=False,header=target.tell()==0)


def dump(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def save(result):
    # 不同旧函数的诊断列统一后追加；逐人表只保留共同元数据。
    pred=result['pred'];append('predictions.csv',pred[m.a.META+KEYS+['reverse','score','probability','prediction']])
    done=result['done'];append('completed.csv',pd.DataFrame([{key:done[key] for key in KEYS+['C','selection_AUC','seconds','final_n_iter','warning_count','fit_count']}]))
    diagnostic=result['diagnostics'].reindex(columns=KEYS+['role','inner_fold','C','AUC','validation_pairs','n_actual_train','n_iter','n_fun','objective','gradient_max','success','message','n_inner_train','n_validation']);append('selection_fits.csv',diagnostic)
    if len(result['warnings']):append('warnings.csv',result['warnings'].reindex(columns=KEYS+['category','message','count']))
    for name,records in [('inner_splits.jsonl',result['splits']),('fit_provenance.jsonl',[result['provenance']])]:
        with (OUT/name).open('a') as target:
            for row in records:target.write(json.dumps(row)+'\n')


def parallel(work):
    from joblib import Parallel,delayed,parallel_config
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for result in Parallel(n_jobs=5,return_as='generator_unordered')(delayed(fit_job)(j) for j in work):save(result);print('TRAIN_DONE '+json.dumps({key:result['done'][key] for key in ['job_id','seconds','C','warning_count']}),flush=True)


def prepare():
    from importlib.metadata import version
    OUT.mkdir();table=m.subjects();table.to_csv(OUT/'subjects.csv',index=False);work=plans();(OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in work));preflight().to_csv(OUT/'pair_preflight.csv',index=False)
    sources=['pair_estimator_confirmation.py','pair_estimator_confirmation_report.py','within_site_pair_model.py','bridge_design.py','abide_class_pure_sites.py','bridge_design_report.py','tests/test_pair_estimator_confirmation.py','docs/tasks/T13B_pair_estimator_confirmation.md','AGENTS.md']
    inputs=[cache/name for cache in CACHES.values() for name in ['fisher_z.npy','row_order.csv']]+[Path('outputs')/task/name for task,names in [('T12A',['subjects.csv','splits.jsonl','predictions.csv','completed.csv']),('T12C',['splits.jsonl','predictions.csv','completed.csv']),('T12D/V-HO',['subjects.csv','splits.jsonl','predictions.csv','completed.csv']),('T13A',['splits.jsonl','predictions.csv','completed.csv','solver_fits.csv','inner_splits.jsonl'])] for name in names]
    dump('config.json',dict(task='T13B',stage='prepared',created_at=datetime.now().astimezone().isoformat(),seeds=list(range(50)),jobs=1800,new_jobs=1700,reused_T13A_jobs=100,expected_new_fits=35200,CPU_process_limit=6,workers=5,BLAS_threads=1,feature_cache={k:str(v) for k,v in CACHES.items()},L3_C=m.GRID,C0_C=m.a.GRID,L3='direct T13A fit_job;unchanged objective/solver;only cache and min(5,b) folds supplied;site-by-diagnosis stratification;pooled validation pair AUC',C0='direct T12C fit_job with HO cache;original diagnosis-stratified5fold4C validation subject AUC',score='L3 raw decision;C0/B0/C2 frozen or original probability;classification L3>=0,LR>=.5',intervals='2.5/97.5 repeat variability range,not CI;positive proportion difference>1e-12',male='SEX=1;within-site both diagnoses>=3',BNU_watch_not_applicable=True,no_dependencies_installed=True,references='CC200:C0/C2 T12C,B0 T12A;HO:B0 T12D V-HO;CC200 L3 k2b5 both directions T13A;C2 reverse never reported',versions={key:version(key) for key in ['numpy','pandas','scipy','scikit-learn','joblib']},source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources},input_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},splits_sha256=hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()))
    for source,name in [('/tmp/T13B_pytest_red.log','pytest_red.log'),('/tmp/T13B_pytest_green.log','pytest_task.log')]:shutil.copyfile(source,OUT/name)


def timing():
    prepare();print('START T13B timing',flush=True);tick=perf_counter();work=[j for j in plans() if not j['reuse_T13A']];selected=[j for j in work if j['repeat'] in [0,16]]
    parallel(selected);done=pd.read_csv(OUT/'completed.csv');rates=done.groupby(['feature','method','k','b','direction']).seconds.mean();remaining=[j for j in work if j['job_id'] not in set(done.job_id)];seconds=sum(rates[j['feature'],j['method'],j['k'],j['b'],j['direction']] for j in remaining);point=perf_counter()-tick+seconds/5+120
    result=dict(point_hours=point/3600,with_25percent_margin_hours=point*1.25/3600,pilot_wall_seconds=perf_counter()-tick,pilot_jobs=len(selected),remaining_jobs=len(remaining),pilot_results_reused=True,formula='mean actual end-to-end seconds at repeats0+16 separately for each atlas/method/k/b/direction;remaining sum/5 +pilot wall+120s')
    dump('timing.json',result);cfg=json.loads((OUT/'config.json').read_text());cfg['stage']='timed';dump('config.json',cfg);print('TIMING_COMPLETE '+json.dumps(result),flush=True)


def run():
    cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='running',started_at=datetime.now().astimezone().isoformat());dump('config.json',cfg);done=set(pd.read_csv(OUT/'completed.csv').job_id);work=[j for j in plans() if not j['reuse_T13A'] and j['job_id'] not in done];print('START T13B remaining_jobs='+str(len(work)),flush=True);tick=perf_counter();parallel(work);cfg.update(stage='training_complete',formal_seconds=perf_counter()-tick,finished_at=datetime.now().astimezone().isoformat());dump('config.json',cfg);print('ALL_TRAINING_COMPLETE jobs=1700 reused_T13A=100',flush=True)


if __name__=='__main__':
    import sys
    {'timing':timing,'run':run}[sys.argv[1]]()
