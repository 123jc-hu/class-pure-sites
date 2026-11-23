"""T14-B第二阶段：固定ABIDE-II数据适配和原函数的参数入口。"""
from pathlib import Path
import sys,json
REPO=Path(__file__).resolve().parent
# -I下仅显式加入固定可信代码目录，不把工作目录或数据目录加入搜索路径。
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
import numpy as np
import pandas as pd
import abide_class_pure_sites as a
import bridge_design as c
import same_class_control as s
import site_pair_validation as b
import within_site_pair_model as m

OUT=REPO/'outputs/T14B/stage2'
CACHE=Path.home()/'datasets_cache/ABIDE_derived/T14B'


def table():
    f=pd.read_csv(REPO/'outputs/T14B/stage1/subjects.csv',dtype={'participant_id':str});assert f.DCAN_ID.tolist()==pd.read_csv(CACHE/'row_order.csv').DCAN_ID.tolist()
    f['FILE_ID']=f.DCAN_ID;f['SITE_ID']=f.collection;f['site']=f.collection.str.removeprefix('ABIDEII-').str.replace(r'_\d+$','',regex=True);f['DX_GROUP']=np.where(f.label.eq(1),1,2);f['SEX']=f.sex;f['AGE_AT_SCAN']=f.age;f['func_mean_fd']=np.nan;f['n_zero_roi']=f.n_zero_parcels;f['n_timepoints']=f.n_kept;f['construction_eligible']=~f.collection.isin(['ABIDEII-KUL_3','ABIDEII-UCLA_1'])
    return f


def plans():
    t=table();pool=t.loc[t.construction_eligible];primary=[]
    for repeat in range(50):
        for j in a.repeat_design(pool,repeat,large_sites={'NYU','OHSU','KKI'},test_site_count=4,settings=[(1,20),(2,20),(4,20)]):
            if j['arm']=='FD':continue
            primary.append(j|dict(no_zero_roi=False,job_id=f'r{repeat}_{j["arm"]}_k{j["k"]}_n{j["n"]}'))
    base=[j for j in primary if j['arm']=='P' and j['k'] in [2,4]];cross=[j for j in c.plans(t,base,budgets=(2,5)) if j['method']=='C0'];control,feasible=s.plans(t,cross)
    return primary,cross,control,feasible


META=['job_id','part','method','repeat','seed','k','n','b','direction','source_job_id']


def tasks():
    primary,cross,control,_=plans();work=[dict(part='A',method=j['arm'],job=j) for j in primary]
    work.extend(dict(part='B',method=method,job=j) for j in primary if j['arm']=='P' and j['k'] in [2,4] for method in ['E_subj','E_LSPO'])
    work.extend(dict(part='C',method=j['method'],job=j) for j in cross+control)
    work.extend(dict(part='D',method='L3',job=j|dict(model='L3',method='L3',setting='secondary',job_id=j['job_id'].replace('C0_','L3_',1))) for j in cross)
    for w in work:w['task_id']=w['part']+'_'+(f'{w["method"]}_r{w["job"]["repeat"]}_k{w["job"]["k"]}' if w['part']=='B' else w['job']['job_id'])
    return work


def fit_task(w,data_override=None):
    from time import perf_counter
    assert sys.flags.isolated
    tick=perf_counter();j=w['job'];data=(table(),np.load(CACHE/'fisher_z.npy',mmap_mode='r')) if data_override is None else data_override
    if w['part'] in ('A','H'):result=a.fit_job(j,data=data)
    elif w['part']=='B':result=b.fit_job((j,w['method']),data_override=data)
    elif w['part']=='C':result=s.fit_job(j,data=data) if w['method']=='S0' else c.fit_job(j,data=data)
    else:result=m.fit_job(j,n_splits=min(5,j['b']),data=data)
    context=dict(job_id=w['task_id'],part=w['part'],method=w['method'],repeat=j['repeat'],seed=j['seed'],k=j['k'],n=j.get('n',20),b=j.get('b',0),direction=j.get('direction','none'),source_job_id=j['job_id'])
    for key in ['pred','diagnostics','warnings']:
        for name,value in context.items():result[key][name]=value
    for row in result['splits']:row.update(context)
    for name,value in context.items():result.setdefault('folds',pd.DataFrame())[name]=value
    if 'score' not in result['pred']:result['pred']['score']=result['pred'].probability
    if w['part']=='B':result['pred']['reverse']=False
    else:result['pred']['outer_fold']=j.get('outer_fold',-1)
    result['done'].update(context);result['done'].update(seconds=perf_counter()-tick,n_train=len(j['train']),n_test=len(j['test']),n_reverse=len(j['reverse']),fit_count=21*len(result['folds']) if w['part']=='B' else 5*min(5,j['b'])+1 if w['part']=='D' else 21,estimate_meanfold=float(result['folds'].AUC.mean()) if w['part']=='B' else None,estimate_pooled=result['done']['estimate'] if w['part']=='B' and w['method']=='E_subj' else None)
    return result


def append(name,f):
    with (OUT/name).open('a') as target:f.to_csv(target,index=False,header=target.tell()==0)


def save(result):
    name='estimation_predictions.csv' if result['done']['part']=='B' else 'predictions.csv';append(name,result['pred'][a.META+META+['reverse','outer_fold','score','probability','prediction']])
    columns=META+['C','selection_AUC','best_inner_AUC','seconds','n_train','n_test','n_reverse','fit_count','final_n_iter','warning_count','estimate_meanfold','estimate_pooled'];append('completed.csv',pd.DataFrame([result['done']]).reindex(columns=columns))
    columns=META+['inner_fold','outer_fold','C','AUC','inner_mean_AUC','inner_sd_AUC']+[f'fold{i}_AUC' for i in range(5)]+['role','n_actual_train','n_train','n_validation','validation_pairs','n_iter','n_fun','objective','gradient_max','success','message'];append('selection_fits.csv',result['diagnostics'].reindex(columns=columns))
    if len(result['folds']):append('estimation_folds.csv',result['folds'].reindex(columns=META+['outer_fold','AUC','n_train','n_validation','C','selection_AUC','final_n_iter']))
    if len(result['warnings']):append('warnings.csv',result['warnings'].reindex(columns=META+['category','message','count']))
    with (OUT/'inner_splits.jsonl').open('a') as f:
        for row in result['splits']:f.write(json.dumps(row)+'\n')


def parallel(work):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    # spawn继承-I；4个工作进程+父进程+资源跟踪器不超过6个进程。
    remaining=iter(work[4:])
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(fit_task,w) for w in work[:4]}
        while pending:
            finished,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                pending.remove(future);result=future.result();save(result);print('TRAIN_DONE '+json.dumps({k:result['done'][k] for k in ['job_id','seconds','warning_count']}),flush=True);w=next(remaining,None)
                if w is not None:pending.add(pool.submit(fit_task,w))


def prepare():
    import hashlib,shutil
    from datetime import datetime
    from importlib.metadata import version
    from design_rule_baselines import leakage
    OUT.mkdir(parents=True);t=table();t.to_csv(OUT/'subjects.csv',index=False);t.loc[t.construction_eligible].to_csv(OUT/'construction_pool.csv',index=False);t.loc[~t.construction_eligible].to_csv(OUT/'not_in_construction.csv',index=False);primary,cross,control,feasible=plans();work=tasks()
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(w)+'\n' for w in work));pd.DataFrame(feasible).to_csv(OUT/'S0_feasibility.csv',index=False);pd.DataFrame(feasible).groupby(['k','b','direction']).feasible.agg(available='sum',total='count').reset_index().to_csv(OUT/'S0_available_repeats.csv',index=False)
    lookup=pd.Index(t.FILE_ID);pre=[];mixed=[]
    for j in cross:
        for fold,(tr,va) in enumerate(m.inner_folds(t,lookup.get_indexer(j['train']),j['seed'],min(5,j['b']))):
            v=t.loc[va];left,_=m.pairs(v.label.to_numpy(),v.site.to_numpy());assert len(left)>0;pre.append(dict(source_job_id=j['job_id'],fold=fold,validation_pairs=len(left)))
    pd.DataFrame(pre).to_csv(OUT/'pair_preflight.csv',index=False)
    for dataset,subjects,designs,ks in [('ABIDE-II',t,primary,[1,2,4]),('ABIDE-I',pd.read_csv(REPO/'outputs/T12A/subjects.csv'),[j for j in a.read_jobs() if not j['no_zero_roi'] and j['n']==40],[4,6])]:
        meta=subjects.set_index('FILE_ID')
        for j in designs:
            if j['arm']!='M' or j['k'] not in ks:continue
            counts=meta.loc[j['train']].groupby('site').label.nunique();single=counts.index[counts.eq(1)].tolist();mixed.append(dict(dataset=dataset,k=j['k'],repeat=j['repeat'],single_class_site_count=len(single),single_class_sites=';'.join(single)))
    raw=pd.DataFrame(mixed);raw.to_csv(OUT/'mixed_site_purity_repeats.csv',index=False);raw.groupby(['dataset','k']).agg(repeats=('repeat','count'),repeats_with_single_class_site=('single_class_site_count',lambda x:x.gt(0).sum()),single_class_site_occurrences=('single_class_site_count','sum')).reset_index().to_csv(OUT/'mixed_site_purity.csv',index=False)
    rows=[]
    for method,source in [('B0',[j for j in primary if j['arm']=='P']),('C0',cross)]:
        for j in source:rows.append(dict(method=method,k=j['k'],b=j.get('b',0),direction=j.get('direction','none'),repeat=j['repeat'],L=leakage(t,lookup.get_indexer(j['train']))))
    lr=pd.DataFrame(rows);lr.to_csv(OUT/'L_repeats.csv',index=False);lr.groupby(['method','k','b','direction']).L.mean().reset_index().to_csv(OUT/'L_table.csv',index=False)
    originals=json.loads(Path('/tmp/T14B2_original_source_sha256.json').read_text());sources=list(originals)+['abide2_replication.py','abide2_replication_report.py','tests/test_abide2_replication.py','docs/tasks/T14B_abide2_replication.md','AGENTS.md'];inputs=[CACHE/name for name in ['fisher_z.npy','row_order.csv']]+[REPO/'outputs/T14B/stage1'/name for name in ['subjects.csv','site_table.csv','feature_sha256.json']]
    for task,name in [('T12A','repeat_metrics.csv'),('T12B','performance_summary.csv'),('T12C2','paired_differences.csv'),('T13B','mean_results.csv'),('T14A','leakage_correlation.csv'),('T14A','all_L.csv'),('T12A','subjects.csv'),('T12A','splits.jsonl')]:inputs.append(REPO/'outputs'/task/name)
    cfg=dict(task='T14B_stage2',stage='prepared',started_at=datetime.now().astimezone().isoformat(),seeds=list(range(50)),stage1_subjects=568,construction_pool=546,sites=13,site_merge='ABIDEII-*_numeric suffix removed;NYU1/2 pooled',large_sites=['NYU','OHSU','KKI'],test_sites=4,n=20,k=[1,2,4],budgets=[2,5],LR_C=a.GRID,L3_C=m.GRID,E_subj='mean outer fold AUC primary;pooled OOF report only',truth='equal-test-site AUC',R1='gapk1>=.03;gapk4<gapk1;P reverse k4>k1',R2='k2 mean signed subject bias>=.10;mean absolute LSPO error<subject for k2,4',R3='twoway b5 separate k2/k4;paired C0-S0 test+reverse means>0 only if>=20 feasible repeats',R4='k2b5 L3 reverse>=.5 and>C0 in both directions;twoway L3-C0 test>=-.01',R5='all11 cells Spearman(L,C0/B0 native reverse AUC)<=-.8',intervals='2.5/97.5 repeat variability percentiles,not CI;positive fraction>1e-12 only',M_boundary='user confirmed keep repeat2 k4 SDSU0ASD/2TD;original algorithm ceil sampling then trim;no resampling',workers=4,process_limit=6,worker_start='stdlib spawn inherits-I;fixed trusted code path only',new_dependencies=[],ABIDEI_original_source_sha256=originals,source_sha256={name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in sources},input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},splits_sha256=hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest(),versions={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn']},jobs=len(work),S0_jobs=len(control),expected_fits=54650+21*len(control),BNU_watch_not_applicable=True)
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    for source,name in [('/tmp/T14B2_pytest_red.log','pytest_red.log'),('/tmp/T14B2_pytest_task.log','pytest_task.log'),('/tmp/T14B2_pytest_all.log','pytest_all.log')]:shutil.copyfile(source,OUT/name)
    print('PREPARED '+json.dumps(dict(jobs=len(work),S0_jobs=len(control),fits=cfg['expected_fits'])),flush=True)


def timing():
    from time import perf_counter
    prepare();print('START T14B stage2 timing',flush=True);tick=perf_counter();work=tasks();groups={}
    for w in work:
        j=w['job'];key=(w['part'],w['method'],j['k'],j.get('b',0),j.get('direction','none'));groups.setdefault(key,[]).append(w)
    pilot=[w for group in groups.values() for w in [group[0],group[len(group)//3]]];parallel(pilot);done=pd.read_csv(OUT/'completed.csv');rates=done.groupby(['part','method','k','b','direction']).seconds.mean();remaining=[w for w in work if w['task_id'] not in set(done.job_id)];worker_seconds=sum(rates[w['part'],w['method'],w['job']['k'],w['job'].get('b',0),w['job'].get('direction','none')] for w in remaining);point=perf_counter()-tick+worker_seconds/4+120
    value=dict(point_hours=point/3600,with_25percent_margin_hours=point*1.25/3600,pilot_seconds=perf_counter()-tick,pilot_tasks=len(pilot),remaining_tasks=len(remaining),pilot_reused=True,workers=4,formula='actual mean full task per part/method/k/b/direction at first and one-third available repeat;remaining sum/4+pilot wall+120s');(OUT/'timing.json').write_text(json.dumps(value,indent=2));cfg=json.loads((OUT/'config.json').read_text());cfg['stage']='timed';(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('TIMING_COMPLETE '+json.dumps(value),flush=True)


def run():
    from time import perf_counter
    from datetime import datetime
    cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='running',formal_started_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));done=set(pd.read_csv(OUT/'completed.csv').job_id);work=[w for w in tasks() if w['task_id'] not in done];print('START T14B stage2 remaining='+str(len(work)),flush=True);tick=perf_counter();parallel(work);cfg.update(stage='training_complete',formal_seconds=perf_counter()-tick,formal_finished_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('ALL_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    {'timing':timing,'run':run}[sys.argv[1]]()
