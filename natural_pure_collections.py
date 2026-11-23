"""T15-B：固定天然纯类集合，直接复用C0的嵌套LR。"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import abide2_replication as old
import abide_class_pure_sites as a

OUT=ROOT/'outputs/T15B';KEY=['job_id','pool','evaluation','site_definition','repeat','seed','outer_fold','held_site']


def table():
    t=old.table();t['site']=t.collection.str.removeprefix('ABIDEII-');t['pool_main']=~t.collection.isin(['ABIDEII-NYU_2','ABIDEII-UCLA_1']);t['pool_secondary']=t.collection.ne('ABIDEII-UCLA_1');t['mixed_reference']=~t.collection.isin(['ABIDEII-KUL_3','ABIDEII-NYU_2','ABIDEII-UCLA_1']);t['adult_male_reference']=t.mixed_reference&t.label.eq(1)&t.AGE_AT_SCAN.ge(18)&t.SEX.eq(1);t['watch']=False
    return t[a.META+['collection','watch','pool_main','pool_secondary','mixed_reference','adult_male_reference']]


def plans(secondary_repeats=50):
    t=table();work=[]
    for pool,repeats in [('main',50),('secondary',secondary_repeats)]:
        ids=t.index[t['pool_'+pool]].to_numpy();f=t.loc[ids]
        for repeat in range(repeats):
            for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=repeat).split(ids,f.label)):
                work.append(dict(job_id=f'{pool}_subject_r{repeat}_f{fold}',pool=pool,evaluation='subject',site_definition='separate',repeat=repeat,seed=repeat,outer_fold=fold,held_site='',train=t.loc[ids[tr],'FILE_ID'].tolist(),test=t.loc[ids[te],'FILE_ID'].tolist()))
        for site in sorted(f.site.unique()):
            work.append(dict(job_id=f'{pool}_LOSO_{site}',pool=pool,evaluation='LOSO',site_definition='separate',repeat=-1,seed=0,outer_fold=-1,held_site=site,train=f.loc[f.site.ne(site),'FILE_ID'].tolist(),test=f.loc[f.site.eq(site),'FILE_ID'].tolist()))
    f=t.loc[t.mixed_reference]
    for site in sorted(f.site.unique()):work.append(dict(job_id=f'main_no_KUL_{site}',pool='main',evaluation='LOSO_no_KUL',site_definition='separate',repeat=-1,seed=0,outer_fold=-1,held_site=site,train=f.loc[f.site.ne(site),'FILE_ID'].tolist(),test=f.loc[f.site.eq(site),'FILE_ID'].tolist()))
    f=t.loc[t.pool_secondary];nyu=f.collection.isin(['ABIDEII-NYU_1','ABIDEII-NYU_2'])
    work.append(dict(job_id='secondary_merged_LOSO_NYU',pool='secondary',evaluation='LOSO',site_definition='merged',repeat=-1,seed=0,outer_fold=-1,held_site='NYU',train=f.loc[~nyu,'FILE_ID'].tolist(),test=f.loc[nyu,'FILE_ID'].tolist()))
    for j in work:j['standardizer_fit']=j['train']
    return work


def inner_folds(t,j):
    train=pd.Index(t.FILE_ID).get_indexer(j['train'])
    return [(train[tr],train[va]) for tr,va in StratifiedKFold(5,shuffle=True,random_state=j['seed']).split(train,t.loc[train,'label'])]


def aliases(secondary_repeats=50):
    jobs=plans(secondary_repeats);rows=[]
    for scenario,pool in [('main','main'),('nyu_separate','secondary'),('nyu_merged','secondary')]:
        for j in jobs:
            if j['pool']!=pool or j['evaluation']=='LOSO_no_KUL':continue
            if scenario!='nyu_merged' and j['site_definition']=='merged':continue
            if scenario=='nyu_merged' and j['evaluation']=='LOSO' and j['held_site'] in ['NYU_1','NYU_2']:continue
            rows.append(dict(scenario=scenario,source_job_id=j['job_id'],evaluation=j['evaluation'],repeat=j['repeat'],outer_fold=j['outer_fold'],held_site=j['held_site']))
    return pd.DataFrame(rows)


def fit_job(j):
    from time import perf_counter
    assert sys.flags.isolated
    tick=perf_counter();t=table()
    if j['site_definition']=='merged':t.loc[t.collection.isin(['ABIDEII-NYU_1','ABIDEII-NYU_2']),'site']='NYU'
    x=np.load(old.CACHE/'fisher_z.npy',mmap_mode='r');legacy=j|dict(no_zero_roi=False,arm='F',k=0,n=0,reverse=[]);result=a.fit_job(legacy,data=(t,x));context={key:j[key] for key in KEY}
    assert set(j['train']).isdisjoint(j['test'])
    for name in ['pred','diagnostics','warnings']:
        for key,value in context.items():result[name][key]=value
    for row in result['splits']:row.update(context,standardizer_fit=row['train'])
    L=None
    if j['pool']=='main' and j['evaluation']!='subject' and j['held_site']!='KUL_3':
        from design_rule_baselines import leakage
        L=leakage(t,pd.Index(t.FILE_ID).get_indexer(j['train']))
    result['done'].update(context,seconds=perf_counter()-tick,fit_count=21,L=L);return result


def append(name,f):
    with (OUT/name).open('a') as target:f.to_csv(target,index=False,header=target.tell()==0)


def save(result):
    append('predictions.csv',result['pred'][a.META+['collection','watch']+KEY+['probability','prediction']]);append('completed.csv',pd.DataFrame([result['done']]).reindex(columns=KEY+['C','best_inner_AUC','seconds','n_train','n_test','fit_count','L','final_n_iter','warning_count']))
    append('selection_fits.csv',result['diagnostics'][KEY+['C','inner_mean_AUC','inner_sd_AUC']+[f'fold{i}_AUC' for i in range(5)]])
    if len(result['warnings']):append('warnings.csv',result['warnings'].reindex(columns=KEY+['category','message','count']))
    with (OUT/'inner_splits.jsonl').open('a') as f:
        for row in result['splits']:f.write(json.dumps(row)+'\n')


def parallel(work):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    remaining=iter(work[4:])
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(fit_job,j) for j in work[:4]}
        while pending:
            finished,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                pending.remove(future);result=future.result();save(result);print('TRAIN_DONE '+json.dumps({key:result['done'][key] for key in ['job_id','seconds','warning_count']}),flush=True);j=next(remaining,None)
                if j is not None:pending.add(pool.submit(fit_job,j))


def freeze_splits(secondary_repeats):
    import hashlib
    work=plans(secondary_repeats);(OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in work));aliases(secondary_repeats).to_csv(OUT/'evaluation_aliases.csv',index=False);cfg=json.loads((OUT/'config.json').read_text());cfg.update(secondary_repeats=secondary_repeats,jobs=len(work),expected_fits=21*len(work),splits_sha256=hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest(),aliases_sha256=hashlib.sha256((OUT/'evaluation_aliases.csv').read_bytes()).hexdigest());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))


def prepare():
    import hashlib,shutil
    from datetime import datetime
    from importlib.metadata import version
    OUT.mkdir();t=table();t.to_csv(OUT/'subjects.csv',index=False);t.loc[t.pool_main].to_csv(OUT/'main_pool.csv',index=False);t.loc[t.pool_secondary].to_csv(OUT/'secondary_pool.csv',index=False);t.loc[t.adult_male_reference].groupby('site').size().rename('ASD_n').reset_index().to_csv(OUT/'adult_male_reference_counts.csv',index=False)
    sources=['natural_pure_collections.py','natural_pure_collections_report.py','tests/test_natural_pure_collections.py','abide_class_pure_sites.py','abide2_replication.py','design_rule_baselines.py','docs/tasks/T15B_natural_pure_collections.md','AGENTS.md'];inputs=[old.CACHE/name for name in ['fisher_z.npy','row_order.csv']]+[ROOT/'outputs/T14B/stage1/subjects.csv']
    cfg=dict(task='T15B',stage='prepared',created_at=datetime.now().astimezone().isoformat(),main_subjects=547,main_mixed_subjects=526,secondary_subjects=567,main_sites=14,secondary_separate_sites=15,secondary_merged_sites=14,main_repeats=50,secondary_repeats=50,seeds=list(range(50)),LOSO_seed=0,inner='direct T12A select_model via fit_job;diagnosis StratifiedKFold5 shuffle random_state=outer seed;train-only StandardScaler;mean probability AUC;first C on ties',C=a.GRID,LR='sklearn defaults:L2,lbfgs,max_iter100,tol1e-4,no class weights',workers=4,process_limit=6,BLAS_threads=1,python_isolated=True,pre_run_prediction='DiD>0 because within-site ASD training subjects may expose KUL_3 shortcut;not a guarantee',criterion='K1 only main KUL_3 Sens DiD mean over50 repeats>=.10',control='13 mixed sites ASD,equal site means;NYU_2 and KUL_3 excluded from control',adult_male='50 mixed-reference ASD,age>=18,SEX1;pooled by subjects,not equal sites;predictions reused,no retraining',secondary='NYU_2 target20;KUL_3 incidental target21;both compare to base13 mixed-reference ASD;NYU_merged LOSO excludes all73 NYU subjects',reuse='same567 cohort implies identical diagnosis-only subject folds/training/scalers for both site definitions;nonNYU LOSO also identical;main +KUL analysis2 uses main LOSO fits',intervals='repeat2.5/97.5 variability ranges,not CI;analysis2 range=min/max over13 sites,no hypothesis test',dependencies_installed=[],versions={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn']},BNU_watch_not_applicable=True,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));freeze_splits(50)
    for source,name in [('/tmp/T15B_pytest_red.log','pytest_red.log'),('/tmp/T15B_pytest_task.log','pytest_task.log'),('/tmp/T15B_pytest_all.log','pytest_all.log')]:shutil.copyfile(source,OUT/name)


def timing():
    from time import perf_counter
    prepare();tick=perf_counter();print('START T15B timing',flush=True);selected=[j for j in plans() if j['evaluation']=='subject' and j['repeat'] in [0,16] and j['outer_fold']==0]
    for pool in ['main','secondary']:selected.extend([j for j in plans() if j['pool']==pool and j['evaluation']=='LOSO' and j['held_site'] in ['KKI_1','KUL_3']])
    selected.extend([next(j for j in plans() if j['evaluation']=='LOSO_no_KUL' and j['held_site']=='KKI_1'),next(j for j in plans() if j['site_definition']=='merged')]);parallel(selected);done=pd.read_csv(OUT/'completed.csv');rate=float((done.seconds/done.n_train).mean())
    def estimate(repeats):
        remaining=[j for j in plans(repeats) if j['job_id'] not in set(done.job_id)]
        return perf_counter()-tick+rate*sum(len(j['train']) for j in remaining)/4+180
    full=estimate(50);repeats=20 if full>10800 else 50;point=estimate(repeats);freeze_splits(repeats);value=dict(full50_point_hours=full/3600,point_hours=point/3600,with_25percent_margin_hours=point*1.25/3600,pilot_seconds=perf_counter()-tick,pilot_jobs=len(selected),pilot_reused=True,secondary_repeats=repeats,workers=4,formula='mean actual end-to-end seconds/trainN from main/secondary repeated0+16 and actual LOSO sizes;remaining actual trainN sum/4+pilot wall+180s');(OUT/'timing.json').write_text(json.dumps(value,indent=2));cfg=json.loads((OUT/'config.json').read_text());cfg['stage']='timed';(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('TIMING_COMPLETE '+json.dumps(value),flush=True)


def run():
    from time import perf_counter
    from datetime import datetime
    cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='running',started_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));done=set(pd.read_csv(OUT/'completed.csv').job_id);work=[j for j in plans(cfg['secondary_repeats']) if j['job_id'] not in done];print('START T15B remaining='+str(len(work)),flush=True);tick=perf_counter();parallel(work);cfg.update(stage='training_complete',formal_seconds=perf_counter()-tick,finished_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('ALL_TRAINING_COMPLETE jobs='+str(cfg['jobs']),flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    {'timing':timing,'run':run}[sys.argv[1]]()
