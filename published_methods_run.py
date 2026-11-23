"""T19 timing and main experiment on the fixed ABIDE-I partitions."""
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
import json
import sys
import warnings

sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import published_methods as task
from site_pair_validation import subject_folds

OUT = task.OUT/'rev2'
WORKERS = 5
MODELS = ['Parisot-HO','Parisot-CC200','Kunda']
MH = (50,150,300)
GROUP = ['model','design','k','kind']


def work(table, n_repeats):
    output = []
    for j in task.main_jobs():
        if j['repeat']>=n_repeats:
            continue
        common = {k:j[k] for k in ('job_id','repeat','seed','k','test_sites')}
        common['source_job_id'] = common.pop('job_id')
        common['design'] = j['method']
        split = dict(train=j['train'],test=j['test'],reverse=j['reverse'])
        rows = [common|split|dict(kind='main',fold=-1)]
        if j['method']=='B0' and j['k']==2:
            train = pd.Index(table.FILE_ID).get_indexer(j['train'])
            for fold,(tr,te) in enumerate(subject_folds(table,train,j['seed'])):
                rows.append(common|dict(kind='E_subj',fold=fold,test_sites=[],train=table.FILE_ID.iloc[tr].tolist(),
                                        test=table.FILE_ID.iloc[te].tolist(),reverse=[]))
        for row in rows:
            for model in MODELS:
                output.append(row|dict(model=model,job_id=f"{model}/{j['job_id']}/{row['fold']}"))
    assert len(output)==33*n_repeats and len({j['job_id'] for j in output})==len(output)
    return output


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def input_data(table,models=MODELS):
    data = {}
    if 'Parisot-CC200' in models:
        assert pd.read_csv(task.a.CACHE/'row_order.csv').FILE_ID.equals(table.FILE_ID)
        data['Parisot-CC200'] = np.load(task.a.CACHE/'fisher_z.npy',mmap_mode='r')
    rho = task.correlation(task.read_timeseries(table.FILE_ID,'ho'))
    i,j = np.triu_indices(rho.shape[1],1)
    ho = np.arctanh(rho[:,i,j]); del rho
    corr = task.correlation(task.read_timeseries(table.FILE_ID,'cc200'))
    return data|{'Parisot-HO':ho,'Kunda':corr}


def run_job(job):
    start = perf_counter()
    n_train = len(job['train'])
    n_test = len(job['test'])
    ids = pd.Index(TABLE.FILE_ID)
    rows = ids.get_indexer(job['train']+job['test']+job['reverse'])
    assert np.min(rows)>=0 and len(set(rows))==len(rows)
    table = TABLE.iloc[rows]
    raw_site = table.SITE_ID.to_numpy()
    labels = table.label.to_numpy()
    sex = table.SEX.to_numpy()
    train = np.arange(n_train)
    x = DATA[job['model']][rows]
    params,dimensions,notes = {},[],[]
    if job['model']=='Kunda':
        dimensions = [h for h in MH if h<=len(rows)]
        phenotype = np.column_stack([sex==1,sex==2,table.AGE_AT_SCAN])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            score,pred,params = task.kunda_ridge(task.tpe(x,train),raw_site,phenotype,train,labels[train],dimensions)
        notes = [str(w.message) for w in captured]
    else:
        score,pred = task.parisot_fit(x,raw_site,sex,train,labels[train])
    context = {k:job[k] for k in ('job_id','source_job_id','repeat','seed','design','k','kind','fold','model')}
    target = table.iloc[n_train:][['FILE_ID','site','label']]
    frame = target.assign(**context,reverse=np.arange(len(target))>=n_test,score=score[n_train:],prediction=pred[n_train:])
    receipt = context|dict(seconds=perf_counter()-start,n_nodes=len(rows),n_train=n_train,n_test=n_test,
        n_reverse=len(job['reverse']),h_grid=json.dumps(dimensions),warnings=json.dumps(notes))|dict.fromkeys(
        ['mu','h','alpha','inner_accuracy'])|params
    return receipt,frame


def append(name,frame):
    with (OUT/name).open('a') as target:
        frame.to_csv(target,index=False,header=target.tell()==0)


def parallel(jobs):
    with get_context('fork').Pool(WORKERS) as pool:
        for row,frame in pool.imap_unordered(run_job,jobs):
            append('predictions.csv',frame)
            append('completed.csv',pd.DataFrame([row]))
            print(f"TRAIN_DONE {row['job_id']} seconds={row['seconds']:.3f}",flush=True)


def timing(core=False):
    global TABLE,DATA
    gate = json.loads((OUT/'config.json').read_text())
    assert gate['status']=='fidelity_passed'
    print('START T19 timing repeats 0–1' + (' full-grid jobs' if core else ''),flush=True)
    begin = perf_counter()
    TABLE = task.a.subjects()
    full = work(TABLE,50)
    DATA = input_data(TABLE)
    prep = perf_counter()-begin
    pilot = [j for j in full if j['repeat']<2]
    core_path = OUT/'timing_core.json'
    if core:
        pilot = [j for j in pilot if j['model']!='Kunda' or len(j['train']+j['test']+j['reverse'])>=max(MH)]
        parallel(pilot)
        write_json(core_path,dict(pilot_seconds=perf_counter()-begin,jobs=len(pilot)))
        print(f'FINISH T19 full-grid timing jobs={len(pilot)}',flush=True)
        return
    observed = set(pd.read_csv(OUT/'completed.csv').job_id) if core_path.exists() else set()
    parallel([j for j in pilot if j['job_id'] not in observed])
    wall = perf_counter()-begin
    if core_path.exists():
        wall += json.loads(core_path.read_text())['pilot_seconds']
    done = pd.read_csv(OUT/'completed.csv')
    assert len(done)==len(pilot)==66 and set(done.job_id)=={j['job_id'] for j in pilot}
    pace = done.groupby(GROUP).seconds.mean()
    projections = {n:wall+prep+sum(pace[tuple(j[k] for k in GROUP)] for j in full
                  if 2<=j['repeat']<n)/WORKERS for n in (20,50)}
    n = 50 if projections[50]<=3*3600 else 20
    plan = [j for j in full if j['repeat']<n]
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in plan))
    result = dict(repeats=n,workers=WORKERS,seeds=list(range(n)),jobs=len(plan),pilot_jobs=len(done),
        pilot_seconds=wall,input_preparation_seconds=prep,projected_seconds=projections,retain_pilot_predictions=True,
        estimate='pilot wall + second input prep + sum of stratum mean times for remaining jobs / workers',
        kunda_phenotypes=['AGE_AT_SCAN','SEX'],kunda_h_grid=list(MH),kunda_mu_grid=[.5,.75,1.],
        kunda_alpha_grid=[.25,.5,.75],
        dimension_rule='Use only h in {50, 150, 300} with h <= the number of subjects entering MIDA. '
                       'Only the subject count determines the grid. No values are added.',
        graph_and_mida_sites='SITE_ID',evaluation_sites='T12A pooled-institution site',E_subj='mean of five fold AUCs',
        cc200_cache=str(task.a.CACHE),ho_features='official per-subject Nilearn correlation, then upper triangle Fisher z')
    write_json(OUT/'main_config.json',result)
    print('FINISH T19 timing '+json.dumps(projections),flush=True)


def run():
    global TABLE,DATA
    start = perf_counter()
    TABLE = task.a.subjects()
    cfg = json.loads((OUT/'main_config.json').read_text())
    jobs = list(map(json.loads,(OUT/'splits.jsonl').read_text().splitlines()))
    observed = set(pd.read_csv(OUT/'completed.csv').job_id)
    remaining = [j for j in jobs if j['job_id'] not in observed]
    assert len(jobs)==cfg['jobs'] and observed<={j['job_id'] for j in jobs}
    print(f"START T19 main repeats={cfg['repeats']} completed={len(observed)} remaining={len(remaining)}",flush=True)
    DATA = input_data(TABLE)
    parallel(remaining)
    cfg['main_elapsed_seconds'] = perf_counter()-start
    write_json(OUT/'main_config.json',cfg)
    print(f"FINISH T19 main jobs={len(jobs)} seconds={cfg['main_elapsed_seconds']:.1f}",flush=True)


if __name__=='__main__':
    {'timing_core':lambda:timing(core=True),'timing':timing,'run':run}[sys.argv[1]]()
