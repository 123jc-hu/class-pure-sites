"""T17-A 第二阶段，以 ADHD-200 名单复用 ABIDE 的拟合。"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from hashlib import file_digest
from itertools import permutations
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parent
sys.path.insert(0,str(REPO))

import numpy as np
import pandas as pd
import abide2_replication as old
from abide2_replication import a,b,c,m,s
from adhd200_prepare import CACHE,ID

STAGE1 = REPO/'outputs/T17A'
OUT = STAGE1/'stage2'
EXTRA = {'WashU','Pittsburgh'}
WORKERS = 4
SOURCES = ['abide2_replication.py','abide_class_pure_sites.py','bridge_design.py','same_class_control.py',
           'site_pair_validation.py','within_site_pair_model.py','design_rule_baselines.py',
           'adhd200_replication.py','adhd200_replication_report.py','tests/test_adhd200_replication.py',
           'docs/tasks/T17A_adhd200_prepare.md','AGENTS.md']
ABIDE_REFS = ['T12A/repeat_metrics.csv','T12B/performance_summary.csv','T12C2/paired_differences.csv',
              'T13B/mean_results.csv','T14A/all_L.csv','T14B/stage2/repeat_metrics.csv',
              'T14B/stage2/performance_summary.csv','T14B/stage2/C_paired_differences.csv',
              'T14B/stage2/L_outcomes.csv']


def table():
    f = pd.read_csv(STAGE1/'subjects.csv',dtype={ID:str})
    f = f.loc[f.include].reset_index(drop=True)
    sites = f.site.str.replace(r'^Peking_[123]$','Peking',regex=True)
    return f.assign(FILE_ID=f[ID],SITE_ID=f.site,site=sites,label=f.label.astype(int),
                    DX_GROUP=np.where(f.label.eq(1),1,2),SEX=f.Gender,AGE_AT_SCAN=f.Age,
                    func_mean_fd=np.nan,watch=False,construction_eligible=~sites.isin(EXTRA))


def natural_plans(t):
    jobs = []

    def add(pool,evaluation,repeat,fold,held,train,test,scopes):
        name = f'H_{pool}_{evaluation}_{repeat}_{fold}_{held}'
        j = dict(job_id=name,pool=pool,evaluation=evaluation,scopes=scopes,held_site=held,
                 repeat=repeat,outer_fold=fold,seed=0 if evaluation=='LOSO' else repeat,
                 train=train,test=test,standardizer_fit=train,reverse=[],k=0,n=0,arm='F',no_zero_roi=False)
        jobs.append(dict(part='H',method=evaluation,task_id=name,job=j))

    for pool,n in [('main',50),('pittsburgh',20)]:
        f = t.loc[t.site.ne('Pittsburgh')] if pool=='main' else t
        for r in range(n):
            for fold,(train,test) in enumerate(b.subject_folds(t,f.index.to_numpy(),r)):
                scopes = ['main','peking_separate'] if pool=='main' and r<20 else [pool]
                add(pool,'subject',r,fold,'',t.loc[train,'FILE_ID'].tolist(),t.loc[test,'FILE_ID'].tolist(),scopes)
        for site in sorted(f.site.unique()):
            held = f.site.eq(site)
            scopes = ['main','peking_separate'] if pool=='main' and site!='Peking' else [pool]
            add(pool,'LOSO',-1,-1,site,f.loc[~held,'FILE_ID'].tolist(),f.loc[held,'FILE_ID'].tolist(),scopes)
    main = t.loc[t.site.ne('Pittsburgh')]
    for site in sorted(main.loc[main.site.eq('Peking'),'SITE_ID'].unique()):
        held = main.SITE_ID.eq(site)
        add('peking_separate','LOSO',-1,-1,site,main.loc[~held,'FILE_ID'].tolist(),
            main.loc[held,'FILE_ID'].tolist(),['peking_separate'])
    return jobs


def construction_plans(t):
    pool = t.loc[t.construction_eligible]
    pairs = list(permutations(sorted(pool.site.unique()),2))
    primary = []
    for r in range(60):
        for job in a.repeat_design(pool,r,settings=[(1,20)],site_pair=pairs[r%20]):
            if job['arm']!='FD':
                key = f"r{r}_{job['arm']}_k{job['k']}_n{job['n']}"
                primary.append(job|dict(job_id=key,no_zero_roi=False))
    pure = [j for j in primary if j['arm']=='P']
    cross = [j for j in c.plans(t,pure,budgets=(2,5)) if j['method']=='C0']
    control,feasibility = s.plans(t,cross)
    return primary,cross,control,feasibility


def construction_tasks(primary,cross,control):
    work = [dict(part='A',method=j['arm'],job=j) for j in primary]
    work.extend(dict(part='B',method='E_subj',job=j) for j in primary if j['arm']=='P')
    work.extend(dict(part='C',method=j['method'],job=j) for j in cross+control)
    work.extend(dict(part='D',method='L3',job=j|dict(model='L3',method='L3',setting='secondary',
                     job_id=j['job_id'].replace('C0_','L3_',1))) for j in cross)
    for w in work:
        w['task_id'] = w['part']+'_'+w['job']['job_id']
    return work


def fit_task(w,data_override):
    result = old.fit_task(w,data_override=data_override)
    for row in result['splits']:
        # A/H 的每次内层拟合也经过同一个 sklearn Pipeline。
        if w['part'] in ('A','H'):
            row['standardizer_fit'] = row['train']
    return result


def worker_setup():
    global DATA
    assert sys.flags.isolated
    t = table()
    order = pd.read_csv(CACHE/'row_order.csv',dtype=str)[ID]
    x = np.load(CACHE/'fisher_z.npy',allow_pickle=False,mmap_mode='r')
    assert t.FILE_ID.tolist()==order.tolist() and x.shape==(865,17955)
    DATA = t,x


def train(w):
    return fit_task(w,DATA)


def parallel(work):
    old.OUT = OUT
    remaining = iter(work[WORKERS:])
    with ProcessPoolExecutor(WORKERS,mp_context=get_context('spawn'),initializer=worker_setup) as pool:
        active = {pool.submit(train,w) for w in work[:WORKERS]}
        while active:
            finished,_ = wait(active,return_when=FIRST_COMPLETED)
            for future in finished:
                active.remove(future)
                result = future.result()
                old.save(result)
                row = result['done']
                print(f"TRAIN_DONE {row['job_id']} seconds={row['seconds']:.3f}",flush=True)
                w = next(remaining,None)
                if w is not None:
                    active.add(pool.submit(train,w))


def digest(path):
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def prepare():
    from design_rule_baselines import leakage
    t = table()
    primary,cross,control,feasibility = construction_plans(t)
    work = natural_plans(t)+construction_tasks(primary,cross,control)
    assert len(work)==1314 and len({w['task_id'] for w in work})==len(work)
    index = pd.Index(t.FILE_ID)
    for j in cross:
        train = index.get_indexer(j['train'])
        for _,va in m.inner_folds(t,train,j['seed'],min(5,j['b'])):
            held = t.loc[va]
            assert len(m.pairs(held.label.to_numpy(),held.site.to_numpy())[0])>0,j['job_id']
    l_rows = []
    for name,cells in [('B0',[j for j in primary if j['arm']=='P']),('C0',cross)]:
        for j in cells:
            l_rows.append(dict(method=name,repeat=j['repeat'],k=1,b=j.get('b',0),
                               direction=j.get('direction','none'),L=leakage(t,index.get_indexer(j['train']))))
    OUT.mkdir()
    pd.DataFrame(l_rows).to_csv(OUT/'L_repeats.csv',index=False)
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(w)+'\n' for w in work))
    t[a.META+['construction_eligible','watch']].to_csv(OUT/'subjects.csv',index=False)
    pd.DataFrame(feasibility).to_csv(OUT/'S0_feasibility.csv',index=False)
    inputs = [STAGE1/'subjects.csv',CACHE/'fisher_z.npy',CACHE/'row_order.csv',OUT/'splits.jsonl',
              *(REPO/'outputs'/p for p in ABIDE_REFS)]
    config = dict(stage='prepared',jobs=len(work),workers=WORKERS,processes_max=6,blas_threads=1,
                  seeds=dict(natural_subject=list(range(50)),natural_LOSO=0,sensitivity=list(range(20)),
                             construction=list(range(60)),bridge='[r,1,3]',same_class='[r,1,4]'),
                  LR_C=a.GRID,L3_C=m.GRID,
                  sensitivity='male and 7<=Age<=13 filter saved predictions; different Peking LOSO / Pittsburgh membership refit',
                  source_sha256={s:digest(REPO/s) for s in SOURCES},input_sha256={str(p):digest(p) for p in inputs})
    dump('config.json',config)
    return work


def time_group(w):
    j = w['job']
    return w['part'],w['method'],j.get('pool',''),j.get('b',0),j.get('direction','')


def timing():
    work = prepare()
    groups = {}
    for w in work:
        groups.setdefault(time_group(w),[]).append(w)
    sample = [w for key,rows in groups.items()
              for w in (rows if key[:2]==('H','LOSO') else [rows[0],rows[len(rows)//3]])]
    print(f'START T17-A stage2 timing tasks={len(sample)}',flush=True)
    tick = perf_counter()
    parallel(sample)
    elapsed = perf_counter()-tick
    done = pd.read_csv(OUT/'completed.csv').set_index('job_id')
    average = {key:float(done.loc[[w['task_id'] for w in sample if time_group(w)==key],'seconds'].mean())
               for key in groups}
    remaining = [w for w in work if w['task_id'] not in done.index]
    points = {part:sum(average[time_group(w)] for w in remaining if w['part'] in part)
              /WORKERS for part in ('H','ABCD')}
    point = elapsed+sum(points.values())+120
    result = dict(pilot_tasks=len(sample),pilot_seconds=elapsed,remaining_tasks=len(remaining),
                  natural_remaining_seconds=points['H'],construction_remaining_seconds=points['ABCD'],
                  point_hours=point/3600,pilot_reused=True)
    dump('timing.json',result)
    print('TIMING_COMPLETE '+json.dumps(result),flush=True)


def run():
    config = json.loads((OUT/'config.json').read_text())
    for key in ['source_sha256','input_sha256']:
        for p,h in config[key].items():
            assert digest(REPO/p)==h,p
    work = [json.loads(line) for line in (OUT/'splits.jsonl').read_text().splitlines()]
    done = set(pd.read_csv(OUT/'completed.csv').job_id)
    remaining = [w for w in work if w['task_id'] not in done]
    assert len(work)==config['jobs'] and len(work)-len(done)==len(remaining)
    print(f'START T17-A stage2 remaining={len(remaining)}',flush=True)
    tick = perf_counter()
    parallel(remaining)
    dump('config.json',config|dict(stage='training_complete',formal_seconds=perf_counter()-tick))
    print('ALL_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    {'timing':timing,'run':run}[sys.argv[1]]()
