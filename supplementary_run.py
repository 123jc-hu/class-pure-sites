"""T18 修订一计时和正式运行入口。"""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from itertools import product
import json
from multiprocessing import get_context
from pathlib import Path
from shutil import copytree
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import supplementary_analyses as task
import supplementary_site_probe as probe
import supplementary_inference as stats

OUT = task.OUT
TIMING = OUT/'timing_rev1'
WORKERS = 4


def worker_setup():
    global GROUPS,INDEXED_ANCHORS,DESIGNS
    GROUPS = probe.populations()
    t = GROUPS['ABIDE-I_all'][1]
    INDEXED_ANCHORS = {j['job_id']:j for j in task.anchors()}
    DESIGNS = {f"{j['method']}_r{j['repeat']}":j for r in range(50) for j in task.same_design(t,r)}


def execute(job):
    tick = perf_counter()
    kind = job['kind']
    x,t = GROUPS['ABIDE-I_all'][:2]
    frame,splits,extra = None,[],[]
    meta = job.copy()
    if kind=='A1':
        name,r,f = (job[key] for key in ('group','repeat','fold'))
        x,t,y,ids,binary = GROUPS[name]
        null = job['mode']=='null'
        if null:
            y = probe.permuted_labels(y,ids,list(GROUPS).index(name),r)
        fold_seed = 0 if null else r
        train,test = probe.folds(y,ids,fold_seed)[f]
        result = probe.probe_fold(x,y,train,test,fold_seed,binary,fixed_c=job.get('fixed_c'))
        meta.update(C=result['C'],warning_count=result['convergence_warnings'],other_warnings=result['other_warnings'])
        frame = t.loc[test,['FILE_ID','site','label']].assign(group=name,mode=job['mode'],repeat=r,
                  fold=f,target_site=y[test],site_prediction=result['score'])
        all_ids = t.FILE_ID.to_numpy()
        splits = [job|dict(train=all_ids[train].tolist(),test=all_ids[test].tolist(),
                  inner=[row|{field:all_ids[row[field]].tolist() for field in ['train','validation','standardizer_fit']}
                         for row in result['inner']])]
    elif kind=='A2':
        j = INDEXED_ANCHORS[job['anchor']]
        groups = task.same_diagnosis(t,j)
        frame = task.a2_predict(x,t,j,job['method'],groups)
        splits = [job|dict(train=j['train'],evaluation=groups)]
    elif kind=='O':
        j = INDEXED_ANCHORS[job['anchor']]
        split = task.normative_plan(t,j)
        frame,extra = task.normative_predict(x,t,j,split)
        splits = [job|split]
    elif kind=='S':
        j = DESIGNS[job['design']]
        frame,done,inner = task.split_predict(x,t,j)
        splits = [job|j|dict(inner=inner)]
        meta.update(C=done['C'],warning_count=done['warning_count'])
    else:
        stats.inference(TIMING/'part2')
    meta['seconds'] = perf_counter()-tick
    return meta,frame,splits,extra


def parallel(jobs,save):
    remaining = iter(jobs[WORKERS:])
    with ProcessPoolExecutor(WORKERS,mp_context=get_context('spawn'),initializer=worker_setup) as pool:
        active = {pool.submit(execute,job) for job in jobs[:WORKERS]}
        while active:
            finished,_ = wait(active,return_when=FIRST_COMPLETED)
            for future in finished:
                active.remove(future)
                result = future.result()
                save(*result)
                row = result[0]
                print(f"TRAIN_DONE {row['id']} seconds={row['seconds']:.3f}",flush=True)
                job = next(remaining,None)
                if job is not None:
                    active.add(pool.submit(execute,job))


def a1_job(group,mode,r,f,**options):
    suffix = str(options['fixed_c']) if 'fixed_c' in options else ''
    return dict(kind='A1',id=f'A1/{group}/{mode}/{r}/{f}/{suffix}',group=group,mode=mode,repeat=r,fold=f,**options)


def anchors_work(js,repeat=None):
    work = []
    for j in js:
        if repeat is not None and j['repeat']!=repeat:
            continue
        key = j['job_id']
        for method in [j['method']]+(['L3'] if j['b']==5 else []):
            work.append(dict(kind='A2',id=f'A2/{key}/{method}',anchor=key,method=method))
        work.append(dict(kind='O',id=f'O/{key}',anchor=key))
    return work


def split_work(repeats):
    return [dict(kind='S',id=f'S/{method}_r{r}',design=f'{method}_r{r}')
            for r,method in product(repeats,['P_same','P_cross','M_same'])]


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def table_of_designs(groups):
    return {name:dict(subjects=len(ids),sites=len(np.unique(y[ids])),binary=binary)
            for name,(_,_,y,ids,binary) in groups.items()}


def timing():
    tick = perf_counter()
    TIMING.mkdir(exist_ok=True)
    groups = probe.populations()
    designs = table_of_designs(groups)
    anchors = task.anchors()
    queued = [a1_job(name,'observed',0,0) for name in groups]
    queued += [a1_job(name,'null',0,0,fixed_c=c) for name in groups if not designs[name]['binary'] for c in task.a.GRID]
    queued += anchors_work(anchors,repeat=0)+split_work([0])+[dict(kind='I',id='I/statistics')]
    dump(TIMING/'config.json',dict(designs=designs,workers=WORKERS,max_iter=probe.MAX_ITER,seed_observed=0,
                                  seed_null=[2026,'group_index',0],fold=0,work=queued))
    completed = TIMING/'jobs.jsonl'
    rows = [json.loads(s) for s in completed.read_text().splitlines()] if completed.exists() else []
    finished = {r['id'] for r in rows}
    assert len(finished)==len(rows) and finished<={j['id'] for j in queued}
    pending = [j for j in queued if j['id'] not in finished]
    print(f'START T18 rev1 timing jobs={len(pending)}, reused={len(rows)}',flush=True)

    def save(row,frame,splits,extra):
        rows.append(row)
        with completed.open('a') as f:
            f.write(json.dumps(row)+'\n')

    parallel(pending,save)
    raw = pd.DataFrame(rows)
    observed = raw.loc[raw.kind.eq('A1')&raw['mode'].eq('observed')].set_index('group')
    null = raw.loc[raw.kind.eq('A1')&raw['mode'].eq('null')]
    projection = []
    for name,r in observed.iterrows():
        d = null.loc[null.group.eq(name)].set_index('fixed_c').seconds
        seconds = 25*r.seconds
        projection.append(dict(group=name,timing_C=r.C,observed_seconds=seconds,observed_fold_seconds=r.seconds,
                               null_seconds=500*d.loc[r.C] if len(d) else 0.,
                               null_seconds_min=500*d.min() if len(d) else 0.,
                               null_seconds_max=500*d.max() if len(d) else 0.))
    g = pd.DataFrame(projection)
    g.to_csv(TIMING/'A1_timing_projection.csv',index=False)
    point1 = (g.observed_seconds+g.null_seconds).sum()/WORKERS
    point2 = raw.loc[raw.kind.eq('I'),'seconds'].item()
    point3 = raw.loc[raw.kind.eq('O'),'seconds'].sum()*50/WORKERS
    point4 = raw.loc[raw.kind.eq('S'),'seconds'].sum()*50/WORKERS
    a2 = raw.loc[raw.kind.eq('A2'),'seconds'].sum()*50/WORKERS
    points = dict(part1=point1+a2,part2=point2,part3=point3,part4=point4)
    value = dict(timing_jobs=len(rows),timing_last_session_seconds=perf_counter()-tick,point_seconds=points,
                 total_point_hours=sum(points.values())/3600,
                 A1_seconds=point1,A2_seconds=a2,
                 total_hours_min_profile=float((sum(points.values())+(g.null_seconds_min-g.null_seconds).sum()/WORKERS)/3600),
                 total_hours_max_profile=float((sum(points.values())+(g.null_seconds_max-g.null_seconds).sum()/WORKERS)/3600),
                 C_estimate='pilot first observed fold; null times profiled at all four Cs; formal uses the mode of 25 folds')
    dump(OUT/'timing_rev1.json',value)
    print('TIME_DONE '+json.dumps(value),flush=True)


def append(path,frame):
    with path.open('a') as f:
        frame.to_csv(f,index=False,header=f.tell()==0)


def save(row,frame,splits,extra):
    part = OUT/{'A1':'part1','A2':'part1','O':'part3','S':'part4'}[row['kind']]
    stem = row['kind'].lower()
    append(part/(stem+'_predictions.csv'),frame)
    with (part/(stem+'_splits.jsonl')).open('a') as f:
        for split in splits:
            f.write(json.dumps(split)+'\n')
    if extra:
        append(part/'nu_selection.csv',pd.DataFrame(extra))
    done = {key:row.get(key,'') for key in ['id','kind','group','mode','repeat','fold','C','warning_count',
                                           'other_warnings','seconds']}
    append(OUT/'completed.csv',pd.DataFrame([done]))


def run():
    timer = json.loads((OUT/'timing_rev1.json').read_text())
    assert timer['total_point_hours']<=3
    groups = probe.populations()
    designs = table_of_designs(groups)
    for p in ('part1','part3','part4'):
        (OUT/p).mkdir()
    anchors = task.anchors()
    dump(OUT/'part1'/'config.json',dict(designs=designs,seeds=list(range(5)),permutations=100,
            perm_seed='[2026, group_index, repeat]',max_iter=probe.MAX_ITER,C_grid=task.a.GRID))
    dump(OUT/'part3'/'config.json',dict(seeds=list(range(50)),pca_components=30,nu=[.05,.1,.2],
            calibration='per-site floor half of test TD, repeat seed'))
    dump(OUT/'part4'/'config.json',dict(seeds=list(range(50)),M_same_even=[7,6],M_same_odd=[6,7]))
    availability = []
    t = groups['ABIDE-I_all'][1]
    for anchor in anchors:
        for dx,roles in task.same_diagnosis(t,anchor).items():
            availability.append({k:anchor[k] for k in ['job_id','method','k','b','direction','repeat']}|
                 dict(label=dx,**{k:len(v) for k,v in roles.items()},eligible=min(map(len,roles.values()))>=5))
    pd.DataFrame(availability).to_csv(OUT/'part1'/'A2_availability.csv',index=False)
    copytree(TIMING/'part2',OUT/'part2')
    dump(OUT/'part2'/'config.json',dict(bootstrap_seeds=[[2026,0],[2026,1]],draws=2000,
         sources='existing ABIDE-I/II and ADHD-200 predictions and metrics'))
    work = anchors_work(anchors)+split_work(range(50))
    obs = [a1_job(name,'observed',r,f) for name,r,f in product(groups,range(5),range(5))]
    print(f'START T18 rev1 formal {len(obs)+len(work)} nested and other jobs; 4500 fixed-C folds follow',flush=True)
    tick = perf_counter()
    parallel(obs,save)
    votes = pd.read_csv(OUT/'completed.csv')
    constants = {name:float(values.C.mode().min()) for name,values in votes.groupby('group')}
    dump(OUT/'part1'/'fixed_C.json',constants)
    null=[a1_job(name,'null',r,f,fixed_c=constants[name])
          for name,r,f in product(groups,range(100),range(5)) if not designs[name]['binary']]
    parallel(null+work,save)
    dump(OUT/'formal_status.json',dict(stage='training_complete',seconds=perf_counter()-tick,
                                     work=len(obs)+len(null)+len(work)))
    print('TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    {'timing':timing,'run':run}[sys.argv[1]]()
