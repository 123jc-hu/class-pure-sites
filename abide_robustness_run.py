"""T12-D调度：旧LR函数与固定CNN共享既定名单，不增加方法。"""
from pathlib import Path
from datetime import datetime
import hashlib,json,math,shutil
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import abide_robustness as d

a,b,c,s=d.a,d.b,d.c,d.s
OUT=d.OUT
VARIANTS=['V-QC','V-HO','V-CNN']


def read(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def configure(variant):
    a.OUT=OUT/variant if variant=='V-HO' else d.ORIGINAL;a.CACHE=d.CACHE if variant=='V-HO' else d.CC
    b.OUT=OUT/variant/'B';c.OUT=OUT/variant/'C'


def tasks(variant):
    root=OUT/variant;curve=read(root/'splits.jsonl');work=[dict(variant=variant,part='A',job=j,method=j['arm']) for j in curve]
    work.extend(dict(variant=variant,part='B',job=j,method=m) for j in curve if j['arm']=='P' and j['k'] in ([2,4] if variant=='V-CNN' else [2,4,6]) for m in ['E_subj','E_LSPO'])
    work.extend(dict(variant=variant,part='C',job=j,method=j['method']) for j in read(root/'C/splits.jsonl'))
    return work


def task_id(t):
    j=t['job'];name=f'{t["method"]}_r{j["repeat"]}_k{j["k"]}' if t['part']=='B' else j['job_id']
    return t['variant']+'/'+t['part']+'/'+name


def prepare():
    from importlib.metadata import version
    original=[j for j in read(d.ORIGINAL/'splits.jsonl') if not j['no_zero_roi'] and (j['arm']=='F' or j['arm'] in ['P','M'] and j['n']==40)]
    for variant in VARIANTS:
        root=OUT/variant;(root/'B').mkdir(parents=True);(root/'C').mkdir()
        table=d.qc_table() if variant=='V-QC' else pd.read_csv(d.ORIGINAL/'subjects.csv')
        if variant=='V-HO':
            # 只新增HO数组；QC训练/预测索引仍直接映射原CC200缓存。
            d.CACHE.mkdir(parents=True,exist_ok=True);values=[];counts=[];timepoints=[];hashes={};tick=perf_counter()
            for row in table.itertuples():
                path=a.ROOT/'rois_ho'/(row.FILE_ID+'_rois_ho.1D');x=np.loadtxt(path,skiprows=1);v,n=d.ho_features(x)
                values.append(v);counts.append(n);timepoints.append(len(x));hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            np.save(d.CACHE/'fisher_z.npy',np.array(values));table[['FILE_ID']].to_csv(d.CACHE/'row_order.csv',index=False);table['n_zero_roi']=counts;table['n_timepoints']=timepoints
            write(root/'HO_extraction.json',dict(seconds=perf_counter()-tick,shape=[len(table),5995],dropped_ROI='#3455,index82',raw_sha256=hashes))
        curve=d.curve_plans(table) if variant=='V-QC' else [j for j in original if j['repeat']<(20 if variant=='V-CNN' else 50)]
        if variant=='V-QC':cross,controls,feasible=d.bridge_plans(table,curve)
        else:
            cross=[j for j in read('outputs/T12C/splits.jsonl') if j['method']=='C0' and j['k']==2 and j['b']==5 and j['repeat']<(20 if variant=='V-CNN' else 50)]
            controls=[j for j in read('outputs/T12C2/splits.jsonl') if j['k']==2 and j['b']==5 and j['repeat']<(20 if variant=='V-CNN' else 50)]
            f=pd.read_csv('outputs/T12C2/feasibility.csv');feasible=f.loc[f.k.eq(2)&f.b.eq(5)&f.repeat.lt(20 if variant=='V-CNN' else 50)].to_dict('records')
        table.to_csv(root/'subjects.csv',index=False);(root/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in curve));(root/'C/splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in cross+controls));pd.DataFrame(feasible).to_csv(root/'C/feasibility.csv',index=False)
        counts=pd.DataFrame(curve).groupby(['arm','k']).repeat.nunique().reset_index(name='available_repeats');counts.to_csv(root/'feasible_curve_repeats.csv',index=False)
        pd.DataFrame(feasible).groupby('direction').feasible.agg(available='sum',candidate='count').reset_index().to_csv(root/'C/feasible_repeats.csv',index=False)
        configured=dict(task='T12D',variant=variant,stage='prepared',seeds=list(range(20 if variant=='V-CNN' else 50)),subjects=len(table),feature_cache=str(d.CACHE if variant=='V-HO' else d.CC),CC200_recomputed=False,QC_fit_row_mapping='original1035-row table/cache;every train/test/reverse ID is from regenerated QC-only lists',BNU_watch_not_applicable=True,CPU_workers=5,CPU_process_limit=6,BLAS_threads=1,C=a.GRID if variant!='V-CNN' else None,estimated_procedure='LR:original nested4C;CNN:fixed50epoch without inner hyperparameter selection',E_subj='mean outer-fold AUC;pooled OOF report only',pairing='Q3 only same feasible C0/S0 repeats;B0 reverse restricted to current cross-class bridge removal',intervals='2.5/97.5 repeat variability ranges,not CI',positive_fraction_tolerance=1e-12,versions={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','joblib','torch']})
        if variant=='V-CNN':
            shutil.copyfile(root/'config.json',root/'pilot_config.json');configured['pilot']=json.loads((root/'pilot_config.json').read_text());configured['paper_deviation']=configured['pilot']['paper_deviation']
        write(root/'config.json',configured)
    pilot=pd.read_csv(OUT/'V-CNN/completed.csv');pilot['global_id']='V-CNN/A/'+pilot.job_id;pilot[['global_id','seconds']].to_csv(OUT/'completed.csv',index=False)
    sources=['abide_robustness.py','abide_robustness_run.py','abide_robustness_report.py','abide_class_pure_sites.py','abide_class_pure_sites_report.py','site_pair_validation.py','site_pair_validation_report.py','bridge_design.py','bridge_design_report.py','same_class_control.py','tests/test_abide_robustness.py','docs/tasks/T12D_robustness.md','AGENTS.md']
    inputs=[d.CC/'fisher_z.npy',d.CC/'row_order.csv',a.ROOT/'Phenotypic_V1_0b_preprocessed1.csv',a.ROOT/'README_local.md']+[Path('outputs')/task/name for task,names in [('T12A',['splits.jsonl','subjects.csv','repeat_metrics.csv']),('T12B',['performance_estimates.csv']),('T12C',['splits.jsonl']),('T12C2',['splits.jsonl','feasibility.csv','paired_repeat_differences.csv'])] for name in names]
    generated=[OUT/v/name for v in VARIANTS for name in ['subjects.csv','splits.jsonl','C/splits.jsonl']]+[d.CACHE/'fisher_z.npy',d.CACHE/'row_order.csv']
    write(OUT/'provenance.json',dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},generated_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in generated}))


def fit_lr(task):
    configure(task['variant']);j=task['job']
    result=a.fit_job(j) if task['part']=='A' else b.fit_job((j,task['method'])) if task['part']=='B' else (s.fit_job(j) if j['method']=='S0' else c.fit_job(j))
    return task,result


def record(global_id,seconds):
    a.OUT=OUT;a.append('completed.csv',pd.DataFrame([dict(global_id=global_id,seconds=seconds)]))


def save_lr(task,result):
    root=OUT/task['variant'];a.OUT=root;b.OUT=root/'B';c.OUT=root/'C'
    if task['part']=='A':a.save_result(result)
    elif task['part']=='B':b.save_result(result)
    else:
        result['pred']=result['pred'].rename(columns={'is_bridge':'is_added','is_same_class_added':'is_added'});c.save_result(result)
    record(task_id(task),result['done']['seconds']);print('TRAIN_DONE '+json.dumps(dict(global_id=task_id(task),seconds=result['done']['seconds'])),flush=True)


def parallel(work):
    from joblib import Parallel,delayed,parallel_config
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for task,result in Parallel(n_jobs=5,return_as='generator_unordered')(delayed(fit_lr)(t) for t in work):save_lr(task,result)


def cnn_sizes():
    table=pd.read_csv(d.ORIGINAL/'subjects.csv');lookup=pd.Index(table.FILE_ID);sizes=[]
    for t in tasks('V-CNN'):
        j=t['job']
        if t['part']=='B':
            tr=lookup.get_indexer(j['train']);folds=b.subject_folds(table,tr,j['seed']) if t['method']=='E_subj' else b.site_pairs(table,tr)
            sizes.extend((len(a),len(v)) for a,v in folds)
        else:sizes.append((len(j['train']),len(j['test'])+len(j['reverse'])))
    return sizes


def timing():
    assert json.loads((OUT/'V-CNN/pilot_gate.json').read_text())['passed']
    print('START T12D LR timing and HO extraction',flush=True);prepare();result={}
    for variant in ['V-QC','V-HO']:
        work=tasks(variant);pilot=[]
        for arm in ['P','M','F']:pilot.append(next(t for t in work if t['part']=='A' and t['method']==arm and t['job']['k']==(0 if arm=='F' else 1)))
        for k in [2,4,6]:
            for method in ['E_subj','E_LSPO']:pilot.append(next(t for t in work if t['part']=='B' and t['job']['k']==k and t['method']==method))
        for method in ['C0','S0']:
            for direction in ['oneway','twoway']:pilot.append(next(t for t in work if t['part']=='C' and t['method']==method and t['job']['direction']==direction))
        parallel(pilot);times=pd.read_csv(OUT/'completed.csv').set_index('global_id').seconds;lookup={(t['part'],t['method'],t['job']['k'] if t['part']=='B' else t['job'].get('direction','')):float(times[task_id(t)]) for t in pilot}
        seconds=0.
        for t in work:
            key=(t['part'],t['method'],t['job']['k'] if t['part']=='B' else t['job'].get('direction',''));seconds+=lookup[key]
        result[variant]=dict(point_hours=(seconds/5+60)/3600,jobs=len(work),pilot_jobs=len(pilot),worker_seconds=seconds)
    pilot=pd.read_csv(OUT/'V-CNN/completed.csv');sizes=cnn_sizes();train_steps=sum(math.ceil(n/14)*50 for n,_ in sizes);inference=sum(math.ceil(m/14) for _,m in sizes)
    units=sum(math.ceil(n/14)*50+math.ceil(m/14)/3 for n,m in zip(pilot.n_train,[len(j['test']) for j in read(OUT/'V-CNN/pilot_splits.jsonl')]))
    rate=float(pilot.seconds.sum()/units);point=rate*(train_steps+inference/3)+.2*len(sizes)+60
    result['V-CNN']=dict(point_hours=point/3600,fits=len(sizes),pilot_fits=5,train_steps=train_steps,inference_batches=inference,seconds_per_training_batch_equivalent=rate,formula='5 actual F pilots;50*ceil(train/14)+ceil(test/14)/3;+.2s per fit+60s;no additional CNN pilot')
    result['total_point_hours']=sum(result[v]['point_hours'] for v in VARIANTS);result['with_25percent_margin_hours']=result['total_point_hours']*1.25
    write(OUT/'timing.json',result);print('TIMING_COMPLETE '+json.dumps(result),flush=True)


def loss_context(job,fit_id,arm):
    return dict(job_id=fit_id,no_zero_roi=False,repeat=job['repeat'],seed=job['seed'],arm=arm,k=job['k'],n=job.get('n',40))


def cnn_estimator(task,finished):
    root=OUT/'V-CNN/B';j=task['job'];method=task['method'];table=pd.read_csv(d.ORIGINAL/'subjects.csv');x=np.load(d.CC/'fisher_z.npy',mmap_mode='r');lookup=pd.Index(table.FILE_ID);tr=lookup.get_indexer(j['train']);name=task_id(task).split('/')[-1]
    folds=b.subject_folds(table,tr,j['seed']) if method=='E_subj' else b.site_pairs(table,tr);context=dict(job_id=name,method=method,repeat=j['repeat'],seed=j['seed'],k=j['k'],n=40,T12A_job_id=j['job_id'])
    for fold,(train,target) in enumerate(folds):
        fit_id=name+f'_f{fold}';global_id='V-CNN/B/'+fit_id
        if global_id in finished:continue
        probability,details=d.cnn_fit(x,table,train,target,j['seed'],loss_context(j,fit_id,method));pred=table.loc[target].copy()
        for key,value in context.items():pred[key]=value
        pred['outer_fold']=fold;pred['probability']=probability;pred['prediction']=pred.probability.ge(.5);b.OUT=root;b.append('estimation_predictions.csv',pred)
        b.append('estimation_folds.csv',pd.DataFrame([context|dict(outer_fold=fold,AUC=roc_auc_score(pred.label,probability),n_train=len(train),n_validation=len(target))]))
        with (root/'inner_splits.jsonl').open('a') as f:f.write(json.dumps(context|dict(role='estimate_outer',fold=fold,train=table.loc[train,'FILE_ID'].tolist(),validation=table.loc[target,'FILE_ID'].tolist(),standardizer_fit=table.loc[train,'FILE_ID'].tolist()))+'\n')
        record(global_id,details['seconds']);print('TRAIN_DONE '+json.dumps(dict(global_id=global_id,**details)),flush=True)
    f=pd.read_csv(root/'estimation_predictions.csv',float_precision='round_trip');f=f.loc[f.job_id.eq(name)];folds=pd.read_csv(root/'estimation_folds.csv',float_precision='round_trip');folds=folds.loc[folds.job_id.eq(name)]
    estimate=roc_auc_score(f.label,f.probability) if method=='E_subj' else float(folds.AUC.mean());b.OUT=root;b.append('completed.csv',pd.DataFrame([context|dict(estimate=estimate)]))


def cnn_bridge(task):
    j=task['job'];root=OUT/'V-CNN/C';table=pd.read_csv(d.ORIGINAL/'subjects.csv');lookup=pd.Index(table.FILE_ID);x=np.load(d.CC/'fisher_z.npy',mmap_mode='r');train=lookup.get_indexer(j['train']);target=lookup.get_indexer(j['test']+j['reverse'])
    probability,details=d.cnn_fit(x,table,train,target,j['seed'],loss_context(j,j['job_id'],j['method']+'_'+j['direction']));pred=table.loc[target].copy();context={key:j[key] for key in c.CONTEXT}
    for key,value in context.items():pred[key]=value
    pred['reverse']=np.r_[np.zeros(len(j['test']),bool),np.ones(len(j['reverse']),bool)];pred['probability']=probability;pred['prediction']=pred.probability.ge(.5);pred['is_added']=False;c.OUT=root;c.append('predictions.csv',pred);c.append('completed.csv',pd.DataFrame([context|details]));record(task_id(task),details['seconds']);print('TRAIN_DONE '+json.dumps(dict(global_id=task_id(task),**details)),flush=True)


def run():
    print('START T12D formal',flush=True);start=perf_counter();finished=set(pd.read_csv(OUT/'completed.csv').global_id)
    write(OUT/'run_status.json',dict(stage='running',started_at=datetime.now().astimezone().isoformat()))
    for variant in ['V-QC','V-HO']:
        cfg=json.loads((OUT/variant/'config.json').read_text());cfg.update(stage='running',started_at=datetime.now().astimezone().isoformat());write(OUT/variant/'config.json',cfg)
        parallel([t for t in tasks(variant) if task_id(t) not in finished]);cfg.update(stage='training_complete',finished_at=datetime.now().astimezone().isoformat());write(OUT/variant/'config.json',cfg)
    cfg=json.loads((OUT/'V-CNN/config.json').read_text());cfg.update(stage='running',started_at=datetime.now().astimezone().isoformat());write(OUT/'V-CNN/config.json',cfg)
    for t in tasks('V-CNN'):
        if t['part']=='B':cnn_estimator(t,finished)
        elif task_id(t) not in finished:
            if t['part']=='A':
                d.cnn_outer(t['job']);q=pd.read_csv(OUT/'V-CNN/completed.csv').iloc[-1];record(task_id(t),float(q.seconds))
            else:cnn_bridge(t)
    cfg.update(stage='training_complete',finished_at=datetime.now().astimezone().isoformat());write(OUT/'V-CNN/config.json',cfg)
    write(OUT/'run_status.json',dict(stage='training_complete',seconds=perf_counter()-start,finished_at=datetime.now().astimezone().isoformat()));print('ALL_TRAINING_COMPLETE T12D',flush=True)


if __name__=='__main__':
    import sys
    {'timing':timing,'run':run}[sys.argv[1]]()
