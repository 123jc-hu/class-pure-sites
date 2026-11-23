"""T15-A：固定名单上的线性GD-ERM与按站点GroupDRO。"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import abide_class_pure_sites as a

OUT=ROOT/'outputs/T15A/rev1';GRID=a.GRID;STEPS=2000
KEY=['job_id','source_job_id','design','method','eta','k','b','direction','repeat','seed','n_pure','n_added','n_train','T12A_job_id']


def plans():
    pure=[j for j in a.read_jobs() if not j['no_zero_roi'] and j['arm']=='P' and j['n']==40 and j['k'] in [2,4]]
    bridge=[j for j in map(json.loads,(ROOT/'outputs/T12C/splits.jsonl').read_text().splitlines()) if j['method']=='C0' and j['b']==5 and j['direction']=='twoway'];work=[]
    for design,source in [('B0',pure),('bridge_b5',bridge)]:
        for j in source:
            for method,eta in [('GD_ERM',0.),('GroupDRO',.01)]+([('GroupDRO_eta0.1',.1)] if design=='B0' else []):
                work.append(j|dict(job_id=f'{design}_{method}_r{j["repeat"]}_k{j["k"]}',source_job_id=j['job_id'],design=design,method=method,eta=eta,b=0 if design=='B0' else 5,direction='none' if design=='B0' else 'twoway',n_pure=80,n_added=0 if design=='B0' else len(j['bridge']),n_train=len(j['train']),T12A_job_id=j['job_id'] if design=='B0' else j['T12A_job_id'],bridge=[] if design=='B0' else j['bridge']))
    return work


def inner_folds(table,j):
    train=pd.Index(table.FILE_ID).get_indexer(j['train']);y=table.loc[train,'label'].to_numpy()
    return [(train[tr],train[va]) for tr,va in StratifiedKFold(5,shuffle=True,random_state=j['seed']).split(train,y)]


def site_groups(table,train):return np.unique(table.loc[train,'site'].to_numpy(),return_inverse=True)


def optimize(z,y,groups,method,eta,Cs):
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits
    x=torch.from_numpy(z);target=torch.from_numpy(y.astype(float))[:,None];C=torch.tensor(Cs,dtype=torch.float64);n,p=x.shape
    w=torch.zeros((p,len(Cs)),dtype=torch.float64,requires_grad=True);b=torch.zeros(len(Cs),dtype=torch.float64,requires_grad=True)
    mapping=torch.tensor(groups[None,:]==np.arange(groups.max()+1)[:,None],dtype=torch.float64);mapping/=mapping.sum(1,keepdim=True);q=torch.full((len(mapping),len(Cs)),1/len(mapping),dtype=torch.float64)
    optimizer=torch.optim.Adam([w,b],lr=.01,foreach=False);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:1-s/STEPS)
    # 各列是独立C路径；对列求和保留每条路径的梯度，未取跨C平均。
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True);logits=x@w+b;loss=binary_cross_entropy_with_logits(logits,target.expand_as(logits),reduction='none');group_loss=mapping@loss
        if method=='GD_ERM':risk=loss.mean(0)
        else:
            # 作者loss.py:54–63：q先用脱离梯度的组CE更新，再更新模型。
            with torch.no_grad():q.mul_(torch.exp(eta*group_loss.detach()));q.div_(q.sum(0,keepdim=True))
            risk=(q*group_loss).sum(0)
        (risk+(w*w).sum(0)/(2*C*n)).sum().backward();optimizer.step();scheduler.step()
    with torch.no_grad():
        logits=x@w+b;loss=binary_cross_entropy_with_logits(logits,target.expand_as(logits),reduction='none');group_loss=mapping@loss
        info=[dict(C=float(C[i]),q_max=float(q[:,i].max()),q_entropy=float(torch.special.entr(q[:,i]).sum()),worst_group_loss=float(group_loss[:,i].max()),mean_group_loss=float(group_loss[:,i].mean()),mean_subject_loss=float(loss[:,i].mean()),q=q[:,i].tolist(),group_losses=group_loss[:,i].tolist()) for i in range(len(Cs))]
    return w.detach().numpy(),b.detach().numpy(),info


def fit_job(j):
    from time import perf_counter
    from scipy.special import expit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    tick=perf_counter();table=pd.read_csv(ROOT/'outputs/T12A/subjects.csv');assert table.FILE_ID.tolist()==pd.read_csv(a.CACHE/'row_order.csv').FILE_ID.tolist()
    x=np.load(a.CACHE/'fisher_z.npy',mmap_mode='r');lookup=pd.Index(table.FILE_ID);train=lookup.get_indexer(j['train']);target=lookup.get_indexer(j['test']+j['reverse']);context={k:j[k] for k in KEY};records=[];splits=[];values=[]
    assert (train>=0).all() and (target>=0).all() and set(train).isdisjoint(target)
    for fold,(tr,va) in enumerate(inner_folds(table,j)):
        scale=StandardScaler().fit(x[tr]);names,groups=site_groups(table,tr)
        w,b,info=optimize(scale.transform(x[tr]),table.loc[tr,'label'].to_numpy(),groups,j['method'],j['eta'],GRID);prob=expit(scale.transform(x[va])@w+b);auc=[roc_auc_score(table.loc[va,'label'],prob[:,i]) for i in range(4)];values.append(auc)
        records.extend(context|v|dict(role='selection',inner_fold=fold,AUC=auc[i],group_sites=names.tolist(),n_actual_train=len(tr),steps=STEPS) for i,v in enumerate(info));splits.append(context|dict(inner_fold=fold,train=table.loc[tr,'FILE_ID'].tolist(),validation=table.loc[va,'FILE_ID'].tolist(),standardizer_fit=table.loc[tr,'FILE_ID'].tolist(),group_site=dict(zip(table.loc[tr,'FILE_ID'],names[groups]))))
    means=np.mean(values,axis=0);C=GRID[int(np.argmax(means))];scale=StandardScaler().fit(x[train]);names,groups=site_groups(table,train);w,b,info=optimize(scale.transform(x[train]),table.loc[train,'label'].to_numpy(),groups,j['method'],j['eta'],[C]);score=(scale.transform(x[target])@w+b)[:,0];assert np.isfinite(score).all()
    if j['method']=='GD_ERM' and j['repeat']<3:info[0]['training_objective']=info[0]['mean_subject_loss']+float((w*w).sum())/(2*C*len(train))
    records.append(context|info[0]|dict(role='final',inner_fold=-1,AUC=None,group_sites=names.tolist(),n_actual_train=len(train),steps=STEPS));pred=table.loc[target,a.META].copy()
    for k,v in context.items():pred[k]=v
    pred['reverse']=np.r_[np.zeros(len(j['test']),bool),np.ones(len(j['reverse']),bool)];pred['score']=score;pred['probability']=expit(score);pred['prediction']=pred.probability.ge(.5)
    return dict(pred=pred,done=context|dict(C=C,selection_AUC=float(means.max()),seconds=perf_counter()-tick,fit_count=21,q_max=info[0]['q_max'],q_entropy=info[0]['q_entropy'],worst_group_loss=info[0]['worst_group_loss'],mean_group_loss=info[0]['mean_group_loss']),diagnostics=records,splits=splits,groups=context|dict(standardizer_fit=j['train'],group_site=dict(zip(j['train'],names[groups]))))


def append(name,frame):
    with (OUT/name).open('a') as f:frame.to_csv(f,index=False,header=f.tell()==0)


def save(result):
    append('predictions.csv',result['pred']);append('completed.csv',pd.DataFrame([result['done']]))
    with (OUT/'fit_diagnostics.jsonl').open('a') as f:
        for row in result['diagnostics']:f.write(json.dumps(row)+'\n')
    for name,rows in [('inner_splits.jsonl',result['splits']),('final_groups.jsonl',[result['groups']])]:
        with (OUT/name).open('a') as f:
            for row in rows:f.write(json.dumps(row)+'\n')


def setup_worker():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)


def parallel(work):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    remaining=iter(work[4:])
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=setup_worker) as pool:
        pending={pool.submit(fit_job,j) for j in work[:4]}
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                pending.remove(future);result=future.result();save(result);print('TRAIN_DONE '+json.dumps({k:result['done'][k] for k in ['job_id','C','seconds']}),flush=True);job=next(remaining,None)
                if job is not None:pending.add(pool.submit(fit_job,job))


def references():
    work=[j for j in plans() if j['method']=='GD_ERM'];sources={}
    for task in ['T12A','T12C','T13B']:
        files=['predictions.csv','reference_predictions.csv'] if task=='T13B' else ['predictions.csv']
        for name in files:
            path=ROOT/'outputs'/task/name;f=pd.read_csv(path,float_precision='round_trip',low_memory=False)
            if task=='T13B':f=f.loc[f.feature.eq('CC200')&f.method.eq('L3')&f.b.eq(5)&f.direction.eq('twoway')]
            sources[str(path)]={key:g for key,g in f.groupby('job_id')}
    rows=[]
    for j in work:
        methods=['sklearn_ERM']+(['L3'] if j['design']=='bridge_b5' else [])
        for method in methods:
            task='T12A' if j['design']=='B0' else 'T12C';source_job=j['source_job_id'];path=ROOT/'outputs'/task/'predictions.csv'
            if method=='L3':
                path=ROOT/'outputs/T13B'/('reference_predictions.csv' if j['k']==2 else 'predictions.csv');source_job=f'CC200_L3_r{j["repeat"]}_k{j["k"]}_b5_twoway'
            f=sources[str(path)][source_job].copy();assert set(f.FILE_ID)==set(j['test']+j['reverse']);context={k:j[k] for k in KEY};context.update(job_id=f'REF_{method}_{j["design"]}_r{j["repeat"]}_k{j["k"]}',method=method,source_job_id=source_job)
            for k,v in context.items():f[k]=v
            if method=='sklearn_ERM':f['score']=f.probability
            f['source_file']=str(path);rows.append(f[a.META+KEY+['reverse','score','probability','prediction','source_file']])
    result=pd.concat(rows,ignore_index=True);result.to_csv(OUT/'reference_predictions.csv',index=False)


def prepare():
    import hashlib,shutil
    from datetime import datetime
    from importlib.metadata import version
    OUT.mkdir();snapshots=OUT/'original_sources';snapshots.mkdir()
    for path in Path('/tmp/T15A_rev1_original_sources').iterdir():shutil.copyfile(path,snapshots/(path.name+'.snapshot'))
    shutil.copyfile('/tmp/T15A_rev1_original_outputs_sha256.json',OUT/'original_outputs_sha256.json')
    table=pd.read_csv(ROOT/'outputs/T12A/subjects.csv');table.to_csv(OUT/'subjects.csv',index=False);work=plans();assert len(work)==500
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in work));references()
    sources=['group_dro.py','group_dro_report.py','tests/test_group_dro.py','abide_class_pure_sites.py','bridge_design_report.py','docs/tasks/T15A_group_dro.md','AGENTS.md'];inputs=[a.CACHE/name for name in ['fisher_z.npy','row_order.csv']]+[ROOT/'outputs'/task/name for task,names in [('T12A',['subjects.csv','splits.jsonl','predictions.csv']),('T12C',['splits.jsonl','predictions.csv']),('T13B',['predictions.csv','reference_predictions.csv'])] for name in names]
    cfg=dict(task='T15A_rev1',stage='prepared',created_at=datetime.now().astimezone().isoformat(),seeds=list(range(50)),jobs=500,expected_fits=10500,GD_ERM_jobs=200,GroupDRO_jobs=200,sensitivity_jobs=100,feature='ABIDE-I CC20019900,T12A cache unchanged',C=GRID,steps=STEPS,optimizer=dict(name='torch Adam',lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,scheduler='LambdaLR(lambda s:1-s/2000);step after Adam;first actual lr=.01,last actual lr=.000005,after last step=0'),initial='zero coefficients and intercept;uniform q for each actual training-site group',dtype='float64',penalty='||w||^2/(2*C*N_actual_train);intercept unpenalized',group='site;bridges stay in own site;recompute groups within each inner train',eta_main=.01,eta_sensitivity=.1,generalization_adjustment=False,group_update='q <- q*exp(eta*detached_group_mean_CE);normalize;then Adam',C_batching='four independent columns;sum objectives across columns,no division across Cs;each has independent Adam state and q',inner='diagnosis StratifiedKFold5 shuffle seed=repeat;train-only StandardScaler per fold;mean validation probability AUC;first C on ties;subject CV,not site-held-out',workers=4,process_limit=6,BLAS_and_torch_threads=1,spawn=True,python_isolated=True,G0='all4cells mean(abs(per-repeat GD_ERM-sklearn_ERM test within-site AUC))<.01;full GD_ERM before remaining GroupDRO',H1='each B0 k2/4 mean GroupDRO-GD_ERM reverse AUC<.05;both required',H2='bridge_b5 paired differences report only;L3 frozen reference',pre_run_prediction='B0 pure sites allow shortcut to reduce loss in every group;expect H1,no guarantee',intervals='repeat2.5/97.5 variability percentiles,not CI;positive fraction>1e-12 only',BNU_watch_not_applicable=True,dependencies_installed=[],official_source='https://github.com/kohpangwei/group_DRO/blob/master/loss.py#L54-L63',versions={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','torch']},source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},splits_sha256=hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest(),references_sha256=hashlib.sha256((OUT/'reference_predictions.csv').read_bytes()).hexdigest())
    cfg.update(revision='rev1:linear decay 2000steps;all500 retrained,no old model predictions reused',previous_result_dir=str(OUT.parent),previous_output_hashes=json.loads((OUT/'original_outputs_sha256.json').read_text()),original_source_snapshots_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshots.iterdir()})
    assert (OUT/'splits.jsonl').read_bytes()==(OUT.parent/'splits.jsonl').read_bytes()
    assert (OUT/'reference_predictions.csv').read_bytes()==(OUT.parent/'reference_predictions.csv').read_bytes()
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    for source,name in [('/tmp/T15A_rev1_pytest_red.log','pytest_red.log'),('/tmp/T15A_rev1_pytest_task.log','pytest_task.log'),('/tmp/T15A_rev1_pytest_all.log','pytest_all.log')]:shutil.copyfile(source,OUT/name)


def timing():
    from time import perf_counter
    prepare();tick=perf_counter();print('START T15A timing',flush=True);work=plans();selected=[j for j in work if j['repeat'] in [0,16]];parallel(selected);done=pd.read_csv(OUT/'completed.csv');rates=done.groupby(['design','method','k']).seconds.mean();remaining=[j for j in work if j['job_id'] not in set(done.job_id)];point=perf_counter()-tick+sum(rates[j['design'],j['method'],j['k']] for j in remaining)/4+180
    value=dict(point_hours=point/3600,with_25percent_margin_hours=point*1.25/3600,pilot_seconds=perf_counter()-tick,pilot_jobs=len(selected),remaining_jobs=len(remaining),pilot_reused=True,workers=4,formula='actual end-to-end repeats0+16 per design/method/k;remaining sum/4+pilot wall+180s verification reserve');(OUT/'timing.json').write_text(json.dumps(value,indent=2));cfg=json.loads((OUT/'config.json').read_text());cfg['stage']='timed';(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('TIMING_COMPLETE '+json.dumps(value),flush=True)


def run():
    from time import perf_counter
    from datetime import datetime
    from group_dro_report import g0
    cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='running',started_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));tick=perf_counter();work=plans();done=set(pd.read_csv(OUT/'completed.csv').job_id);print('START T15A GD_ERM gate phase',flush=True)
    parallel([j for j in work if j['method']=='GD_ERM' and j['job_id'] not in done])
    if not g0():
        cfg.update(stage='G0_failed',formal_seconds=perf_counter()-tick,finished_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('G0_STOP no GroupDRO interpretation',flush=True);return
    print('G0_PASSED all4cells;START remaining GroupDRO',flush=True);done=set(pd.read_csv(OUT/'completed.csv').job_id);parallel([j for j in work if j['job_id'] not in done]);cfg.update(stage='training_complete',formal_seconds=perf_counter()-tick,finished_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('ALL_TRAINING_COMPLETE jobs=500',flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    {'timing':timing,'run':run}[sys.argv[1]]()
