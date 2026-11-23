"""T16-C 第一阶段的训练入口。"""
from datetime import datetime
import json
from math import ceil
from multiprocessing import get_context
from multiprocessing.connection import wait
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
import deep_pair_pilot as pilot
import group_dro

OUT = ROOT/'outputs/T16C/rev1'
FORMAL = OUT/'formal'
CNN = pilot.cnn
META = pilot.META
KEY = ['job_id', 'k', 'b', 'direction', 'repeat', 'seed']
HEADS = ['E-CE', 'E-GDRO', 'E-L3']
DEEP = ['D-ERM', 'D-GDRO', 'D-DANN']
STAGES = ['encoder'] + HEADS + DEEP
LEGACY = ROOT/'outputs/T16B/rev1'
REFS = [
    ('outputs/T13B/predictions.csv', {'L3':'linear_L3'}, True),
    ('outputs/T13B/reference_predictions.csv', {'C0':'linear_C0','L3':'linear_L3'}, True),
    ('outputs/T15A/rev1/predictions.csv', {'GroupDRO':'linear_GDRO'}, True),
    ('outputs/T12D/V-CNN/C/predictions.csv', {'C0':'D-ERM_old'}, True),
    ('outputs/T16B/rev1/predictions.csv', {'A-ext-CE':'E-CE_old','A-ext-L3':'E-L3_old'}, False),
]


def job_key(j):
    return j['k'], j['direction'], j['repeat']


def plans(table):
    lookup = table.set_index('FILE_ID')
    jobs = [j for j in map(json.loads, pilot.ROSTER.read_text().splitlines())
            if j['method']=='C0' and j['b']==5 and j['k'] in (2,4)]
    assert len(jobs)==200
    for j in jobs:
        train_sites = set(lookup.loc[j['train'],'site'])
        test_sites = set(lookup.loc[j['test'],'site'])
        assert train_sites.isdisjoint(test_sites)
        assert set(lookup.loc[j['reverse'],'site']).issubset(train_sites)
        j['external'] = table.loc[~table.site.isin(train_sites | test_sites), 'FILE_ID'].tolist()
        j['unlabeled_pool'] = j['train']+j['external']
        j['confirmatory'] = not (j['k']==2 and j['direction']=='twoway' and j['repeat']<10)
        indices(table,j)
    return sorted(jobs, key=job_key)


def indices(table,j):
    ids = pd.Index(table.FILE_ID)
    train = ids.get_indexer(j['train'])
    target = ids.get_indexer(j['test']+j['reverse'])
    pool = ids.get_indexer(j['unlabeled_pool'])
    external = pool[len(train):]
    assert min(train.min(),target.min(),pool.min())>=0
    assert len(pool)==len(set(pool)) and len(target)==len(set(target))
    assert np.array_equal(train,pool[:len(train)]) and not set(pool)&set(target)
    assert set(table.loc[external,'site']).isdisjoint(table.loc[np.r_[train,target],'site'])
    return train,target,pool


def unlabeled_batch(pool,rng):
    return rng.choice(pool,size=14,replace=False)


def references(table,jobs):
    lookup = table.set_index('FILE_ID')
    by_job = {job_key(j):j for j in jobs}
    for path in [pilot.DEEP_ROSTER,LEGACY/'splits.jsonl']:
        for j in map(json.loads,path.read_text().splitlines()):
            if j['method']!='C0' or j['b']!=5 or job_key(j) not in by_job:
                continue
            source = by_job[job_key(j)]
            for field in ['train','test','reverse','bridge','seed']:
                assert j[field]==source[field], (job_key(j),field)
            if path.parent==LEGACY:
                assert source['external']==j['external']
    columns = META+['method','repeat','seed','reverse','probability','prediction']
    frames = []
    for filename, tags, has_cell in REFS:
        path = ROOT/filename
        f = pd.read_csv(path,usecols=lambda c:c in columns+['k','b','direction','feature','score'],
                        float_precision='round_trip')
        if 'feature' in f:
            f = f.loc[f.feature.eq('CC200')].drop(columns='feature')
        if not has_cell:
            f = f.assign(k=2,b=5,direction='twoway')
        f = f.loc[f.method.isin(tags)&f.b.eq(5)&f.k.isin([2,4])].copy()
        if 'score' not in f:
            f['score'] = f.probability
        if 'linear_C0' in tags:
            f.loc[f.method.eq('C0'),'score'] = f.loc[f.method.eq('C0'),'probability']
        f['method'] = f.method.map(tags)
        f['source'] = filename
        frames.append(f)
    result = pd.concat(frames,ignore_index=True)
    assert not result.duplicated(['k','direction','repeat','method','FILE_ID']).any()
    for (k,direction,repeat,method),g in result.groupby(['k','direction','repeat','method']):
        j = by_job[k,direction,repeat]
        assert set(g.seed)=={j['seed']}
        assert set(g.loc[g.reverse,'FILE_ID'])==set(j['reverse'])
        assert set(g.loc[~g.reverse,'FILE_ID'])==set(j['test'])
        for field in META[1:]:
            np.testing.assert_array_equal(g[field],lookup.loc[g.FILE_ID,field])
        assert np.isfinite(g.score).all() and g.probability.between(0,1).all()
    counts = result.groupby('method').apply(lambda f:f[['k','direction','repeat']].drop_duplicates().shape[0],
                                           include_groups=False)
    assert counts.to_dict()=={'D-ERM_old':40,'E-CE_old':10,'E-L3_old':10,'linear_C0':200,
                              'linear_GDRO':100,'linear_L3':200}
    for _,g in result.loc[result.method.eq('D-ERM_old')].groupby('direction'):
        assert set(g.k)=={2} and set(g.repeat)==set(range(20))
    return result


def fit_head(embedding, table, j, train, target, method):
    if method in ('E-CE','E-L3'):
        return pilot.fit_head(embedding,table,train,target,j['seed'],method[2:])
    y = table.label.to_numpy()
    x = embedding.astype(np.float64)
    inner = []
    for fold,(tr,va) in enumerate(group_dro.inner_folds(table,j)):
        assert not set(tr)&set(va) and set(tr)|set(va)==set(train)
        assert not (set(tr)|set(va))&set(target)
        scale = StandardScaler().fit(x[tr])
        _, groups = group_dro.site_groups(table,tr)
        w,b,_ = group_dro.optimize(scale.transform(x[tr]),y[tr],groups,'GroupDRO',.01,group_dro.GRID)
        p = expit(scale.transform(x[va])@w+b)
        auc = [roc_auc_score(y[va],p[:,i]) for i in range(len(group_dro.GRID))]
        ids = table.loc[tr,'FILE_ID'].tolist()
        inner.append(dict(fold=fold,train=ids,validation=table.loc[va,'FILE_ID'].tolist(),
                          standardizer_fit=ids,validation_auc=dict(zip(map(str,group_dro.GRID),auc))))
    means = np.mean([list(f['validation_auc'].values()) for f in inner],axis=0)
    C = group_dro.GRID[int(np.argmax(means))]
    scale = StandardScaler().fit(x[train])
    names,groups = group_dro.site_groups(table,train)
    w,b,info = group_dro.optimize(scale.transform(x[train]),y[train],groups,'GroupDRO',.01,[C])
    return dict(C=C,inner=inner,score=(scale.transform(x[target])@w+b)[:,0],
                scaler_mean=scale.mean_,group_weights=dict(zip(names,info[0]['q'])))


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, strength):
        ctx.strength = strength
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength*gradient,None


class DANN(CNN.BrainNetCNN):
    def __init__(self, n_sites):
        super().__init__()
        self.domain = nn.Sequential(pilot.xavier(nn.Linear(128,128)),nn.LeakyReLU(.33),
                                    pilot.xavier(nn.Linear(128,n_sites)))

    def forward(self,x,strength=None):
        features = pilot.Encoder.forward(self,x)
        label_logits = self.out(F.leaky_relu(self.fc2(features),.33))
        if strength is None:
            return label_logits
        reversed_features = ReverseGradient.apply(features,strength)
        return label_logits, self.domain(F.layer_norm(reversed_features,(128,)))


def sgd(model):
    weights = [p for name,p in model.named_parameters() if name.endswith('weight')]
    biases = [p for name,p in model.named_parameters() if name.endswith('bias')]
    return torch.optim.SGD([dict(params=weights,lr=1e-4,weight_decay=5e-4),
                            dict(params=biases,lr=2e-4,weight_decay=0.)],momentum=.9)


@torch.no_grad()
def site_probe(x,table,pool,model,scale,method):
    tick = perf_counter()
    model.eval()
    embedding = np.concatenate([
        pilot.Encoder.forward(model,pilot.cuda_matrices(x,scale,ids)).cpu().numpy()
        for ids in pilot.batches(pool)
    ])
    rows = table.loc[pool,['FILE_ID','site']].assign(method=method)
    rows['predicted_site'] = cross_val_predict(LogisticRegression(),embedding,rows.site,cv=5)
    rows['correct'] = rows.site.eq(rows.predicted_site)
    pilot.append('site_probe_predictions',rows)
    return dict(probe_accuracy=float(rows.correct.mean()),probe_chance=1/rows.site.nunique(),
                probe_n=len(rows),probe_seconds=perf_counter()-tick)


def deep_baseline(x,table,train,target,pool,j,method,folder):
    repeat = j['repeat']
    if method=='D-ERM':
        CNN.OUT = folder
        (folder/'V-CNN').mkdir()
        # 取原训练函数中的骨干，保持参照训练及预测路径一致。
        def capture(module,args):
            if isinstance(module,CNN.BrainNetCNN):
                models.append(module)
                handle.remove()
        models = []
        append = CNN.a.append

        def save_loss(name,rows):
            append(name,rows)
            loss = rows.iloc[0]
            if not np.isfinite(loss.weighted_train_loss):
                raise RuntimeError(f"NONFINITE_LOSS {j['job_id']} D-ERM epoch={loss.epoch} loss={loss.weighted_train_loss}")

        CNN.a.append = save_loss
        with nn.modules.module.register_module_forward_pre_hook(capture) as handle:
            p,_ = CNN.cnn_fit(x,table,train,target,repeat,dict(method=method,repeat=repeat))
        CNN.a.append = append
        log = pd.read_csv(folder/'V-CNN/epoch_losses.csv',float_precision='round_trip')
        assert len(log)==50 and np.isfinite(log.weighted_train_loss).all()
        probe = site_probe(x,table,pool,models[0],CNN.edge_scaler(x,train),method)
        return p,dict(final_loss=float(log.weighted_train_loss.iloc[-1]))|probe

    pilot.configure(repeat)
    lrng = np.random.default_rng(repeat)
    urng = np.random.default_rng(repeat)
    labels = table.label.to_numpy()
    y = torch.as_tensor(labels[train],device='cuda')
    training_ids = pool if method=='D-DANN' else train
    scale = CNN.edge_scaler(x,training_ids)
    matrix = pilot.cuda_matrices(x,scale,training_ids)
    sites,groups = group_dro.site_groups(table,training_ids)
    group = torch.as_tensor(groups,device='cuda')
    position = np.full(len(table),-1)
    position[training_ids] = np.arange(len(training_ids))
    if method=='D-DANN':
        model = DANN(len(sites)).cuda()
        weight = torch.tensor(len(train)/(2*np.bincount(labels[train],minlength=2)),
                              device='cuda',dtype=torch.float32)
    else:
        model = CNN.BrainNetCNN().cuda()
        log_q = torch.zeros(len(sites),device='cuda')
        onehot = F.one_hot(group,len(sites)).float()
    opt = sgd(model)
    total_steps = 50*ceil(len(train)/14)
    step = 0
    for epoch in range(1,51):
        tick = perf_counter()
        total, domain_accuracy = 0.,0.
        chunks = pilot.batches(lrng.permutation(len(train)))
        for ids in chunks:
            opt.zero_grad()
            if method=='D-DANN':
                u = position[unlabeled_batch(pool,urng)]
                strength = 2/(1+np.exp(-10*step/(total_steps-1)))-1
                selected = np.r_[ids,u]
                logits, domain = model(matrix[selected],strength)
                cls = F.cross_entropy(logits[:len(ids)],y[ids],weight=weight)
                dom = F.cross_entropy(domain,group[selected])
                loss = cls+dom
                domain_accuracy += float(domain.argmax(1).eq(group[selected]).sum())
            else:
                ce = F.cross_entropy(model(matrix[ids]),y[ids],reduction='none')
                observed = onehot[ids].T
                site_loss = (observed@ce)/observed.sum(1).clamp_min(1)
                log_q = F.log_softmax(log_q+.01*site_loss.detach(),dim=0)
                loss = site_loss@log_q.exp()
            total += pilot.finite_loss(loss,repeat,method,epoch)
            loss.backward()
            opt.step()
            step += 1
        final_loss = total/len(chunks)
        pilot.loss_record(repeat,method,epoch,final_loss,len(chunks),perf_counter()-tick)
        if method=='D-DANN':
            accuracy = domain_accuracy/(len(train)+14*len(chunks))
            pilot.append('domain_accuracy',[dict(repeat=repeat,epoch=epoch,accuracy=accuracy)])
    model.eval()
    with torch.no_grad():
        p = np.concatenate([model(pilot.cuda_matrices(x,scale,ids)).softmax(1)[:,1].cpu().numpy()
                            for ids in pilot.batches(target)])
    probe = site_probe(x,table,pool,model,scale,method) if method=='D-DANN' else {}
    return p,dict(final_loss=final_loss,domain_accuracy=accuracy if method=='D-DANN' else None)|probe


def save_predictions(table,j,target,*,method,values,logit=False):
    frame = table.loc[target,META].assign(**{key:j[key] for key in KEY},method=method,watch=False)
    frame['reverse'] = np.arange(len(target))>=len(j['test'])
    frame['score'] = values
    frame['probability'] = expit(values) if logit else values
    frame['prediction'] = values>=0 if logit else values>=.5
    assert frame.probability.between(0,1).all()
    pilot.append('predictions',frame)
    old_path = {'D-ERM':'D-ERM_old','E-CE':'E-CE_old','E-L3':'E-L3_old'}.get(method)
    if old_path:
        ref = pd.read_csv(OUT/'reference_predictions.csv',float_precision='round_trip')
        rows = ref.loc[ref.method.eq(old_path)&ref.k.eq(j['k'])&ref.direction.eq(j['direction'])&
                       ref.repeat.eq(j['repeat'])]
        if len(rows):
            saved = pd.read_csv(pilot.OUT/'predictions.csv',float_precision='round_trip')
            frame = saved.loc[saved.method.eq(method)]
            other = rows.set_index('FILE_ID').loc[frame.FILE_ID]
            np.testing.assert_array_equal(frame.reverse,other.reverse)
            difference = float(np.max(np.abs(frame.probability.to_numpy()-other.probability.to_numpy())))
            score_difference = float(np.max(np.abs(frame.score.to_numpy()-other.score.to_numpy())))
            equal = bool(np.array_equal(frame.prediction,other.prediction))
            pilot.append('consistency',[{key:j[key] for key in KEY}|dict(method=method,n=len(frame),
                         max_abs_probability_difference=difference,max_abs_score_difference=score_difference,
                         predictions_identical=equal)])
            assert difference==score_difference==0. and equal,(method,difference,score_difference,equal)


def run_job(job,group):
    entered = perf_counter()
    torch.set_num_threads(1)
    table,x = pilot.data()
    train,target,pool = indices(table,job)
    folder = group/job['job_id']
    folder.mkdir()
    pilot.OUT = folder

    def start():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return perf_counter()

    def record(stage,tick,**details):
        torch.cuda.synchronize()
        row = {k:job[k] for k in KEY}|dict(stage=stage,seconds=perf_counter()-tick,
              peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
              peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
              final_loss=None,C=None,domain_accuracy=None,probe_accuracy=None,probe_chance=None,
              probe_n=None,probe_seconds=None)|details
        pilot.append('completed',[row])
        print('TRAIN_DONE '+json.dumps(row),flush=True)

    tick = start()
    encoder,scale,loss = pilot.pretrain(x,pool,job['seed'],'encoder')
    embedding = np.empty((len(table),128),dtype=np.float32)
    relevant = np.r_[train,target]
    embedding[relevant] = pilot.encode(encoder,x,scale,relevant)
    del encoder
    record('encoder',tick,final_loss=loss)
    for method in HEADS:
        tick = start()
        fit = fit_head(embedding,table,job,train,target,method)
        save_predictions(table,job,target,method=method,values=fit['score'],logit=True)
        with (folder/'head_inner_splits.jsonl').open('a') as out:
            for f in fit['inner']:
                out.write(json.dumps(dict(method=method)|f)+'\n')
        if method=='E-GDRO':
            (folder/'group_weights.json').write_text(json.dumps(fit['group_weights'],indent=2))
        record(method,tick,C=fit['C'])
    for method in DEEP:
        tick = start()
        p,diag = deep_baseline(x,table,train,target,pool,job,method,folder)
        save_predictions(table,job,target,method=method,values=p)
        record(method,tick,**diag)
    (folder/'seconds.txt').write_text(str(perf_counter()-entered)+'\n')


def worker(jobs,group):
    for job in jobs:
        print(f"JOB_START {job['job_id']}",flush=True)
        run_job(job,group)


def run_workers(lanes,group):
    processes = [get_context('spawn').Process(target=worker,args=(jobs,group),name=f'lane-{i}')
                 for i,jobs in enumerate(lanes)]
    tick = perf_counter()
    for p in processes:
        p.start()
    active = {p.sentinel:p for p in processes}
    while active:
        for sentinel in wait(list(active)):
            p = active.pop(sentinel)
            p.join()
            if p.exitcode:
                for remaining in active.values():
                    if remaining.is_alive():
                        remaining.terminate()
                    remaining.join()
                raise RuntimeError(f'T16-C {p.name} exited {p.exitcode}, other worker stopped')
    return perf_counter()-tick


def write_inputs(table,jobs):
    OUT.mkdir()
    references(table,jobs).to_csv(OUT/'reference_predictions.csv',index=False)
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))
    inputs = [pilot.ROSTER,pilot.DEEP_ROSTER,LEGACY/'splits.jsonl',CNN.CC/'fisher_z.npy',
              CNN.CC/'row_order.csv',CNN.ORIGINAL/'subjects.csv',OUT/'splits.jsonl',
              OUT/'reference_predictions.csv']+[ROOT/path for path,_,_ in REFS]
    sources = [ROOT/path for path in ['deep_abide1.py','deep_pair_pilot.py','group_dro.py','abide_robustness.py',
               'within_site_pair_model.py','abide_class_pure_sites.py','tests/test_deep_abide1.py',
               'docs/tasks/T16C_p1b_deep_abide1.md','AGENTS.md']]
    config = dict(stage='timing',pilot_repeat=0,full_jobs=200,seeds=list(range(50)),pilot_workers=[1,2],
        GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,
        precision='T12-D deterministic FP32 no TF32',encoder='T16-B rev1 A-ext',
        consistency='max abs difference 0, D-ERM references repeat 0–19 per k2 direction',
        D_DANN='affine-free LayerNorm at discriminator input; remaining settings are unchanged',
        D_GDRO='unweighted per-subject CE averaged within observed groups, absent group update multiplier 1, global renormalization',
        site_probe='FC128 eval features of labeled train plus external pool, with each model’s own scaler; default sklearn LR and site-stratified unshuffled 5-fold CV; uniform chance 1/n_sites',
        hashes={str(p):pilot.digest(p) for p in inputs+sources})
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')


def timing(workers,k):
    assert sys.flags.isolated and pilot.AE_LR==1e-4
    if workers==1 and k==2:
        table = pilot.table()
        write_inputs(table,plans(table))
    jobs = [json.loads(line) for line in (OUT/'splits.jsonl').read_text().splitlines()]
    group = OUT/f'workers_{workers}'/f'k{k}'
    group.mkdir(parents=True)
    sampled = [j for j in jobs if j['repeat']==0 and j['k']==k]
    print(f'START T16-C rev1 timing workers={workers} k={k}',flush=True)
    seconds = run_workers([sampled[i::workers] for i in range(workers)],group)
    (group/'wall_seconds.txt').write_text(str(seconds)+'\n')
    print(f'TIMING_COMPLETE workers={workers} k={k} wall_seconds={seconds:.3f}',flush=True)


def project(jobs,sampled,stages,job_seconds,workers):
    encoder_times = stages.loc[stages.stage.eq('encoder')].set_index(['k','direction']).seconds
    sample_pools = {job_key(j)[:2]:len(j['unlabeled_pool']) for j in sampled}
    projected = []
    for j in jobs:
        key = job_key(j)[:2]
        ratio = ceil(len(j['unlabeled_pool'])/14)/ceil(sample_pools[key]/14)
        projected.append({field:j[field] for field in KEY}|
                         dict(seconds=float(job_seconds[key]+encoder_times.loc[key]*(ratio-1))))
    pd.DataFrame(projected).to_csv(OUT/f'projected_jobs_{workers}.csv',index=False)
    estimates = {}
    for name,skip_timed in [('full',False),('remaining',True)]:
        lanes = [0.]*workers
        for j in sorted(projected,key=lambda j:j['seconds'],reverse=True):
            if not skip_timed or j['repeat']!=0:
                lanes[int(np.argmin(lanes))] += j['seconds']
        estimates[f'projected_{name}_hours'] = max(lanes)/3600
    return estimates


def summarize_timing():
    jobs = [json.loads(line) for line in (OUT/'splits.jsonl').read_text().splitlines()]
    sampled = [j for j in jobs if j['repeat']==0]
    all_stages, all_checks, predictions, site_predictions, throughput = [], [], [], [], {}
    for workers in (1,2):
        groups = {k:OUT/f'workers_{workers}'/f'k{k}' for k in (2,4)}
        folders = {job_key(j)[:2]:groups[j['k']]/j['job_id'] for j in sampled}
        stages = pd.concat([pd.read_csv(p/'completed.csv') for p in folders.values()])
        checks = pd.concat([pd.read_csv(p/'consistency.csv') for (k,_),p in folders.items() if k==2])
        pred = pd.concat([pd.read_csv(p/'predictions.csv',float_precision='round_trip') for p in folders.values()])
        probe = pd.concat([pd.read_csv(p/'site_probe_predictions.csv') for p in folders.values()],
                          keys=list(folders))
        assert len(stages)==28 and len(checks)==4
        all_stages.append(stages.assign(workers=workers))
        all_checks.append(checks.assign(workers=workers))
        predictions.append(pred)
        site_predictions.append(probe)
        seconds = {k:float((p/'wall_seconds.txt').read_text()) for k,p in groups.items()}
        job_seconds = {key:float((p/'seconds.txt').read_text()) for key,p in folders.items()}
        peak = stages.groupby('job_id').peak_reserved_MiB.max().nlargest(workers).sum()
        throughput[workers] = dict(wall_seconds=seconds,jobs_per_hour=4*3600/sum(seconds.values()),
            sum_worker_peak_reserved_MiB=float(peak),**project(jobs,sampled,stages,job_seconds,workers))
    pd.concat(all_checks).to_csv(OUT/'consistency.csv',index=False)
    pd.concat(all_stages).to_csv(OUT/'timing_stages.csv',index=False)
    pd.testing.assert_frame_equal(*predictions,check_exact=True)
    pd.testing.assert_frame_equal(*site_predictions,check_exact=True)
    best = max(throughput,key=lambda w:throughput[w]['jobs_per_hour'])
    result = dict(workers=throughput,selected_workers=best,predictions_identical=True,
                  site_probe_predictions_identical=True,prediction_rows_compared=len(predictions[0]),
                  site_probe_rows_compared=len(site_predictions[0]))
    (OUT/'timing.json').write_text(json.dumps(result,indent=2)+'\n')
    config_path = OUT/'config.json'
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps(config|dict(stage='timing_complete_waiting_confirmation'),indent=2)+'\n')
    print('TIMING_COMPLETE '+json.dumps(result),flush=True)


def prepare_formal():
    config = json.loads((OUT/'config.json').read_text())
    assert config['stage']=='timing_complete_waiting_confirmation'
    changed_source = str(Path(__file__).resolve())
    for path,digest in config['hashes'].items():
        if path!=changed_source:
            assert pilot.digest(path)==digest,path
    jobs = {j['job_id']:j for j in map(json.loads,(OUT/'splits.jsonl').read_text().splitlines())}
    reused = {str(Path('workers_2')/f"k{j['k']}"/key):key for key,j in jobs.items() if j['repeat']==0}
    assert len(jobs)==200 and len(reused)==4
    reuse_hashes = {}
    for path,key in reused.items():
        folder = OUT/path
        done = pd.read_csv(folder/'completed.csv')
        pred = pd.read_csv(folder/'predictions.csv')
        probe = pd.read_csv(folder/'site_probe_predictions.csv')
        assert set(done.job_id)=={key} and done.stage.tolist()==STAGES
        assert len(pred)==6*(len(jobs[key]['test'])+len(jobs[key]['reverse']))
        assert len(probe)==2*len(jobs[key]['unlabeled_pool'])
        assert (folder/'seconds.txt').is_file()
        reuse_hashes.update({str(f):pilot.digest(f) for f in folder.rglob('*') if f.is_file()})
    projected = pd.read_csv(OUT/'projected_jobs_2.csv').sort_values('seconds',ascending=False,kind='stable')
    assert set(projected.job_id)==set(jobs) and len(projected)==200
    lanes = [[],[]]
    seconds = [0.,0.]
    for row in projected.loc[projected.repeat.ne(0)].itertuples():
        lane = int(np.argmin(seconds))
        lanes[lane].append(row.job_id)
        seconds[lane] += row.seconds
    assert set(lanes[0]+lanes[1])==set(jobs)-set(reused.values()) and sum(map(len,lanes))==196
    FORMAL.mkdir()
    config.update(stage='prepared',workers=2,cpu_threads_per_worker=1,reused=reused,
                  reused_hashes=reuse_hashes,lanes=lanes,lane_projected_seconds=seconds,
                  expected_new_jobs=196,expected_new_training_stages=1372,
                  prepared_at=datetime.now().astimezone().isoformat(),
                  nonfinite_GDRO_DANN_encoder='per batch',nonfinite_ERM='at original per-epoch loss write')
    config['hashes'][changed_source] = pilot.digest(changed_source)
    config['hashes'][str(OUT/'projected_jobs_2.csv')] = pilot.digest(OUT/'projected_jobs_2.csv')
    (FORMAL/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(f"FORMAL_PREPARED 4 reused, {[len(lane) for lane in lanes]} new, {max(seconds)/3600:.3f} h")


def run_formal():
    path = FORMAL/'config.json'
    config = json.loads(path.read_text())
    assert config['stage']=='prepared' and torch.cuda.get_device_name()==config['GPU']
    for file,digest in (config['hashes']|config['reused_hashes']).items():
        assert pilot.digest(file)==digest,file
    jobs = {j['job_id']:j for j in map(json.loads,(OUT/'splits.jsonl').read_text().splitlines())}
    group = FORMAL/'jobs'
    group.mkdir()
    config.update(stage='running',started_at=datetime.now().astimezone().isoformat())
    path.write_text(json.dumps(config,indent=2)+'\n')
    print('START T16-C rev1 formal, 4 reused, 196 new, 2 GPU workers',flush=True)
    seconds = run_workers([[jobs[key] for key in lane] for lane in config['lanes']],group)
    config.update(stage='training_complete',new_wall_seconds=seconds,
                  ended_at=datetime.now().astimezone().isoformat())
    path.write_text(json.dumps(config,indent=2)+'\n')
    print(f'FORMAL_TRAINING_COMPLETE 196 new jobs, wall_seconds={seconds:.3f}',flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    if sys.argv[1]=='summary':
        summarize_timing()
    elif sys.argv[1]=='prepare':
        prepare_formal()
    elif sys.argv[1]=='run':
        run_formal()
    else:
        timing(*map(int,sys.argv[1:]))
