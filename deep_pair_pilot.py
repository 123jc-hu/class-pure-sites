"""T16-B：去噪编码器加线性头、端到端站点内配对。"""
from datetime import datetime
from hashlib import file_digest
from math import ceil
from pathlib import Path
from time import perf_counter
import json
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.nn import functional as F
import abide_robustness as cnn
import within_site_pair_model as pair

OUT = ROOT/'outputs/T16B/rev1'
ROSTER = ROOT/'outputs/T12C/splits.jsonl'
DEEP_ROSTER = ROOT/'outputs/T12D/V-CNN/C/splits.jsonl'
REFS = {
    'deep_ERM': (ROOT/'outputs/T12D/V-CNN/C/predictions.csv', 'C0'),
    'linear_C0': (ROOT/'outputs/T12C/predictions.csv', 'C0'),
    'linear_L3': (ROOT/'outputs/T13B/reference_predictions.csv', 'L3'),
}
META = ['FILE_ID', 'site', 'DX_GROUP', 'SEX', 'label']
METHODS = ['A-train-L3', 'A-train-CE', 'A-ext-L3', 'A-ext-CE', 'B']
BATCH = 14
AE_EPOCHS = 100
AE_LR = 1e-4
B_EPOCHS = 50


def table():
    return pd.read_csv(cnn.ORIGINAL/'subjects.csv')


def cell(j):
    return j['method']=='C0' and j['k']==2 and j['b']==5 and j['direction']=='twoway' and j['repeat'] in range(10)


def plans(subjects):
    lookup = subjects.set_index('FILE_ID')
    jobs = [j for j in map(json.loads, ROSTER.read_text().splitlines()) if cell(j)]
    comparator = {j['repeat']:j for j in map(json.loads, DEEP_ROSTER.read_text().splitlines()) if cell(j)}
    assert len(jobs) == len(comparator) == 10
    for j in jobs:
        old = comparator[j['repeat']]
        for field in ['train','test','reverse','bridge','seed']:
            assert j[field] == old[field], (j['repeat'], field)
        train, test = lookup.loc[j['train']], lookup.loc[j['test']]
        assert set(train.site).isdisjoint(test.site)
        assert set(lookup.loc[j['reverse'], 'site']).issubset(train.site)
        external = subjects.loc[~subjects.site.isin(set(train.site)|set(test.site)), 'FILE_ID'].tolist()
        j.update(A_train=j['train'], external=external, A_ext=j['train']+external)
        indices(subjects, j)
    return sorted(jobs, key=lambda j:j['repeat'])


def indices(subjects, job):
    lookup = pd.Index(subjects.FILE_ID)
    train = lookup.get_indexer(job['train'])
    target = lookup.get_indexer(job['test']+job['reverse'])
    external = lookup.get_indexer(job['external'])
    assert min(train.min(), target.min(), external.min()) >= 0
    assert len(train)==len(set(train))==100 and len(target)==len(set(target))
    assert not set(train)&set(target) and not set(external)&(set(train)|set(target))
    assert set(subjects.loc[external,'site']).isdisjoint(subjects.loc[np.r_[train,target],'site'])
    return train, target, external


def pair_indices(subjects, train):
    left, right = pair.pairs(subjects.loc[train,'label'].to_numpy(),
                             subjects.loc[train,'site'].to_numpy())
    assert len(left)==400 and len(np.unique(np.r_[left,right]))==len(train)
    return left, right


def references(subjects, jobs):
    lookup = subjects.set_index('FILE_ID')
    frames = []
    columns = META+['job_id','method','k','b','direction','repeat','seed','reverse','probability','prediction']
    for method, (path, tag) in REFS.items():
        extra = ['feature','score'] if tag=='L3' else []
        f = pd.read_csv(path, usecols=columns+extra, float_precision='round_trip')
        f = f.loc[f.method.eq(tag)&f.k.eq(2)&f.b.eq(5)&f.direction.eq('twoway')&f.repeat.isin(range(10))].copy()
        if tag=='L3':
            f = f.loc[f.feature.eq('CC200')].drop(columns='feature')
        else:
            f['score'] = f.probability
        assert set(f.repeat)==set(range(10))
        for j in jobs:
            q = f.loc[f.repeat.eq(j['repeat'])]
            assert q.FILE_ID.is_unique and set(q.seed)=={j['seed']}
            assert set(q.loc[q.reverse,'FILE_ID'])==set(j['reverse'])
            assert set(q.loc[~q.reverse,'FILE_ID'])==set(j['test'])
            for field in META[1:]:
                np.testing.assert_array_equal(q[field], lookup.loc[q.FILE_ID,field])
        assert np.isfinite(f.score).all() and f.probability.between(0,1).all()
        frames.append(f.assign(method=method, watch=False))
    return pd.concat(frames, ignore_index=True)


def fit_head(embedding, subjects, train, target, seed, head):
    y = subjects.label.to_numpy()
    assert not set(train)&set(target)
    inner = []
    if head=='L3':
        folds = pair.inner_folds(subjects, train, seed, 5)
        grid = pair.GRID
        validation = []
        for tr, va in folds:
            values = []
            for C in grid:
                fitted = pair.fit_model(embedding, subjects, tr, 'L3', C)
                score = pair.decision(fitted, embedding, subjects, va)
                values.append(pair.pair_auc(y[va], subjects.loc[va,'site'].to_numpy(), score))
            validation.append(values)
        mean = np.mean(validation, axis=0)
        C = grid[int(np.argmax(mean))]
        fitted = pair.fit_model(embedding, subjects, train, 'L3', C)
        score = pair.decision(fitted, embedding, subjects, target)
        scaler_mean = fitted['scale'].mean_
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        folds = [(train[tr],train[va]) for tr, va in cv.split(train, y[train])]
        fitted = cnn.a.select_model(embedding, y, train, seed)
        grid = cnn.a.GRID
        C = fitted.best_params_['lr__C']
        validation = [fitted.cv_results_[f'split{i}_test_score'] for i in range(5)]
        score = fitted.decision_function(embedding[target])
        scaler_mean = fitted.best_estimator_.named_steps['scale'].mean_
    for fold, (tr, va) in enumerate(folds):
        assert set(tr).isdisjoint(va) and set(tr)|set(va)==set(train)
        tr_ids = subjects.loc[tr,'FILE_ID'].tolist()
        inner.append(dict(fold=fold, train=tr_ids, validation=subjects.loc[va,'FILE_ID'].tolist(),
                          standardizer_fit=tr_ids, validation_auc=dict(zip(map(str,grid),map(float,validation[fold])))))
    return dict(C=C, inner=inner, score=score, scaler_mean=scaler_mean)


def configure(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def xavier(layer):
    bound = np.sqrt(3/(layer.weight.numel()/layer.weight.shape[0]))
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.zeros_(layer.bias)
    return layer


class Encoder(cnn.BrainNetCNN):
    def __init__(self):
        super().__init__()
        del self.fc2, self.out

    def forward(self, x):
        x = F.leaky_relu(self.e1(x), .33)
        x = F.leaky_relu(self.e2(x), .33)
        x = F.leaky_relu(self.e2n(x), .33)
        x = F.leaky_relu(self.n2g(x), .33).flatten(1)
        x = F.dropout(x, .5, self.training)
        return F.dropout(F.leaky_relu(self.fc1(x), .33), .5, self.training)


def batches(order):
    return np.array_split(order, np.arange(BATCH, len(order), BATCH))


def cuda_matrices(x, scale, ids):
    return torch.from_numpy(cnn.edge_matrices(x, scale, ids)).cuda()


@torch.no_grad()
def encode(model, x, scale, ids):
    model.eval()
    return np.concatenate([model(cuda_matrices(x,scale,idx)).cpu().numpy() for idx in batches(ids)])


def append(name, rows):
    with (OUT/f'{name}.csv').open('a') as dest:
        pd.DataFrame(rows).to_csv(dest,index=False,header=dest.tell()==0)


def loss_record(repeat, stage, epoch, value, steps, seconds):
    append('epoch_losses',[dict(repeat=repeat,stage=stage,epoch=epoch,loss=value,batches=steps,seconds=seconds)])


def finite_loss(loss, repeat, stage, epoch):
    value = loss.item()
    if not np.isfinite(value):
        loss_record(repeat, stage, epoch, value, 0, 0)
        raise RuntimeError(f'NONFINITE_LOSS repeat={repeat} stage={stage} epoch={epoch} loss={value}')
    return value


def pretrain(x, pool, repeat, stage):
    configure(repeat)
    scale = cnn.edge_scaler(x, pool)
    targets = torch.from_numpy(scale.transform(x[pool]).astype(np.float32)).cuda()
    row, col = (torch.as_tensor(p, device='cuda') for p in cnn.a.PAIRS)
    encoder = Encoder().cuda()
    decoder = xavier(nn.Linear(128, 19900)).cuda()
    weights = list(encoder.parameters())+list(decoder.parameters())
    opt = torch.optim.Adam(weights, lr=AE_LR, weight_decay=5e-4)
    rng = np.random.default_rng(repeat)
    mask_rng = torch.Generator(device='cuda').manual_seed(repeat)
    losses = []
    for epoch in range(1, AE_EPOCHS+1):
        tick = perf_counter()
        total = 0.
        chunks = batches(rng.permutation(len(pool)))
        for ids in chunks:
            clean = targets[ids]
            mask = torch.rand(clean.shape, device='cuda', generator=mask_rng).topk(3980, sorted=False).indices
            masked = clean.clone().scatter_(1, mask, 0.)
            matrix = clean.new_zeros((len(ids), 1, 200, 200))
            matrix[:,0,row,col] = masked
            matrix[:,0,col,row] = masked
            opt.zero_grad()
            loss = F.mse_loss(decoder(encoder(matrix)), clean)
            total += finite_loss(loss, repeat, stage, epoch)*len(ids)
            loss.backward()
            opt.step()
        losses.append(total/len(pool))
        loss_record(repeat,stage,epoch,losses[-1],len(chunks),perf_counter()-tick)
        if len(losses)>=4 and losses[-3]<losses[-2]<losses[-1] and losses[-3]>2*losses[0]:
            raise RuntimeError(f'DIVERGENCE repeat={repeat} stage={stage} epoch={epoch} recent_losses={losses[-3:]}')
    if not all(torch.isfinite(param).all().item() for param in weights):
        raise RuntimeError(f'NONFINITE_PARAMETERS repeat={repeat} stage={stage}')
    encoder.requires_grad_(False).eval()
    return encoder, scale, losses[-1]


def end_to_end(x, subjects, train, target, repeat):
    configure(repeat)
    scale = cnn.edge_scaler(x, train)
    features = cuda_matrices(x, scale, train)
    left, right = pair_indices(subjects, train)
    model = cnn.BrainNetCNN()
    model.out = xavier(nn.Linear(30, 1))
    model.cuda()
    weights = [p for name,p in model.named_parameters() if name.endswith('weight')]
    biases = [p for name,p in model.named_parameters() if name.endswith('bias')]
    opt = torch.optim.SGD([dict(params=weights,lr=1e-4,weight_decay=5e-4),
                           dict(params=biases,lr=2e-4,weight_decay=0)],momentum=.9)
    rng = np.random.default_rng(repeat)
    for epoch in range(1, B_EPOCHS+1):
        tick = perf_counter()
        total = 0.
        chunks = batches(rng.permutation(len(left)))
        for ids in chunks:
            opt.zero_grad()
            logits = model(features[np.r_[left[ids],right[ids]]]).flatten()
            case, control = logits.chunk(2)
            loss = F.softplus(control-case).mean()
            total += finite_loss(loss,repeat,'B',epoch)*len(ids)
            loss.backward()
            opt.step()
        value = total/len(left)
        loss_record(repeat,'B',epoch,value,len(chunks),perf_counter()-tick)
    return encode(model, x, scale, target).ravel(), value


def save_predictions(job, subjects, target, score, method):
    assert np.isfinite(score).all()
    rows = subjects.loc[target,META].assign(repeat=job['repeat'], seed=job['seed'], method=method,
        reverse=np.arange(len(target))>=len(job['test']), score=score, probability=expit(score),
        prediction=score>=0, watch=False)
    append('predictions',rows)


def mark(repeat, stage, seconds, loss=None, C=None):
    row = dict(repeat=repeat,stage=stage,seconds=seconds,final_loss=loss,C=C)
    append('completed',[row])
    print('TRAIN_DONE '+json.dumps(row),flush=True)


def fit_repeat(job, subjects, x):
    repeat = job['repeat']
    train, target, external = indices(subjects, job)
    for stage, pool in [('A-train',train),('A-ext',np.r_[train,external])]:
        tick = perf_counter()
        encoder, scale, loss = pretrain(x,pool,repeat,stage)
        embed = np.empty((len(subjects),128),dtype=np.float32)
        to_encode = np.r_[train,target]
        embed[to_encode] = encode(encoder,x,scale,to_encode)
        del encoder
        mark(repeat,stage,perf_counter()-tick,loss=loss)
        for head in ['L3','CE']:
            tick = perf_counter()
            method = f'{stage}-{head}'
            result = fit_head(embed,subjects,train,target,repeat,head)
            save_predictions(job,subjects,target,result['score'],method)
            with (OUT/'head_inner_splits.jsonl').open('a') as out:
                for f in result['inner']:
                    out.write(json.dumps(dict(repeat=repeat,method=method)|f)+'\n')
            mark(repeat,method,perf_counter()-tick,C=result['C'])
    tick = perf_counter()
    score, loss = end_to_end(x,subjects,train,target,repeat)
    save_predictions(job,subjects,target,score,'B')
    mark(repeat,'B',perf_counter()-tick,loss=loss)


def digest(path):
    with Path(path).open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def dump(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def data():
    subjects = table()
    assert subjects.FILE_ID.tolist()==pd.read_csv(cnn.CC/'row_order.csv').FILE_ID.tolist()
    x = np.load(cnn.CC/'fisher_z.npy',mmap_mode='r')
    return subjects,x


def timing():
    import shutil

    subjects,x = data()
    jobs = plans(subjects)
    ref = references(subjects,jobs)
    OUT.mkdir()
    ref.to_csv(OUT/'reference_predictions.csv',index=False)
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))
    files = [ROSTER,DEEP_ROSTER,cnn.CC/'fisher_z.npy',cnn.CC/'row_order.csv',
             cnn.ORIGINAL/'subjects.csv',OUT/'splits.jsonl',OUT/'reference_predictions.csv']
    files += [path for path,_ in REFS.values()]
    files += [ROOT/name for name in ['deep_pair_pilot.py','deep_pair_pilot_report.py','abide_robustness.py','within_site_pair_model.py',
              'abide_class_pure_sites.py','docs/tasks/T16B_deep_pair_pilot.md','tests/test_deep_pair_pilot.py','AGENTS.md']]
    dump('config.json',dict(revision=1, stage='timing', seeds=list(range(10)), cell='k2 b5 twoway',
         train_n=100, train_sites=4, pairs=400, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), dtype='float32, TF32 off, deterministic algorithms',
         autoencoder=dict(epochs=AE_EPOCHS,batch=14,mask_count=3980,mask='uniform without replacement per sample; symmetric',
                          loss='MSE all standardized upper-triangle edges',optimizer=f'Adam lr={AE_LR:g} decay=.0005',
                          divergence='three consecutive strictly increasing epoch means, each above twice epoch 1'),
         pair_model=dict(epochs=B_EPOCHS,pairs_per_batch=14,optimizer='T12-D SGD including separate weight and bias groups'),
         architecture='T12-D BrainNetCNN; AE truncates after activated FC128 with inherited dropout; linear decoder and scalar final head use T12-D fan-in Xavier',
         training_scalers='edges fit on actual method training IDs; A uses AE pool; heads fit on inner train or outer train',
         hashes={str(path):digest(path) for path in files}))
    shutil.copyfile('/tmp/T16B_rev1_pytest_task.log',OUT/'pytest_task.log')
    # Largest pool is repeat 0, so the timing covers its full 51 batches per epoch.
    print('START T16B timing repeat=0',flush=True)
    tick = perf_counter()
    fit_repeat(jobs[0],subjects,x)
    wall = perf_counter()-tick
    completed = pd.read_csv(OUT/'completed.csv').set_index('stage')
    ratio = sum(ceil(len(j['A_ext'])/BATCH) for j in jobs[1:])/ceil(len(jobs[0]['A_ext'])/BATCH)
    point = (wall + completed.loc['A-ext','seconds']*ratio +
             9*completed.loc[completed.index!='A-ext','seconds'].sum()+120)/3600
    result = dict(point_hours=point,with_25percent_margin_hours=1.25*point,
                  pilot_repeat=0,pilot_seconds=wall,remaining_repeats=9,
                  pilot_stages=completed.seconds.to_dict(),
                  formula='actual single GPU repeat 0; extension time scaled by AE batch counts; add 120 seconds for report')
    dump('timing.json',result)
    print('TIMING_COMPLETE '+json.dumps(result),flush=True)


def run():
    config = json.loads((OUT/'config.json').read_text())
    assert json.loads((OUT/'timing.json').read_text())['point_hours']<=3.
    for path, expected in config['hashes'].items():
        assert digest(path)==expected,path
    jobs = list(map(json.loads,(OUT/'splits.jsonl').read_text().splitlines()))
    subjects,x = data()
    config.update(stage='running',started_at=datetime.now().astimezone().isoformat())
    dump('config.json',config)
    print('START T16B repeats=1..9',flush=True)
    tick = perf_counter()
    for job in jobs[1:]:
        fit_repeat(job,subjects,x)
    config.update(stage='training_complete',formal_seconds=perf_counter()-tick,
                  finished_at=datetime.now().astimezone().isoformat())
    dump('config.json',config)
    print('ALL_TRAINING_COMPLETE repeats=10',flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    {'timing':timing,'run':run}[sys.argv[1]]()
