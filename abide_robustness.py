"""T12-D：复用既有ABIDE实验，仅替换图谱、被试池或固定CNN。"""
from pathlib import Path
import hashlib,json
from time import perf_counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

import abide_class_pure_sites as a
import site_pair_validation as b
import bridge_design as c
import same_class_control as s

OUT=Path('outputs/T12D');ORIGINAL=Path('outputs/T12A')
CC=Path.home()/'datasets_cache/ABIDE_derived/T12A'
CACHE=Path.home()/'datasets_cache/ABIDE_derived/T12D'
QC=['qc_rater_1','qc_anat_rater_2','qc_func_rater_2','qc_anat_rater_3','qc_func_rater_3']


def ho_features(x):
    return a.fisher_features(np.delete(x,82,axis=1))  # 本地HO表头核对：第82列（零起点）为#3455。


def qc_table():
    table=pd.read_csv(ORIGINAL/'subjects.csv');raw=pd.read_csv(a.ROOT/'Phenotypic_V1_0b_preprocessed1.csv')
    return table.loc[table.FILE_ID.isin(raw.loc[raw[QC].ne('fail').all(axis=1),'FILE_ID'])].reset_index(drop=True)


def curve_plans(table):
    jobs=[]
    for repeat in range(50):
        for j in a.repeat_design(table,repeat,site_curve_only=True):
            if j['arm']=='FD':continue
            jobs.append(j|dict(no_zero_roi=False,job_id=f'nz0_r{repeat}_{j["arm"]}_k{j["k"]}_n{j["n"]}'))
    return jobs


def bridge_plans(table,jobs):
    work=[j for j in jobs if j['arm']=='P' and j['k'] in [2,4]]
    cross=[j for j in c.plans(table,work) if j['method']=='C0' and j['k']==2 and j['b']==5]
    controls,feasibility=s.plans(table,cross)
    return cross,controls,feasibility


def edge_scaler(x,train):return StandardScaler().fit(x[train])


def edge_matrices(x,scale,indices):
    z=scale.transform(x[indices]);matrix=np.zeros((len(z),1,200,200),np.float32)
    matrix[:,0,a.PAIRS[0],a.PAIRS[1]]=z;matrix[:,0,a.PAIRS[1],a.PAIRS[0]]=z
    return matrix


class E2E(nn.Module):
    def __init__(self,in_channels):
        super().__init__();self.row=nn.Conv2d(in_channels,32,(1,200));self.col=nn.Conv2d(in_channels,32,(200,1))

    def forward(self,x):return self.row(x)+self.col(x)


class BrainNetCNN(nn.Module):
    def __init__(self):
        super().__init__();self.e1=E2E(1);self.e2=E2E(32);self.e2n=nn.Conv2d(32,64,(1,200));self.n2g=nn.Conv2d(64,256,(200,1));self.fc1=nn.Linear(256,128);self.fc2=nn.Linear(128,30);self.out=nn.Linear(30,2)
        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                bound=np.sqrt(3/(layer.weight.numel()/layer.weight.shape[0]));nn.init.uniform_(layer.weight,-bound,bound);nn.init.zeros_(layer.bias)

    def forward(self,x):
        x=F.leaky_relu(self.e1(x),.33);x=F.leaky_relu(self.e2(x),.33);x=F.leaky_relu(self.e2n(x),.33);x=F.leaky_relu(self.n2g(x),.33).flatten(1)
        x=F.dropout(x,.5,self.training);x=F.leaky_relu(self.fc1(x),.33);x=F.dropout(x,.5,self.training);x=F.leaky_relu(self.fc2(x),.33)
        return self.out(x)


def cnn_fit(x,table,train,target,seed,context):
    tick=perf_counter();torch.set_num_threads(1);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    assert set(train).isdisjoint(target)
    scale=edge_scaler(x,train);z=torch.from_numpy(edge_matrices(x,scale,train)).cuda();y=torch.tensor(table.loc[train,'label'].to_numpy(),device='cuda',dtype=torch.long)
    counts=np.bincount(table.loc[train,'label'],minlength=2);weight=torch.tensor(len(train)/(2*counts),device='cuda',dtype=torch.float32)
    model=BrainNetCNN().cuda();weights=[p for name,p in model.named_parameters() if name.endswith('weight')];biases=[p for name,p in model.named_parameters() if name.endswith('bias')]
    optimizer=torch.optim.SGD([dict(params=weights,lr=1e-4,weight_decay=5e-4),dict(params=biases,lr=2e-4,weight_decay=0)],momentum=.9)
    rng=np.random.default_rng(seed);a.OUT=OUT/'V-CNN'
    for epoch in range(50):
        model.train();total=0.;denominator=0.
        for indices in np.array_split(rng.permutation(len(train)),np.arange(14,len(train),14)):
            optimizer.zero_grad();loss=F.cross_entropy(model(z[indices]),y[indices],weight=weight);loss.backward();optimizer.step()
            normalizer=weight[y[indices]].sum().item();total+=loss.item()*normalizer;denominator+=normalizer
        a.append('epoch_losses.csv',pd.DataFrame([context|dict(epoch=epoch+1,weighted_train_loss=total/denominator)]))
    model.eval();prob=[]
    with torch.no_grad():
        for indices in np.array_split(target,np.arange(14,len(target),14)):
            v=torch.from_numpy(edge_matrices(x,scale,indices)).cuda();prob.extend(model(v).softmax(1)[:,1].cpu().tolist())
    torch.cuda.synchronize();seconds=perf_counter()-tick
    return np.array(prob),dict(seconds=seconds,epochs=50,n_train=len(train),class_weight_TD=float(weight[0]),class_weight_ASD=float(weight[1]))


def cnn_outer(job):
    table=pd.read_csv(ORIGINAL/'subjects.csv');x=np.load(CC/'fisher_z.npy',mmap_mode='r');assert table.FILE_ID.tolist()==pd.read_csv(CC/'row_order.csv').FILE_ID.tolist()
    lookup=pd.Index(table.FILE_ID);train=lookup.get_indexer(job['train']);target=lookup.get_indexer(job['test']+job['reverse'])
    assert (train>=0).all() and (target>=0).all()
    context={key:job[key] for key in ['job_id','no_zero_roi','repeat','seed','arm','k','n']}
    probability,details=cnn_fit(x,table,train,target,job['seed'],context);pred=table.loc[target].copy()
    for key,value in context.items():pred[key]=value
    pred['reverse']=np.r_[np.zeros(len(job['test']),bool),np.ones(len(job['reverse']),bool)];pred['probability']=probability;pred['prediction']=pred.probability.ge(.5)
    a.append('predictions.csv',pred);a.append('completed.csv',pd.DataFrame([context|details]));print('TRAIN_DONE '+json.dumps(context|details),flush=True)


def pilot():
    import shutil,sys
    path=OUT/'V-CNN';path.mkdir(parents=True);table=pd.read_csv(ORIGINAL/'subjects.csv');table.to_csv(path/'subjects.csv',index=False)
    original=[json.loads(line) for line in (ORIGINAL/'splits.jsonl').read_text().splitlines()]
    jobs=[j for j in original if not j['no_zero_roi'] and j['arm']=='F' and j['repeat']<5]
    (path/'pilot_splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))
    config=dict(task='T12D',variant='V-CNN',stage='pilot_running',started_at=datetime.now().astimezone().isoformat(),pilot_repeats=list(range(5)),formal_repeats=list(range(20)),architecture='E2E32,E2E32,E2N64,N2G256,FC128,FC30,2 logits',activation='LeakyReLU .33 hidden layers',dropout='p=.5 after N2G256 and FC128,user-confirmed paper sec2.3',initialization='official Caffe Xavier FAN_IN uniform[-sqrt(3/fan_in),sqrt(3/fan_in)];bias zero',optimizer='SGD momentum .9;weight lr1e-4 decay5e-4;bias lr2e-4 decay0 per official layers.py;no scheduler',paper_deviation='only learning rate:1e-4 instead of .01,bias2e-4;user-approved task revision2026-11-20 after training-only stability diagnosis,no validation/test AUC used',batch_size=14,epochs=50,early_stopping=False,loss='CrossEntropy,N/(2*class_count) training-only weights,PyTorch weighted-mean batch reduction',seed='repeat for initialization and numpy default_rng batch shuffles;reuse seed inside estimator fits',precision='float32,no AMP,no TF32;deterministic algorithms and cuDNN;CUBLAS_WORKSPACE_CONFIG=:4096:8',standardization='train-only sklearn StandardScaler on19900 edges;mirror to symmetric200x200,diagonal0',gate='mean five test within-site AUC >=.60;otherwise stop entire task,no tuning',python=sys.executable,torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),BNU_watch_not_applicable=True,
        sources=dict(paper='https://www.cs.sfu.ca/~hamarneh/ecopy/neuroimage2017.pdf',official_layers='https://github.com/jeremykawahara/ann4brains/blob/master/ann4brains/layers.py'),
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['abide_robustness.py','tests/test_abide_robustness.py','abide_class_pure_sites.py','site_pair_validation.py','bridge_design.py','same_class_control.py','docs/tasks/T12D_robustness.md','AGENTS.md']},input_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [CC/'fisher_z.npy',CC/'row_order.csv',ORIGINAL/'subjects.csv',ORIGINAL/'splits.jsonl']})
    (path/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2));shutil.copyfile('/tmp/T12D_pytest_red.log',OUT/'pytest_red.log');shutil.copyfile('/tmp/T12D_pytest_green.log',OUT/'pytest_task.log')
    print('START T12D BrainNetCNN F pilot repeats0-4',flush=True);tick=perf_counter()
    for j in jobs:cnn_outer(j)
    pred=pd.read_csv(path/'predictions.csv',float_precision='round_trip');rows=[]
    for repeat,f in pred.groupby('repeat'):
        site=f.groupby('site').apply(lambda g:roc_auc_score(g.label,g.probability),include_groups=False);rows.append(dict(repeat=int(repeat),within_site_AUC=float(site.mean()),pooled_AUC=roc_auc_score(f.label,f.probability)))
    metrics=pd.DataFrame(rows);metrics.to_csv(path/'pilot_metrics.csv',index=False);mean=float(metrics.within_site_AUC.mean());passed=mean>=.60
    result=dict(mean_within_site_AUC=mean,threshold=.60,passed=passed,pilot_seconds=perf_counter()-tick,gate_repeats=5)
    (path/'pilot_gate.json').write_text(json.dumps(result,indent=2));config.update(stage='pilot_passed' if passed else 'stopped_gate_failed',finished_at=datetime.now().astimezone().isoformat());(path/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2));print('PILOT_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':
    import sys
    {'pilot':pilot}[sys.argv[1]]()
