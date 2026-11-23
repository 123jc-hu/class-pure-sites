"""T19 revision 2 numerical fidelity, with all five paper phenotypes."""
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
import json
import subprocess
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import published_methods as task
from abide_class_pure_sites_report import markdown

OUT = task.OUT/'rev2'
BENCHMARK = {'Parisot':{'Acc':.704},'Kunda':{'Acc':.730,'AUC':.780}}
WORKERS = 5
PHENOTYPES = ['SEX','EYE_STATUS_AT_SCAN','HANDEDNESS_CATEGORY','AGE_AT_SCAN','FIQ']


def fidelity_data():
    raw = pd.read_csv(task.a.ROOT/'Phenotypic_V1_0b_preprocessed1.csv')
    order = list(map(int,(task.KUNDA/'subject_ids.txt').read_text().split()))
    kunda = raw.loc[raw.FILE_ID.ne('no_filename')].set_index('SUB_ID').loc[order].reset_index()
    assert len(kunda)==1035 and kunda.FILE_ID.is_unique
    corr = task.correlation(task.read_timeseries(kunda.FILE_ID,'cc200'))
    return kunda,corr,kunda_default_phenotypes(kunda)


def kunda_default_phenotypes(table):
    # Upstream phenotype imputation and indicator columns, including eye status per II-F.
    sex = table.SEX
    eye = table.EYE_STATUS_AT_SCAN
    hand = table.HANDEDNESS_CATEGORY.fillna('').str.strip().replace(
        {'':'R','-9999':'R','Mixed':'Ambi','L->R':'Ambi'})
    fiq = table.FIQ.replace(-9999,100).fillna(100)
    assert sex.isin([1,2]).all() and eye.isin([1,2]).all() and hand.isin(['Ambi','L','R']).all()
    return np.column_stack([sex==1,sex==2,table.AGE_AT_SCAN,fiq,hand=='Ambi',hand=='L',hand=='R',eye==1,eye==2])


def fidelity_fold(job):
    method,fold,train,test = job
    tick = perf_counter()
    table,x,phenotypes = DATA
    target = table.iloc[test]
    labels = table.DX_GROUP.eq(1).to_numpy().astype(int)
    site = table.SITE_ID.to_numpy()
    y = labels[train]
    assert set(train).isdisjoint(test)
    score,pred,params = task.kunda_ridge(task.tpe(x,train),site,phenotypes,train,y)
    truth = labels[test]
    frame = target[['FILE_ID','SITE_ID']].assign(method=method,fold=fold,label=truth,score=score[test],prediction=pred[test])
    stats = dict(method=method,fold=fold,n=len(test),Acc=accuracy_score(truth,pred[test]),
        Sens=recall_score(truth,pred[test]),Spec=recall_score(truth,pred[test],pos_label=0),
        BA=balanced_accuracy_score(truth,pred[test]),AUC=roc_auc_score(truth,score[test]),seconds=perf_counter()-tick)|params
    return frame,stats


def provenance():
    cfg = json.loads((task.OUT/'config.json').read_text())
    for name,path in [('Kunda',task.KUNDA),('Parisot_TF1_dependency',task.PARISOT_FORK)]:
        cfg['upstream'][name] = dict(path=str(path),commit=subprocess.check_output(
            ['git','-C',str(path),'rev-parse','HEAD'],text=True).strip())
    cfg['versions'] = {p:version(p) for p in ('nilearn','yacs','packaging','requests','numpy','scipy','scikit-learn','torch')}
    cfg['fidelity']['Parisot']['reused_from'] = str(task.OUT/'fidelity_metrics.csv')
    cfg['fidelity']['Kunda']['phenotypes'] = PHENOTYPES
    cfg['fidelity']['Kunda']['phenotype_imputation'] = {'FIQ':100,'hand':'R','Mixed_or_L->R':'Ambi'}
    cfg.update(revision=2,status='fidelity_running',fidelity_reference=BENCHMARK,fidelity_threshold_pp=3)
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    return cfg


def run():
    global DATA
    cfg = provenance()
    tick = perf_counter()
    print('START T19 revision 2 fidelity, 10 Kunda folds',flush=True)
    DATA = fidelity_data()
    table,x,_ = DATA
    order = StratifiedKFold(10,shuffle=True,random_state=1234)
    ids = table.FILE_ID.to_numpy()
    jobs = [('Kunda',fold,tr,te) for fold,(tr,te) in enumerate(reversed(list(order.split(x,table.DX_GROUP))))]
    splits = [dict(method=method,fold=fold,train=ids[tr].tolist(),test=ids[te].tolist()) for method,fold,tr,te in jobs]
    previous = map(json.loads,(task.OUT/'fidelity_splits.jsonl').read_text().splitlines())
    assert splits==[s for s in previous if s['method']=='Kunda']
    (OUT/'fidelity_splits.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in splits))
    frames,stats = [],[]
    with ProcessPoolExecutor(WORKERS,mp_context=get_context('fork')) as pool:
        for frame,fold in pool.map(fidelity_fold,jobs):
            frames.append(frame)
            stats.append(fold)
            print(f"FOLD_DONE {fold['method']} {fold['fold']} seconds={fold['seconds']:.1f}",flush=True)
    pred = pd.concat(frames,ignore_index=True)
    assert not pred.duplicated(['method','FILE_ID']).any()
    assert len(pred)==1035
    pred.to_csv(OUT/'fidelity_predictions.csv',index=False)
    metrics = pd.DataFrame(stats)
    metrics.to_csv(OUT/'fidelity_folds.csv',index=False)
    rows = []
    for stat in ('Acc','Sens','Spec','BA','AUC'):
        v = metrics[stat].mean()
        ref = BENCHMARK['Kunda'].get(stat)
        delta = None if ref is None else 100*(v-ref)
        rows.append(dict(method='Kunda',metric=stat,replication=v,paper=ref,difference_pp=delta,
                         within_3pp=None if ref is None else abs(delta)<=3))
    parisot = pd.read_csv(task.OUT/'fidelity_metrics.csv').query('method == "Parisot"')
    result = pd.concat([pd.DataFrame(rows),parisot],ignore_index=True)
    result.to_csv(OUT/'fidelity_metrics.csv',index=False)
    passed = bool(result.within_3pp.dropna().astype(bool).all())
    cfg.update(status='fidelity_passed' if passed else 'fidelity_failed',numeric_fidelity_run=True,
               fidelity_elapsed_seconds=perf_counter()-tick)
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (OUT/'FIDELITY_REPORT.md').write_text('# T19 修订二 数值保真\n\n'+markdown(result.fillna(''))+
        '\n\nParisot 引用修订一已通过的结果。Kunda 仅重跑原 1,035 人、原十折。'
        '按任务及论文 II-F 使用性别、眼状态、利手、年龄、FIQ。公开 HEAD 的 MIDA 分支省略眼状态，'
        '本次按任务明确列出的五项输入。TPE 的默认迭代上限与其余模型设置沿用修订一。\n')
    print(f'FINISH T19 fidelity pass={passed} seconds={perf_counter()-tick:.1f}',flush=True)


if __name__=='__main__':
    run()
