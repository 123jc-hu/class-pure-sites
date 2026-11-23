"""Verify T19 predictions and apply the prespecified G1–G3 rules."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score

from abide_class_pure_sites_report import markdown
from published_methods_run import MODELS, OUT, task

KEY = ['model','design','k','repeat']
PRED = ['FILE_ID','site','label','reverse','score','prediction']


def read_predictions(path):
    labels = {'False':0,'True':1,'0':0,'1':1}
    return pd.read_csv(path,float_precision='round_trip',converters={'prediction':labels.__getitem__})


def read_lr(repeats):
    base = task.ROOT/'outputs'
    common = ['FILE_ID','site','label','reverse','probability','prediction','job_id','repeat','k']
    t12 = pd.read_csv(base/'T12A/predictions.csv',usecols=common+['no_zero_roi','arm','n'],float_precision='round_trip')
    t12 = t12.query('not no_zero_roi and arm in ["P","M"] and k in [2,4] and n==40 and repeat<@repeats')
    t12 = t12.rename(columns={'job_id':'source_job_id'}).assign(design=t12.arm.map({'P':'B0','M':'M'}))
    t13 = pd.read_csv(base/'T13B/reference_predictions.csv',usecols=common+['source_job_id','feature','method','b','direction'],
                      float_precision='round_trip')
    t13 = t13.query('feature=="CC200" and method=="C0" and b==5 and direction=="twoway" and repeat<@repeats')
    t13 = t13.assign(design='C0')
    result = pd.concat([t12,t13],ignore_index=True).rename(columns={'probability':'score'}).assign(model='LR')
    result['prediction'] = result.prediction.astype(int)
    return result[['source_job_id',*KEY,*PRED]]


def auc(frame):
    return roc_auc_score(frame.label,frame.score)


def main_metrics(frame):
    rows = []
    for key,g in frame.groupby(KEY+['reverse']):
        rows.append(dict(zip(KEY+['reverse'],key))|dict(n=len(g),Acc=accuracy_score(g.label,g.prediction),
            Sens=recall_score(g.label,g.prediction),Spec=recall_score(g.label,g.prediction,pos_label=0),
            BA=balanced_accuracy_score(g.label,g.prediction),pooled_AUC=auc(g),
            within_site_AUC=np.nan if key[-1] else np.mean([auc(s) for _,s in g.groupby('site')])))
    return pd.DataFrame(rows)


def verify(cfg, completed, pred):
    jobs = {j['job_id']:j for j in map(json.loads,(OUT/'splits.jsonl').read_text().splitlines())}
    assert len(jobs)==len(completed)==cfg['jobs'] and set(completed.job_id)==set(jobs)
    assert not pred.duplicated(['job_id','FILE_ID']).any() and set(pred.job_id)==set(jobs)
    table = task.a.subjects().set_index('FILE_ID')
    assert np.isfinite(pred.score).all() and np.array_equal(pred.label,table.loc[pred.FILE_ID,'label'])
    for key,g in pred.groupby('job_id'):
        j = jobs[key]
        assert list(g.loc[~g.reverse,'FILE_ID'])==j['test'] and list(g.loc[g.reverse,'FILE_ID'])==j['reverse']
        assert set(j['train']).isdisjoint(g.FILE_ID)
    for r in completed.query('model=="Kunda"').itertuples():
        j = jobs[r.job_id]
        n = sum(len(j[key]) for key in ('train','test','reverse'))
        assert n==r.n_nodes and json.loads(r.h_grid)==[h for h in cfg['kunda_h_grid'] if h<=n]
        assert int(r.h) in json.loads(r.h_grid)
        assert float(r.mu) in cfg['kunda_mu_grid'] and float(r.alpha) in cfg['kunda_alpha_grid']


def report():
    cfg = json.loads((OUT/'main_config.json').read_text())
    completed = pd.read_csv(OUT/'completed.csv',keep_default_na=False)
    pred = read_predictions(OUT/'predictions.csv')
    verify(cfg,completed,pred)
    n = cfg['repeats']
    main = pred.query('kind=="main"')
    lr = read_lr(n)
    cols = ['source_job_id','FILE_ID','site','label','reverse']
    assert len(lr)==len(main[main.model==MODELS[0]])
    paired = lr[cols].merge(main.loc[main.model==MODELS[0],cols],on=cols,validate='one_to_one')
    assert len(lr)==len(paired)
    raw = main_metrics(pd.concat([main,lr],ignore_index=True))
    mean = raw.groupby(['model','design','k','reverse'])[
        ['within_site_AUC','pooled_AUC','Acc','Sens','Spec','BA']].mean().reset_index()

    estimates = []
    for (model,repeat),g in pred.query('kind=="E_subj"').groupby(['model','repeat']):
        assert g.FILE_ID.is_unique and g.groupby('fold').size().to_dict()==dict.fromkeys(range(5),16)
        estimates.append(dict(model=model,repeat=repeat,E_subj=np.mean([auc(f) for _,f in g.groupby('fold')]),
                              E_subj_pooled=auc(g)))
    estimate_ref = pd.read_csv(task.ROOT/'outputs/T12B/performance_estimates.csv').query(
        'method in ["E_subj","E_subj_pooled"] and k==2 and repeat<@n')
    estimate_ref = estimate_ref.pivot(index='repeat',columns='method',values='estimate').reset_index().assign(model='LR')
    estimates = pd.concat([pd.DataFrame(estimates),estimate_ref],ignore_index=True)
    truth = raw.query('not reverse and design=="B0" and k==2')[['model','repeat','within_site_AUC']]
    assert len(estimates)==len(truth)==4*n
    estimates = estimates.merge(truth,on=['model','repeat'],validate='one_to_one')
    estimates['delta'] = estimates.E_subj-estimates.within_site_AUC

    rev = raw.loc[raw.reverse].pivot(index=['model','k','repeat'],columns='design',values='pooled_AUC')
    site = raw.loc[~raw.reverse].pivot(index=['model','k','repeat'],columns='design',values='within_site_AUC')
    assert len(rev)==len(site)==8*n
    paired_delta = pd.DataFrame({'C0_minus_B0_reverse':rev.C0-rev.B0,'M_minus_B0_site':site.M-site.B0}).reset_index()
    changes = paired_delta.groupby(['model','k']).mean(numeric_only=True).drop(columns='repeat')
    g2 = estimates.groupby('model').mean(numeric_only=True).drop(columns='repeat')
    rows = []
    for model in [*MODELS,'LR']:
        g1 = rev.loc[model].groupby('k').B0.mean()
        g3 = changes.loc[model].C0_minus_B0_reverse
        rows.append(dict(model=model,B0_reverse_k2=g1[2],B0_reverse_k4=g1[4],G1=bool(g1.lt(.5).all()),
            E_subj=g2.at[model,'E_subj'],E_subj_pooled=g2.at[model,'E_subj_pooled'],B0_site_k2=g2.at[model,'within_site_AUC'],
            E_subj_minus_B0=g2.at[model,'delta'],G2=bool(g2.at[model,'delta']>=.10),C0_minus_B0_k2=g3[2],
            C0_minus_B0_k4=g3[4],G3=bool(g3.gt(0).all())))
    checks = pd.DataFrame(rows)
    dims = completed.loc[completed.model.eq('Kunda'),
        ['job_id','repeat','design','k','kind','fold','n_nodes','h_grid','h']].copy()
    dims['excluded_h'] = dims.n_nodes.map(lambda n:json.dumps([h for h in cfg['kunda_h_grid'] if h>n]))
    dims = dims.sort_values(['repeat','k','design','fold'])
    removed = dims.groupby(['design','k','kind','excluded_h']).size().reset_index(name='fits')
    for name,frame in [('repeat_metrics',raw),('main_table',mean),('E_subj',estimates),('paired_differences',paired_delta),('criteria',checks)]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    dims.to_csv(OUT/'mida_h_by_job.csv',index=False)
    n_warnings = sum(bool(json.loads(x)) for x in completed.warnings)
    (OUT/'REPORT.md').write_text('# T19 修订二 主实验\n\n'
        f'运行重复 0–{n-1}，完成 {len(completed)} 个模型格次，复用计时拟合。'
        f'核对了全部 {len(pred):,} 条逐人预测与原名单、隔离及标签。共有 {n_warnings} 个格次记录警告。\n\n'
        'Kunda 主实验有两处相对原方法的偏离。第一，只加入年龄和性别，省略眼状态、利手和 FIQ。'
        '第二，只搜索原网格 {50, 150, 300} 中不超过进入 MIDA 的被试数 n 的 h。该规则只由人数决定。'
        'μ 和 α 网格沿用原值。MIDA 和图构建用原始采集站点编码，指标按 T12-A 的合并站点计算。'
        'E_subj 取同一名单五折 AUC 的等权均值，合并 AUC 仅作参照。LR 引用 T12-A、T12-B 和 T13-B 的既有结果。'
        'HO 连接沿用 Parisot 保真计算方式，CC200 图卷积对照读取 T12-A 缓存。\n\n'
        +markdown(checks)+'\n\n## 主表\n\n'+markdown(mean.fillna(''))+
        '\n\n## 配对差值\n\n'+markdown(changes.reset_index())+
        '\n\n## Kunda h 网格\n\n'+markdown(removed)+
        '\n\n每个格次的被试数、候选 h、剔除值及选定值见 [逐格记录](mida_h_by_job.csv)。\n')
    print(checks.to_string(index=False))


if __name__=='__main__':
    report()
