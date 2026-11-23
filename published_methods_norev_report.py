"""Verify T19 noRev against the original splits and apply N1/N2."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from abide_class_pure_sites_report import markdown
from published_methods_norev import BASE, GROUP, MODELS, OUT, jobs, task
from published_methods_report import main_metrics, read_predictions


def report():
    cfg = json.loads((OUT/'config.json').read_text())
    planned = list(map(json.loads,(OUT/'splits.jsonl').read_text().splitlines()))
    assert planned==jobs()
    by_id = {j['job_id']:j for j in planned}
    pred = read_predictions(OUT/'predictions.csv')
    done = pd.read_csv(OUT/'completed.csv',keep_default_na=False)
    assert len(by_id)==len(done)==cfg['jobs']==400 and set(done.job_id)==set(by_id)
    assert not pred.duplicated(['job_id','FILE_ID']).any() and set(pred.job_id)==set(by_id)
    assert np.isfinite(pred.score).all()
    table = task.a.subjects().set_index('FILE_ID')
    assert np.array_equal(pred.label,table.loc[pred.FILE_ID,'label'])
    meta = done.set_index('job_id')
    for key,g in pred.groupby('job_id'):
        j = by_id[key]
        assert list(g.loc[~g.reverse,'FILE_ID'])==j['test'] and list(g.loc[g.reverse,'FILE_ID'])==j['reverse']
        assert len(set(j['train']+j['test']+j['reverse']))==len(j['train'])+len(g)
        n = len(j['train'])+len(j['test'])
        assert n==meta.at[key,'n_nodes'] and len(j['reverse'])==meta.at[key,'n_reverse']
        if j['model']=='Kunda':
            grid = [h for h in cfg['kunda_h_grid'] if h<=n]
            assert json.loads(meta.at[key,'h_grid'])==grid and int(meta.at[key,'h']) in grid
            assert float(meta.at[key,'mu']) in cfg['kunda_mu_grid'] and float(meta.at[key,'alpha']) in cfg['kunda_alpha_grid']

    cols = ['job_id','FILE_ID','site','label','reverse']
    previous = pd.read_csv(BASE/'predictions.csv',usecols=cols).loc[lambda rows:rows.job_id.isin(by_id)]
    assert len(previous)==len(pred)==len(previous.merge(pred[cols],on=cols,validate='one_to_one'))
    key = GROUP+['repeat','reverse']
    raw = main_metrics(pred)
    old = pd.read_csv(BASE/'repeat_metrics.csv').query('model in @MODELS and design in ["B0","C0"]')
    joined = raw.merge(old[key+['n','within_site_AUC','pooled_AUC']],on=key,validate='one_to_one',suffixes=('','_rev2'))
    assert len(raw)==len(old)==len(joined)==800 and joined.n.eq(joined.n_rev2).all()
    joined['AUC'] = joined.within_site_AUC.where(~joined.reverse,joined.pooled_AUC)
    joined['AUC_rev2'] = joined.within_site_AUC_rev2.where(~joined.reverse,joined.pooled_AUC_rev2)
    joined['delta'] = joined.AUC-joined.AUC_rev2
    mean = joined.groupby(GROUP+['reverse'])[['AUC','AUC_rev2','delta','Acc','Sens','Spec','BA','pooled_AUC']].mean().reset_index()
    assert joined.groupby(GROUP+['reverse']).size().eq(50).all()
    rev = mean.loc[mean.reverse & mean.design.eq('B0')].set_index(['model','k']).AUC
    n1 = [rev['Kunda',2],rev['Kunda',4]]
    n2 = rev['Parisot-HO',4]
    criteria = pd.DataFrame([dict(criterion='N1',k2=n1[0],k4=n1[1],passed=max(n1)<.5),
                             dict(criterion='N2',k2=np.nan,k4=n2,passed=n2<.5)])
    kunda = done.loc[done.model.eq('Kunda')].copy()
    kunda['excluded_h'] = kunda.n_nodes.map(lambda n:json.dumps([h for h in cfg['kunda_h_grid'] if h>n]))
    for name,f in [('repeat_metrics',joined),('main_table',mean),('criteria',criteria),
                   ('mida_h_by_job',kunda[['job_id','repeat','design','k','n_nodes','h_grid','excluded_h','h']])]:
        f.to_csv(OUT/f'{name}.csv',index=False)
    warned = kunda.warnings.ne('[]').sum()
    (OUT/'REPORT.md').write_text('# T19 修订三 noRev\n\n'
        '事后提出的假设，按运行前设定的 N1 和 N2 判读。B0 与双向 C0（b=5）使用修订二相同名单，'
        '每格重复 0–49。Parisot 使用主图谱 HO。评估时反向集的真实诊断只用于计算指标。\n\n'
        '## 样本外扩展\n\n'
        '- Kunda 用原训练集拟合 TPE，只在训练与测试站点节点上估计 MIDA 核和 W。岭模型沿用原来的 K·W 表示。'
        '反向集先经冻结的 TPE 和 RBF 带宽得到交叉核，再减去拟合核列均值并乘 W，同时等价平移岭截距。'
        '这一平移等价于原 K·W 的样本外投影。年龄、性别和按拟合节点数限制 h 的规则沿用修订二。\n'
        '- Parisot 冻结训练部分拟合的 RFE、拟合图的核带宽及 GCN 权重。反向节点在训练后一次性接入，'
        '按原性别、站点与相关权重规则连边，保留原图的邻接子块，重新计算归一化和 Chebyshev 支持矩阵后前向一次。'
        '测试站点分数取接入反向集前的图。样本外路径为本任务对作者实现的扩展。\n\n'
        '## 判读\n\n'+markdown(criteria.fillna(''))+'\n\n'
        '下表 AUC 对测试站点取站点内 AUC 等权平均，对反向集取合并 AUC。差值按重复配对，为 noRev − 修订二。\n\n'
        +markdown(mean)+'\n\n'
        f'核对 {len(done)} 份拟合名单、{len(pred):,} 条逐人分数及修订二对应评估名单。'
        f'{warned} 个 Kunda 拟合记录 TPE 警告，设置沿用原方案。逐格 h 记录见 [mida_h_by_job.csv](mida_h_by_job.csv)。\n')
    print(criteria.to_string(index=False))


if __name__=='__main__':
    report()
