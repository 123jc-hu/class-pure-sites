"""T18 第二部分，仅使用已经保存的配对结果和预测。"""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from abide_class_pure_sites_report import markdown
from supplementary_analyses import OUT, ROOT

PREVIOUS = ROOT/'outputs'
TEST = 'within_site_AUC'
REVERSE = 'reverse_AUC'


def read(source):
    return pd.read_csv(PREVIOUS/(source+'.csv'),float_precision='round_trip')


def corrected(frame):
    values = frame.difference.to_numpy()
    n = len(values)
    mean,sd = values.mean(),values.std(ddof=1)
    tr,te = frame[['n_train','n_eval']].mean()
    multiplier = 1/n+te/tr
    with np.errstate(divide='ignore',invalid='ignore'):
        effect = np.divide(mean,sd)
        stat = effect/np.sqrt(multiplier)
    if sd==0 and mean==0:
        stat,effect = 0.,0.
    return dict(J=n,n_train=tr,n_test=te,mean=mean,sd=sd,effect=effect,
                multiplier=multiplier,t=stat,p=2*t.sf(abs(stat),n-1))


def holm(series):
    order = series.sort_values(kind='stable')
    adjusted = np.minimum(1.,np.maximum.accumulate(order.to_numpy()*np.arange(len(order),0,-1)))
    return pd.Series(adjusted,index=order.index).reindex(series.index)


def pure_sizes(dataset):
    if dataset=='ABIDE-I':
        return read('T12A/completed').query('not no_zero_roi and arm == "P" and n == 40')
    source = 'T14B/stage2' if dataset=='ABIDE-II' else 'T17A/stage2'
    return read(source+'/completed').query('part == "A" and method == "P"')


def paired_tests():
    ages = {name:pure_sizes(name) for name in ('ABIDE-I','ABIDE-II','ADHD-200')}
    rows = []
    repetitions = []

    def add(family,dataset,contrast,metric,cell,frame,sizes):
        v = frame[['repeat','difference']].merge(sizes,on='repeat',validate='one_to_one')
        assert len(v)==len(frame) and v.notna().all().all()
        key = dict(family=family,dataset=dataset,contrast=contrast,metric=metric,cell=cell)
        rows.append(key|corrected(v))
        repetitions.append(v.assign(**key))

    def size(dataset,k,metric=TEST):
        f = ages[dataset].query('k == @k')
        return f[['repeat','n_train','n_reverse' if metric==REVERSE else 'n_test']].rename(
            columns={'n_reverse':'n_eval','n_test':'n_eval'})

    def bridge_rows(family,dataset,source,f,sizes_source,average_pure=False):
        n = read(sizes_source).query('method == "C0"')
        for (contrast,k,b,direction,metric),g in f.groupby(['contrast','k','b','direction','metric']):
            local = n.query('k == @k and b == @b and direction == @direction')
            local = local[['repeat','n_train','n_reverse' if metric==REVERSE else 'n_test']].rename(
                columns={'n_reverse':'n_eval','n_test':'n_eval'})
            if average_pure:
                fixed = size(dataset,k).set_index('repeat').n_train
                local = local.assign(n_train=(local.n_train+local.repeat.map(fixed))/2)
            add(family,dataset,contrast,metric,f'k{k}/b{b}/{direction}',g,local)

    i = 'ABIDE-I'
    ii = 'ABIDE-II'
    adhd = 'ADHD-200'
    f = read('T12A/paired_repeat_differences').query('not no_zero_roi and n == 40')
    for k,g in f.groupby('k'):
        add(1,i,'M-P',TEST,f'k{k}',g.rename(columns={TEST:'difference'}),size(i,k))
    for family,dataset,source,ks in [(2,i,'T12B',[2,4,6]),(6,ii,'T14B/stage2',[2,4])]:
        est = read(source+'/performance_estimates')
        for k in ks:
            g = est.loc[est.k.eq(k)]
            bias = g.loc[g.method.eq('E_subj')].rename(columns={'error_within_site_AUC':'difference'})
            if dataset==i or k==2:
                add(family,dataset,'E_subj-truth',TEST,f'k{k}',bias,size(dataset,k))
            mae = g.pivot(index='repeat',columns='method',values='abs_error_within_site_AUC')
            v = (mae.E_LSPO-mae.E_subj).rename('difference').reset_index()
            add(family,dataset,'abs(E_LSPO-truth)-abs(E_subj-truth)',TEST,f'k{k}',v,size(dataset,k))
    for contrast,source in [('C0-B0','T12C'),('C0-S0','T12C2')]:
        f = read(source+'/paired_repeat_differences').query('contrast == @contrast and b == 5 and metric in [@TEST,@REVERSE]')
        bridge_rows(3,i,source,f,'T12C/completed',average_pure=contrast=='C0-B0')
    f = read('T13B/paired_repeat_differences').query(
        'feature == "CC200" and contrast == "L3-C0" and metric in [@TEST,@REVERSE]')
    bridge_rows(4,i,'T13B',f,'T12C/completed')
    f = read('T12B/paired_repeat_differences').query('contrast in ["B1-B0","B2-B0"] and metric in [@TEST,@REVERSE]')
    for (name,k,metric),g in f.groupby(['contrast','k','metric']):
        add(5,i,{'B1-B0':'LSPO-selected-B0','B2-B0':'NAP-B0'}[name],metric,f'k{k}',g,size(i,k,metric))
    f = read('T15A/rev1/paired_repeat_differences').query(
        'design == "B0" and contrast == "GroupDRO-sklearn_ERM" and metric in [@TEST,@REVERSE]')
    for (k,metric),g in f.groupby(['k','metric']):
        add(5,i,'GroupDRO-B0',metric,f'k{k}',g,size(i,k,metric))
    f = read('T14B/stage2/paired_repeat_differences').query('contrast == "M-P" and metric == @TEST')
    comparisons = f.pivot(index='repeat',columns='k',values='difference')
    add(6,ii,'M-P',TEST,'k1',comparisons[1].rename('difference').reset_index(),size(ii,1))
    for kind,wide in [(TEST,comparisons),(REVERSE,read('T14B/stage2/repeat_metrics').query(
                     'method == "P" and reverse').pivot(index='repeat',columns='k',values='pooled_AUC'))]:
        n = pd.concat([size(ii,k,kind) for k in [1,4]]).groupby('repeat',as_index=False).mean()
        add(6,ii,'gap_k4-gap_k1' if kind==TEST else 'P_k4-P_k1',kind,'k4-k1',
            (wide[4]-wide[1]).rename('difference').reset_index(),n)
    for dataset,source,k in [(ii,'T14B/stage2',2),(adhd,'T17A/stage2',1)]:
        f = read(source+'/paired_repeat_differences')
        groups = f.query('k == @k and b == 5 and contrast == "L3-C0" and metric in [@TEST,@REVERSE]')
        if dataset==adhd:
            groups = pd.concat([groups,f.query(
                'b == 5 and direction == "twoway" and contrast == "C0-S0" and metric in [@TEST,@REVERSE]')])
        bridge_rows(6,dataset,source,groups,source+'/completed')
    table = pd.DataFrame(rows)
    table['p_Holm'] = table.groupby('family').p.transform(holm)
    return table,pd.concat(repetitions,ignore_index=True)


def draw_weights(meta,seed,n=2000):
    rng = np.random.default_rng(seed)
    weights = np.zeros((n,len(meta)),dtype=np.float64)
    for indices in meta.reset_index(drop=True).groupby(['site','label']).indices.values():
        size = len(indices)
        weights[:,indices] = rng.multinomial(size,np.full(size,1/size),size=n)
    return weights


def bootstrap(scores,target,first,second,seed):
    matrix = scores.pivot(index='FILE_ID',columns='repeat',values='score')
    meta = scores.drop_duplicates('FILE_ID').set_index('FILE_ID').loc[matrix.index]
    w = draw_weights(meta,seed)
    x = matrix.to_numpy()
    g = []
    for left,right in (first,second):
        p,q = np.flatnonzero(left(meta)),np.flatnonzero(right(meta))
        d = x[p,None,:]-x[None,q,:]
        wins = (d>0)+.5*(d==0)
        loso = matrix.columns.get_loc(-1)
        contrast = np.delete(wins,loso,axis=2).mean(2)-wins[:,:,loso]
        g.append((w[:,p]@contrast*w[:,q]).sum(1)/(len(p)*len(q)))
    value = g[0]-g[1]
    return pd.DataFrame(dict(target=target,bootstrap=np.arange(len(w)),target_drop=g[0],mixed_drop=g[1],DiD=value))


def natural_bootstraps():
    f = read('T15B/evaluation_predictions').query('scenario == "main" and evaluation in ["subject","LOSO"]')
    f = f.rename(columns={'probability':'score'})
    td = lambda g:g.label.eq(0)
    pure = lambda g:g.label.eq(1)&g.collection.eq('ABIDEII-KUL_3')
    mix = lambda g:g.label.eq(1)&g.collection.ne('ABIDEII-KUL_3')
    k = bootstrap(f,'KUL_3',(pure,td),(mix,td),[2026,0])
    f = pd.read_csv(PREVIOUS/'T17A/stage2/predictions.csv',dtype={'FILE_ID':str},float_precision='round_trip')
    jobs = [json.loads(line) for line in (PREVIOUS/'T17A/stage2/splits.jsonl').read_text().splitlines()]
    ids = [w['task_id'] for w in jobs if w['part']=='H' and 'main' in w['job']['scopes']]
    f = f.loc[f.job_id.isin(ids)]
    positives = lambda g:g.label.eq(1)
    target = lambda g:g.label.eq(0)&g.site.eq('WashU')
    control = lambda g:g.label.eq(0)&g.site.ne('WashU')
    wash = bootstrap(f,'WashU',(positives,target),(positives,control),[2026,1])
    return pd.concat([k,wash],ignore_index=True)


def inference(path=OUT/'part2'):
    path.mkdir()
    tests,repeat = paired_tests()
    tests.to_csv(path/'corrected_tests.csv',index=False)
    repeat.to_csv(path/'paired_repeats.csv',index=False)
    raw = natural_bootstraps()
    raw.to_csv(path/'natural_bootstrap_draws.csv',index=False)
    intervals = raw.groupby('target',as_index=False).DiD.agg(ci_low=lambda x:x.quantile(.025),
                                                            ci_high=lambda x:x.quantile(.975),draws='size')
    intervals['point'] = intervals.target.map({'WashU':read('T17A/stage2/DiD_table').query(
        'analysis == "main" and quantity == "DiD"')['mean'].item(),
        'KUL_3':read('T15B/addendum1/auc_summary').query('cohort == "all" and comparison == "DiD"').drop_or_DiD_mean.item()})
    intervals.to_csv(path/'natural_bootstrap_intervals.csv',index=False)
    f = read('T14A/leakage_cells').query('feature == "CC200"').reset_index(drop=True)
    values = []
    for j in f.index:
        g = f.drop(index=j)
        values.append(spearmanr(g.mean_L,g.native_reverse_AUC).statistic)
    f.assign(leave_one_cell_rho=values).to_csv(path/'L_leave_one_cell.csv',index=False)
    (path/'REPORT.md').write_text('# 第二部分 校正配对检验与被试自助法\n\n'+markdown(tests)+'\n\n'
        +markdown(intervals)+f'\n\nL 的留一格 ρ 范围 [{min(values):.6f}, {max(values):.6f}]。\n\n'
        '同一对比的重复先配对，使用双侧 t 检验及族内 Holm 校正。真值对比使用 B0 外层训练和测试人数。'
        '训练人数不同的 C0−B0 取两方均值，反向比较取配对反向集人数。'
        '差中差的 2,000 次自助法按站点和原诊断分层，同一份抽样用于所有重复及两种评估。'
        '预测固定，区间反映被试抽样变异。\n')
