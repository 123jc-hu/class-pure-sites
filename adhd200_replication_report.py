"""T17-A 的 H1、R1–R5 与既有 ABIDE 数值并列。"""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import adhd200_replication as d
import bridge_design_report as bridge
import site_pair_validation_report as cv_report
from abide_class_pure_sites_report import markdown

OUT = d.OUT
AUC = ['within_site_AUC','pooled_AUC','male_within_site_AUC','reverse_AUC']
CELL = ['k','b','direction']


def summarize(frame,keys,column):
    return frame.groupby(keys)[column].agg(mean='mean',range_low=lambda s:s.quantile(.025),
              range_high=lambda s:s.quantile(.975),repeats='count').reset_index()


def verify(config,work,pred,done,t):
    assert config['stage']=='training_complete' and (OUT/'exit_code.txt').read_text().strip()=='0'
    for field in ('source_sha256','input_sha256'):
        for path,digest in config[field].items():
            assert d.digest(d.REPO/path)==digest,path
    assert len(done)==len(work)==1314 and done.job_id.is_unique and set(done.job_id)==set(work)
    assert np.isfinite(pred.score).all() and not pred.duplicated(['job_id','FILE_ID']).any()
    meta = t.set_index('FILE_ID')
    for key,f in pred.groupby('job_id'):
        j = work[key]['job']
        assert set(f.loc[~f.reverse,'FILE_ID'])==set(j['test']) and set(f.loc[f.reverse,'FILE_ID'])==set(j['reverse'])
        for c in ['site','SEX','label']:
            np.testing.assert_array_equal(f[c],meta.loc[f.FILE_ID,c])
    fold_rows = (OUT/'inner_splits.jsonl').read_text().splitlines()
    outer = {}
    for r in map(json.loads,fold_rows):
        w = work[r['job_id']]
        j = w['job']
        tr,va = set(r['train']),set(r['validation'])
        fit = set(r.get('standardizer_fit',r['train']))
        assert tr.isdisjoint(va) and tr|va<=set(j['train']) and fit==tr
        assert (tr|va).isdisjoint(j['test']+j['reverse'])
        if r.get('role')=='estimate_outer':
            outer[r['job_id'],r['fold']] = va
    estimates = pd.read_csv(OUT/'estimation_predictions.csv',dtype={'FILE_ID':str})
    assert len(outer)==300 and not estimates.duplicated(['job_id','FILE_ID']).any()
    for key,f in estimates.groupby(['job_id','outer_fold']):
        assert set(f.FILE_ID)==outer[key]
    return dict(jobs=len(work),fits=int(done.fit_count.sum()),prediction_rows=len(pred),
                estimation_prediction_rows=len(estimates),inner_and_estimation_split_records=len(fold_rows),
                warning_count=int(done.warning_count.sum()),all_prediction_and_split_memberships_verified=True)


def natural(pred,work):
    chunks = []
    for key,f in pred.loc[pred.part.eq('H')].groupby('job_id'):
        j = work[key]['job']
        chunks.extend(f.assign(scope=scope) for scope in j['scopes'])
    full = pd.concat(chunks,ignore_index=True)
    assert not full.duplicated(['scope','method','repeat','FILE_ID']).any()
    for (scope,_,_),g in full.groupby(['scope','method','repeat']):
        assert len(g)==(865 if scope=='pittsburgh' else 769)
    results = []
    for name,scope in [('main','main'),('male','main'),('age_7_13','main'),
                       ('peking_separate','peking_separate'),('pittsburgh','pittsburgh')]:
        f = full.loc[full.scope.eq(scope)]
        if name!='main':
            f = f.loc[f.repeat.lt(20)]
        if name=='male':
            f = f.loc[f.SEX.eq(1)]
        if name=='age_7_13':
            f = f.loc[f.AGE_AT_SCAN.between(7,13)]
        pure = 'Pittsburgh' if scope=='pittsburgh' else 'WashU'
        for (evaluation,r),g in f.groupby(['method','repeat']):
            mixed = ~g.site.isin(d.EXTRA)
            positive = mixed&g.label.eq(1)
            for contrast,negative in [('target_TD',g.site.eq(pure)&g.label.eq(0)),('mixed_TD',mixed&g.label.eq(0))]:
                chosen = g.loc[positive|negative]
                sens = chosen.loc[chosen.label.eq(1),'prediction'].mean()
                spec = 1-chosen.loc[chosen.label.eq(0),'prediction'].mean()
                results.append(dict(analysis=name,target=pure,evaluation=evaluation,repeat=r,contrast=contrast,
                                    n_ADHD=int(positive.sum()),n_TD=int(negative.sum()),
                                    AUC=roc_auc_score(chosen.label,chosen.score),Acc=chosen.prediction.eq(chosen.label).mean(),
                                    Sens=sens,Spec=spec,BA=(sens+spec)/2))
    raw = pd.DataFrame(results)
    keys = ['analysis','target','evaluation','contrast','n_ADHD','n_TD']
    auc = summarize(raw,keys,'AUC').rename(columns={'mean':'mean_AUC'})
    means = raw.groupby(keys,as_index=False)[['Acc','Sens','Spec','BA']].mean()
    auc = auc.merge(means,on=keys,validate='one_to_one')
    differences = []
    for (analysis,target),f in raw.groupby(['analysis','target']):
        held = f.loc[f.evaluation.eq('LOSO')].set_index('contrast').AUC
        subject = f.loc[f.evaluation.eq('subject')].pivot(index='repeat',columns='contrast',values='AUC')
        assert len(subject)==(50 if analysis=='main' else 20)
        gaps = subject-held
        differences.extend(dict(analysis=analysis,target=target,repeat=r,target_drop=v.target_TD,
                                mixed_drop=v.mixed_TD,DiD=v.target_TD-v.mixed_TD) for r,v in gaps.iterrows())
    did = pd.DataFrame(differences)
    did_table = summarize(did.melt(id_vars=['analysis','target','repeat'],var_name='quantity',value_name='value'),
                          ['analysis','target','quantity'],'value')
    raw.to_csv(OUT/'natural_repeat_auc.csv',index=False)
    auc.to_csv(OUT/'natural_auc_table.csv',index=False)
    did.to_csv(OUT/'DiD_repeats.csv',index=False)
    did_table.to_csv(OUT/'DiD_table.csv',index=False)
    return auc,did_table


def construction(pred,done,work):
    f = pred.loc[pred.part.ne('H')].copy()
    jobs = {key:w['job'] for key,w in work.items()}
    for c in ['n_pure','n_added','n_train','T12A_job_id']:
        values = {key:j.get(c,len(j['train']) if c in ('n_train','n_pure') else 0 if c=='n_added' else j['job_id'])
                  for key,j in jobs.items()}
        f[c] = f.job_id.map(values)
    f['probability'] = f.score
    bridge.OUT = OUT
    runs,main = bridge.method_metrics(f,repeats=None,n_test_sites=3)
    truth = runs.loc[runs.method.eq('P')].assign(method='B0')
    cv_report.OUT = OUT
    estimates = cv_report.estimates(done.assign(estimate=done.estimate_pooled),truth,expected_rows=120)
    raw = []
    test = runs.loc[~runs.reverse&runs.method.isin(['M','P'])]
    for metric in bridge.METRICS:
        wide = test.pivot(index='repeat',columns='method',values=metric)
        raw.extend(dict(contrast='M-P',k=1,b=0,direction='none',metric=metric,repeat=r,difference=v)
                   for r,v in (wide.M-wide.P).items())
    for (k,b,direction),group in runs.loc[runs.b.gt(0)].groupby(CELL):
        for metric in AUC:
            reverse = metric=='reverse_AUC'
            values = group.loc[group.reverse.eq(reverse)].pivot(index='repeat',columns='method',
                                      values='pooled_AUC' if reverse else metric)
            for left,right in [('C0','S0'),('L3','C0')]:
                diff = (values[left]-values[right]).dropna()
                raw.extend(dict(contrast=f'{left}-{right}',k=k,b=b,direction=direction,
                                metric=metric,repeat=r,difference=v) for r,v in diff.items())
    raw = pd.DataFrame(raw)
    keys = ['contrast']+CELL+['metric']
    pair = summarize(raw,keys,'difference')
    pair['positive_fraction'] = raw.groupby(keys).difference.apply(lambda s:s.gt(1e-12).mean()).to_numpy()
    raw.to_csv(OUT/'paired_repeat_differences.csv',index=False)
    pair.to_csv(OUT/'paired_differences.csv',index=False)
    lr = pd.read_csv(OUT/'L_repeats.csv')
    reverse = runs.loc[runs.reverse&runs.method.isin(['P','C0'])].replace({'method':{'P':'B0'}})
    leakage = lr.merge(reverse[['method']+CELL+['repeat','pooled_AUC']],
                       on=['method']+CELL+['repeat'],validate='one_to_one')
    lt = leakage.groupby(['method']+CELL,as_index=False).agg(L=('L','mean'),
                           native_reverse_AUC=('pooled_AUC','mean'),repeats=('repeat','size'))
    assert len(lt)==5 and lt.repeats.eq(60).all()
    lt.to_csv(OUT/'L_table.csv',index=False)
    feasible = pd.read_csv(OUT/'S0_feasibility.csv').groupby(CELL,as_index=False).feasible.agg(
        available='sum',total='size')
    feasible.to_csv(OUT/'S0_available_repeats.csv',index=False)
    samples = runs.loc[runs.method.eq('S0'),CELL+['repeat']].drop_duplicates()
    matched = runs.loc[runs.method.isin(['C0','S0'])].merge(samples,on=CELL+['repeat'],validate='many_to_one')
    matched.groupby(CELL+['method','reverse'])[bridge.METRICS].mean().reset_index().to_csv(
        OUT/'C_matched_main_table.csv',index=False)
    return runs,main,pair,estimates,lt,feasible


def criteria(did,pair,estimates,lt):
    rows = []

    def add(rule,cell,point,threshold,status,details=''):
        status = ('通过' if status else '未通过') if isinstance(status,(bool,np.bool_)) else status
        rows.append(dict(criterion=rule,cell=cell,point=point,threshold=threshold,status=status,details=details))

    def delta(contrast,b,direction,metric):
        return pair.loc[pair.contrast.eq(contrast)&pair.b.eq(b)&pair.direction.eq(direction)&pair.metric.eq(metric)].iloc[0]

    h = did.loc[did.analysis.eq('main')&did.quantity.eq('DiD')].iloc[0]
    add('H1','WashU DiD',h['mean'],.10,h['mean']>=.10,f'范围 [{h.range_low:.6f}, {h.range_high:.6f}]')
    gap = delta('M-P',0,'none','within_site_AUC')['mean']
    add('R1','M−P 测试站点内 AUC',gap,.03,gap>=.03)
    add('R1','随 k 的代价与反向集趋势',np.nan,np.nan,'不可评估','仅 k=1')
    r = estimates.loc[estimates.method.eq('E_subj')&estimates.truth.eq('within_site_AUC')].iloc[0]
    add('R2','E_subj−真值',r.mean_bias,.10,r.mean_bias>=.10,'5 个外折 AUC 等权均值')
    add('R2','LSPO 相对误差',np.nan,np.nan,'不可评估','需要至少两对训练站点')
    test = delta('C0-S0',5,'twoway','within_site_AUC')
    reverse = delta('C0-S0',5,'twoway','reverse_AUC')
    status = (test['mean']>0 and reverse['mean']>0) if test.repeats>=20 else '只报告'
    add('R3','b5 双向 C0−S0',test['mean'],0.,status,
        f"反向差 {reverse['mean']:+.6f}，{test.repeats} 次配对，两项均须 > 0")
    for way in ['oneway','twoway']:
        base = lt.loc[lt.method.eq('C0')&lt.b.eq(5)&lt.direction.eq(way),'native_reverse_AUC'].item()
        diff = delta('L3-C0',5,way,'reverse_AUC')['mean']
        test = delta('L3-C0',5,way,'within_site_AUC')['mean']
        point = base+diff
        add('R4',f'b5 {way} L3 反向集 AUC',point,.50,
            point>=.50 and diff>0 and (way=='oneway' or test>=-.01),
            f'C0 反向 AUC {base:.6f}，L3−C0 测试站点内 AUC {test:+.6f}')
    add('R5','L 与反向集 AUC 的相关',np.nan,np.nan,'不可评估','只有 5 个格子，直接报告 L 与 AUC')
    return pd.DataFrame(rows)


def comparison(runs,pair,estimates,lt):
    root = d.REPO/'outputs'
    ii = root/'T14B/stage2'
    ai = pd.read_csv(root/'T12A/repeat_metrics.csv').query('not no_zero_roi and (k == 2 and n == 40 or arm == "F")')
    aii = pd.read_csv(ii/'repeat_metrics.csv').query('part == "A" and (k == 2 or method == "F")')
    rows = []

    def add(part,cell,metric,adhd,i,ii):
        rows.append(dict(part=part,cell=cell,metric=metric,ADHD200_k1=adhd,ABIDE_I_k2=i,ABIDE_II_k2=ii))

    for method in ['P','M','F']:
        for rev,metric in [(False,'within_site_AUC')]+([(True,'pooled_AUC')] if method=='P' else []):
            add('A',method,'reverse_AUC' if rev else metric,
                runs.loc[runs.method.eq(method)&runs.reverse.eq(rev),metric].mean(),
                ai.loc[ai.arm.eq(method)&ai.reverse.eq(rev),metric].mean(),
                aii.loc[aii.method.eq(method)&aii.reverse.eq(rev),metric].mean())
    for metric in ['estimate','true_AUC','mean_bias','mean_absolute_error']:
        refs = [pd.read_csv(path).query('method == "E_subj" and truth == "within_site_AUC" and k == 2')[metric].item()
                for path in (root/'T12B/performance_summary.csv',ii/'performance_summary.csv')]
        add('B','E_subj',metric,estimates.query('method == "E_subj" and truth == "within_site_AUC"')[metric].item(),*refs)
    ci = pd.read_csv(root/'T12C2/paired_differences.csv').query('k == 2 and contrast == "C0-S0"')
    cii = pd.read_csv(ii/'C_paired_differences.csv').query('k == 2 and contrast == "C0-S0"')
    di = pd.read_csv(root/'T13B/mean_results.csv').query('k == 2 and feature == "CC200"')
    dii = pd.read_csv(ii/'repeat_metrics.csv').query('k == 2 and method in ["C0","L3"]')
    for b in [2,5]:
        for way in ['oneway','twoway']:
            cell = f'b{b} {way}'
            for metric in ['within_site_AUC','reverse_AUC']:
                query = 'b == @b and direction == @way and metric == @metric'
                delta = pair.query('contrast == "C0-S0"').query(query)['mean'].item()
                add('C',cell,metric,delta,ci.query(query)['mean'].item(),cii.query(query)['mean'].item())
                rev = metric=='reverse_AUC'
                field = 'pooled_AUC' if rev else metric
                query = 'b == @b and direction == @way and method == @method'
                for method in ['C0','L3']:
                    add('D',f'{method} {cell}',metric,runs.query(query).query('reverse == @rev')[field].mean(),
                        di.query(query)[metric].item(),dii.query(query).query('reverse == @rev')[field].mean())
    ei = pd.read_csv(root/'T14A/all_L.csv').query('k == 2 and feature == "CC200" and allocation in ["none","equal"]')
    eii = pd.read_csv(ii/'L_outcomes.csv').query('k == 2')
    for r in lt.itertuples():
        query = 'method == @r.method and b == @r.b and direction == @r.direction'
        add('E',f'{r.method} b{r.b} {r.direction}','L',r.L,
            ei.query(query).mean_L.item(),eii.query(query).L.item())
        prior = ai.loc[ai.arm.eq('P')&ai.reverse,'pooled_AUC'].mean() if r.method=='B0' else di.query(query).reverse_AUC.item()
        add('E',f'{r.method} b{r.b} {r.direction}','reverse_AUC',r.native_reverse_AUC,
            prior,eii.query(query).native_reverse_AUC.item())
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'ABIDE_comparison.csv',index=False)
    return frame


def report():
    config = json.loads((OUT/'config.json').read_text())
    work = {w['task_id']:w for w in map(json.loads,(OUT/'splits.jsonl').read_text().splitlines())}
    pred = pd.read_csv(OUT/'predictions.csv',dtype={'FILE_ID':str},float_precision='round_trip')
    done = pd.read_csv(OUT/'completed.csv',float_precision='round_trip')
    verification = verify(config,work,pred,done,d.table())
    auc,did = natural(pred,work)
    runs,main,pair,estimates,lt,feasible = construction(pred,done,work)
    judge = criteria(did,pair,estimates,lt)
    judge.to_csv(OUT/'criteria.csv',index=False)
    c = comparison(runs,pair,estimates,lt)
    d.dump('verification.json',verification)
    status = {key:('未通过' if f.status.eq('未通过').any() else '通过' if f.status.eq('通过').any()
                   else f.status.iloc[0]) for key,f in judge.groupby('criterion',sort=False)}
    text = '# T17-A 第二阶段 ADHD-200\n\n'+markdown(judge)+'\n\n'
    text += '## 天然纯类站点\n\n'+markdown(auc)+'\n\n'+markdown(did)+'\n\n'
    text += '## 构造设计主表\n\n'+markdown(main.loc[main.metric.isin(AUC[:3])])+'\n\n'
    text += '## 估计偏差、可配对重复及配对差值\n\n'+markdown(estimates)+'\n\n'+markdown(feasible)+'\n\n'+markdown(pair)+'\n\n'
    text += '## 站点泄漏指数\n\n'+markdown(lt)+'\n\n'
    text += '## ABIDE 参考并列\n\n'+markdown(c)+'\n\n'
    text += '''ABIDE-I 对照为 CC200、每类 n=40、5 个测试站点。ABIDE-II 对照为 Gordon、每类 n=20、4 个测试站点。P、M 和桥接格子的 ABIDE 值取 k=2，F 取既有全训练参照。表格只作描述。C 各组只用各自可配对重复。

主分析合并 Peking，WashU 和 Pittsburgh 均未进入构造设计。H1 的男性和年龄敏感性按保存预测筛选，年龄按 [7,13] 岁。缺失性别保留在全体。Peking 分开复用相同被试折和非 Peking 的 LOSO，只重训三个 Peking LOSO。Pittsburgh 分析的 4 名 ADHD 留在拟合池，两个合并 AUC 都只用五个混合站点的 ADHD 为正类。LOSO 选参种子固定 0。

构造的站点对按 (TD, ADHD) 字典序依次循环。E_subj 按 5 个外折 AUC 的均值判读，合并 OOF 和合并测试真值只报告。L3 选参和评分使用原 T13-B 函数。C0 和 S0 用同一剩余反向集，只比较可配对的重复。L 表用 B0 和 C0 各自的原生反向集。

范围为重复 2.5%–97.5% 分位数。正差比例以 1e−12 排除浮点舍入。男性站点内 AUC 纳入两类各至少 3 名男性的测试站点。ADHD 是正类。BNU 的观察被试不在本数据中。
'''
    (OUT/'REPORT.md').write_text(text)
    d.dump('config.json',config|dict(stage='complete',decision=status))
    print('REPORT_COMPLETE '+json.dumps(status,ensure_ascii=False),flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    report()
