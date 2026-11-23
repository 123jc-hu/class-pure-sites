"""T12-A：50次重复的变异范围、配对混杂代价及固定判读。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,recall_score,roc_auc_score

import abide_class_pure_sites as a

OUT=a.OUT;GROUP=['no_zero_roi','arm','k','n','reverse']
METRICS=['within_site_AUC','pooled_AUC','BA','Acc','Sens','Spec','male_within_site_AUC']


def markdown(table):
    cols=list(table.columns);lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in table.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def summarize(pred,repeats=50):
    rows=[];sites=[]
    for job_id,f in pred.groupby('job_id',sort=False):
        context={k:f[k].iloc[0] for k in ['job_id','no_zero_roi','repeat','seed','arm','k','n']}
        for reverse,g in f.groupby('reverse'):
            row=context|dict(reverse=reverse,n_subjects=len(g),pooled_AUC=roc_auc_score(g.label,g.probability),BA=balanced_accuracy_score(g.label,g.prediction),Acc=accuracy_score(g.label,g.prediction),Sens=recall_score(g.label,g.prediction),Spec=recall_score(g.label,g.prediction,pos_label=0))
            if not reverse:
                all_auc=[];male_auc=[];skipped=[]
                for site,s in g.groupby('site'):
                    value=roc_auc_score(s.label,s.probability);all_auc.append(value)
                    m=s.loc[s.SEX.eq(1)];n1=int(m.label.eq(1).sum());n0=int(m.label.eq(0).sum());skip=min(n1,n0)<3
                    if skip:skipped.append(site)
                    else:male_auc.append(roc_auc_score(m.label,m.probability))
                    sites.append(context|dict(site=site,within_site_AUC=value,n_ASD=int(s.label.eq(1).sum()),n_TD=int(s.label.eq(0).sum()),male_within_site_AUC=np.nan if skip else male_auc[-1],male_ASD=n1,male_TD=n0,male_skipped=skip))
                assert len(all_auc)==5
                row.update(within_site_AUC=np.mean(all_auc),male_within_site_AUC=np.mean(male_auc),n_male_sites=len(male_auc),n_male_skipped=len(skipped),male_skipped_sites=';'.join(skipped))
            rows.append(row)
    runs=pd.DataFrame(rows);runs.to_csv(OUT/'repeat_metrics.csv',index=False);pd.DataFrame(sites).to_csv(OUT/'site_metrics.csv',index=False)
    long=runs.melt(id_vars=GROUP+['repeat'],value_vars=METRICS,var_name='metric',value_name='value').dropna(subset=['value'])
    summary=long.groupby(GROUP+['metric']).value.agg(mean='mean',range_low=lambda v:v.quantile(.025),range_high=lambda v:v.quantile(.975),std='std',repeats='count').reset_index()
    if repeats is not None:assert summary.repeats.eq(repeats).all()
    summary.to_csv(OUT/'main_table.csv',index=False)
    pairs=runs.loc[runs.arm.isin(['P','M'])&~runs.reverse].pivot(index=['no_zero_roi','repeat','k','n'],columns='arm',values=METRICS)
    paired=pd.DataFrame({metric:pairs[metric,'M']-pairs[metric,'P'] for metric in METRICS}).reset_index();paired.to_csv(OUT/'paired_repeat_differences.csv',index=False)
    long=paired.melt(id_vars=['no_zero_roi','repeat','k','n'],var_name='metric',value_name='value')
    differences=long.groupby(['no_zero_roi','k','n','metric']).value.agg(mean='mean',range_low=lambda v:v.quantile(.025),range_high=lambda v:v.quantile(.975),std='std',repeats='count').reset_index()
    differences.to_csv(OUT/'paired_differences.csv',index=False)
    return summary,differences


def decision(summary,differences):
    rows=[]
    for sens in [False,True]:
        f=summary.loc[summary.no_zero_roi.eq(sens)&~summary.reverse&summary.metric.eq('within_site_AUC')]
        d=differences.loc[differences.no_zero_roi.eq(sens)&differences.metric.eq('within_site_AUC')]
        def score(arm,k,n):return float(f.loc[f.arm.eq(arm)&f.k.eq(k)&f.n.eq(n),'mean'].iloc[0])
        def cost(k,n):return float(d.loc[d.k.eq(k)&d.n.eq(n),'mean'].iloc[0])
        p1=cost(1,40)>=.05;gain=score('P',6,40)-score('P',1,40);drop=cost(1,40)-cost(6,40);p2=gain>0 and drop>0
        result='推论得到支持，后续方法设计须另行下达' if p1 and p2 else '在本ABIDE连接平台上纯类站点危害不大，回报讨论' if not p1 else '危害存在，但增加站点不能缓解，推论不成立，回报讨论'
        row=dict(no_zero_roi=sens,P1_cost_k1=cost(1,40),P1=p1,P2_P_gain_k6_minus_k1=gain,P2_cost_drop_k1_minus_k6=drop,P2=p2,decision=result)
        if not sens:
            gp=score('P',2,80)-score('P',2,20);gm=score('M',2,80)-score('M',2,20);row.update(P3_P_gain_n80_minus_n20=gp,P3_M_gain_n80_minus_n20=gm,P3=gp<gm)
        rows.append(row)
    check=pd.DataFrame(rows);check.to_csv(OUT/'criteria.csv',index=False);return check


def figure(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    f=summary.loc[~summary.no_zero_roi&~summary.reverse&summary.metric.eq('within_site_AUC')]
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained');values=[]
    for ax,kind,x in [(axes[0],'sites','k'),(axes[1],'subjects','n')]:
        g=f.loc[f.n.eq(40)] if kind=='sites' else f.loc[f.k.eq(2)]
        for arm,color,marker in [('P','#D55E00','o'),('M','#0072B2','s')]:
            s=g.loc[g.arm.eq(arm)].sort_values(x);ax.plot(s[x],s['mean'],color=color,marker=marker,label='Pure sites (P)' if arm=='P' else 'Mixed sites (M)');ax.fill_between(s[x],s.range_low,s.range_high,color=color,alpha=.15)
            values.extend(s.assign(panel=kind).to_dict('records'))
        ax.set(xlabel='Sites per diagnosis (k)' if kind=='sites' else 'Subjects per diagnosis (n)',ylabel='Mean within-site AUC',ylim=(0,1),xticks=[1,2,4,6] if kind=='sites' else [20,40,80]);ax.axhline(.5,color='.6',ls=':',lw=.8);ax.spines[['top','right']].set_visible(False)
    ref=f.loc[f.arm.eq('F')].iloc[0];axes[0].axhline(ref['mean'],color='#009E73',ls='--',label='Full training (F)');axes[0].axhspan(ref.range_low,ref.range_high,color='#009E73',alpha=.08);axes[0].legend(frameon=False,fontsize=8)
    values.append(ref.to_dict()|dict(panel='sites_F_reference'))
    for ext in ['png','pdf','svg']:fig.savefig(OUT/('class_pure_sites.'+ext),dpi=200)
    plt.close(fig);pd.DataFrame(values).to_csv(OUT/'figure_values.csv',index=False)
    (OUT/'figure_caption.md').write_text('主分析：50个种子0–49。左n=40，右k=2，共享k=2/n=40条件。每重复先在5个保留站点内分别算AUC后等权平均；曲线为50次均值，色带为重复指标2.5%–97.5%分位范围。F使用同一重复12个训练站点全体，其水平线/浅带为50次均值/分位范围。重复共用被试，色带只是变异范围，不是独立样本置信区间；不是bootstrap。P/M同一组站点及每类人数；纵轴0–1。\n')


def verify(pred):
    cfg=json.loads((OUT/'config.json').read_text());assert cfg['stage'] in ['training_complete','complete'] and (OUT/'exit_code.txt').read_text().strip()=='0'
    for section in ['source_sha256','cache_sha256']:
        for p,d in cfg[section].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d,p
    for p,d in json.loads((OUT/'input_sha256.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d,p
    assert hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()==cfg['splits_sha256']
    table=pd.read_csv(OUT/'subjects.csv');meta=table.set_index('FILE_ID',drop=False);jobs=a.read_jobs();done=pd.read_csv(OUT/'completed.csv');diag=pd.read_csv(OUT/'fit_diagnostics.csv');inner=[json.loads(s) for s in (OUT/'inner_splits.jsonl').read_text().splitlines()]
    assert len(jobs)==len(done)==1100 and not done.job_id.duplicated().any() and set(done.job_id)=={j['job_id'] for j in jobs};assert len(diag)==4400 and len(inner)==5500
    assert len(pred)==sum(len(j['test'])+len(j['reverse']) for j in jobs) and not pred.duplicated(['job_id','FILE_ID']).any()
    assert pred.probability.between(0,1).all();np.testing.assert_array_equal(pred.prediction,pred.probability.ge(.5))
    groups=dict(tuple(pred.groupby('job_id')));lookup={j['job_id']:j for j in jobs};splits={}
    for j in jobs:
        g=groups[j['job_id']];assert set(g.loc[~g.reverse,'FILE_ID'])==set(j['test']) and set(g.loc[g.reverse,'FILE_ID'])==set(j['reverse'])
        tr=meta.loc[j['train']];te=meta.loc[j['test']];assert set(tr.site).isdisjoint(te.site) and set(tr.index).isdisjoint(te.index)
        if j['no_zero_roi']:assert meta.loc[j['train']+j['test']+j['reverse'],'n_zero_roi'].eq(0).all()
        for column in a.META+['n_zero_roi','n_timepoints']:pd.testing.assert_series_equal(g[column].reset_index(drop=True),meta.loc[g.FILE_ID,column].reset_index(drop=True),check_dtype=False,check_names=False)
        splits[j['job_id']]=list(StratifiedKFold(5,shuffle=True,random_state=j['seed']).split(tr,tr.label))
    for s in inner:
        job=lookup[s['job_id']];tr=meta.loc[job['train']];aa,bb=splits[s['job_id']][s['inner_fold']]
        assert s['train']==tr.iloc[aa].index.tolist() and s['validation']==tr.iloc[bb].index.tolist() and set(s['train']).isdisjoint(s['validation'])
        assert not set(job['test'])&(set(s['train'])|set(s['validation']))
    for row in done.itertuples():
        g=diag.loc[diag.job_id.eq(row.job_id)];assert row.C==g.iloc[int(np.argmax(g.inner_mean_AUC.to_numpy()))].C
    return dict(subjects=1035,sites=17,zero_ROI_subjects=int(table.n_zero_roi.gt(0).sum()),max_zero_ROIs=int(table.n_zero_roi.max()),outer_jobs=1100,LR_fits=23100,inner_splits=5500,predictions=len(pred),main_outer_jobs=int((~done.no_zero_roi).sum()),sensitivity_outer_jobs=int(done.no_zero_roi.sum()),warning_count=int(done.warning_count.sum()),all_subject_site_splits_and_metadata_verified=True,all_frozen_inputs_sources_cache_hashes_verified=True,inner_scaling='GridSearchCV Pipeline,not prefit',head_motion_imputation='training-part median inside each Pipeline fit')


def report(summary,diff,check,v):
    cfg=json.loads((OUT/'config.json').read_text());primary=summary.loc[summary.metric.eq('within_site_AUC')&~summary.reverse]
    paired=diff.loc[diff.metric.eq('within_site_AUC')]
    other=summary.loc[(summary.arm.isin(['F','FD'])&summary.metric.isin(['within_site_AUC','pooled_AUC','male_within_site_AUC']))|(summary.reverse&summary.metric.eq('pooled_AUC'))]
    text='# T12-A ABIDE纯类站点平台\n\n'+check.iloc[0].decision+'。本轮只做平台与基线，不启动T12-B或新增方法。\n\n'
    text+='**所有范围均为50次重复的2.5%与97.5%分位数。重复共用被试，它们是变异范围，不是独立样本置信区间，也没有做bootstrap。**\n\n'
    text+='## P1–P3判读\n\n'+markdown(check)+'\n\n## 主要指标：站点内AUC\n\n'+markdown(primary)+'\n\n## 混杂代价M−P\n\n'+markdown(paired)+'\n\n## 反向集、头动与全量参照\n\n'+markdown(other)+'\n\n'
    text+='## 男性结果\n\n'+markdown(summary.loc[summary.metric.eq('male_within_site_AUC')])+'\n\n'
    text+='## 固定实现与边界\n\n- 全部1035人，DX_GROUP=1为ASD，SEX=1为男。LEUVEN/UCLA/UM合并_1/_2为同一机构，共17站点；每次从13小站点选5个作测试，4大站点始终留在12站点训练池。无年龄或质量额外排除，BNU watch规则不适用。\n'
    text+='- CC200 Pearson相关，±.999999截断后Fisher z，按上三角行序19900维、float64。零方差ROI涉及的相关置0；46人主分析保留，敏感性仅删除这46人重做4个站点数条件。原始时间序列只读，派生矩阵和行序仅存指定ABIDE_derived/T12A。\n'
    text+='- 每重复numpy default_rng(r)，先抽测试站点，依固定顺序(k,n)=(1,40),(2,40),(4,40),(6,40),(2,20),(2,80)构造；同一条件P后M，k2/n40只算一次。在所有同时满足P/M每站点配额的有向ASD/TD不相交站点分配中均匀抽一个，站点内无放回抽样再逐类随机修剪。P/M人数和站点集合相同，不强求被试相同。敏感性重用相同种子/测试站点，删除46人后重新计算可行分配并抽人。\n'
    text+='- 普通sklearn L2 LR：默认lbfgs、max_iter=100、tol=1e-4、无类别权重；C固定.001/.01/.1/1。内部按被试标签5折分层、shuffle、种子r，按合并验证AUC选C，完全相同取网格中首个。StandardScaler位于Pipeline内，每个内折重新拟合；最终再用全外层训练集拟合。未按站点分组选C，正是任务规定的普通使用者流程；未调参、未加新方法。\n'
    text+='- 唯一缺失头动Yale_0050610只在FD参照中按用户确认的训练部分中位数填补；SimpleImputer位于Pipeline内，各内折及外层重拟合。其余连接分析保留全部1035人。FD使用F的相同训练名单，始终仅一个头动特征。\n'
    text+='- 主要指标先按5个测试站点分别AUC，再等权平均，合并AUC另外报告；男性也等权，某站点男性任一类<3时跳过，逐重复/逐站点的跳过人数与标记见repeat_metrics/site_metrics。反向集只对P，取所选站点的相反诊断者，不与训练重叠；仅报告合并AUC。\n'
    text+='- 两条曲线共享k2/n40；F和FD主分析每重复仅一次，敏感性不新增F/FD。全部21次LR拟合/外折（4C×5内折+最终1次）按固定流程执行。计时的6个完整外折及预测直接复用，其余1094个运行，不重复计入研究样本。五个CPU工作进程、BLAS各1线程，未用GPU、不装依赖。\n'
    text+=f'- 总耗时点估{cfg["timing"]["point_hours"]*60:.2f}分钟、含25%余量{cfg["timing"]["with_25percent_margin_hours"]*60:.2f}分钟，按用户≤3小时授权直接tmux运行。实际正式起止{cfg["started_at"]}至{cfg["finished_at"]}。收敛与其他警告计数为{v["warning_count"]}，不据此修改设置。\n\n'
    text+='核对：'+json.dumps(v,ensure_ascii=False)+'\n\n输出包括config/subjects/zero_variance_ROI/site_counts、完整splits/inner_splits、逐人predictions、completed、fit_diagnostics、重复指标/逐站点指标/主表/配对差值/criteria、以及class_pure_sites.png/pdf/svg与数值/图注。\n'
    (OUT/'REPORT.md').write_text(text)


if __name__=='__main__':
    from sklearn.model_selection import StratifiedKFold
    pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');v=verify(pred);summary,diff=summarize(pred);check=decision(summary,diff);figure(summary);report(summary,diff,check,v)
    a.dump('verification.json',v);cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='complete',report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());a.dump('config.json',cfg)
    print('REPORT_COMPLETE '+check.iloc[0].decision,flush=True)
