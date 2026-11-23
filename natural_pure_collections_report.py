"""T15-B：固定集合、等权站点主DiD与合并成人男性敏感性。"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,balanced_accuracy_score
import natural_pure_collections as n
from abide_class_pure_sites_report import markdown

OUT=n.OUT


def verify(pred,done):
    cfg=json.loads((OUT/'config.json').read_text());assert cfg['stage']=='training_complete' and (OUT/'exit_code.txt').read_text().strip()=='0'
    for section in ['source_sha256','input_sha256']:
        for name,h in cfg[section].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    assert hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()==cfg['splits_sha256'] and hashlib.sha256((OUT/'evaluation_aliases.csv').read_bytes()).hexdigest()==cfg['aliases_sha256']
    jobs={j['job_id']:j for j in n.plans(cfg['secondary_repeats'])};assert len(done)==cfg['jobs'] and done.job_id.is_unique and set(done.job_id)==set(jobs) and done.fit_count.sum()==cfg['expected_fits']
    assert not pred.duplicated(['job_id','FILE_ID']).any() and len(pred)==sum(len(j['test']) for j in jobs.values()) and np.isfinite(pred.probability).all();np.testing.assert_array_equal(pred.prediction,pred.probability.ge(.5))
    t=n.table();meta=t.set_index('FILE_ID',drop=False)
    for key,f in pred.groupby('job_id'):
        j=jobs[key];assert set(f.FILE_ID)==set(j['test']) and not set(f.FILE_ID)&set(j['train'])
        expected=meta.loc[f.FILE_ID,n.a.META+['collection','watch']].copy()
        if j['site_definition']=='merged':expected.loc[expected.collection.isin(['ABIDEII-NYU_1','ABIDEII-NYU_2']),'site']='NYU'
        for col in n.a.META+['collection','watch']:np.testing.assert_array_equal(f[col],expected[col] if col!='FILE_ID' else expected.index)
    fits=pd.read_csv(OUT/'selection_fits.csv',float_precision='round_trip');assert len(fits)==4*len(jobs)
    for q in done.itertuples():
        f=fits.loc[fits.job_id.eq(q.job_id)].set_index('C').loc[n.a.GRID];assert q.C==n.a.GRID[int(np.argmax(f.inner_mean_AUC.to_numpy()))]
    splits=[json.loads(line) for line in (OUT/'inner_splits.jsonl').read_text().splitlines()];assert len(splits)==5*len(jobs);lookup=pd.Index(t.FILE_ID)
    for s in splits:
        j=jobs[s['job_id']];a,b=n.inner_folds(t,j)[s['inner_fold']];assert s['train']==lookup[a].tolist() and s['validation']==lookup[b].tolist() and s['standardizer_fit']==s['train'];assert set(s['train']).isdisjoint(s['validation']) and not set(s['train']+s['validation'])&set(j['test'])
    result=dict(unique_jobs=len(jobs),LR_fits=int(done.fit_count.sum()),physical_predictions=len(pred),inner_records=len(splits),all_metadata_C_selection_and_isolation_verified=True,source_and_cache_hashes_unchanged=True,warnings=int(done.warning_count.sum()),python_isolated=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));return result


def expand(pred):
    aliases=pd.read_csv(OUT/'evaluation_aliases.csv',keep_default_na=False);by={key:f for key,f in pred.groupby('job_id')};frames=[]
    for row in aliases.itertuples():
        f=by[row.source_job_id].copy();f['scenario']=row.scenario;f['source_job_id']=row.source_job_id
        if row.scenario=='nyu_merged':f.loc[f.collection.isin(['ABIDEII-NYU_1','ABIDEII-NYU_2']),'site']='NYU';f['site_definition']='merged'
        frames.append(f)
    result=pd.concat(frames,ignore_index=True);assert not result.duplicated(['scenario','evaluation','repeat','FILE_ID']).any();t=n.table()
    for (scenario,_,_),f in result.groupby(['scenario','evaluation','repeat']):assert set(f.FILE_ID)==set(t.loc[t.pool_main if scenario=='main' else t.pool_secondary,'FILE_ID'])
    result.to_csv(OUT/'evaluation_predictions.csv',index=False);return result


def group_stat(f,collection,adult=False):
    target=f.loc[f.collection.eq(collection)&f.label.eq(1)];control=f.loc[~f.collection.isin(['ABIDEII-KUL_3','ABIDEII-NYU_2','ABIDEII-UCLA_1'])&f.label.eq(1)]
    if adult:control=control.loc[control.AGE_AT_SCAN.ge(18)&control.SEX.eq(1)]
    values=[]
    for name,g in [('target',target),('control',control)]:
        for metric,column in [('Sens','prediction'),('mean_ASD_probability','probability')]:
            value=g[column].mean() if name=='target' or adult else g.groupby('site')[column].mean().mean()
            values.append(dict(group=name,metric=metric,value=float(value),n=len(g),n_sites=g.site.nunique(),aggregation='subjects' if name=='target' or adult else 'equal_mixed_sites'))
    return pd.DataFrame(values)


def analysis1(pred):
    rows=[];groups=[]
    for scenario,f in pred.groupby('scenario'):
        targets=['ABIDEII-KUL_3'] if scenario=='main' else ['ABIDEII-NYU_2','ABIDEII-KUL_3'];lo=f.loc[f.evaluation.eq('LOSO')]
        for target in targets:
            for analysis,adult in [('primary' if scenario=='main' else 'secondary',False)]+([('adult_male_sensitivity',True)] if scenario=='main' else []):
                base=group_stat(lo,target,adult).set_index(['group','metric']);groups.append(base.reset_index().assign(scenario=scenario,target=target,analysis=analysis,evaluation='LOSO',repeat=-1))
                for repeat,s in f.loc[f.evaluation.eq('subject')].groupby('repeat'):
                    a=group_stat(s,target,adult).set_index(['group','metric']);groups.append(a.reset_index().assign(scenario=scenario,target=target,analysis=analysis,evaluation='subject',repeat=repeat))
                    for metric in ['Sens','mean_ASD_probability']:
                        tv=float(a.loc[('target',metric),'value']);cv=float(a.loc[('control',metric),'value']);tl=float(base.loc[('target',metric),'value']);cl=float(base.loc[('control',metric),'value']);rows.append(dict(scenario=scenario,target=target,analysis=analysis,repeat=int(repeat),metric=metric,target_subject=tv,target_LOSO=tl,control_subject=cv,control_LOSO=cl,target_drop=tv-tl,control_drop=cv-cl,DiD=(tv-tl)-(cv-cl),target_n=int(a.loc[('target',metric),'n']),control_n=int(a.loc[('control',metric),'n']),control_sites=int(a.loc[('control',metric),'n_sites'])))
    raw=pd.DataFrame(rows);raw.to_csv(OUT/'DiD_repeats.csv',index=False);pd.concat(groups,ignore_index=True).to_csv(OUT/'group_repeat_metrics.csv',index=False)
    long=raw.melt(id_vars=['scenario','target','analysis','repeat','metric','target_n','control_n','control_sites'],value_vars=['target_subject','target_LOSO','control_subject','control_LOSO','target_drop','control_drop','DiD'],var_name='quantity',value_name='value');summary=long.groupby(['scenario','target','analysis','metric','quantity','target_n','control_n','control_sites']).value.agg(mean='mean',range_low=lambda x:x.quantile(.025),range_high=lambda x:x.quantile(.975),repeats='count').reset_index();summary.to_csv(OUT/'analysis1_summary.csv',index=False);return raw,summary


def analysis2(pred,done):
    rows=[];choices=done.set_index('job_id')
    for site,f in pred.loc[pred.pool.eq('main')&pred.evaluation.ne('subject')&pred.held_site.ne('KUL_3')].groupby('held_site'):
        yes=f.loc[f.evaluation.eq('LOSO')].sort_values('FILE_ID');no=f.loc[f.evaluation.eq('LOSO_no_KUL')].sort_values('FILE_ID');np.testing.assert_array_equal(yes.FILE_ID,no.FILE_ID);y=roc_auc_score(yes.label,yes.probability);z=roc_auc_score(no.label,no.probability);yc=choices.loc[yes.job_id.iloc[0]];nc=choices.loc[no.job_id.iloc[0]];rows.append(dict(site=site,test_n=len(yes),with_KUL_AUC=y,without_KUL_AUC=z,paired_difference=y-z,with_KUL_L=float(yc.L),without_KUL_L=float(nc.L),with_KUL_C=yc.C,without_KUL_C=nc.C,with_KUL_train_n=yc.n_train,without_KUL_train_n=nc.n_train))
    result=pd.DataFrame(rows);assert len(result)==13;result.to_csv(OUT/'analysis2_paired_sites.csv',index=False);summary=pd.DataFrame([dict(sites=13,mean_paired_difference=result.paired_difference.mean(),min_paired_difference=result.paired_difference.min(),max_paired_difference=result.paired_difference.max())]);summary.to_csv(OUT/'analysis2_summary.csv',index=False);return result,summary


def overall(pred):
    rows=[]
    for (scenario,evaluation,repeat),f in pred.groupby(['scenario','evaluation','repeat']):
        rows.append(dict(scenario=scenario,evaluation=evaluation,repeat=repeat,n=len(f),Acc=accuracy_score(f.label,f.prediction),Sens=recall_score(f.label,f.prediction),Spec=recall_score(f.label,f.prediction,pos_label=0),BA=balanced_accuracy_score(f.label,f.prediction),pooled_AUC=roc_auc_score(f.label,f.probability)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'overall_repeat_metrics.csv',index=False);result.melt(id_vars=['scenario','evaluation','repeat','n'],var_name='metric',value_name='value').groupby(['scenario','evaluation','metric']).value.agg(mean='mean',range_low=lambda x:x.quantile(.025),range_high=lambda x:x.quantile(.975),repeats='count').reset_index().to_csv(OUT/'overall_summary.csv',index=False)


def report():
    pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');done=pd.read_csv(OUT/'completed.csv',float_precision='round_trip');verification=verify(pred,done);logical=expand(pred);raw,summary=analysis1(logical);sites,impact=analysis2(pred,done);overall(logical)
    main=raw.loc[raw.scenario.eq('main')&raw.analysis.eq('primary')&raw.metric.eq('Sens')];sensitivity=raw.loc[raw.analysis.eq('adult_male_sensitivity')&raw.metric.eq('Sens')];point=float(main.DiD.mean());sens=float(sensitivity.DiD.mean());opposite=bool(point*sens<0);passed=point>=.1;criteria=pd.DataFrame([dict(criterion='K1',scenario='main',metric='Sens',DiD_mean=point,threshold=.1,repeats=50,passed=passed)]);criteria.to_csv(OUT/'criteria.csv',index=False)
    decision=dict(K1=bool(passed),main_DiD=point,adult_male_DiD=sens,adult_male_reference_n=50,sensitivity_opposite_direction=opposite,secondary='report only,no K1',conclusion='K1满足：按预设，真实天然纯类集合在按被试CV下识别率被高估。' if passed else 'K1不满足：只描述天然纯类集合存在，不声称本案例达到预设高估标准。');(OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2));cfg=json.loads((OUT/'config.json').read_text())
    main_table=summary.loc[summary.quantity.isin(['target_subject','target_LOSO','control_subject','control_LOSO','target_drop','control_drop','DiD'])];text='# T15-B 天然纯类集合案例\n\n'+decision['conclusion']+'\n\n## 判读\n\n'+markdown(criteria)+'\n\n## 分析1：敏感度、概率与DiD\n\n'+markdown(main_table)+'\n\n## 分析2：KUL进入训练的影响与L\n\n'+markdown(sites)+'\n\n'+markdown(impact)+'\n\n## 成年男性敏感性人数\n\n'+markdown(pd.read_csv(OUT/'adult_male_reference_counts.csv'))+'\n\n## 固定口径与限制\n\n'
    text+='- 只读T14B第一阶段568人Gordon352/61776维缓存，UCLA_1不用。主池547=526混合+21KUL_3，NYU_2不在主池，NYU_1是独立混合站点；次要池567人包含KUL_3与NYU_2。未重新提取特征或安装依赖。\n'
    text+='- C0训练直接调用abide_class_pure_sites.fit_job/select_model：训练专用StandardScaler+原4C默认LR。外层按诊断五折种子repeat，内层亦诊断五折shuffle种子repeat；LOSO内层固定0。外层测试者从不参与内层或Scaler拟合。每次训练结束即写逐人预测、4C验证分数、内折与参数。\n'
    text+='- 主对照为13个混合站点ASD的站点等权Sens/平均概率；KUL目标21人。下降=按被试OOF−LOSO；DiD=目标下降−对照下降。LOSO每站只训练一次，其固定预测用于每个重复的比较；50种子变异范围不包含重新抽站点或被试的不确定性，不是置信区间。概率DiD只报告；固定K1仅主分析Sens DiD均值≥.10，不作新显著性检验。\n'
    text+='- 成年男性敏感性仅替换主对照：18岁以上SEX=1、50名ASD、7站点，按人合并，不重新训练；报告人数和方向，不应用K1。方向与主分析相反：'+str(opposite)+'。这只是粗年龄/性别子集，不是精确年龄匹配或因果识别。\n'
    text+='- NYU_2次要目标为其20名ASD，KUL_3的21人数值顺带报告，均与基础13混合站点ASD比较（始终排除NYU_2和KUL_3）。合并NYU时LOSO留出NYU_1+NYU_2共73人；目标仍只统计NYU_2的20人，对照中的NYU_1 ASD使用同一合并NYU留出模型。次要不做分析2或成人敏感性，不作K1判读。\n'
    text+='- 次要两定义的567人群、诊断标签、外内层种子完全相同，LR不使用站点输入，故全部被试CV拟合共用；LOSO非NYU训练/测试名单也完全相同，复用13个模型，仅合并NYU新增一个拟合。主分析2加KUL侧复用主LOSO，去KUL侧新拟合13个模型。evaluation_aliases.csv和evaluation_predictions.csv逐条保存复用关系，未把复用计为独立重复。\n'
    text+='- 分析2按13个混合测试站点分别计算AUC配对差；范围为13站点的最小/最大值，非置信区间。L直接调用原design_rule_baselines.leakage，只用本格实际训练者的站点×类别组成；不引入新指标或方法。总五项分类指标另存在overall_summary.csv，只为全部人群的辅助描述，不对纯ASD集合计算AUC。\n'
    text+='- Python始终-I，固定可信代码路径显式加入，数据目录不加入搜索路径；4个spawn工作进程及父进程/资源跟踪器不超过6，BLAS单线程，无GPU。ABIDE不含BNU观察被试，watch统一False，no_watch不适用。不作图或启动后续任务。\n\n计时：'+(OUT/'timing.json').read_text()+'\n\n正式运行秒数：'+str(cfg['formal_seconds'])+'\n\n验证：'+json.dumps(verification,ensure_ascii=False)+'\n'
    (OUT/'REPORT.md').write_text(text);cfg.update(stage='complete',decision=decision);(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('REPORT_COMPLETE '+json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    report()
