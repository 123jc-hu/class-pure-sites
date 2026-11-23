"""T12-D：沿用原指标与配对汇总，只判读预先给定的Q1–Q3。"""
import hashlib,json
from pathlib import Path

import numpy as np
import pandas as pd

import abide_robustness_run as r
import abide_class_pure_sites_report as ar
import site_pair_validation_report as br
import bridge_design_report as cr

OUT=r.OUT


def summarize_variant(variant):
    root=OUT/variant;pred=pd.read_csv(root/'predictions.csv',float_precision='round_trip');ar.OUT=root
    summary,differences=ar.summarize(pred,repeats=None);runs=pd.read_csv(root/'repeat_metrics.csv',float_precision='round_trip')
    truth=runs.loc[runs.arm.eq('P')].copy();truth['method']='B0';br.OUT=root/'B'
    performance=br.estimates(pd.read_csv(br.OUT/'completed.csv',float_precision='round_trip'),truth,expected_rows=None)
    controls=[j for j in r.read(root/'C/splits.jsonl') if j['method']=='S0'];cross={j['job_id']:j for j in r.read(root/'C/splits.jsonl') if j['method']=='C0'}
    kept={j['job_id'] for j in controls}|{j['C0_job_id'] for j in controls};cp=pd.read_csv(root/'C/predictions.csv',float_precision='round_trip');cp=cp.loc[cp.job_id.isin(kept)];base=[]
    for s in controls:
        j=cross[s['C0_job_id']];f=pred.loc[pred.job_id.eq(j['T12A_job_id'])&pred.FILE_ID.isin(j['test']+j['reverse'])].copy()
        assert set(f.FILE_ID)==set(j['test']+j['reverse'])
        context={key:j[key] for key in r.c.CONTEXT}|dict(job_id=j['job_id'].replace('C0_','B0_',1),method='B0',n_added=0,n_train=80)
        for key,value in context.items():f[key]=value
        base.append(f)
    cr.OUT=root/'C';c_runs,c_summary=cr.method_metrics(pd.concat([cp,*base],ignore_index=True),repeats=None);r.s.OUT=root/'C';paired=r.s.paired_summary(c_runs)
    rows=[]
    def cost(k):return float(differences.loc[differences.k.eq(k)&differences.metric.eq('within_site_AUC'),'mean'].iloc[0])
    def reverse(k):return float(summary.loc[summary.arm.eq('P')&summary.k.eq(k)&summary.reverse&summary.metric.eq('pooled_AUC'),'mean'].iloc[0])
    q1=dict(cost_k1=cost(1),cost_k6=cost(6),reverse_k1=reverse(1),reverse_k6=reverse(6));rows.append(dict(variant=variant,criterion='Q1',passed=q1['cost_k1']>=.03 and q1['cost_k6']<q1['cost_k1'] and q1['reverse_k6']>q1['reverse_k1'],details=json.dumps(q1)))
    f=performance.loc[performance.truth.eq('within_site_AUC')];bias=float(f.loc[f.method.eq('E_subj')&f.k.eq(2),'mean_bias'].iloc[0]);mae=f.loc[f.method.isin(['E_subj','E_LSPO'])].pivot(index='k',columns='method',values='mean_absolute_error')
    rows.append(dict(variant=variant,criterion='Q2',passed=bool(bias>=.10 and mae.E_LSPO.lt(mae.E_subj).all()),details=json.dumps(dict(bias_k2=bias,MAE={str(k):dict(subject=q.E_subj,LSPO=q.E_LSPO) for k,q in mae.iterrows()}))))
    q=paired.loc[paired.contrast.eq('C0-S0')&paired.direction.eq('twoway')].set_index('metric');test=float(q.loc['within_site_AUC','mean']);reverse_gain=float(q.loc['reverse_AUC','mean'])
    rows.append(dict(variant=variant,criterion='Q3',passed=test>0 and reverse_gain>0,details=json.dumps(dict(test_difference=test,reverse_difference=reverse_gain,repeats=int(q.loc['within_site_AUC','repeats'])))))
    criteria=pd.DataFrame(rows);criteria.to_csv(root/'criteria.csv',index=False);return summary,performance,paired,criteria


def comparison(variant,summary,performance,paired):
    old=pd.read_csv('outputs/T12A/repeat_metrics.csv',float_precision='round_trip');old=old.loc[~old.no_zero_roi];rows=[]
    for q in summary.loc[summary.metric.eq('within_site_AUC')|summary.reverse&summary.metric.eq('pooled_AUC')].itertuples():
        f=old.loc[old.arm.eq(q.arm)&old.k.eq(q.k)&old.n.eq(q.n)&old.reverse.eq(q.reverse)];f=f.loc[f.repeat.lt(20)] if variant=='V-CNN' else f;metric='within_site_AUC' if not q.reverse else 'pooled_AUC'
        rows.append(dict(variant=variant,part='A',metric=f'{q.arm}_k{q.k}_'+('reverse_AUC' if q.reverse else 'within_site_AUC'),new=q.mean,original=float(f[metric].mean()),new_repeats=q.repeats,original_repeats=len(f)))
    old=pd.read_csv('outputs/T12B/performance_estimates.csv',float_precision='round_trip');old=old.loc[old.method.isin(['E_subj','E_LSPO'])];old=old.loc[old.repeat.lt(20)] if variant=='V-CNN' else old
    for q in performance.loc[performance.truth.eq('within_site_AUC')&performance.method.isin(['E_subj','E_LSPO'])].itertuples():
        f=old.loc[old.method.eq(q.method)&old.k.eq(q.k)]
        for metric,new,baseline in [('bias',q.mean_bias,f.error_within_site_AUC.mean()),('MAE',q.mean_absolute_error,f.abs_error_within_site_AUC.mean())]:rows.append(dict(variant=variant,part='B',metric=f'{q.method}_k{q.k}_{metric}',new=new,original=float(baseline),new_repeats=q.repeats,original_repeats=len(f)))
    old=pd.read_csv('outputs/T12C2/paired_repeat_differences.csv',float_precision='round_trip');old=old.loc[old.k.eq(2)&old.b.eq(5)];old=old.loc[old.repeat.lt(20)] if variant=='V-CNN' else old
    for q in paired.itertuples():
        f=old.loc[old.direction.eq(q.direction)&old.metric.eq(q.metric)&old.contrast.eq(q.contrast)];rows.append(dict(variant=variant,part='C',metric=q.contrast+'_'+q.direction+'_'+q.metric,new=q.mean,original=float(f.difference.mean()),new_repeats=q.repeats,original_repeats=len(f)))
    return pd.DataFrame(rows)


def verify():
    provenance=json.loads((OUT/'provenance.json').read_text())
    for section in ['source_sha256','input_sha256','generated_sha256']:
        for f,h in provenance[section].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    expected=set();checks=[]
    for variant in r.VARIANTS:
        root=OUT/variant;table=pd.read_csv(root/'subjects.csv');meta=table.set_index('FILE_ID',drop=False);all_ids=set(meta.index);a_pred=pd.read_csv(root/'predictions.csv',float_precision='round_trip');c_pred=pd.read_csv(root/'C/predictions.csv',float_precision='round_trip');ep=pd.read_csv(root/'B/estimation_predictions.csv',float_precision='round_trip');ef=pd.read_csv(root/'B/estimation_folds.csv',float_precision='round_trip');mapping=pd.Index(pd.read_csv(r.d.ORIGINAL/'subjects.csv').FILE_ID)
        assert a_pred.probability.between(0,1).all() and c_pred.probability.between(0,1).all() and ep.probability.between(0,1).all()
        assert not a_pred.duplicated(['job_id','FILE_ID']).any() and not c_pred.duplicated(['job_id','FILE_ID']).any() and not ep.duplicated(['job_id','outer_fold','FILE_ID']).any()
        for t in r.tasks(variant):
            j=t['job'];assert set(j['train']+j['test']+j['reverse']).issubset(all_ids)
            if t['part'] in ['A','C']:
                f=(a_pred if t['part']=='A' else c_pred).loc[lambda x:x.job_id.eq(j['job_id'])];assert set(f.loc[~f.reverse,'FILE_ID'])==set(j['test']) and set(f.loc[f.reverse,'FILE_ID'])==set(j['reverse']) and set(j['train']).isdisjoint(f.FILE_ID)
                for col in r.a.META:pd.testing.assert_series_equal(f[col].reset_index(drop=True),meta.loc[f.FILE_ID,col].reset_index(drop=True),check_names=False,check_dtype=False)
                expected.add(r.task_id(t));continue
            tr=mapping.get_indexer(j['train']);whole=pd.read_csv(r.d.ORIGINAL/'subjects.csv');folds=r.b.subject_folds(whole,tr,j['seed']) if t['method']=='E_subj' else r.b.site_pairs(whole,tr);name=r.task_id(t).split('/')[-1]
            for fold,(train,validation) in enumerate(folds):
                f=ep.loc[ep.job_id.eq(name)&ep.outer_fold.eq(fold)];assert set(f.FILE_ID)==set(mapping[validation]) and set(mapping[train]).isdisjoint(f.FILE_ID)
                expected.add(r.task_id(t)+f'_f{fold}' if variant=='V-CNN' else r.task_id(t))
            assert len(ef.loc[ef.job_id.eq(name)])==len(folds)
        for frame in [a_pred,c_pred,ep]:np.testing.assert_array_equal(frame.prediction,frame.probability.ge(.5))
        checks.append(dict(variant=variant,subjects=len(table),A_jobs=a_pred.job_id.nunique(),C_jobs=c_pred.job_id.nunique(),estimation_folds=len(ef),predictions=len(a_pred)+len(c_pred)+len(ep)))
    done=pd.read_csv(OUT/'completed.csv');assert set(done.global_id)==expected and not done.global_id.duplicated().any()
    losses=pd.read_csv(OUT/'V-CNN/epoch_losses.csv');assert np.isfinite(losses.weighted_train_loss).all();assert losses.groupby('job_id').epoch.agg(list).apply(lambda x:x==list(range(1,51))).all();assert len(losses)==50*sum(x.startswith('V-CNN/') for x in expected)
    result=dict(global_completion_records=len(done),CNN_models=len(losses)//50,CNN_epochs=len(losses),all_probabilities_finite=True,all_split_IDs_and_metadata_verified=True,source_and_original_cache_hashes_unchanged=True,variants=checks);r.write(OUT/'verification.json',result);return result


def report():
    verification=verify();comparisons=[];criteria=[];text='# T12-D 核心链条的稳健性\n\n'
    for variant in r.VARIANTS:
        summary,performance,paired,check=summarize_variant(variant);criteria.append(check);comparisons.append(comparison(variant,summary,performance,paired));text+='## '+variant+'\n\n'+ar.markdown(check)+'\n\n### A：测试站点内及反向AUC\n\n'+ar.markdown(summary.loc[summary.metric.eq('within_site_AUC')|summary.reverse&summary.metric.eq('pooled_AUC')])+'\n\n### B：性能估计（折AUC均值为主）\n\n'+ar.markdown(performance.loc[performance.truth.eq('within_site_AUC')])+'\n\n### C：配对差值\n\n'+ar.markdown(paired)+'\n\n'
    check=pd.concat(criteria,ignore_index=True);check.to_csv(OUT/'criteria.csv',index=False);comparison_table=pd.concat(comparisons,ignore_index=True);comparison_table.to_csv(OUT/'numerical_comparison.csv',index=False)
    conclusion='三种变体均满足Q1–Q3，按预设核心链条稳健。' if check.passed.all() else '存在未满足项，按预设回报讨论：'+','.join(check.loc[~check.passed].apply(lambda x:x.variant+'/'+x.criterion,axis=1))+'。'
    text=text.replace('# T12-D 核心链条的稳健性\n\n','# T12-D 核心链条的稳健性\n\n'+conclusion+'\n\n'+ar.markdown(check)+'\n\n')
    text+='## 与原设置的数值对照\n\n'+ar.markdown(comparison_table)+'\n\n## 固定口径及限制\n\n'
    text+='- 所有范围为重复2.5%–97.5%变异范围，不是置信区间；正差比例沿用差值>1e−12的数值判等，均值/范围/Q判据不变。CNN前20重复，LR50重复，QC不可行格子直接跳过、报告实际人数；QC与原设置被试池不同，数值对照是描述性比较。CNN原设置参照重算既有结果的前20重复统计，没有重训原参照。\n'
    text+='- QC直接调用原划分及桥接/同类补采函数，拟合时把这些QC-only ID映射回只读原CC200缓存，不复制或重算该数组。HO只删#3455后调用原Fisher特征函数，零方差相关=0；HO/CNN的A/C/C2名单直接读原输出，未重新生成。LR仅调用a/b/c/s旧拟合函数，四C及内外折流程不变。\n'
    text+='- CNN固定E2E32×2/E2N64/N2G256/FC128/30/2；LeakyReLU .33，dropout .5在N2G256、FC128后。固定50epoch，不早停、不内层选参；E_subj与E_LSPO估计的就是此固定流程。边Scaler与类频率权重均只在当前实际训练者上拟合。每个外估计折重新初始化/拟合，不访问其验证者或原外层测试者；损失逐epoch、预测与完成记录逐训练立即写盘。\n'
    text+='- 学习率由原文.01改为1e−4（偏置2e−4），这是任务预定超参数中对原文的唯一偏离。依据为只看训练稳定性的诊断；旧NaN试跑未产生AUC，修订未看验证/测试结果，不继续调参。其他初始化/偏置SGD/损失/精度设置沿已冻结的pilot_config.json。\n'
    text+='- C的B0直接重用本变体A中的P预测，反向集过滤到同一真实桥接扣除名单，不重新训练。C0/S0原始预测均保存；配对及Q3只取S0可行的相同重复，男女主分析均保留。ABIDE不含BNU观察被试，因此不构造no_watch。\n'
    text+='- 所有任务完成后才读epoch明细并核对；无新依赖，不读restricted目录，不作图。本任务止于回报，不启动写作或后续任务。\n\n## 运行核对\n\n'+json.dumps(verification,ensure_ascii=False)+'\n\n计时：'+json.dumps(json.loads((OUT/'timing.json').read_text()),ensure_ascii=False)+'\n'
    (OUT/'REPORT.md').write_text(text);r.write(OUT/'decision.json',dict(all_passed=bool(check.passed.all()),conclusion=conclusion));print('REPORT_COMPLETE '+conclusion,flush=True)


if __name__=='__main__':report()
