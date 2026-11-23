"""T15-A：G0停止门、固定配对比较和训练诊断。"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import group_dro as g
import bridge_design_report as cr
from abide_class_pure_sites_report import markdown

OUT=g.OUT;METRICS=['within_site_AUC','male_within_site_AUC','pooled_AUC','reverse_AUC']


def metrics(pred):
    cr.OUT=OUT;context=pred.drop_duplicates('job_id').set_index('job_id');f=pred.copy();f['probability']=np.where(f.method.eq('L3'),f.score,f.probability);runs,summary=cr.method_metrics(f)
    runs['design']=runs.job_id.map(context.design);summary['design']=np.where(summary.b.eq(0),'B0','bridge_b5');return runs,summary


def g0():
    pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');ref=pd.read_csv(OUT/'reference_predictions.csv',float_precision='round_trip');pred=pred.loc[pred.method.eq('GD_ERM')];ref=ref.loc[ref.method.eq('sklearn_ERM')];assert pred.job_id.nunique()==200
    runs,_=metrics(pd.concat([pred,ref],ignore_index=True));runs.to_csv(OUT/'G0_repeat_metrics.csv',index=False);rows=[];raw=[]
    for (design,k),f in runs.loc[~runs.reverse].groupby(['design','k']):
        v=f.pivot(index='repeat',columns='method',values='within_site_AUC');assert len(v)==50;d=v.GD_ERM-v.sklearn_ERM;value=float(abs(d).mean());rows.append(dict(criterion='G0',design=design,k=k,mean_absolute_paired_difference=value,mean_signed_difference=float(d.mean()),threshold=.01,passed=value<.01,repeats=50));raw.extend(dict(design=design,k=k,repeat=int(i),difference=float(x)) for i,x in d.items())
    result=pd.DataFrame(rows);result.to_csv(OUT/'G0.csv',index=False);pd.DataFrame(raw).to_csv(OUT/'G0_paired_repeat_differences.csv',index=False);passed=bool(result.passed.all())
    if not passed:(OUT/'REPORT.md').write_text('# T15-A GroupDRO\n\nG0未通过，按预设停止；不解读GroupDRO结果，不改步数、学习率或C网格。\n\n'+markdown(result)+'\n\nGD-ERM是固定2000步衰减Adam，与sklearn优化器不同；未达到预设实现核对门槛。计时中的GroupDRO结果仅保留原始记录，不作科学判读。\n')
    return passed


def verify(pred,done):
    cfg=json.loads((OUT/'config.json').read_text());assert cfg['stage'] in ['training_complete','G0_failed'] and (OUT/'exit_code.txt').read_text().strip()=='0'
    for section in ['source_sha256','input_sha256']:
        for name,h in cfg[section].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    for section in ['previous_output_hashes','original_source_snapshots_sha256']:
        for name,h in cfg[section].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    assert hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()==cfg['splits_sha256'] and hashlib.sha256((OUT/'reference_predictions.csv').read_bytes()).hexdigest()==cfg['references_sha256']
    work={j['job_id']:j for j in g.plans()};assert done.job_id.is_unique and not pred.duplicated(['job_id','FILE_ID']).any() and np.isfinite(pred.score).all()
    if cfg['stage']=='training_complete':assert len(done)==500 and set(done.job_id)==set(work) and done.fit_count.sum()==10500
    table=pd.read_csv(OUT/'subjects.csv');meta=table.set_index('FILE_ID')
    for key,f in pred.groupby('job_id'):
        j=work[key];assert set(f.loc[~f.reverse,'FILE_ID'])==set(j['test']) and set(f.loc[f.reverse,'FILE_ID'])==set(j['reverse']) and not set(f.FILE_ID)&set(j['train'])
        for col in g.a.META:np.testing.assert_array_equal(f[col],meta.loc[f.FILE_ID,col] if col!='FILE_ID' else meta.loc[f.FILE_ID].index)
    fits=pd.DataFrame(map(json.loads,(OUT/'fit_diagnostics.jsonl').read_text().splitlines()));assert len(fits)==21*len(done) and fits.steps.eq(g.STEPS).all()
    for q in done.itertuples():
        f=fits.loc[fits.job_id.eq(q.job_id)];v=f.loc[f.role.eq('selection')].pivot(index='inner_fold',columns='C',values='AUC').reindex(columns=g.GRID);assert v.shape==(5,4) and q.C==g.GRID[int(np.argmax(v.to_numpy().mean(axis=0)))];assert f.loc[f.role.eq('final'),'C'].iloc[0]==q.C
    for row in fits.itertuples():
        q=np.array(row.q);loss=np.array(row.group_losses);assert len(q)==len(row.group_sites) and np.isfinite(loss).all() and np.isfinite(q).all();np.testing.assert_allclose(q.sum(),1.,atol=1e-14);np.testing.assert_allclose([q.max(),loss.max(),loss.mean()],[row.q_max,row.worst_group_loss,row.mean_group_loss],atol=1e-14)
    splits=[json.loads(s) for s in (OUT/'inner_splits.jsonl').read_text().splitlines()];assert len(splits)==5*len(done);lookup=pd.Index(table.FILE_ID)
    for s in splits:
        j=work[s['job_id']];tr,va=g.inner_folds(table,j)[s['inner_fold']];assert s['train']==lookup[tr].tolist() and s['validation']==lookup[va].tolist() and s['standardizer_fit']==s['train'];assert set(s['train']).isdisjoint(s['validation']) and not set(s['train']+s['validation'])&set(j['test']+j['reverse']);assert s['group_site']==meta.loc[s['train'],'site'].to_dict()
    final=[json.loads(s) for s in (OUT/'final_groups.jsonl').read_text().splitlines()];assert len(final)==len(done)
    for s in final:assert s['standardizer_fit']==work[s['job_id']]['train'] and s['group_site']==meta.loc[s['standardizer_fit'],'site'].to_dict()
    result=dict(jobs=len(done),fits=len(fits),Adam_steps=g.STEPS*len(fits),predictions=len(pred),inner_splits=len(splits),all_C_choices_groups_scalers_and_isolation_verified=True,all_inputs_sources_and_frozen_references_unchanged=True,original_outputs_unchanged=True,all_values_finite=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));return result,fits


def report():
    cfg=json.loads((OUT/'config.json').read_text());pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');done=pd.read_csv(OUT/'completed.csv',float_precision='round_trip');verification,fits=verify(pred,done)
    if cfg['stage']=='G0_failed':
        assert not g0();(OUT/'decision.json').write_text(json.dumps(dict(G0=False,H1=None,H2='not interpreted',stop='G0 failed'),indent=2));print('REPORT_COMPLETE G0_FAILED;no GroupDRO interpretation',flush=True);return
    assert g0();ref=pd.read_csv(OUT/'reference_predictions.csv',float_precision='round_trip');runs,summary=metrics(pd.concat([pred,ref],ignore_index=True));runs.to_csv(OUT/'repeat_metrics.csv',index=False);summary.to_csv(OUT/'main_table.csv',index=False);pairs=[]
    for (design,k),f in runs.groupby(['design','k']):
        methods=[('GD_ERM','sklearn_ERM'),('GroupDRO','GD_ERM'),('GroupDRO','sklearn_ERM')]+([('GroupDRO_eta0.1','GD_ERM'),('GroupDRO_eta0.1','sklearn_ERM')] if design=='B0' else [])
        for metric in METRICS:
            rev=metric=='reverse_AUC';name='pooled_AUC' if rev else metric;v=f.loc[f.reverse.eq(rev)].pivot(index='repeat',columns='method',values=name)
            for left,right in methods:pairs.extend(dict(design=design,k=k,contrast=left+'-'+right,metric=metric,repeat=int(i),difference=float(x)) for i,x in (v[left]-v[right]).items())
    raw=pd.DataFrame(pairs);assert len(raw)==3200 and raw.difference.notna().all();raw.to_csv(OUT/'paired_repeat_differences.csv',index=False);paired=raw.groupby(['design','k','contrast','metric']).difference.agg(mean='mean',range_low=lambda x:x.quantile(.025),range_high=lambda x:x.quantile(.975),positive_fraction=lambda x:x.gt(1e-12).mean(),repeats='count').reset_index();assert len(paired)==64 and paired.repeats.eq(50).all();paired.to_csv(OUT/'paired_differences.csv',index=False)
    criteria=pd.read_csv(OUT/'G0.csv');extra=[]
    for k in [2,4]:
        q=paired.loc[paired.design.eq('B0')&paired.k.eq(k)&paired.contrast.eq('GroupDRO-GD_ERM')&paired.metric.eq('reverse_AUC')].iloc[0];extra.append(dict(criterion='H1',design='B0',k=k,mean_signed_difference=q['mean'],threshold=.05,passed=bool(q['mean']<.05),repeats=50,details='reverse AUC paired mean<.05'))
        for contrast in ['GroupDRO-GD_ERM','GroupDRO-sklearn_ERM']:
            for metric in ['within_site_AUC','reverse_AUC']:
                q=paired.loc[paired.design.eq('bridge_b5')&paired.k.eq(k)&paired.contrast.eq(contrast)&paired.metric.eq(metric)].iloc[0];extra.append(dict(criterion='H2',design='bridge_b5',k=k,mean_signed_difference=q['mean'],threshold=None,passed=None,repeats=50,details=contrast+'/'+metric+';report only'))
    criteria=pd.concat([criteria,pd.DataFrame(extra)],ignore_index=True);criteria.to_csv(OUT/'criteria.csv',index=False);H1=all(q['passed'] for q in extra if q['criterion']=='H1');decision=dict(G0=True,H1=bool(H1),H2='report only',conclusion='两个k均满足H1：所测的GroupDRO也没有减少站点依赖。' if H1 else '至少一个k未满足H1：如实报告GroupDRO收益，C5需据此修改。');(OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    done.groupby(['design','method','k','C']).size().rename('count').reset_index().to_csv(OUT/'selected_C_distribution.csv',index=False);final=fits.loc[fits.role.eq('final')];final.to_csv(OUT/'final_diagnostics.csv',index=False);diag=final.melt(id_vars=['job_id','design','method','k','repeat'],value_vars=['q_max','q_entropy','worst_group_loss','mean_group_loss','mean_subject_loss'],var_name='diagnostic',value_name='value').groupby(['design','method','k','diagnostic']).value.agg(mean='mean',range_low=lambda x:x.quantile(.025),range_high=lambda x:x.quantile(.975)).reset_index();diag.to_csv(OUT/'diagnostic_summary.csv',index=False)
    main=summary.loc[(~summary.reverse&summary.metric.isin(METRICS[:3]))|(summary.reverse&summary.metric.eq('pooled_AUC'))]
    text='# T15-A 修订一 GroupDRO对照：ABIDE-I CC200\n\n'+decision['conclusion']+'\n\n## G0/H1/H2\n\n'+markdown(criteria)+'\n\n## 主表\n\n'+markdown(main)+'\n\n## 配对差值\n\n'+markdown(paired)+'\n\n## 训练诊断（最终选中模型）\n\n'+markdown(diag)+'\n\n## C分布\n\n'+markdown(pd.read_csv(OUT/'selected_C_distribution.csv'))+'\n\n## 计时\n\n'+(OUT/'timing.json').read_text()+'\n\n正式训练秒数：'+str(cfg['formal_seconds'])+'\n\n## 固定实现与限制\n\n'
    text+='- 只读T12A CC200缓存及T12A P(n40)/T12C C0双向b5的原名单；50次重复、k2/4，不重新划分、提特征或重训sklearn ERM。L3并列引用T13B已有逐人预测（k2为其T13A复用记录），评估人群逐格完全相同。ABIDE无BNU观察被试，no_watch不适用。\n'
    text+='- 全批量float64、零初始化、截距不惩罚，Adam lr=.01、betas=(.9,.999)、eps=1e−8，无weight decay/早停，LambdaLR(lambda s:1−s/2000)在每次Adam更新后调用，共2000步；实际第一步lr=.01、最后一步lr=.000005，最后调度到0；显式惩罚||w||²/(2CN)。四C在独立列中计算，各列有独立参数、Adam矩和q，目标对列求和而非取均值；最终只训练选中的C。\n'
    text+='- GroupDRO按[作者loss.py第54–63行](https://github.com/kohpangwei/group_DRO/blob/master/loss.py#L54-L63)：本站点平均CE更新脱离梯度的q后再更新模型，主eta=.01；eta=.1只在B0报告，不参与H1。无generalization adjustment或loss normalization。组按当前训练部分的实际site，桥接者仍归本站点。q诊断是第2000次q更新后的值；训练损失是第2000次Adam更新后重算的不含惩罚CE；平均组损失按站点等权，另存被试等权CE。\n'
    text+='- 修订一仅改变优化器日程；固定lr在C≤.01时未收敛，依据为任务说明中的同C目标核对。本轮全部500任务重新训练，原输出保留且逐文件哈希复核，旧源码快照存在本目录original_sources。q仍每步使用固定eta，因此2000步可能比1000步更集中；eta=.1敏感性只报告更强的一端，门槛不变。\n'
    text+='- 内层仍为诊断分层的被试五折，与C0相同，并非站点留出；每折及最终Scaler仅拟合该训练部分。按验证概率AUC的五折均值选原四C，打平取首个。预测GD/sklearn用原概率，L3的AUC用原decision分数；分类GD/sklearn阈值.5，L3沿原预测。\n'
    text+='- 新测试五站点AUC等权；男性SEX=1且每站点各诊断至少3人。合并AUC、Acc/Sens/Spec/BA另表保存。区间是50重复2.5%–97.5%变异范围，不是独立样本置信区间；只在正差比例用1e−12判等。G0是配对绝对差的均值，不是均值差绝对值；H1按有向均值<.05，不取绝对值，不据变异范围增加显著性判读。H2只报告，不作新判定。\n'
    text+='- 运行前预测已写入config：纯类站点捷径可同时降低各组损失，预期H1满足；预测不是保证。H1仅支持所测线性模型、固定优化设置和被试内选参下的有限结论，不推广到所有域泛化方法。4个CPU spawn工作进程，BLAS/torch单线程；无GPU、新依赖或图。\n\n验证：'+json.dumps(verification,ensure_ascii=False)+'\n'
    (OUT/'REPORT.md').write_text(text);cfg.update(stage='complete',decision=decision);(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('REPORT_COMPLETE '+json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    report()
