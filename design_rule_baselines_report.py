"""T14-A：公共反向集分配比较、原生反向集泄漏相关和固定G判读。"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import design_rule_baselines as d
import bridge_design_report as cr
from abide_class_pure_sites_report import markdown

OUT=d.OUT;GROUP=['feature','method','allocation','k','b','direction'];AUC=['within_site_AUC','pooled_AUC','male_within_site_AUC']


def references():
    parts=[];choices=[];by={}
    for path in ['outputs/T13B/predictions.csv','outputs/T13B/reference_predictions.csv']:
        f=pd.read_csv(path,float_precision='round_trip',low_memory=False)
        by.update({key:(path,g) for key,g in f.groupby('job_id')})
    selected=pd.read_csv('outputs/T13B/selected_C.csv').set_index('job_id')
    for j in d.b.plans():
        if j['method']!='L3':continue
        for method in ['C0']+(['L3'] if j['b']==5 else [])+(['C2'] if j['feature']=='CC200' and j['b']==5 else []):
            key=j['job_id'].replace('_L3_','_'+method+'_');path,g=by[key];f=g.copy();f['allocation']='equal';f['reference_file']=path;f['reference_job_id']=key;parts.append(f[d.m.a.META+d.KEYS+['reverse','score','probability','prediction','reference_file','reference_job_id']]);choices.append(dict(job_id=key,C=float(selected.loc[key,'C']),reference=True))
    for feature,task in [('CC200','T12A'),('HO','T12D/V-HO')]:
        path=f'outputs/{task}/predictions.csv';f=pd.read_csv(path,float_precision='round_trip');old={key:g for key,g in f.groupby('job_id')};choice=pd.read_csv(f'outputs/{task}/completed.csv').set_index('job_id')
        for j in d.c.original_jobs():
            key=f'{feature}_B0_r{j["repeat"]}_k{j["k"]}';g=old[j['job_id']].copy();assert set(g.FILE_ID)==set(j['test']+j['reverse']);context=dict(feature=feature,job_id=key,source_job_id=j['job_id'],method='B0',model='B0',setting='secondary',k=j['k'],b=0,direction='none',repeat=j['repeat'],seed=j['seed'],n_pure=80,n_added=0,n_train=80,T12A_job_id=j['job_id'],allocation='none')
            for name,value in context.items():g[name]=value
            g['score']=g.probability;g['reference_file']=path;g['reference_job_id']=j['job_id'];parts.append(g[d.m.a.META+d.KEYS+['reverse','score','probability','prediction','reference_file','reference_job_id']]);choices.append(dict(job_id=key,C=float(choice.loc[j['job_id'],'C']),reference=True))
    result=pd.concat(parts,ignore_index=True);assert result.job_id.nunique()==2000 and not result.loc[result.method.eq('C2'),'reverse'].any();result.to_csv(OUT/'reference_predictions.csv',index=False)
    new=pd.read_csv(OUT/'completed.csv');new['reference']=False;pd.concat([new,pd.DataFrame(choices)],ignore_index=True).to_csv(OUT/'selected_C.csv',index=False);return result


def verify(pred):
    cfg=json.loads((OUT/'config.json').read_text());assert cfg['stage']=='training_complete' and (OUT/'exit_code.txt').read_text().strip()=='0'
    for section in ['source_sha256','input_sha256','neuroCombat_source_sha256']:
        for name,h in cfg[section].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    assert hashlib.sha256((OUT/'predicted_L.csv').read_bytes()).hexdigest()==cfg['predicted_L_sha256']
    assert hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()==cfg['splits_sha256']
    jobs={j['job_id']:j for j in d.jobs()};done=pd.read_csv(OUT/'completed.csv',float_precision='round_trip');assert len(done)==2000 and done.job_id.is_unique and set(done.job_id)==set(jobs) and done.fit_count.sum()==46000
    assert not pred.duplicated(['job_id','FILE_ID']).any() and np.isfinite(pred.score).all();meta=d.m.subjects().set_index('FILE_ID')
    for key,f in pred.groupby('job_id'):
        j=jobs[key];rev=j['reverse'] if not j['method'].startswith('ComBat') or (j['method']=='ComBat-noEB' and cfg['report_noEB_reverse']) else []
        assert set(f.FILE_ID)==set(j['test']+rev) and set(f.FILE_ID).isdisjoint(j['train']) and set(f.loc[f.reverse,'FILE_ID'])==set(rev)
        for col in d.m.a.META[1:]:np.testing.assert_array_equal(f[col],meta.loc[f.FILE_ID,col])
    fits=pd.read_csv(OUT/'selection_fits.csv',float_precision='round_trip');assert len(fits)==44800
    for q in done.itertuples():
        f=fits.loc[fits.job_id.eq(q.job_id)];grid=d.m.GRID if q.method in ['L1','L3'] else d.c.a.GRID;v=f.loc[f.inner_fold.ge(0)].pivot(index='inner_fold',columns='C',values='AUC').reindex(columns=grid);assert v.shape==(5,len(grid)) and q.C==grid[int(np.argmax(v.to_numpy().mean(axis=0)))]
        if q.method in ['L1','L3']:assert f.success.all()
    inner=[json.loads(line) for line in (OUT/'inner_splits.jsonl').read_text().splitlines()];assert len(inner)==10000
    for s in inner:
        j=jobs[s['job_id']];a=set(s['train']);v=set(s['validation']);assert a.isdisjoint(v) and a|v==set(j['train']) and (a|v).isdisjoint(j['test']+j['reverse']) and s['standardizer_fit']==s['train']
        if j['method'].startswith('ComBat'):assert s['ComBat_fit']==j['train']
    final=[json.loads(line) for line in (OUT/'fit_provenance.jsonl').read_text().splitlines()];assert len(final)==2000
    for f in final:assert f['standardizer_fit']==jobs[f['job_id']]['train'] and f['dropped_pure']==jobs[f['job_id']]['dropped_pure']
    result=dict(new_jobs=2000,new_fits=46000,inner_folds=10000,predictions=len(pred),warnings=int(done.warning_count.sum()),all_C_and_train_only_fits_verified=True,raw_inputs_sources_and_predicted_L_unchanged=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));return result


def report():
    pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');verification=verify(pred);ref=references();allpred=pd.concat([pred,ref],ignore_index=True);runs=[];summaries=[];sites=[];cfg=json.loads((OUT/'config.json').read_text())
    for (feature,allocation),f in allpred.groupby(['feature','allocation']):
        cr.OUT=OUT;r,s=cr.method_metrics(f.drop(columns='probability').rename(columns={'score':'probability'}))
        for method in ['ComBat-EB']+([] if cfg['report_noEB_reverse'] else ['ComBat-noEB']):
            r.loc[r.method.eq(method),[x for x in cr.METRICS if x!='within_site_AUC']]=np.nan;s=s.loc[~(s.method.eq(method)&s.metric.ne('within_site_AUC'))]
        r.loc[r.method.eq('C2'),['Acc','BA','Sens','Spec']]=np.nan;s=s.loc[~(s.method.eq('C2')&~s.metric.str.endswith('AUC'))]
        site=pd.read_csv(OUT/'site_metrics.csv')
        for g in [r,s,site]:g['feature']=feature;g['allocation']=allocation
        runs.append(r);summaries.append(s);sites.append(site)
    runs=pd.concat(runs,ignore_index=True);summary=pd.concat(summaries,ignore_index=True);runs.to_csv(OUT/'repeat_metrics.csv',index=False);summary.to_csv(OUT/'main_table.csv',index=False);pd.concat(sites,ignore_index=True).to_csv(OUT/'site_metrics.csv',index=False)
    by={key:g for key,g in allpred.groupby('job_id')};common=[]
    for j in d.designs():
        for method in ['C0','L3']:
            key=j['job_id']+'_'+method if j['allocation']=='concentrated' else f'{j["feature"]}_{method}_r{j["repeat"]}_k{j["k"]}_b5_{j["direction"]}';f=by[key].set_index('FILE_ID').loc[j['common_reverse']];assert len(f)==len(j['common_reverse'])
            common.append({name:j[name] for name in ['feature','allocation','k','b','direction','repeat']}|dict(method=method,n_evaluated=len(f),common_reverse_AUC=roc_auc_score(f.label,f.score)))
    common=pd.DataFrame(common);common.to_csv(OUT/'common_reverse_metrics.csv',index=False);raw=[]
    for (feature,k,direction,method),f in common.groupby(['feature','k','direction','method']):
        v=f.pivot(index='repeat',columns='allocation',values='common_reverse_AUC');raw.extend(dict(feature=feature,k=k,b=5,direction=direction,contrast=method+':equal-concentrated',metric='common_reverse_AUC',repeat=int(r),difference=float(x)) for r,x in (v.equal-v.concentrated).items())
    for (feature,k,direction),f in runs.loc[runs.b.eq(5)&runs.allocation.eq('concentrated')].groupby(['feature','k','direction']):
        for metric,rev,name in [('within_site_AUC',False,'within_site_AUC'),('reverse_AUC',True,'pooled_AUC')]:
            v=f.loc[f.reverse.eq(rev)].pivot(index='repeat',columns='method',values=name);raw.extend(dict(feature=feature,k=k,b=5,direction=direction,contrast='concentrated:L3-C0',metric=metric,repeat=int(r),difference=float(x)) for r,x in (v.L3-v.C0).items())
    for (feature,k,direction),f in runs.loc[runs.b.eq(5)&runs.allocation.eq('equal')&runs.method.isin(['L1','L3'])].groupby(['feature','k','direction']):
        v=f.loc[~f.reverse].pivot(index='repeat',columns='method',values='within_site_AUC');raw.extend(dict(feature=feature,k=k,b=5,direction=direction,contrast='equal:L1-L3',metric='within_site_AUC',repeat=int(r),difference=float(x)) for r,x in (v.L1-v.L3).items())
    raw=pd.DataFrame(raw);assert len(raw)==2000;raw.to_csv(OUT/'paired_repeat_differences.csv',index=False);paired=raw.groupby(['feature','k','b','direction','contrast','metric']).difference.agg(mean='mean',range_low=lambda x:x.quantile(.025),range_high=lambda x:x.quantile(.975),positive_fraction=lambda x:x.gt(1e-12).mean(),mean_absolute_difference=lambda x:x.abs().mean(),repeats='count').reset_index();assert len(paired)==40 and paired.repeats.eq(50).all();paired.to_csv(OUT/'paired_differences.csv',index=False)
    # G2严格使用各格原生反向集；G1另用上面的公共反向集。
    native=runs.loc[runs.reverse&runs.method.isin(['C0','B0'])].groupby(GROUP).pooled_AUC.mean().rename('native_reverse_AUC').reset_index();leak=pd.read_csv(OUT/'all_L.csv');cells=leak.merge(native,on=GROUP,validate='one_to_one');assert len(cells)==36;cells['evidence']=np.where(cells.allocation.eq('concentrated'),'new_confirmatory','existing_exploratory');cells.to_csv(OUT/'leakage_cells.csv',index=False)
    correlation=[];criteria=[]
    for feature in ['CC200','HO']:
        f=cells.loc[cells.feature.eq(feature)];rho=float(spearmanr(f.mean_L,f.native_reverse_AUC).statistic);correlation.append(dict(feature=feature,cells=len(f),Spearman_rho=rho,passed=rho<=-.8));criteria.append(dict(feature=feature,criterion='G2',k=None,point=rho,passed=rho<=-.8))
        q=paired.loc[paired.feature.eq(feature)&paired.k.eq(4)&paired.direction.eq('twoway')&paired.contrast.eq('C0:equal-concentrated')&paired.metric.eq('common_reverse_AUC')].iloc[0];criteria.append(dict(feature=feature,criterion='G1',k=4,point=float(q['mean']),passed=bool(q['mean']>0)))
        for k in [2,4]:
            q=paired.loc[paired.feature.eq(feature)&paired.k.eq(k)&paired.direction.eq('twoway')&paired.contrast.eq('equal:L1-L3')].iloc[0];criteria.append(dict(feature=feature,criterion='G4',k=k,point=float(q.mean_absolute_difference),passed=bool(q.mean_absolute_difference<.005)))
    pd.DataFrame(correlation).to_csv(OUT/'leakage_correlation.csv',index=False);criteria=pd.DataFrame(criteria);criteria.to_csv(OUT/'criteria.csv',index=False);g1=bool(criteria.loc[criteria.criterion.eq('G1'),'passed'].all());g2=bool(criteria.loc[criteria.criterion.eq('G2'),'passed'].all());g4=bool(criteria.loc[criteria.criterion.eq('G4'),'passed'].all());text='G1/G2均满足：分配规则和泄漏指数可写进论文，ABIDE-II确认须另行下达。' if g1 and g2 else 'G1不满足：分配规则只保留为理论推论，回报讨论。' if not g1 else 'G1满足但G2不满足：集中分配比较支持，泄漏指数未达预设相关阈值，回报讨论。';decision=dict(G1=g1,G2=g2,G3_EB=False,G3_noEB=bool(cfg['report_noEB_reverse']),G4=g4,decision=text);(OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    main=summary.loc[summary.b.eq(5)|summary.method.eq('B0')];newcells=cells.loc[cells.evidence.eq('new_confirmatory')];oldcells=cells.loc[cells.evidence.eq('existing_exploratory')]
    report='# T14-A 桥接设计规则与对照\n\n'+text+'\n\n## G1/G2/G4判读\n\n'+markdown(criteria)+'\n\n## G3零检验\n\nEB：\n\n'+markdown(pd.read_csv(OUT/'null_summary.csv'))+'\n\nnoEB：\n\n'+markdown(pd.read_csv(OUT/'null_noEB_summary.csv'))+'\n\n## 训练前预测L（未重写）\n\n'+markdown(pd.read_csv(OUT/'predicted_L.csv'))+'\n\n## 主表（其余格子见CSV）\n\n'+markdown(main)+'\n\n## 配对差值\n\n'+markdown(paired)+'\n\n## G2已有格子：探索性\n\n'+markdown(oldcells)+'\n\n## G2新增集中格子：确认部分\n\n'+markdown(newcells)+'\n\n'+markdown(pd.DataFrame(correlation))+'\n\n## 固定实现与限制\n\n'
    report+='- 仅本任务，不作图、不安装其他依赖、不写特征数组、不读restricted、不上传。neuroCombat0.2.12。CC200/HO缓存只读；平均格逐人预测引用T13-B/T12-C，不重新训练。ABIDE没有BNU观察被试，no_watch不适用。\n'
    report+='- ComBat-EB沿用旧零检验，四组均值均0。任务修订提供的原因：EB对站点位置gamma的收缩留下与类别相关的站点残留，几千维累积后被LR利用；这是方法性质，非实现错误。该原因来自一次最小模拟，不是本轮新增诊断实验。noEB同噪声、种子、维数，只改eb=False；HO双向.344638未通过，因此两种ComBat正式数据都不生成反向预测，也不报告反向AUC，仅报告测试站点内AUC。\n'
    report+='- ComBat仅外层训练拟合，保留诊断类别协变量，parametric=True/mean_only=False/ref_batch=None。其输出调用原C0四C/诊断分层五折LR；内层不重拟合ComBat，外层训练中的内层验证者及其诊断协变量已进入调和，是任务规定的非完全嵌套简化。测试原特征不调和，只使用训练Scaler及LR。\n'
    report+='- 集中C0/L3直接调用T13-B包装及T12-C/T13-A实现；L3五C、站点×诊断五折、配对验证AUC。L1先排除所有纯类训练站点，再用实际剩余N拟合Scaler/优化器/内层；直接调用T13-A无惩罚站点截距L1，预测不加站点截距，选参同L3。G4使用逐重复绝对差均值，不是两均值差的绝对值。\n'
    report+='- 集中站点按default_rng([repeat,k,5])先抽排序ASD站点再抽TD站点，单双向ASD名单相同；被试为T12-C b10前缀。G1两分配使用同一公共反向集，G2始终为各格原生反向集。L是在任何结果前保存，36格Spearman每图谱18格；其中14格已有、4格新增，不把整个G2当作独立确认性证据。\n'
    report+='- L1/L3的AUC用decision score，LR用原概率；男性SEX=1、每诊断至少3人的站点。重复范围为2.5%–97.5%变异范围，不是置信区间；正比例仅差值>1e−12，均值和G阈值不变。C2只报告测试AUC。未新增显著性检验、未自动修改论文或启动ABIDE-II。\n\n验证：'+json.dumps(verification,ensure_ascii=False)+'\n\n计时：'+(OUT/'timing.json').read_text()+'\n'
    (OUT/'REPORT.md').write_text(report);cfg['stage']='complete';(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('REPORT_COMPLETE '+json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':report()
