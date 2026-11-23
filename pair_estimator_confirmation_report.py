"""T13-B：冻结参照、逐重复配对，以及E1/E2确认判据。"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

import pair_estimator_confirmation as b
import bridge_design_report as cr
from abide_class_pure_sites_report import markdown

OUT=b.OUT;METRICS=['within_site_AUC','male_within_site_AUC','pooled_AUC','reverse_AUC']


def references():
    inputs={name:pd.read_csv(f'outputs/{name}/predictions.csv',float_precision='round_trip',low_memory=False) for name in ['T12A','T12C','T12D/V-HO','T13A']}
    groups={name:{key:g for key,g in f.groupby('job_id')} for name,f in inputs.items()}
    completed={name:pd.read_csv(f'outputs/{name}/completed.csv',float_precision='round_trip').set_index('job_id') for name in inputs};parts=[];choices=[]
    def add(j,method,source,source_job,test_only=False):
        target=j['test'] if test_only else j['test']+j['reverse'];f=groups[source][source_job].copy();f=f.loc[f.FILE_ID.isin(target)];assert set(f.FILE_ID)==set(target)
        context={key:j[key] for key in b.KEYS};context.update(method=method,model=method,job_id=j['job_id'].replace('_L3_','_'+method+'_'),n_train=80 if method=='B0' else j['n_train'],n_added=0 if method=='B0' else j['n_added'],source_job_id=source_job)
        for key,value in context.items():f[key]=value
        if method!='L3':f['score']=f.probability
        f['source_file']=f'outputs/{source}/predictions.csv';parts.append(f[b.m.a.META+b.KEYS+['reverse','score','probability','prediction','source_file']]);choices.append(context|dict(C=float(completed[source].loc[source_job,'C']),reference=True,source_file=f'outputs/{source}/completed.csv'))
    for j in b.plans():
        if j['method']!='L3':continue
        if j['reuse_T13A']:add(j,'L3','T13A',f'secondary_L3_r{j["repeat"]}_{j["direction"]}')
        if j['feature']=='CC200':
            add(j,'C0','T12C',j['source_job_id']);add(j,'C2','T12C',j['source_job_id'].replace('C0_','C2_',1),True)
        add(j,'B0','T12A' if j['feature']=='CC200' else 'T12D/V-HO',j['T12A_job_id'])
    result=pd.concat(parts,ignore_index=True);assert result.job_id.nunique()==2500 and not result.loc[result.method.eq('C2'),'reverse'].any();result.to_csv(OUT/'reference_predictions.csv',index=False)
    done=pd.read_csv(OUT/'completed.csv',float_precision='round_trip');done['reference']=False;done['source_file']='outputs/T13B/completed.csv';choice=pd.concat([done,pd.DataFrame(choices)],ignore_index=True);assert len(choice)==4200 and choice.job_id.is_unique;choice.to_csv(OUT/'selected_C.csv',index=False);return result


def verify(pred):
    cfg=json.loads((OUT/'config.json').read_text());assert cfg['stage']=='training_complete' and (OUT/'exit_code.txt').read_text().strip()=='0'
    for section in ['source_sha256','input_sha256']:
        for path,h in cfg[section].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    assert hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()==cfg['splits_sha256']
    jobs={j['job_id']:j for j in b.plans() if not j['reuse_T13A']};done=pd.read_csv(OUT/'completed.csv',float_precision='round_trip');assert len(done)==1700 and set(done.job_id)==set(jobs) and not done.job_id.duplicated().any()
    assert done.fit_count.sum()==35200 and not pred.duplicated(['job_id','FILE_ID']).any() and pred.job_id.nunique()==1700 and np.isfinite(pred.score).all()
    meta=b.m.subjects().set_index('FILE_ID')
    for key,f in pred.groupby('job_id'):
        j=jobs[key];assert set(f.FILE_ID)==set(j['test']+j['reverse']) and not set(f.FILE_ID)&set(j['train']);assert set(f.loc[~f.reverse,'FILE_ID'])==set(j['test'])
        for col in b.m.a.META[1:]:np.testing.assert_array_equal(f[col],meta.loc[f.FILE_ID,col])
    pre=pd.read_csv(OUT/'pair_preflight.csv');assert len(pre)==4800 and pre.validation_pairs.gt(0).all()
    fits=pd.read_csv(OUT/'selection_fits.csv',float_precision='round_trip');expected_rows=34600;assert len(fits)==expected_rows
    for q in done.itertuples():
        f=fits.loc[fits.job_id.eq(q.job_id)];grid=b.m.GRID if q.method=='L3' else b.m.a.GRID;selection=f.loc[f.inner_fold.ge(0)].pivot(index='inner_fold',columns='C',values='AUC').reindex(columns=grid)
        assert selection.shape==(min(5,q.b) if q.method=='L3' else 5,len(grid)) and q.C==grid[int(np.argmax(selection.to_numpy().mean(axis=0)))]
        if q.method=='L3':assert f.success.all() and f.loc[f.inner_fold.ge(0),'validation_pairs'].gt(0).all()
    splits=[json.loads(line) for line in (OUT/'inner_splits.jsonl').read_text().splitlines()];expected_splits=7300;assert len(splits)==expected_splits
    for s in splits:
        j=jobs[s['job_id']];tr=set(s['train']);va=set(s['validation']);assert tr.isdisjoint(va) and tr|va==set(j['train']) and not (tr|va)&set(j['test']+j['reverse']) and s['standardizer_fit']==s['train']
    result=dict(new_jobs=len(done),reused_T13A_jobs=100,total_L3_HO_C0_jobs=1800,new_fits=int(done.fit_count.sum()),predictions=len(pred),all_validation_folds_have_pairs=True,preflight_folds=len(pre),new_inner_splits=len(splits),input_sources_cache_unchanged=True,all_C_selections_and_memberships_verified=True,warnings=int(done.warning_count.sum()))
    b.dump('verification.json',result);return result


def report():
    pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');verification=verify(pred);ref=references();combined=pd.concat([pred,ref],ignore_index=True);runs=[];summaries=[];sites=[]
    for feature,f in combined.groupby('feature'):
        cr.OUT=OUT;r,s=cr.method_metrics(f.drop(columns='probability').rename(columns={'score':'probability'}))
        r.loc[r.method.eq('C2'),['Acc','BA','Sens','Spec']]=np.nan;s=s.loc[~(s.method.eq('C2')&~s.metric.str.endswith('AUC'))]
        r['feature']=feature;s['feature']=feature;site=pd.read_csv(OUT/'site_metrics.csv');site['feature']=feature;runs.append(r);summaries.append(s);sites.append(site)
    runs=pd.concat(runs,ignore_index=True);summary=pd.concat(summaries,ignore_index=True);runs.to_csv(OUT/'repeat_metrics.csv',index=False);summary.to_csv(OUT/'main_table.csv',index=False);pd.concat(sites,ignore_index=True).to_csv(OUT/'site_metrics.csv',index=False)
    raw=[]
    for (feature,k,bsize,direction),f in runs.groupby(['feature','k','b','direction']):
        for metric in METRICS:
            reverse=metric=='reverse_AUC';name='pooled_AUC' if reverse else metric;v=f.loc[f.reverse.eq(reverse)].pivot(index='repeat',columns='method',values=name)
            for reference in ['C0','B0']:
                raw.extend(dict(feature=feature,k=k,b=bsize,direction=direction,contrast='L3-'+reference,metric=metric,repeat=int(r),difference=float(value)) for r,value in (v.L3-v[reference]).items())
    raw=pd.DataFrame(raw);assert len(raw)==9600 and raw.difference.notna().all();raw.to_csv(OUT/'paired_repeat_differences.csv',index=False)
    paired=raw.groupby(['feature','k','b','direction','contrast','metric']).difference.agg(mean='mean',range_low=lambda s:s.quantile(.025),range_high=lambda s:s.quantile(.975),positive_fraction=lambda s:s.gt(1e-12).mean(),repeats='count').reset_index();assert len(paired)==192 and paired.repeats.eq(50).all();paired.to_csv(OUT/'paired_differences.csv',index=False)
    rows=[];decisions=[]
    for feature in ['CC200','HO']:
        for k in [2,4]:
            for direction in ['oneway','twoway']:
                q=summary.loc[summary.feature.eq(feature)&summary.method.eq('L3')&summary.k.eq(k)&summary.b.eq(5)&summary.direction.eq(direction)&summary.reverse&summary.metric.eq('pooled_AUC')].iloc[0]
                d=paired.loc[paired.feature.eq(feature)&paired.k.eq(k)&paired.b.eq(5)&paired.direction.eq(direction)&paired.contrast.eq('L3-C0')&paired.metric.eq('reverse_AUC')].iloc[0]
                rows.append(dict(feature=feature,criterion='E1',k=k,b=5,direction=direction,L3_reverse_AUC=float(q['mean']),difference=float(d['mean']),passed=bool(q['mean']>=.5 and d['mean']>0)))
            d=paired.loc[paired.feature.eq(feature)&paired.k.eq(k)&paired.b.eq(5)&paired.direction.eq('twoway')&paired.contrast.eq('L3-C0')&paired.metric.eq('within_site_AUC')].iloc[0];rows.append(dict(feature=feature,criterion='E2',k=k,b=5,direction='twoway',L3_reverse_AUC=None,difference=float(d['mean']),passed=bool(d['mean']>=-.01)))
        e1=all(q['passed'] for q in rows if q['feature']==feature and q['criterion']=='E1');e2=all(q['passed'] for q in rows if q['feature']==feature and q['criterion']=='E2');decisions.append(dict(feature=feature,E1=e1,E2=e2))
    criteria=pd.DataFrame(rows);criteria.to_csv(OUT/'criteria.csv',index=False);confirmed=all(d['E1'] and d['E2'] for d in decisions);decision='两种特征均满足E1/E2：L3确认可作为桥接设计配套估计方法写进论文；本任务不修改论文。' if confirmed else '至少一种特征E1不满足：T13-A的探索性发现未全面确认，回报讨论。' if not all(d['E1'] for d in decisions) else '两种特征E1满足，但至少一种E2不满足：不依赖站点以准确率为代价，回报讨论。';b.dump('decision.json',dict(by_feature=decisions,confirmed=confirmed,decision=decision))
    pre=pd.read_csv(OUT/'pair_preflight.csv');counts=pre.groupby(['feature','k','b','direction','repeat']).training_pairs.first().reset_index();counts.to_csv(OUT/'pair_counts.csv',index=False);counts.groupby(['feature','k','b','direction']).training_pairs.agg(mean='mean',minimum='min',maximum='max').reset_index().to_csv(OUT/'pair_count_summary.csv',index=False)
    main=runs.loc[~runs.reverse].groupby(['feature','method','k','b','direction'])[cr.METRICS].mean().reset_index();reverse=runs.loc[runs.reverse].groupby(['feature','method','k','b','direction']).pooled_AUC.mean().rename('reverse_AUC').reset_index();main=main.merge(reverse,on=['feature','method','k','b','direction'],how='left');main.to_csv(OUT/'mean_results.csv',index=False)
    text='# T13-B 桥接设计下配对估计确认\n\n'+decision+'\n\n## E1/E2判读\n\n'+markdown(criteria)+'\n\n## 主表（50重复均值）\n\n'+markdown(main)+'\n\n## 配对差值\n\n'+markdown(paired)+'\n\n## 配对数\n\n'+markdown(pd.read_csv(OUT/'pair_count_summary.csv'))+'\n\n## 固定口径与限制\n\n'
    text+='- 缓存只读，不重新提取特征。CC200/HO完全共用T12-C训练、桥接、测试和剩余反向集名单；两种图谱都保留全部1035人。男性SEX=1，沿既有每诊断至少3人的测试站点规则。ABIDE没有BNU四名观察被试，no_watch不适用。\n'
    text+='- L3直接调用T13-A的fit_job/fit_model/fit_weight，目标、标准化及优化器均不变；仅为旧函数增加cache/n_splits可选参数，默认保持原值。内层按站点×诊断分层，min(5,b)折；b=2用2折只适用于L3。配对层面验证AUC选五C，五C相等时取列表首项。所有4800个预检验证折均有配对，未跳过重复或折。\n'
    text+='- HO-C0直接调用T12-C的fit_job，只切换到HO缓存，仍为原诊断分层5折、四C及验证被试AUC选参；未改成L3的网格或指标。CC200的C0、C2、B0和L3已有两格均引用冻结预测；HO-B0引用T12-D的V-HO，不重跑。B0反向集按当前桥接名单删人，与L3/C0完全对应。\n'
    text+='- L3 AUC用原始decision score，分类阈值0；C0/B0/C2用其原始概率，阈值.5。范围为50个依赖重复的2.5%–97.5%变异范围，非置信区间；正比例仅用差值>1e−12判定，均值及E阈值不改。无新增显著性检验或图。C2只列测试指标，不报告其反向AUC。\n'
    text+='- E1/E2只按b=5判读；b=2/10、单向测试AUC及配对数只报告。确认结果不等于新的独立外部样本验证；本轮固定方案是在T13-A探索后预先约定的确认格子。未执行新方法、后续路线或论文修改。\n\n验证：'+json.dumps(verification,ensure_ascii=False)+'\n\n计时：'+(OUT/'timing.json').read_text()+'\n'
    (OUT/'REPORT.md').write_text(text);cfg=json.loads((OUT/'config.json').read_text());cfg['stage']='complete';b.dump('config.json',cfg);print('REPORT_COMPLETE '+decision,flush=True)


if __name__=='__main__':report()
