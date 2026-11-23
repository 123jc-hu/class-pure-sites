"""T12-C2：固定桥接实验的等人数同类增补对照。"""
from pathlib import Path
import hashlib,json
from time import perf_counter
from datetime import datetime

import numpy as np
import pandas as pd

import bridge_design as c

SOURCE=Path('outputs/T12C');OUT=Path('outputs/T12C2')
KEY=['k','b','direction','repeat']


def source_plans():
    return [j for j in map(json.loads,(SOURCE/'splits.jsonl').read_text().splitlines()) if j['method']=='C0' and j['b'] in [2,5]]


def plans(table=None,work=None):
    table=c.subjects() if table is None else table;meta=table.set_index('FILE_ID');jobs=[];feasibility=[];orders={}
    for old in source_plans() if work is None else work:
        r,k=old['repeat'],old['k']
        if (r,k) not in orders:
            rng=np.random.default_rng([r,k,4]);labels=meta.loc[old['pure_train']].groupby('site').label.first()
            orders[r,k]={s:rng.permutation(table.loc[table.site.eq(s)&table.label.eq(labels[s])&~table.FILE_ID.isin(old['pure_train']),'FILE_ID'].to_numpy(copy=True)).tolist() for s in sorted(old['asd_sites']+old['td_sites'])}
        sites=sorted(old['td_sites']+(old['asd_sites'] if old['direction']=='twoway' else []))
        counts={s:len(orders[r,k][s]) for s in sites};feasible=all(n>=old['b'] for n in counts.values())
        feasibility.append({key:old[key] for key in KEY}|dict(feasible=feasible,C0_job_id=old['job_id'],available_by_site=json.dumps(counts),shortage_sites=';'.join(s for s,n in counts.items() if n<old['b'])))
        if not feasible:continue
        added=[v for s in sites for v in orders[r,k][s][:old['b']]]
        jobs.append(old|dict(job_id=old['job_id'].replace('C0_','S0_',1),method='S0',C0_job_id=old['job_id'],addition_seed=[r,k,4],cross_class_bridge=old['bridge'],bridge=added,train=old['pure_train']+added,n_added=len(added),n_train=len(old['pure_train'])+len(added)))
    return jobs,feasibility


def fit_job(job,data=None):
    # C0分支就是所需的普通LR；只改输出标记，不复制训练或选参代码。
    result=c.fit_job(job|dict(method='C0'),data=data)
    for name in ['pred','diagnostics']:
        result[name]['method']='S0'
    result['done']['method']='S0'
    if len(result['warnings']):result['warnings']['method']='S0'
    for row in result['splits']:
        row['method']='S0';row['train_added']=row.pop('train_bridge');row['validation_added']=row.pop('validation_bridge')
    result['pred']=result['pred'].rename(columns={'is_bridge':'is_same_class_added'})
    return result


def paired_summary(runs):
    paired=[]
    for (k,b,direction),f in runs.groupby(['k','b','direction']):
        for metric in ['within_site_AUC','pooled_AUC','male_within_site_AUC','reverse_AUC']:
            name='pooled_AUC' if metric=='reverse_AUC' else metric;v=f.loc[f.reverse.eq(metric=='reverse_AUC')].pivot(index='repeat',columns='method',values=name)
            assert not v[['C0','S0','B0']].isna().any().any()
            for left,right in [('C0','S0'),('S0','B0')]:
                paired.extend(dict(k=k,b=b,direction=direction,metric=metric,contrast=left+'-'+right,repeat=int(r),difference=float(d)) for r,d in (v[left]-v[right]).items())
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'paired_repeat_differences.csv',index=False)
    diff=paired.groupby(['k','b','direction','metric','contrast']).difference.agg(mean='mean',range_low=lambda x:x.quantile(.025),range_high=lambda x:x.quantile(.975),positive_fraction=lambda x:x.gt(1e-12).mean(),repeats='count').reset_index();diff.to_csv(OUT/'paired_differences.csv',index=False)
    return diff


def parallel(jobs):
    from joblib import Parallel,delayed,parallel_config
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for result in Parallel(n_jobs=5,return_as='generator_unordered')(delayed(fit_job)(j) for j in jobs):
            c.save_result(result);print('TRAIN_DONE '+json.dumps(result['done']),flush=True)


def timing():
    import shutil,sklearn,scipy,sys
    OUT.mkdir();c.OUT=OUT;jobs,feasibility=plans();table=c.subjects();table.to_csv(OUT/'subjects.csv',index=False)
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs));pd.DataFrame(feasibility).to_csv(OUT/'feasibility.csv',index=False)
    counts=pd.DataFrame(feasibility).groupby(['k','b','direction']).feasible.agg(available='sum',candidate='count').reset_index();counts['skipped']=counts.candidate-counts.available;counts.to_csv(OUT/'feasible_repeats.csv',index=False)
    additions=[]
    for j in jobs:
        f=table.set_index('FILE_ID',drop=False).loc[j['bridge']].copy()
        for key in KEY+['seed','job_id']:f[key]=j[key]
        f['is_same_class_added']=True;additions.append(f)
    pd.concat(additions,ignore_index=True).to_csv(OUT/'added_subjects.csv',index=False)
    sources=['same_class_control.py','tests/test_same_class_control.py','bridge_design.py','abide_class_pure_sites.py','docs/tasks/T12C2_same_class_control.md','AGENTS.md']
    inputs=[str(c.a.CACHE/p) for p in ['fisher_z.npy','row_order.csv']]+[str(SOURCE/p) for p in ['subjects.csv','splits.jsonl','predictions.csv','completed.csv','repeat_metrics.csv','reference_repeat_metrics.csv','baseline_matched_reverse_metrics.csv']]
    c.dump('config.json',dict(task='T12C2',stage='prepared',started_at=datetime.now().astimezone().isoformat(),expected_jobs=len(jobs),seeds=list(range(50)),addition_seed='default_rng([repeat,k,4]);sorted training sites;unused same-class subjects in T12A subjects.csv order;prefix b',k=[2,4],b=[2,5],directions=['oneway','twoway'],C=c.a.GRID,inner='subject-stratified5fold shuffle seed=repeat;train-only StandardScaler;ordinary T12C C0 LR',LR='L2,lbfgs,max_iter100,tol1e-4,no class weights',workers=5,CPU_process_limit=6,BLAS_threads=1,reference_results='C0 and B0 read only;only feasible repeats;identical T12C condition-specific test/reverse IDs',range='2.5/97.5 repeat percentiles;variability range,not confidence interval',positive_fraction_tolerance=1e-12,BNU_watch_not_applicable=True,environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__),source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},input_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in inputs},splits_sha256=hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()))
    shutil.copyfile('/tmp/T12C2_pytest_red.log',OUT/'pytest_red.log')
    print('START T12C2 timing',flush=True);tick=perf_counter()
    selected=[next(j for j in jobs if j['k']==k and j['b']==b and j['direction']==direction) for k,b,direction in [(2,2,'oneway'),(2,5,'oneway'),(2,5,'twoway'),(4,5,'oneway'),(4,5,'twoway')]]
    parallel(selected);done=pd.read_csv(OUT/'completed.csv');point=float((done.seconds/done.n_train).mean()*sum(j['n_train'] for j in jobs)/5+30)
    value=dict(pilot_jobs=len(selected),pilot_wall_seconds=perf_counter()-tick,point_minutes=point/60,with_25percent_margin_minutes=point*1.25/60,pilot_results_reused=True)
    c.dump('timing.json',value);print('TIMING_COMPLETE '+json.dumps(value),flush=True)


def run():
    c.OUT=OUT;cfg=json.loads((OUT/'config.json').read_text());cfg['stage']='running';c.dump('config.json',cfg)
    jobs=c.read_plans();done=set(pd.read_csv(OUT/'completed.csv').job_id);remaining=[j for j in jobs if j['job_id'] not in done]
    print('START T12C2 remaining_jobs='+str(len(remaining)),flush=True);tick=perf_counter();parallel(remaining)
    cfg.update(stage='training_complete',finished_at=datetime.now().astimezone().isoformat(),remaining_run_seconds=perf_counter()-tick);c.dump('config.json',cfg);print('ALL_TRAINING_COMPLETE jobs='+str(len(jobs)),flush=True)


def summarize():
    from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,balanced_accuracy_score
    from abide_class_pure_sites_report import markdown
    c.OUT=OUT;jobs=c.read_plans();pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');done=pd.read_csv(OUT/'completed.csv',float_precision='round_trip');cfg=json.loads((OUT/'config.json').read_text())
    for section in ['source_sha256','input_sha256']:
        for path,h in cfg[section].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    assert hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()==cfg['splits_sha256']
    assert len(done)==len(jobs) and set(done.job_id)=={j['job_id'] for j in jobs} and not done.job_id.duplicated().any()
    assert len(pred)==sum(len(j['test'])+len(j['reverse']) for j in jobs) and not pred.duplicated(['job_id','FILE_ID']).any()
    lookup={j['job_id']:j for j in jobs};meta=c.subjects().set_index('FILE_ID',drop=False);rows=[];sites=[]
    old_pred=pd.read_csv(SOURCE/'predictions.csv',usecols=['job_id','FILE_ID','reverse'],dtype={'job_id':str});old_ids={key:set(f.FILE_ID) for key,f in old_pred.groupby(['job_id','reverse'])}
    for job_id,f in pred.groupby('job_id',sort=False):
        j=lookup[job_id];assert set(j['train']).isdisjoint(f.FILE_ID)
        for col in c.a.META:pd.testing.assert_series_equal(f[col].reset_index(drop=True),meta.loc[f.FILE_ID,col].reset_index(drop=True),check_names=False,check_dtype=False)
        for reverse,g in f.groupby('reverse'):
            assert set(g.FILE_ID)==set(j['reverse'] if reverse else j['test'])==old_ids[j['C0_job_id'],reverse]
            row={key:j[key] for key in c.CONTEXT}|dict(reverse=reverse,n_evaluated=len(g),pooled_AUC=roc_auc_score(g.label,g.probability),Acc=accuracy_score(g.label,g.prediction),Sens=recall_score(g.label,g.prediction),Spec=recall_score(g.label,g.prediction,pos_label=0),BA=balanced_accuracy_score(g.label,g.prediction))
            if not reverse:
                values=[];male=[]
                for site,s in g.groupby('site'):
                    v=roc_auc_score(s.label,s.probability);m=s.loc[s.SEX.eq(1)];skip=m.label.value_counts().reindex([0,1],fill_value=0).min()<3
                    mv=np.nan if skip else roc_auc_score(m.label,m.probability);values.append(v);male.extend([] if skip else [mv]);sites.append({key:j[key] for key in KEY}|dict(site=site,AUC=v,male_AUC=mv,male_skipped=skip))
                assert len(values)==5;row.update(within_site_AUC=np.mean(values),male_within_site_AUC=np.mean(male))
            rows.append(row)
    source_ids={j['C0_job_id'] for j in jobs};ref=pd.read_csv(SOURCE/'repeat_metrics.csv',float_precision='round_trip');ref=ref.loc[ref.job_id.isin(source_ids)];assert len(ref)==2*len(jobs)
    baseline_test=pd.read_csv(SOURCE/'reference_repeat_metrics.csv',float_precision='round_trip').query('method=="B0"').set_index(['k','repeat'])
    baseline_reverse=pd.read_csv(SOURCE/'baseline_matched_reverse_metrics.csv',float_precision='round_trip').set_index(KEY)
    baseline=[]
    for j in jobs:
        context={key:j[key] for key in KEY}|dict(method='B0',n_train=80,n_added=0,job_id=j['job_id'].replace('S0_','B0_',1))
        baseline.append(baseline_test.loc[(j['k'],j['repeat']),:].to_dict()|context|dict(reverse=False))
        q=baseline_reverse.loc[tuple(j[key] for key in KEY)];baseline.append(context|dict(reverse=True,n_evaluated=q.n_evaluated,pooled_AUC=q.reverse_AUC))
    runs=pd.concat([pd.DataFrame(rows),ref,pd.DataFrame(baseline)],ignore_index=True);runs.to_csv(OUT/'repeat_metrics.csv',index=False);pd.DataFrame(sites).to_csv(OUT/'site_metrics.csv',index=False)
    metrics=['within_site_AUC','pooled_AUC','male_within_site_AUC','Acc','Sens','Spec','BA'];group=['method','k','b','direction','reverse']
    long=runs.melt(id_vars=group+['repeat'],value_vars=metrics,var_name='metric',value_name='value').dropna(subset=['value'])
    summary=long.groupby(group+['metric']).value.agg(mean='mean',range_low=lambda x:x.quantile(.025),range_high=lambda x:x.quantile(.975),repeats='count').reset_index();summary.to_csv(OUT/'main_table.csv',index=False)
    diff=paired_summary(runs)
    criteria=[]
    for k in [2,4]:
        for name,direction in [('D1','twoway'),('D2','oneway')]:
            d=diff.loc[diff.k.eq(k)&diff.b.eq(5)&diff.direction.eq(direction)&diff.contrast.eq('C0-S0')].set_index('metric');test=d.loc['within_site_AUC'];rev=d.loc['reverse_AUC']
            criteria.append(dict(criterion=name,k=k,b=5,direction=direction,role='primary' if k==2 else 'replication',repeats=int(test.repeats),test_difference=test['mean'],test_range_low=test.range_low,test_range_high=test.range_high,reverse_difference=rev['mean'],reverse_range_low=rev.range_low,reverse_range_high=rev.range_high,passed=bool(test['mean']>0 and rev['mean']>0)))
    criteria=pd.DataFrame(criteria);criteria.to_csv(OUT/'criteria.csv',index=False)
    inner=[json.loads(line) for line in (OUT/'inner_splits.jsonl').read_text().splitlines()];assert len(inner)==5*len(jobs)
    for s in inner:
        j=lookup[s['job_id']];tr=set(s['train']);va=set(s['validation']);assert tr.isdisjoint(va) and tr|va==set(j['train']) and not (tr|va)&set(j['test']+j['reverse']) and s['standardizer_fit']==s['train'] and not s['additive_fit']
    diagnostics=pd.read_csv(OUT/'selection_fits.csv',float_precision='round_trip');assert len(diagnostics)==20*len(jobs)
    for q in done.itertuples():
        values=diagnostics.loc[diagnostics.job_id.eq(q.job_id)].pivot(index='inner_fold',columns='C',values='AUC').reindex(columns=c.a.GRID);assert q.C==c.a.GRID[int(np.argmax(values.to_numpy().mean(axis=0)))]
    verification=dict(jobs=len(jobs),LR_fits=21*len(jobs),predictions=len(pred),inner_splits=len(inner),selection_fits=len(diagnostics),warnings=int(done.warning_count.sum()),reference_hashes_unchanged=True,identical_C0_S0_evaluation_IDs=True,train_validation_test_isolation=True)
    c.dump('verification.json',verification);done.to_csv(OUT/'selected_parameters.csv',index=False)
    primary=summary.loc[(~summary.reverse&summary.metric.isin(metrics[:3]))|(summary.reverse&summary.metric.eq('pooled_AUC'))]
    interpretation='D1满足：按预设支持跨类桥接的收益超出等人数同类增补。' if criteria.loc[criteria.criterion.eq('D1')&criteria.k.eq(2),'passed'].iloc[0] else 'D1不满足：按预设不能把桥接设计作为独立结论，回报讨论；该点判据不支持排除人数解释。'
    text='# T12-C2 桥接的等人数同类对照\n\n'+interpretation+'\n\n## 判读\n\n'+markdown(criteria)+'\n\n所有范围为可行重复的2.5%–97.5%变异范围，**不是置信区间**。D1/D2严格按预设点均值判读，范围跨零不改变该判据，也不声称统计显著。\n\n## 每格可行重复\n\n'+markdown(pd.read_csv(OUT/'feasible_repeats.csv'))+'\n\n## 相同重复上的主表\n\n'+markdown(primary)+'\n\n## 配对差值\n\n'+markdown(diff)+'\n\n'
    text+='## 固定口径与限制\n\n- T12-A缓存特征、T12-C划分与真实桥接名单只读；C0、B0不重训，只引用同一格可行重复。S0新增同类人员从P未用者抽取，default_rng([repeat,k,4])、站点排序、候选保留subjects.csv顺序，每站只排列一次，b2为b5前缀。不足人数直接跳过，不替换原P。主要格可用重复明显少于50，结论仅适用于这些可行划分，不与T12-C完整50重复均值混比。\n'
    text+='- S0通过薄适配调用bridge_design.fit_job的C0普通LR分支，再将结果标记为S0；保存也复用bridge_design.save_result，未改任何旧代码。四C、五折、种子与训练专用Scaler沿T12-C，未重采原划分或改超参数。训练、验证和评估被试隔离，所有补采者只在训练内。\n'
    text+='- S0和C0逐格逐人使用同一测试集及T12-C真实桥接者扣除后的反向集；B0反向分数直接引用T12-C按该集合计算的结果。不同b/方向/可行重复的评估人群不同，只在格内配对。男性SEX=1，沿T12-C逐站点男女类别计数规则（男性每诊断至少3人），五站点AUC等权；合并AUC及Acc/Sens/Spec/BA另存主表。\n'
    text+='- 正差比例沿用已确认的1e-12数值判等，均值/分位范围/判读不修改。ABIDE不含BNU四名观察被试，因此不额外构造no_watch。只跑S0普通LR，不跑C2、不作图、不增加方法或新依赖。缺少同类余量导致的可行性筛选可能限制推广。\n'
    text+='- 唯一测试先失败后通过；5个正式计时任务复用，5 CPU工作进程加父进程、BLAS单线程。计时和实际运行见timing.json/config.json，完整核对：'+json.dumps(verification,ensure_ascii=False)+'。\n'
    (OUT/'REPORT.md').write_text(text);cfg.update(stage='complete',summarized_at=datetime.now().astimezone().isoformat());c.dump('config.json',cfg);print('REPORT_COMPLETE '+interpretation,flush=True)


if __name__=='__main__':
    import sys
    {'timing':timing,'run':run,'summarize':summarize}[sys.argv[1]]()
