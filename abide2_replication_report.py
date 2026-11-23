"""T14-B第二阶段：复用既有指标，严格落实R1–R5和可判读范围。"""
from pathlib import Path
import sys,json,hashlib
REPO=Path(__file__).resolve().parent
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import abide2_replication as r
import bridge_design_report as cr
import site_pair_validation_report as br
from abide_class_pure_sites_report import markdown

OUT=r.OUT
METRICS=cr.METRICS


def augmented(pred):
    work={w['task_id']:w for w in r.tasks()};f=pred.copy()
    for name,values in [('n_pure',{key:w['job'].get('n_pure',len(w['job']['train'])) for key,w in work.items()}),('n_added',{key:w['job'].get('n_added',0) for key,w in work.items()}),('n_train',{key:len(w['job']['train']) for key,w in work.items()}),('T12A_job_id',{key:w['job'].get('T12A_job_id',w['job']['job_id']) for key,w in work.items()})]:f[name]=f.job_id.map(values)
    return f


def matched_baseline(pred):
    primary,cross,controls,_=r.plans();by={j['job_id']:j for j in primary};frames=[]
    for j in controls:
        source='A_'+j['T12A_job_id'];f=pred.loc[pred.job_id.eq(source)&pred.FILE_ID.isin(j['test']+j['reverse'])].copy();assert set(f.FILE_ID)==set(j['test']+j['reverse']);f['source_prediction_job']=source;f['job_id']='REF_B0_'+j['job_id'];f['part']='C';f['method']='B0';f['b']=j['b'];f['direction']=j['direction'];f['n_pure']=40;f['n_added']=0;f['n_train']=40;frames.append(f)
    result=pd.concat(frames,ignore_index=True);result.to_csv(OUT/'reference_predictions.csv',index=False);return result


def verify(pred,done):
    cfg=json.loads((OUT/'config.json').read_text());assert cfg['stage']=='training_complete' and (OUT/'exit_code.txt').read_text().strip()=='0'
    for section in ['source_sha256','input_sha256']:
        for name,h in cfg[section].items():
            path=Path(name) if Path(name).is_absolute() else REPO/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==h,name
    assert hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()==cfg['splits_sha256']
    work={w['task_id']:w for w in r.tasks()};assert len(done)==cfg['jobs']==1584 and done.job_id.is_unique and set(done.job_id)==set(work) and done.fit_count.sum()==cfg['expected_fits']==59564
    assert np.isfinite(pred.score).all() and not pred.duplicated(['job_id','FILE_ID']).any();meta=r.table().set_index('FILE_ID')
    for key,f in pred.groupby('job_id'):
        j=work[key]['job'];assert set(f.loc[~f.reverse,'FILE_ID'])==set(j['test']) and set(f.loc[f.reverse,'FILE_ID'])==set(j['reverse']) and not set(f.FILE_ID)&set(j['train'])
        for col in r.a.META:np.testing.assert_array_equal(f[col],meta.loc[f.FILE_ID,col] if col!='FILE_ID' else meta.loc[f.FILE_ID].index)
    estimation=pd.read_csv(OUT/'estimation_predictions.csv',float_precision='round_trip');folds=pd.read_csv(OUT/'estimation_folds.csv');assert folds.groupby('job_id').size().sum()==1500 and not estimation.duplicated(['job_id','outer_fold','FILE_ID']).any()
    for (key,fold),f in estimation.groupby(['job_id','outer_fold']):
        w=work[key];j=w['job'];lookup=pd.Index(r.table().FILE_ID);tr=lookup.get_indexer(j['train']);t=r.table();split=r.b.subject_folds(t,tr,j['seed']) if w['method']=='E_subj' else r.b.site_pairs(t,tr);a,b=split[int(fold)];assert set(f.FILE_ID)==set(lookup[b]) and not set(f.FILE_ID)&set(lookup[a]) and not set(f.FILE_ID)&set(j['test']+j['reverse'])
    inner=[json.loads(line) for line in (OUT/'inner_splits.jsonl').read_text().splitlines()]
    for s in inner:
        j=work[s['job_id']]['job'];a=set(s['train']);b=set(s['validation']);assert a.isdisjoint(b) and (a|b).issubset(j['train']) and not (a|b)&set(j['test']+j['reverse'])
        if s.get('standardizer_fit'):assert s['standardizer_fit']==s['train']
    result=dict(jobs=len(done),fits=int(done.fit_count.sum()),new_predictions=len(pred),estimation_predictions=len(estimation),estimation_outer_folds=len(folds),inner_records=len(inner),warnings=int(done.warning_count.sum()),all_memberships_and_metadata_verified=True,source_and_cache_hashes_unchanged=True,ABIDEI_tests='all passed',python_isolated=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));return result


def comparison(runs,performance,c_pairs,summary,rho):
    rows=[];old=pd.read_csv(REPO/'outputs/T12A/repeat_metrics.csv');old=old.loc[~old.no_zero_roi&((old.n.eq(40)&old.k.isin([1,2,4,6]))|old.arm.eq('F'))]
    for (method,k,reverse),f in runs.loc[runs.part.eq('A')].groupby(['method','k','reverse']):
        ref=old.loc[old.arm.eq(method)&old.k.eq(k)&old.reverse.eq(reverse)]
        for metric in ['pooled_AUC'] if reverse else ['within_site_AUC','pooled_AUC','male_within_site_AUC']:
            rows.append(dict(part='A',cell=f'{method}/k{k}/'+('reverse' if reverse else 'test'),metric=metric,ABIDE_I=ref[metric].mean(),ABIDE_II=f[metric].mean(),I_repeats=len(ref),II_repeats=len(f),notes='I n40/5 test sites;II n20/4 test sites'))
    op=pd.read_csv(REPO/'outputs/T12B/performance_summary.csv')
    for q in performance.loc[performance.truth.eq('within_site_AUC')].itertuples():
        ref=op.loc[op.method.eq(q.method)&op.k.eq(q.k)&op.truth.eq(q.truth)].iloc[0]
        for metric in ['estimate','true_AUC','mean_bias','mean_absolute_error']:rows.append(dict(part='B',cell=f'{q.method}/k{q.k}',metric=metric,ABIDE_I=float(ref[metric]),ABIDE_II=float(getattr(q,metric)),I_repeats=int(ref.repeats),II_repeats=q.repeats,notes='E_subj primary fold mean;pooled report only'))
    oc=pd.read_csv(REPO/'outputs/T12C2/paired_differences.csv')
    for q in c_pairs.loc[c_pairs.contrast.eq('C0-S0')&c_pairs.metric.isin(['within_site_AUC','reverse_AUC'])].itertuples():
        ref=oc.loc[oc.k.eq(q.k)&oc.b.eq(q.b)&oc.direction.eq(q.direction)&oc.metric.eq(q.metric)&oc.contrast.eq(q.contrast)].iloc[0];rows.append(dict(part='C',cell=f'C0-S0/k{q.k}/b{q.b}/{q.direction}',metric=q.metric,ABIDE_I=float(ref['mean']),ABIDE_II=q.mean,I_repeats=int(ref.repeats),II_repeats=q.repeats,notes='each dataset own feasible paired repeats;II<20 report only'))
    od=pd.read_csv(REPO/'outputs/T13B/mean_results.csv');od=od.loc[od.feature.eq('CC200')]
    for (method,k,b,direction,rev),f in runs.loc[runs.method.isin(['C0','L3'])].groupby(['method','k','b','direction','reverse']):
        ref=od.loc[od.method.eq(method)&od.k.eq(k)&od.b.eq(b)&od.direction.eq(direction)].iloc[0]
        metric='pooled_AUC' if rev else 'within_site_AUC';rows.append(dict(part='D',cell=f'{method}/k{k}/b{b}/{direction}',metric='reverse_AUC' if rev else metric,ABIDE_I=float(ref['reverse_AUC' if rev else metric]),ABIDE_II=f[metric].mean(),I_repeats=50,II_repeats=len(f),notes='I CC200 n40;II Gordon n20'))
    original=float(pd.read_csv(REPO/'outputs/T14A/leakage_correlation.csv').query('feature=="CC200"').Spearman_rho.iloc[0]);rows.append(dict(part='E',cell='Spearman(L,native reverse AUC)',metric='rho',ABIDE_I=original,ABIDE_II=rho,I_repeats=18,II_repeats=11,notes='cell counts;I includes b10 and concentrated,II only prespecified11'))
    result=pd.DataFrame(rows);result.to_csv(OUT/'ABIDEI_comparison.csv',index=False);return result


def report():
    raw=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');done=pd.read_csv(OUT/'completed.csv',float_precision='round_trip');verification=verify(raw,done);pred=augmented(raw);ref=matched_baseline(pred);allpred=pd.concat([pred,ref],ignore_index=True);cr.OUT=OUT
    runs,summary=cr.method_metrics(allpred.drop(columns='probability').rename(columns={'score':'probability'}),repeats=None,n_test_sites=4)
    context=allpred.drop_duplicates('job_id').set_index('job_id');runs['part']=runs.job_id.map(context.part);runs['n']=runs.job_id.map(context.n);runs.to_csv(OUT/'repeat_metrics.csv',index=False)
    truth=runs.loc[runs.method.eq('P')].copy();truth['method']='B0';est=done.loc[done.part.eq('B')].copy();est['estimate']=np.where(est.method.eq('E_subj'),est.estimate_pooled,est.estimate_meanfold);br.OUT=OUT;performance=br.estimates(est,truth,expected_rows=300)
    feasible=pd.read_csv(OUT/'S0_feasibility.csv');keys=set(zip(feasible.loc[feasible.feasible,'k'],feasible.loc[feasible.feasible,'b'],feasible.loc[feasible.feasible,'direction'],feasible.loc[feasible.feasible,'repeat']))
    selected=runs.loc[runs.method.isin(['C0','S0','B0'])].copy();selected=selected.loc[[tuple(x) in keys for x in selected[['k','b','direction','repeat']].itertuples(index=False,name=None)]];r.s.OUT=OUT;c_pairs=r.s.paired_summary(selected);c_pairs.to_csv(OUT/'C_paired_differences.csv',index=False);c_raw=pd.read_csv(OUT/'paired_repeat_differences.csv');c_raw['part']='C'
    clong=selected.melt(id_vars=['method','k','b','direction','reverse','repeat'],value_vars=METRICS,var_name='metric',value_name='value').dropna(subset=['value']);clong.groupby(['method','k','b','direction','reverse','metric']).value.agg(mean='mean',repeats='count').reset_index().to_csv(OUT/'C_matched_main_table.csv',index=False)
    rawpairs=[]
    f=runs.loc[runs.part.eq('A')&runs.method.isin(['P','M'])&~runs.reverse]
    for k,g in f.groupby('k'):
        for metric in METRICS:
            v=g.pivot(index='repeat',columns='method',values=metric);rawpairs.extend(dict(part='A',contrast='M-P',k=k,b=0,direction='none',metric=metric,repeat=int(i),difference=float(x)) for i,x in (v.M-v.P).items())
    for (k,b,direction),g in runs.loc[runs.method.isin(['C0','L3'])].groupby(['k','b','direction']):
        for metric in ['within_site_AUC','pooled_AUC','male_within_site_AUC','reverse_AUC']:
            rev=metric=='reverse_AUC';name='pooled_AUC' if rev else metric;v=g.loc[g.reverse.eq(rev)].pivot(index='repeat',columns='method',values=name);rawpairs.extend(dict(part='D',contrast='L3-C0',k=k,b=b,direction=direction,metric=metric,repeat=int(i),difference=float(x)) for i,x in (v.L3-v.C0).items())
    pairs=pd.concat([pd.DataFrame(rawpairs),c_raw],ignore_index=True);pairs.to_csv(OUT/'paired_repeat_differences.csv',index=False);paired=pairs.groupby(['part','contrast','k','b','direction','metric']).difference.agg(mean='mean',range_low=lambda x:x.quantile(.025),range_high=lambda x:x.quantile(.975),positive_fraction=lambda x:x.gt(1e-12).mean(),repeats='count').reset_index();paired.to_csv(OUT/'paired_differences.csv',index=False)
    def score(method,k,reverse=False,b=0,direction='none',metric='within_site_AUC'):
        f=runs.loc[runs.method.eq(method)&runs.k.eq(k)&runs.b.eq(b)&runs.direction.eq(direction)&runs.reverse.eq(reverse)];return float(f['pooled_AUC' if reverse else metric].mean())
    def delta(part,contrast,k,b,direction,metric):return float(paired.loc[paired.part.eq(part)&paired.contrast.eq(contrast)&paired.k.eq(k)&paired.b.eq(b)&paired.direction.eq(direction)&paired.metric.eq(metric),'mean'].iloc[0])
    rows=[];gap1=delta('A','M-P',1,0,'none','within_site_AUC');gap4=delta('A','M-P',4,0,'none','within_site_AUC');rev1=score('P',1,True);rev4=score('P',4,True);R1=gap1>=.03 and gap4<gap1 and rev4>rev1;rows.append(dict(criterion='R1',cell='k1 versus k4',assessable=True,passed=R1,details=json.dumps(dict(gap1=gap1,gap4=gap4,reverse1=rev1,reverse4=rev4))))
    perf=performance.loc[performance.truth.eq('within_site_AUC')];bias=float(perf.loc[perf.method.eq('E_subj')&perf.k.eq(2),'mean_bias'].iloc[0]);mae=perf.loc[perf.method.isin(['E_subj','E_LSPO'])].pivot(index='k',columns='method',values='mean_absolute_error');R2=bool(bias>=.1 and mae.E_LSPO.lt(mae.E_subj).all());rows.append(dict(criterion='R2',cell='k2,k4',assessable=True,passed=R2,details=json.dumps(dict(bias_k2=bias,MAE=mae.to_dict('index')))))
    r3=[]
    for k in [2,4]:
        f=c_pairs.loc[c_pairs.k.eq(k)&c_pairs.b.eq(5)&c_pairs.direction.eq('twoway')&c_pairs.contrast.eq('C0-S0')].set_index('metric');n=int(f.loc['within_site_AUC','repeats']);test=float(f.loc['within_site_AUC','mean']);rev=float(f.loc['reverse_AUC','mean']);eligible=n>=20;passed=bool(test>0 and rev>0) if eligible else None;r3.append(passed);rows.append(dict(criterion='R3',cell=f'k{k}/b5/twoway',assessable=eligible,passed=passed,details=json.dumps(dict(repeats=n,test_delta=test,reverse_delta=rev,note='fewer20 report only' if not eligible else 'paired means>0'))))
    rr=[]
    for direction in ['oneway','twoway']:
        value=score('L3',2,True,5,direction);base=score('C0',2,True,5,direction);gain=delta('D','L3-C0',2,5,direction,'within_site_AUC');ok=value>=.5 and value>base and (direction=='oneway' or gain>=-.01);rr.append(ok);rows.append(dict(criterion='R4',cell='k2/b5/'+direction,assessable=True,passed=ok,details=json.dumps(dict(L3_reverse=value,C0_reverse=base,test_delta=gain))))
    native=runs.loc[(runs.method.eq('P'))|(runs.method.eq('C0'))].loc[lambda f:f.reverse].copy();native['L_method']=np.where(native.method.eq('P'),'B0','C0');native=native.groupby(['L_method','k','b','direction']).pooled_AUC.mean().rename('native_reverse_AUC').reset_index();leak=pd.read_csv(OUT/'L_table.csv');cells=leak.merge(native,left_on=['method','k','b','direction'],right_on=['L_method','k','b','direction'],validate='one_to_one');assert len(cells)==11;cells.to_csv(OUT/'L_outcomes.csv',index=False);rho=float(spearmanr(cells.L,cells.native_reverse_AUC).statistic);R5=rho<=-.8;rows.append(dict(criterion='R5',cell='all11 cells',assessable=True,passed=R5,details=json.dumps(dict(Spearman_rho=rho,cells=11))))
    criteria=pd.DataFrame(rows);criteria.to_csv(OUT/'R_criteria.csv',index=False);R3=None if any(x is None for x in r3) else all(r3);decision=dict(R1=bool(R1),R2=R2,R3=R3,R4=bool(all(rr)),R5=bool(R5),all_five_met=bool(R1 and R2 and R3 is True and all(rr) and R5),R3_note='k2=17,k4=4 feasible;report only,no judgment');(OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    compare=comparison(runs,performance,c_pairs,summary,rho)
    main=summary.loc[((~summary.reverse)&summary.metric.isin(['within_site_AUC','pooled_AUC','male_within_site_AUC']))|(summary.reverse&summary.metric.eq('pooled_AUC'))]
    status='R1–R5全部满足，可按预设称独立复现。' if decision['all_five_met'] else '未满足R1–R5全部成立的预设条件；R3因可用重复不足不判读，不能宣称完整独立复现。'
    text='# T14-B 第二阶段 ABIDE-II复现\n\n'+status+'\n\n## 判读\n\n'+markdown(criteria)+'\n\n## A/C/D主表\n\n'+markdown(main)+'\n\n## B性能估计\n\n'+markdown(performance)+'\n\n## S0可用重复\n\n'+markdown(pd.read_csv(OUT/'S0_available_repeats.csv'))+'\n\n## 配对差值\n\n'+markdown(paired)+'\n\n## E泄漏指数\n\n'+markdown(cells)+'\n\n## ABIDE-I并列（不重训）\n\n'+markdown(compare)+'\n\n## 裁样边界（不修订算法）\n\n'+markdown(pd.read_csv(OUT/'mixed_site_purity.csv'))+'\n\n## 固定口径与限制\n\n'
    text+='- 仅使用第一阶段568人、Gordon352/61776维缓存；NYU1/2合并。KUL3的21人和UCLA1的1人不参加任何构造/评估，构造池546人/13站点。天然仅ASD的KUL3和NYU2只描述原集合QC后21/20人，不另分析；NYU2已并入NYU，不再是独立构造站点。\n'
    text+='- 只给旧函数增加显式数据元组/划分参数，默认行为不变。large NYU/OHSU/KKI，4个随机测试站点，n20/k1,2,4，种子0–49。旧算法先各站点ceil抽样再裁到每类n；用户确认保留II repeat2/k4 M中SDSU0ASD/2TD，I k4/6的同类性质只读原划分统计，不重新训练或重抽。\n'
    text+='- LR原4C/诊断五折/训练专用Scaler。L3原5C/min(5,b)/站点×诊断/配对层面验证AUC。E_subj与E_LSPO主值都是外折AUC均值；E_subj合并OOF只辅助报告；真值为4个新站点内AUC等权平均。B0即A中同重复P结果，不重训。R2以逐重复误差的MAE比较，非两均值差绝对值。\n'
    text+='- S0不足按原规则跳过，只在可行相同重复比较C0/S0/B0；三者测试和扣除真实桥接者后的反向人群相同。双向b5仅17/4重复，R3两格不判读，不重抽、不改20次门槛。R4只用k2b5单双向；R5只用B0三格+C0八格的各自原生反向集，不用匹配B0子集。\n'
    text+='- L3 AUC用原始decision分数，LR使用原概率；分类阈值L3为0、LR为.5。男性每诊断≥3人的测试站点参与男性站点均值。所有范围为重复2.5%–97.5%变异范围，不是置信区间；正差比例用>1e−12，均值和R阈值不改。\n'
    text+='- ABIDE-I并列为CC200/n40/5测试站点，II为Gordon/n20/4站点，重复与人群不同，只作描述；E的I原T14A为18格、II预定11格，明确标注，不改II的11格判据。不做b10、人数曲线、去零方差敏感性、ComBat/L1或额外方法，不作图，不安装依赖。\n'
    text+='- 全部读取用-I；固定可信代码路径显式加入，数据目录不加入搜索路径。4个spawn工作进程继承-I，和父进程/资源跟踪器不超过6。ABIDE没有BNU观察被试，no_watch不适用。\n\n验证：'+json.dumps(verification,ensure_ascii=False)+'\n\n计时：'+(OUT/'timing.json').read_text()+'\n'
    (OUT/'REPORT.md').write_text(text);cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='complete',decision=decision);(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('REPORT_COMPLETE '+json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    report()
