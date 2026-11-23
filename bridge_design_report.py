"""T12-C：匹配剩余反向集、均值之比挽回率、固定R1–R3及成本图。"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,balanced_accuracy_score

import bridge_design as c
from abide_class_pure_sites_report import markdown

OUT=c.OUT;GROUP=['method','k','b','direction','reverse'];METRICS=['within_site_AUC','pooled_AUC','male_within_site_AUC','BA','Acc','Sens','Spec']


def references():
    f=pd.read_csv(c.a.OUT/'repeat_metrics.csv',float_precision='round_trip');f=f.loc[~f.no_zero_roi].copy();parts=[]
    for arm,k,n,method in [('P',2,40,'B0'),('P',4,40,'B0'),('P',2,80,'P80'),('M',2,40,'M'),('M',4,40,'M'),('F',0,0,'F')]:
        g=f.loc[f.arm.eq(arm)&f.k.eq(k)&f.n.eq(n)&~f.reverse].copy();g['method']=method;g['n_train']=160 if method=='P80' else 80
        if method=='F':
            done=pd.read_csv(c.a.OUT/'completed.csv');d=done.loc[done.arm.eq('F')&~done.no_zero_roi].set_index('repeat');g['n_train']=g.repeat.map(d.n_train)
        parts.append(g)
    result=pd.concat(parts,ignore_index=True);assert len(result)==300;result.to_csv(OUT/'reference_repeat_metrics.csv',index=False)
    old=c.a.read_jobs();checks=[]
    for j in c.original_jobs():
        if j['k']!=2:continue
        p=next(q for q in old if not q['no_zero_roi'] and q['arm']=='P' and q['n']==80 and q['k']==2 and q['repeat']==j['repeat']);assert set(p['test'])==set(j['test'])
        checks.append(dict(repeat=j['repeat'],test_subjects_identical=True,training_stations_identical=set(p['asd_sites']+p['td_sites'])==set(j['asd_sites']+j['td_sites']),P40_train=80,P80_train=160,partial_pairing=True))
    pd.DataFrame(checks).to_csv(OUT/'R3_partial_pairing.csv',index=False);return result


def method_metrics(pred,repeats=50,n_test_sites=5):
    rows=[];sites=[]
    for job_id,f in pred.groupby('job_id',sort=False):
        ctx={key:f[key].iloc[0] for key in c.CONTEXT}
        for reverse,g in f.groupby('reverse'):
            row=ctx|dict(reverse=reverse,n_evaluated=len(g),pooled_AUC=roc_auc_score(g.label,g.probability),BA=balanced_accuracy_score(g.label,g.prediction),Acc=accuracy_score(g.label,g.prediction),Sens=recall_score(g.label,g.prediction),Spec=recall_score(g.label,g.prediction,pos_label=0))
            if not reverse:
                values=[];male=[];skipped=[]
                for site,s in g.groupby('site'):
                    v=roc_auc_score(s.label,s.probability);values.append(v);m=s.loc[s.SEX.eq(1)];n1=int(m.label.eq(1).sum());n0=int(m.label.eq(0).sum());skip=min(n1,n0)<3
                    if skip:skipped.append(site)
                    else:male.append(roc_auc_score(m.label,m.probability))
                    sites.append(ctx|dict(site=site,AUC=v,male_AUC=np.nan if skip else male[-1],male_ASD=n1,male_TD=n0,male_skipped=skip))
                assert len(values)==n_test_sites;row.update(within_site_AUC=np.mean(values),male_within_site_AUC=np.mean(male),male_skipped=len(skipped),male_skipped_sites=';'.join(skipped))
            rows.append(row)
    runs=pd.DataFrame(rows);runs.to_csv(OUT/'repeat_metrics.csv',index=False);pd.DataFrame(sites).to_csv(OUT/'site_metrics.csv',index=False)
    long=runs.melt(id_vars=GROUP+['repeat'],value_vars=METRICS,var_name='metric',value_name='value').dropna(subset=['value'])
    summary=long.groupby(GROUP+['metric']).value.agg(mean='mean',range_low=lambda x:x.quantile(.025),range_high=lambda x:x.quantile(.975),repeats='count').reset_index()
    if repeats is not None:assert summary.repeats.eq(repeats).all()
    summary.to_csv(OUT/'main_table.csv',index=False);return runs,summary


def matched_baseline_reverse():
    old=pd.read_csv(c.a.OUT/'predictions.csv',float_precision='round_trip');old=old.loc[~old.no_zero_roi&old.arm.eq('P')&old.n.eq(40)&old.k.isin([2,4])&old.reverse];by={(r,k):g.set_index('FILE_ID',drop=False) for (r,k),g in old.groupby(['repeat','k'])};rows=[]
    for j in c.read_plans():
        if j['method']!='C0':continue
        f=by[j['repeat'],j['k']].loc[j['reverse']];rows.append({k:j[k] for k in ['k','b','direction','repeat']}|dict(method='B0',n_evaluated=len(f),reverse_AUC=roc_auc_score(f.label,f.probability),T12A_job_id=j['T12A_job_id']))
    result=pd.DataFrame(rows);assert len(result)==600;result.to_csv(OUT/'baseline_matched_reverse_metrics.csv',index=False);return result


def paired_and_recovery(runs,ref,baseline):
    raw=[];recovery=[];direction_rows=[]
    for k in [2,4]:
        refs=ref.loc[ref.k.eq(k)].pivot(index='repeat',columns='method',values='within_site_AUC').sort_index()
        for b in [2,5,10]:
            for direction in ['oneway','twoway']:
                f=runs.loc[runs.k.eq(k)&runs.b.eq(b)&runs.direction.eq(direction)]
                for method in ['C0','C2']:
                    s=f.loc[f.method.eq(method)&~f.reverse].sort_values('repeat');base=refs.loc[s.repeat,'B0'].to_numpy();mixed=refs.loc[s.repeat,'M'].to_numpy();numerator=s.within_site_AUC.to_numpy()-base;denominator=mixed-base
                    recovery.append(dict(method=method,k=k,b=b,direction=direction,mean_gain=numerator.mean(),mean_mixed_cost=denominator.mean(),recovery=numerator.mean()/denominator.mean(),n_added=int(s.n_added.iloc[0]),n_train=int(s.n_train.iloc[0])))
                for left,right in [('C2','B0'),('C2','C0'),('C0','B0')]:
                    for metric in ['within_site_AUC','pooled_AUC','male_within_site_AUC','reverse_AUC']:
                        reverse=metric=='reverse_AUC';name='pooled_AUC' if reverse else metric;left_values=f.loc[f.method.eq(left)&f.reverse.eq(reverse)].sort_values('repeat');rep=left_values.repeat
                        if right=='B0':
                            q=baseline.loc[baseline.k.eq(k)&baseline.b.eq(b)&baseline.direction.eq(direction)].set_index('repeat').loc[rep] if reverse else ref.loc[ref.method.eq('B0')&ref.k.eq(k)].set_index('repeat').loc[rep]
                            right_score=q['reverse_AUC' if reverse else metric].to_numpy()
                        else:right_score=f.loc[f.method.eq(right)&f.reverse.eq(reverse)].set_index('repeat').loc[rep,name].to_numpy()
                        values=left_values[name].to_numpy()-right_score
                        raw.extend(dict(contrast=left+'-'+right,k=k,b=b,direction=direction,repeat=int(r),metric=metric,difference=v,partial_pairing=False) for r,v in zip(rep,values))
                if k==2 and b==5 and direction=='oneway':
                    f=f.loc[f.method.eq('C2')&~f.reverse].sort_values('repeat');p80=ref.loc[ref.method.eq('P80')].set_index('repeat').loc[f.repeat]
                    for metric in METRICS[:3]:raw.extend(dict(contrast='C2-P80',k=2,b=5,direction='oneway',repeat=int(r),metric=metric,difference=v,partial_pairing=True) for r,v in zip(f.repeat,f[metric].to_numpy()-p80[metric].to_numpy()))
            f=runs.loc[runs.k.eq(k)&runs.b.eq(b)]
            for method in ['C0','C2']:
                one=f.loc[f.method.eq(method)&f.direction.eq('oneway')&~f.reverse].sort_values('repeat');two=f.loc[f.method.eq(method)&f.direction.eq('twoway')&~f.reverse].set_index('repeat').loc[one.repeat]
                one_rev=f.loc[f.method.eq(method)&f.direction.eq('oneway')&f.reverse].pooled_AUC.mean();two_rev=f.loc[f.method.eq(method)&f.direction.eq('twoway')&f.reverse].pooled_AUC.mean()
                direction_rows.append(dict(method=method,k=k,b=b,oneway_AUC=one.within_site_AUC.mean(),twoway_AUC=two.within_site_AUC.mean(),twoway_minus_oneway=(two.within_site_AUC.to_numpy()-one.within_site_AUC.to_numpy()).mean(),oneway_n_added=int(one.n_added.iloc[0]),twoway_n_added=int(two.n_added.iloc[0]),oneway_reverse_AUC=one_rev,twoway_reverse_AUC=two_rev,reverse_pools_differ=True))
    r=pd.DataFrame(raw);r.to_csv(OUT/'paired_repeat_differences.csv',index=False)
    paired=r.groupby(['contrast','k','b','direction','metric','partial_pairing']).difference.agg(mean='mean',range_low=lambda s:s.quantile(.025),range_high=lambda s:s.quantile(.975),positive_fraction=lambda s:s.gt(1e-12).mean(),repeats='count').reset_index();assert len(paired)==147 and paired.repeats.eq(50).all();paired.to_csv(OUT/'paired_differences.csv',index=False)
    recovery=pd.DataFrame(recovery);recovery.to_csv(OUT/'recovery.csv',index=False);pd.DataFrame(direction_rows).to_csv(OUT/'direction_comparison.csv',index=False);return paired,recovery


def decision(summary,paired,recovery):
    rows=[];flags={}
    for k in [2,4]:
        ratio=float(recovery.loc[recovery.method.eq('C2')&recovery.k.eq(k)&recovery.b.eq(5)&recovery.direction.eq('twoway'),'recovery'].iloc[0]);reverse=float(summary.loc[summary.method.eq('C2')&summary.k.eq(k)&summary.b.eq(5)&summary.direction.eq('twoway')&summary.reverse&summary.metric.eq('pooled_AUC'),'mean'].iloc[0]);r1=ratio>=.5 and reverse>=.4
        rows.append(dict(criterion='R1',k=k,direction='twoway',point=ratio,passed=r1,details=f'b5,recovery={ratio:.6f};reverse AUC={reverse:.6f}'));flags[k,'R1']=r1
        for direction in ['oneway','twoway']:
            f=paired.loc[paired.contrast.eq('C2-C0')&paired.k.eq(k)&paired.direction.eq(direction)&paired.b.isin([2,5])&paired.metric.eq('within_site_AUC')].sort_values('b');r2=bool(f['mean'].gt(0).all());rows.append(dict(criterion='R2',k=k,direction=direction,point=np.nan,passed=r2,details=';'.join(f'b{q.b}:{q.mean:+.6f}' for q in f.itertuples())));flags[k,'R2',direction]=r2
    value=float(paired.loc[paired.contrast.eq('C2-P80')&paired.metric.eq('within_site_AUC'),'mean'].iloc[0]);r3=value>=0;rows.append(dict(criterion='R3',k=2,direction='oneway',point=value,passed=r3,details='b5,10 added versus P80 80 added;test subjects same,training site assignments not controlled;partial pairing'))
    table=pd.DataFrame(rows);table.to_csv(OUT/'criteria.csv',index=False)
    result=dict(R1=flags[2,'R1'],R3=r3,R2={d:flags[2,'R2',d] for d in ['oneway','twoway']},k4_R1=flags[4,'R1'],k4_R2={d:flags[4,'R2',d] for d in ['oneway','twoway']})
    result['decision']='R1/R3均满足，桥接设计符合预设建议；后续推广需另行下达' if result['R1'] and r3 else '少量桥接不满足R1，回报讨论' if not result['R1'] else 'R1满足但R3不满足，成本比较未通过，回报讨论'
    c.dump('decision.json',result);return table,result


def figure(summary,runs,ref):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');colors={('C0','oneway'):'#0072B2',('C0','twoway'):'#56B4E9',('C2','oneway'):'#D55E00',('C2','twoway'):'#009E73'};values=[]
    for (method,direction),color in colors.items():
        s=summary.loc[summary.method.eq(method)&summary.direction.eq(direction)&summary.k.eq(2)&~summary.reverse&summary.metric.eq('within_site_AUC')].sort_values('b');label=method+(' one-way' if direction=='oneway' else ' two-way');axes[0].plot(s.b,s['mean'],color=color,marker='o' if direction=='oneway' else 's',label=label);axes[0].fill_between(s.b,s.range_low,s.range_high,color=color,alpha=.08)
        values.extend(dict(panel='bridge_k2',method=method,direction=direction,k=2,b=q.b,x=q.b,mean=q.mean,range_low=q.range_low,range_high=q.range_high) for q in s.itertuples())
        for k,style in [(2,'-'),(4,'--')]:
            f=runs.loc[runs.method.eq(method)&runs.direction.eq(direction)&runs.k.eq(k)&~runs.reverse].groupby('b').agg(n_train=('n_train','mean'),AUC=('within_site_AUC','mean')).reset_index().sort_values('b');axes[1].plot(f.n_train,f.AUC,color=color,ls=style,marker='o' if direction=='oneway' else 's',label=label if k==2 else None)
            values.extend(dict(panel='cost',method=method,direction=direction,k=k,b=q.b,x=q.n_train,mean=q.AUC) for q in f.itertuples())
    for method,color,style in [('B0','#222222',':'),('M','#666666','-.')]:
        value=ref.loc[ref.method.eq(method)&ref.k.eq(2),'within_site_AUC'].mean();axes[0].axhline(value,color=color,ls=style,label=method);values.append(dict(panel='bridge_k2_reference',method=method,k=2,b=0,x=0,mean=value))
    axes[0].set(xlabel='Bridges per selected site (b); k=2',xticks=[0,2,5,10],xlim=(-.3,10.3));axes[0].legend(frameon=False,fontsize=7,ncol=2)
    for method,marker in [('B0','o'),('M','^'),('P80','D'),('F','*')]:
        for k in ([2,4] if method in ['B0','M'] else [2] if method=='P80' else [0]):
            f=ref.loc[ref.method.eq(method)&ref.k.eq(k)];axes[1].scatter(f.n_train.mean(),f.within_site_AUC.mean(),color='#222222' if k!=4 else '#777777',marker=marker,s=45,label=method if k in [0,2] else None,zorder=4);values.append(dict(panel='cost_reference',method=method,k=k,x=f.n_train.mean(),mean=f.within_site_AUC.mean()))
    axes[1].set_xscale('log');axes[1].set(xlabel='Total training subjects (log scale)',xticks=[80,100,160,400,800]);axes[1].xaxis.set_major_formatter(ScalarFormatter());handles,labels=axes[1].get_legend_handles_labels();handles.extend([Line2D([],[],color='.4',ls='-'),Line2D([],[],color='.4',ls='--')]);labels.extend(['k=2','k=4']);axes[1].legend(handles,labels,frameon=False,ncol=2,fontsize=7,loc='upper left')
    for ax in axes:ax.set(ylabel='Test within-site AUC',ylim=(0,1));ax.axhline(.5,color='.7',ls=':',lw=.8);ax.spines[['top','right']].set_visible(False)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/('bridge_design.'+ext),dpi=200)
    plt.close(fig);pd.DataFrame(values).to_csv(OUT/'figure_values.csv',index=False)
    (OUT/'figure_caption.md').write_text('左k2：4条桥接曲线仅含b2/5/10；b0只引用B0，不调用C2。B0/M水平线是T12-A的50重复均值，色带为重复2.5%–97.5%变异范围，不是置信区间。右：横轴总训练人数用对数刻度，全部k2实线/k4虚线桥接方案以及B0/M/P80/F参照，F横坐标为50重复实际训练人数均值。P80与原P40训练站点未保持固定、训练名单非嵌套，只能部分配对；成本仅按人数而非金钱。两图均先逐重复算站点内AUC再平均。各b的反向集分别扣除桥接者，跨b/方向反向人群不同，不把其变化当作完全相同人群上的比较。本图仅展示测试AUC，R1还同时要求反向集标准，需结合判读表。\n')


def verify(pred,done):
    cfg=json.loads((OUT/'config.json').read_text());assert cfg['stage'] in ['training_complete','complete'] and (OUT/'exit_code.txt').read_text().strip()=='0'
    for section in ['source_sha256','input_sha256']:
        for p,h in cfg[section].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    assert hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()==cfg['splits_sha256']
    plans=c.read_plans();assert plans==c.plans();lookup={j['job_id']:j for j in plans};table=c.subjects();meta=table.set_index('FILE_ID',drop=False)
    assert len(done)==1200 and set(done.job_id)==set(lookup) and not done.job_id.duplicated().any();assert len(pred)==sum(len(j['test'])+len(j['reverse']) for j in plans) and not pred.duplicated(['job_id','FILE_ID']).any();assert not pred.is_bridge.any();assert pred.probability.between(0,1).all();np.testing.assert_array_equal(pred.prediction,pred.probability.ge(.5))
    for job_id,g in pred.groupby('job_id'):
        j=lookup[job_id];assert set(g.loc[~g.reverse,'FILE_ID'])==set(j['test']) and set(g.loc[g.reverse,'FILE_ID'])==set(j['reverse'])
        assert set(j['bridge']).isdisjoint(g.FILE_ID) and set(j['train']).isdisjoint(g.FILE_ID)
        for col in c.a.META+['n_zero_roi','n_timepoints']:pd.testing.assert_series_equal(g[col].reset_index(drop=True),meta.loc[g.FILE_ID,col].reset_index(drop=True),check_names=False,check_dtype=False)
    d=pd.read_csv(OUT/'selection_fits.csv',float_precision='round_trip');assert len(d)==24000 and d.C.isin(c.a.GRID).all() and d.AUC.between(0,1).all()
    for q in done.itertuples():
        f=d.loc[d.job_id.eq(q.job_id)].pivot(index='inner_fold',columns='C',values='AUC').reindex(columns=c.a.GRID);assert q.C==c.a.GRID[int(np.argmax(f.to_numpy().mean(axis=0)))]
        if q.method=='C2':assert q.b>0 and q.additive_rank==2*q.k+1 and q.site_effect_sum_max_abs<1e-10
    records=[json.loads(s) for s in (OUT/'inner_splits.jsonl').read_text().splitlines()];assert len(records)==6000
    for s in records:
        j=lookup[s['job_id']];tr=set(s['train']);va=set(s['validation']);assert tr|va==set(j['train']) and tr.isdisjoint(va) and not (tr|va)&set(j['test'])
        assert s['standardizer_fit']==s['train']
        if s['method']=='C2':assert s['additive_fit']==s['train']
    assert d.loc[d.method.eq('C2'),'additive_rank'].isin([5,9]).all()
    return dict(jobs=1200,LR_fits=25200,predictions=len(pred),inner_splits=6000,C_selection_fits=24000,additive_fits=3600,all_bridge_lists_and_seed_sampling_verified=True,all_outer_and_inner_subject_isolation_verified=True,all_C2_designs_full_rank=True,C2_b0_not_called=True,all_source_and_reference_hashes_verified=True,warnings=int(done.warning_count.sum()))


def report(summary,ref,paired,recovery,criteria,result,v):
    cfg=json.loads((OUT/'config.json').read_text());primary=summary.loc[(~summary.reverse&summary.metric.isin(METRICS[:3]))|(summary.reverse&summary.metric.eq('pooled_AUC'))]
    text='# T12-C 桥接被试的设计与校正\n\n'+result['decision']+'。本轮结束后停下，不推广到脑电或启动新任务。\n\n**所有范围都是依赖的50次重复2.5%–97.5%变异范围，不是置信区间。**\n\n## R1–R3\n\n'+markdown(criteria)+'\n\n## 方法主表\n\n'+markdown(primary)+'\n\n## 挽回比例：均值之比\n\n'+markdown(recovery)+'\n\n## 配对差值\n\n'+markdown(paired)+'\n\n## 单向与双向（新增人数不同）\n\n'+markdown(pd.read_csv(OUT/'direction_comparison.csv'))+'\n\n## 只读参照均值\n\n'+markdown(ref.groupby(['method','k'])[['within_site_AUC','pooled_AUC','male_within_site_AUC','n_train']].mean().reset_index())+'\n\n'
    text+='## 固定实现与限制\n\n- 特征及100份P40划分直接读T12-A，不重算或重抽P名单；只按指定default_rng([repeat,k,3])在训练站点名称排序后逐站点排列另一类别候选，取前b人。b嵌套，单/双向共用完全相同的TD桥接。全部原始1035人仍可用，不另定排除或做零方差ROI敏感性；BNU watch不适用。\n'
    text+='- C0用T12-A原Pipeline LR；C2在每个实际训练部分重新拟合StandardScaler与多响应OLS。设计矩阵为截距、ASD指示及S−1个零和站点对比（末站点效应=其他效应负和），因此允许非零TD基线并保留诊断效应。各特征共享设计，按被试普通最小二乘，不另加权、不加先验。只减gamma，不减诊断项。未见过的测试站点只用训练Scaler、不拟合gamma；对于固定线性分类器，站点整体平移不改变站点内排序，合并AUC仅报告。\n'
    text+='- 两法C均固定.001/.01/.1/1，5折按标签分层、shuffle、seed=repeat，选择均值验证AUC；同条件C0/C2内层名单完全一致。C2的Scaler/OLS每内折重拟合，所有3000内折设计及600最终设计均满秩；b0从未调用C2，仅引用B0。LR沿T12-A默认lbfgs/max_iter100/tol1e-4/无类别权重；单向增加TD导致类别不平衡按原流程保留，不做额外校正或调参。\n'
    text+='- 桥接者只用于训练；正式测试和反向预测is_bridge均False，单独bridge_subjects.csv为True。反向集为原反向池扣除当前桥接者；B0不重训，使用其既有逐人概率筛到同一剩余人群后配对。不同b、不同方向的反向池不相同，单/双反向均值只作描述。\n'
    text+='- 挽回率严格按各重复method−B0与M−B0先求差、分别取均值后相除；不用逐次比例平均，不截断负值或>1值，不给该比值伪造重复区间。正比例沿用已确认的数值判等：差值>1e-12；均值、范围和R判据不作阈值修改。\n'
    text+='- R1主k2、k4重复；R2单/双向分别要求b2与b5都为正，k4另作重复检验；b10只报告。R3仅k2/单向b5，新增10人与P80新增80人比较；双方50重复的测试站点和被试相同，训练站点未保持固定、训练名单也不是嵌套扩充，因此只是部分配对，不将其解释为同一站点上多采人数的纯因果效果。\n'
    text+='- M/F/P80/B0均引用T12-A（B0已经在T12-B验证逐人一致），没有重训参照。成本只按总训练人数，未估计患者与对照的货币或招募时间成本；右图对数横轴同时保留全部k2/k4方案和F，其F人数为实际50次训练人数均值。\n'
    text+=f'- 指定唯一测试先失败再实现通过；6个完整计时任务直接复用，正式1194个固定任务用5 CPU工作进程/BLAS1线程，不用GPU、不装依赖。点估{cfg["timing"]["point_hours"]*60:.3f}分钟、含25%余量{cfg["timing"]["with_25percent_margin_hours"]*60:.3f}分钟，按≤3小时授权tmux启动。起止{cfg["started_at"]}至{cfg["finished_at"]}，实际拟合警告{v["warnings"]}。\n\n核对：'+json.dumps(v,ensure_ascii=False)+'\n\n输出包含配置、种子、桥接/训练/验证名单、逐人预测、C及加性设计秩、主表/recovery/paired/criteria、R3部分配对核对、匹配B0反向统计、只读参照、图与图注。\n'
    (OUT/'REPORT.md').write_text(text)


if __name__=='__main__':
    pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');done=pd.read_csv(OUT/'completed.csv',float_precision='round_trip');v=verify(pred,done);ref=references();runs,summary=method_metrics(pred);baseline=matched_baseline_reverse();paired,recovery=paired_and_recovery(runs,ref,baseline);criteria,result=decision(summary,paired,recovery);figure(summary,runs,ref);report(summary,ref,paired,recovery,criteria,result,v)
    done.to_csv(OUT/'selected_parameters.csv',index=False);c.dump('verification.json',v);cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='complete',report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());c.dump('config.json',cfg);print('REPORT_COMPLETE '+result['decision'],flush=True)
