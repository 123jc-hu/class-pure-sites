"""T12-B：嵌套性能估计、逐重复配对差异及两部分独立判读。"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,balanced_accuracy_score

import site_pair_validation as b
from abide_class_pure_sites_report import markdown

OUT=b.OUT;METRICS=['within_site_AUC','pooled_AUC','male_within_site_AUC','BA','Acc','Sens','Spec']


def references():
    old=pd.read_csv(b.a.OUT/'predictions.csv',float_precision='round_trip');f=old.loc[~old.no_zero_roi&~old.reverse&((old.arm.eq('M')&old.k.isin([2,4,6])&old.n.eq(40))|old.arm.eq('F'))].copy()
    f['method']=f.arm;f['T12A_job_id']=f.job_id;f['job_id']=f.method+'_r'+f.repeat.astype(str)+'_k'+f.k.astype(str);f['origin']='T12A_direct_reference';f.to_csv(OUT/'reference_predictions.csv',index=False)
    return f


def method_metrics(pred):
    rows=[];sites=[]
    for (method,k,repeat,reverse),g in pred.groupby(['method','k','repeat','reverse']):
        ctx=dict(method=method,k=k,repeat=repeat,reverse=reverse);row=ctx|dict(pooled_AUC=roc_auc_score(g.label,g.probability),BA=balanced_accuracy_score(g.label,g.prediction),Acc=accuracy_score(g.label,g.prediction),Sens=recall_score(g.label,g.prediction),Spec=recall_score(g.label,g.prediction,pos_label=0))
        if not reverse:
            values=[];male=[];skip=[]
            for site,s in g.groupby('site'):
                v=roc_auc_score(s.label,s.probability);values.append(v);m=s.loc[s.SEX.eq(1)];n1=int(m.label.eq(1).sum());n0=int(m.label.eq(0).sum());excluded=min(n1,n0)<3
                if excluded:skip.append(site)
                else:male.append(roc_auc_score(m.label,m.probability))
                sites.append(ctx|dict(site=site,AUC=v,male_AUC=np.nan if excluded else male[-1],male_ASD=n1,male_TD=n0,male_skipped=excluded))
            assert len(values)==5
            row.update(within_site_AUC=np.mean(values),male_within_site_AUC=np.mean(male),male_skipped=len(skip),male_skipped_sites=';'.join(skip))
        rows.append(row)
    runs=pd.DataFrame(rows);runs.to_csv(OUT/'method_repeat_metrics.csv',index=False);pd.DataFrame(sites).to_csv(OUT/'site_metrics.csv',index=False)
    long=runs.melt(id_vars=['method','k','repeat','reverse'],value_vars=METRICS,var_name='metric',value_name='value').dropna(subset=['value'])
    summary=long.groupby(['method','k','reverse','metric']).value.agg(mean='mean',range_low=lambda s:s.quantile(.025),range_high=lambda s:s.quantile(.975),repeats='count').reset_index();assert summary.repeats.eq(50).all();summary.to_csv(OUT/'main_table.csv',index=False)
    pairs=[]
    for left,right in [('B1','B0'),('B2','B1'),('B2','B0'),('B3','B0'),('M','B2')]:
        for k in [2,4,6]:
            for reverse in [False,True]:
                if reverse and left=='M':continue
                f=runs.loc[runs.k.eq(k)&runs.reverse.eq(reverse)];a=f.loc[f.method.eq(left)].sort_values('repeat');c=f.loc[f.method.eq(right)].sort_values('repeat');np.testing.assert_array_equal(a.repeat,c.repeat)
                for metric in ['pooled_AUC'] if reverse else METRICS[:3]:
                    for r,value in zip(a.repeat,a[metric].to_numpy()-c[metric].to_numpy()):pairs.append(dict(contrast=left+'-'+right,k=k,repeat=r,metric='reverse_AUC' if reverse else metric,difference=value))
    raw=pd.DataFrame(pairs);raw.to_csv(OUT/'paired_repeat_differences.csv',index=False)
    paired=raw.groupby(['contrast','k','metric']).difference.agg(mean='mean',range_low=lambda s:s.quantile(.025),range_high=lambda s:s.quantile(.975),positive_fraction=lambda s:s.gt(1e-12).mean(),repeats='count').reset_index();paired.to_csv(OUT/'paired_differences.csv',index=False)
    ref=[]
    for k in [2,4,6]:
        f=runs.loc[runs.k.eq(k)&~runs.reverse].pivot(index='repeat',columns='method',values='within_site_AUC')
        for method in ['B0','B1','B2']:ref.append(dict(k=k,method=method,M_gap=float((f.M-f[method]).mean())))
    pd.DataFrame(ref).to_csv(OUT/'M_reference_gaps.csv',index=False)
    return runs,summary,paired


def estimates(done,runs,expected_rows=450):
    truth=runs.loc[runs.method.eq('B0')&~runs.reverse,['k','repeat','within_site_AUC','pooled_AUC']]
    raw=done.loc[done.method.str.startswith('E_'),['job_id','method','k','repeat','estimate']].copy()
    fold=pd.read_csv(OUT/'estimation_folds.csv',float_precision='round_trip').merge(truth,on=['k','repeat'],validate='many_to_one');fold.to_csv(OUT/'estimation_fold_results.csv',index=False)
    means=fold.groupby('job_id',sort=False).AUC.agg(lambda values:np.mean(values.to_numpy()));primary=raw.copy();primary['estimate']=primary.job_id.map(means);primary['aggregation']='mean_outer_fold_AUC'
    pooled=raw.loc[raw.method.eq('E_subj')].copy();pooled['method']='E_subj_pooled';pooled['aggregation']='pooled_OOF_AUC_report_only'
    f=pd.concat([primary,pooled],ignore_index=True).merge(truth,on=['k','repeat'],validate='many_to_one')
    if expected_rows is not None:assert len(f)==expected_rows
    rows=[];correlations=[]
    for truth_name in ['within_site_AUC','pooled_AUC']:
        f['error_'+truth_name]=f.estimate-f[truth_name];f['abs_error_'+truth_name]=abs(f['error_'+truth_name])
        for (method,k),g in f.groupby(['method','k']):
            error=g.estimate-g[truth_name];lo,hi=np.quantile(error,[.025,.975]);r=pearsonr(g.estimate,g[truth_name]).statistic
            rows.append(dict(method=method,k=k,truth=truth_name,estimate=g.estimate.mean(),true_AUC=g[truth_name].mean(),mean_bias=error.mean(),range_low=lo,range_high=hi,mean_absolute_error=abs(error).mean(),repeats=len(g)))
            correlations.append(dict(method=method,k=k,truth=truth_name,Pearson_r=r,repeats=len(g)))
    f.to_csv(OUT/'performance_estimates.csv',index=False);summary=pd.DataFrame(rows);summary.to_csv(OUT/'performance_summary.csv',index=False);pd.DataFrame(correlations).to_csv(OUT/'performance_correlations.csv',index=False)
    return summary


def decision(performance,paired):
    p=performance.loc[performance.truth.eq('within_site_AUC')];rows=[]
    v1=float(p.loc[p.method.eq('E_subj')&p.k.eq(2),'mean_bias'].iloc[0]);rows.append(dict(criterion='V1',k=2,point=v1,passed=v1>=.1,details='E_subj mean signed error >=+.10'))
    v2=[]
    for k in [2,4,6]:
        s=float(p.loc[p.method.eq('E_subj')&p.k.eq(k),'mean_absolute_error'].iloc[0]);l=float(p.loc[p.method.eq('E_LSPO')&p.k.eq(k),'mean_absolute_error'].iloc[0]);v2.append(l<s);rows.append(dict(criterion='V2',k=k,point=l-s,passed=l<s,details=f'MAE LSPO={l:.6f};subject={s:.6f}'))
    def difference(contrast,k,metric):return float(paired.loc[paired.contrast.eq(contrast)&paired.k.eq(k)&paired.metric.eq(metric),'mean'].iloc[0])
    m2=[]
    for k in [2,4]:
        s=difference('B1-B0',k,'within_site_AUC');r=difference('B1-B0',k,'reverse_AUC');rows.append(dict(criterion='M1',k=k,point=s,passed=s>=0 and r>0,details=f'reverse gain={r:.6f};k2 main,k4 replication'))
        s=difference('B2-B1',k,'within_site_AUC');r=difference('B2-B1',k,'reverse_AUC');m2.append(s>0 and r>0);rows.append(dict(criterion='M2',k=k,point=s,passed=s>0 and r>0,details=f'reverse gain={r:.6f}'))
    check=pd.DataFrame(rows);check.to_csv(OUT/'criteria.csv',index=False)
    conclusion=dict(V1=v1>=.1,V2=all(v2),M2=all(m2),evaluation='V1/V2均满足，支持站点对留出作为评估工具；后续T12-C须另行下达' if v1>=.1 and all(v2) else '高估存在，但LSPO没有全部更准，回报讨论' if v1>=.1 else '未满足性能估计预设条件，回报讨论',method='NAP组件满足预设保留条件' if all(m2) else '线性的站点子空间去除未满足预设条件，回报讨论')
    b.dump('decision.json',conclusion);return check,conclusion


def figure(performance,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained');values=[]
    raw=pd.read_csv(OUT/'performance_estimates.csv')
    for name,color,marker in [('E_subj','#D55E00','o'),('E_LSPO','#0072B2','s'),('Truth','#222222','D')]:
        vals=[]
        for k in [2,4,6]:
            f=raw.loc[raw.k.eq(k)&raw.method.eq('E_subj' if name=='Truth' else name)];v=f.within_site_AUC if name=='Truth' else f.estimate;lo,hi=np.quantile(v,[.025,.975]);vals.append((k,v.mean(),lo,hi));values.append(dict(panel='estimation',method=name,k=k,mean=v.mean(),range_low=lo,range_high=hi))
        x,m,lo,hi=np.array(vals).T;axes[0].plot(x,m,color=color,marker=marker,label='E_subj (fold mean)' if name=='E_subj' else name);axes[0].fill_between(x,lo,hi,color=color,alpha=.1)
    colors={'B0':'#0072B2','B1':'#D55E00','B2':'#009E73','B3':'#CC79A7','M':'#222222'}
    for method,color in colors.items():
        f=summary.loc[summary.method.eq(method)&summary.metric.eq('within_site_AUC')&~summary.reverse].sort_values('k');axes[1].plot(f.k,f['mean'],color=color,marker='D' if method=='M' else 'o',label=method)
        values.extend(dict(panel='method_test',method=method,k=q.k,mean=q.mean,range_low=q.range_low,range_high=q.range_high) for q in f.itertuples())
        if method!='M':
            f=summary.loc[summary.method.eq(method)&summary.metric.eq('pooled_AUC')&summary.reverse].sort_values('k');axes[1].plot(f.k,f['mean'],color=color,ls='--')
            values.extend(dict(panel='method_reverse',method=method,k=q.k,mean=q.mean,range_low=q.range_low,range_high=q.range_high) for q in f.itertuples())
    axes[0].legend(frameon=False,fontsize=8)
    handles,labels=axes[1].get_legend_handles_labels();handles.extend([Line2D([],[],color='.4',ls='-',label='Test within-site'),Line2D([],[],color='.4',ls='--',label='Reverse pooled')]);labels.extend(['Test within-site','Reverse pooled']);axes[1].legend(handles,labels,frameon=False,ncol=2,fontsize=8,loc='upper left')
    for ax in axes:ax.set(xlabel='Sites per diagnosis (k)',ylabel='AUC',xticks=[2,4,6],ylim=(0,1));ax.axhline(.5,color='.7',ls=':',lw=.8);ax.spines[['top','right']].set_visible(False)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/('site_pair_validation.'+ext),dpi=200)
    plt.close(fig);pd.DataFrame(values).to_csv(OUT/'figure_values.csv',index=False)
    (OUT/'figure_caption.md').write_text('左：每重复嵌套估计E_subj（5个外折AUC等权均值；合并OOF仅另表报告）、E_LSPO（外k²站点对AUC等权平均）与B0在5个新混合站点上的站点内AUC。色带为50重复2.5%–97.5%变异范围，不是独立样本置信区间。右：B0–B3测试站点内AUC为实线、原P站点相反诊断者的反向集合并AUC为同色虚线；M实线直接引用T12-A。均先逐重复计算再对50次平均，n40，k2/4/6。两图纵轴0–1；E_LSPO训练少一对站点，E_subj少1/5被试，且估计评分与站点内真值并非相同构造。\n')


def verify(pred,done):
    cfg=json.loads((OUT/'config.json').read_text());assert cfg['stage'] in ['training_complete','complete'] and (OUT/'exit_code.txt').read_text().strip()=='0'
    for section in ['source_sha256','input_sha256']:
        for p,h in cfg[section].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    table,_=b.data();meta=table.set_index('FILE_ID',drop=False);lookup=pd.Index(table.FILE_ID);jobs=b.jobs();by={(j['repeat'],j['k']):j for j in jobs}
    expected={f'{method}_r{j["repeat"]}_k{j["k"]}' for j in jobs for method in b.METHODS};assert len(done)==900 and set(done.job_id)==expected and not done.job_id.duplicated().any();assert np.isfinite(done.seconds).all()
    assert len(pred)==297176 and not pred.duplicated(['job_id','FILE_ID']).any() and pred.probability.between(0,1).all();np.testing.assert_array_equal(pred.prediction,pred.probability.ge(.5))
    for (method,r,k),g in pred.groupby(['method','repeat','k']):
        j=by[r,k];assert set(g.loc[~g.reverse,'FILE_ID'])==set(j['test']) and set(g.loc[g.reverse,'FILE_ID'])==set(j['reverse'])
        for col in b.a.META+['n_zero_roi','n_timepoints']:pd.testing.assert_series_equal(g[col].reset_index(drop=True),meta.loc[g.FILE_ID,col].reset_index(drop=True),check_names=False,check_dtype=False)
    old=pd.read_csv(b.a.OUT/'predictions.csv',float_precision='round_trip');old=old.loc[~old.no_zero_roi&old.arm.eq('P')&old.n.eq(40)&old.k.isin([2,4,6])];new=pred.loc[pred.method.eq('B0')];m=new.merge(old,on=['FILE_ID','repeat','k','reverse'],suffixes=('_new','_old'),validate='one_to_one');assert len(m)==74294;assert (m.probability_new-m.probability_old).abs().max()==0;np.testing.assert_array_equal(m.prediction_new,m.prediction_old)
    ep=pd.read_csv(OUT/'estimation_predictions.csv',float_precision='round_trip');ef=pd.read_csv(OUT/'estimation_folds.csv');d=pd.read_csv(OUT/'selection_fits.csv',float_precision='round_trip');assert len(ep)==60000 and len(ef)==3550 and len(d)==99400
    assert ep.probability.between(0,1).all() and not ep.duplicated(['job_id','outer_fold','FILE_ID']).any();assert d.C.isin(b.a.GRID).all() and d.AUC.between(0,1).all()
    outer={}
    for j in jobs:
        tr=lookup.get_indexer(j['train'])
        for method in ['E_subj','E_LSPO']:
            job_id=f'{method}_r{j["repeat"]}_k{j["k"]}';pairs=b.subject_folds(table,tr,j['seed']) if method=='E_subj' else b.site_pairs(table,tr)
            for fold,(a,c) in enumerate(pairs):outer[job_id,fold]=(a,c)
            g=ep.loc[ep.job_id.eq(job_id)];assert g.FILE_ID.value_counts().eq(1 if method=='E_subj' else j['k']).all() and set(g.FILE_ID)==set(j['train'])
    inner=[json.loads(s) for s in (OUT/'inner_splits.jsonl').read_text().splitlines()];assert len(inner)==28400
    for s in inner:
        j=by[s['repeat'],s['k']];full=lookup.get_indexer(j['train'])
        if s['role']=='estimate_outer':wanted=outer[s['job_id'],s['fold']]
        elif s['role']=='B0_inner':
            parent=full if s['method']=='B0' else outer[s['job_id'],s['outer_fold']][0];wanted=b.subject_folds(table,parent,j['seed'])[s['fold']]
        else:wanted=(b.subject_folds(table,full,j['seed']) if s['method']=='B3' else b.site_pairs(table,full))[s['fold']]
        aa,cc=wanted;assert s['train']==table.loc[aa,'FILE_ID'].tolist() and s['validation']==table.loc[cc,'FILE_ID'].tolist();assert set(s['train']).isdisjoint(s['validation']) and not set(j['test'])&(set(s['train'])|set(s['validation']))
        if s['method']=='B2':assert s['U_fit']==s['train']
    assert d.loc[d.method.eq('B2')&d.k.eq(2),'U_rank'].eq(0).all()
    for row in done.loc[done.method.isin(['B1','B2','B3'])].itertuples():
        q=d.loc[d.job_id.eq(row.job_id)].pivot(index='inner_fold',columns='C',values='AUC').reindex(columns=b.a.GRID);assert row.C==b.a.GRID[int(np.argmax(q.to_numpy().mean(axis=0)))]
        if row.method=='B2':assert 0<=row.U_rank<=2*row.k-2
    for k in [2]:
        q=done.loc[done.k.eq(k)&done.method.isin(['B1','B2'])].pivot(index='repeat',columns='method',values='C');np.testing.assert_array_equal(q.B1,q.B2)
    return dict(method_jobs=600,estimation_jobs=300,total_jobs=900,LR_fits=103550,method_predictions=len(pred),estimation_predictions=len(ep),estimate_outer_folds=len(ef),inner_and_outer_split_records=len(inner),C_selection_fit_records=len(d),B0_reproduction_rows=len(m),B0_max_probability_difference=0.,all_metadata_and_splits_verified=True,NAP_inner_k2_all_empty=True,all_current_frozen_source_input_hashes_verified=True,training_code_and_features_unchanged=True,task_document_revision_user_authorized=True,warning_count=int(done.warning_count.sum()))


def report(performance,summary,paired,check,conclusion,v):
    cfg=json.loads((OUT/'config.json').read_text());main=summary.loc[(~summary.reverse&summary.metric.isin(['within_site_AUC','pooled_AUC','male_within_site_AUC']))|(summary.reverse&summary.metric.eq('pooled_AUC'))]
    text='# T12-B 纯类站点性能估计与方法比较\n\n'+conclusion['evaluation']+'。'+conclusion['method']+'。本任务结束后停下，不启动后续任务。\n\n'
    text+='**区间均为50次重复的2.5%–97.5%分位变异范围，不是独立样本置信区间。各次重复共享被试，不作显著性判断。**\n\n## 预设判读\n\n'+markdown(check)+'\n\n## 性能估计偏差与MAE\n\n'+markdown(performance)+'\n\n## 跨50次重复的Pearson相关\n\n'+markdown(pd.read_csv(OUT/'performance_correlations.csv'))+'\n\n## 方法结果\n\n'+markdown(main)+'\n\n## 配对差值\n\n'+markdown(paired)+'\n\n## 与混合参照M的站点内AUC差距\n\n'+markdown(pd.read_csv(OUT/'M_reference_gaps.csv'))+'\n\n## B1/B2所选C及最终U秩分布\n\n'+markdown(pd.read_csv(OUT/'parameter_distribution.csv').loc[lambda f:f.method.isin(['B1','B2'])])+'\n\n'
    text+='## 固定口径与限制\n\n- 只读取T12-A的fisher_z.npy/row_order.csv和splits.jsonl；只取主分析n40、k2/4/6的150份P名单，未重算特征或重新抽样，未做零方差ROI敏感性；BNU watch不适用。M与F直接引用，不重训。B0用T12-A原select_model，预检及最终74294条测试/反向预测最大差0。\n'
    text+='- B0及所有估计中的内部C均按标签分层5折、shuffle、种子repeat，在训练内重拟合Pipeline StandardScaler和原L2 LR。E_subj外层也按被试5折，按用户运行中修订，用保存的5个外折AUC等权均值判读；合并80人OOF的AUC作为E_subj_pooled在同表只报告。E_LSPO外层穷举k²个ASD/TD站点对，每对预测单独算AUC后等权平均。保存每对/折预测、选择C和全部内部名单。汇总修订不改变任何拟合或预测；completed.csv的原E_subj估计字段仍是合并法，新的主要均值来自estimation_folds.csv，任务原/修订SHA见aggregation_revision.json。\n'
    text+='- E_LSPO训练少一对站点（每类k−1个），E_subj少1/5被试，两者训练量不同；LSPO评分为跨两个纯类站点的AUC，主要真值是混合新站点内AUC，构造也不同。偏保守不等于已证明无偏；V2逐重复先取绝对误差再平均，不能用绝对平均偏差代替。相关按用户确认每k/估计器/两类真值分别跨50次算Pearson r，不另做相关检验。\n'
    text+='- B1按站点对均值AUC选原4C；B2同样选参，但各内折只用该折训练站点拟合Scaler和NAP，再在全P重拟合最终基。δ为每站点均值减去同类站点均值的等权平均；SVD阈值=max(delta.shape)×float64 epsilon×最大奇异值，取全部数值非零方向，不调秩。k2内层每类单站点、U为空，B1/B2选C完全相同。\n'
    text+='- B3按用户确认：内层每训练站点参数只拟合训练被试，再变换验证者；最终P站点参数同时用于P与反向集，反向者不参与拟合；5个新测试站点分别用本站点全体测试者拟合无标签均值/标准差。未加第二遍全局缩放。B3目标端属于无标签的传导式处理，与B0/B1/B2纯训练端估计须区别。\n'
    text+='- 所有LR保留T12-A默认lbfgs、max_iter100、tol1e-4、无类别权重，固定C=.001/.01/.1/1；相同AUC取第一个C，不热启动、不调设置。B1/B2每内折同一Scaler/NAP对4C复用，数学上与重新拟合相同。预测先逐人再逐站点/逐重复汇总；男性任一类<3则跳站点。M没有纯类反向集，M−B2的反向差值不定义。\n'
    text+='- M1按k2主格子、k4重复格子分别报告；M2要求k2和k4两项指标都提高，k6只描述。第一部分V1/V2与第二部分M2分别判读，不互相替代。方法paired表报告逐重复左减右的均值、分位范围及差值>1e−12的重复比例（按用户确认，|差值|≤1e−12视为相等；不改均值、范围或判读）。F只作全量参照，不把共用被试当独立样本。\n'
    text+=f'- 先通过唯一组合测试，再完成150个B0逐人复现和10个完整计时任务，复用这160个结果，正式跑余740个。五个CPU工作进程、BLAS各1线程，不用GPU、不装依赖。全900任务点估{cfg["timing"]["point_hours"]*60:.3f}分钟、含25%余量{cfg["timing"]["with_25percent_margin_hours"]*60:.3f}分钟，按≤3小时授权tmux/3h timeout执行，起止{cfg["started_at"]}至{cfg["finished_at"]}。\n'
    text+='- 计时汇总最初因不同方法完成/选择记录的CSV列未统一而耗时错列；在正式运行前固定可空字段列序，并从160份原始完成JSON恢复记录，360行选择记录补空outer_fold。仅修复序列化，预测、拟合、种子不变；混合方法写入/读回检查、B0逐人复现复核及全仓测试通过后才正式启动。\n\n核对：'+json.dumps(v,ensure_ascii=False)+'\n\n输出配置/原划分引用/全部内外名单、逐人方法预测与估计预测、每个外折结果和两种真值、所选C/U秩、主表/配对表/判读、图与图注；具体见各CSV和verification.json。\n'
    (OUT/'REPORT.md').write_text(text)


if __name__=='__main__':
    pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');done=pd.read_csv(OUT/'completed.csv',float_precision='round_trip');v=verify(pred,done)
    evaluated=pd.concat([pred.assign(origin='T12B_refit'),references()[pred.columns.tolist()+['origin']]],ignore_index=True);evaluated.to_csv(OUT/'evaluated_predictions.csv',index=False)
    runs,summary,paired=method_metrics(evaluated);performance=estimates(done,runs);check,conclusion=decision(performance,paired);figure(performance,summary)
    parameters=done.loc[done.method.isin(['B0','B1','B2','B3']),['job_id','method','repeat','k','C','U_rank','selection_AUC','final_n_iter']];parameters.to_csv(OUT/'selected_parameters.csv',index=False)
    parameters.groupby(['method','k','C','U_rank']).size().reset_index(name='count').to_csv(OUT/'parameter_distribution.csv',index=False);report(performance,summary,paired,check,conclusion,v)
    b.dump('verification.json',v);cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='complete',report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());b.dump('config.json',cfg);print('REPORT_COMPLETE '+conclusion['evaluation']+';'+conclusion['method'],flush=True)
