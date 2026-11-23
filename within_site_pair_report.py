"""T13-A：只读参照、同重复配对和预先约定的N1/N2。"""
from pathlib import Path
import json,hashlib

import numpy as np
import pandas as pd

import within_site_pair_model as m
import bridge_design_report as cr
from abide_class_pure_sites_report import markdown

OUT=m.OUT;METRICS=['within_site_AUC','male_within_site_AUC','pooled_AUC']


def references():
    old=pd.read_csv('outputs/T12A/predictions.csv',float_precision='round_trip');old=old.loc[~old.no_zero_roi];primary,secondary=m.source_jobs();parts=[]
    for j in primary:
        f=old.loc[old.job_id.eq(j['job_id'])].copy();context=dict(job_id=f'reference_F_orig_r{j["repeat"]}',setting='main',model='F_orig',method='F_orig',k=0,b=0,direction='main',repeat=j['repeat'],seed=j['seed'],n_pure=len(j['train']),n_added=0,n_train=len(j['train']),T12A_job_id=j['job_id'],source_job_id=j['job_id'])
        for key,value in context.items():f[key]=value
        parts.append(f)
    bridge=pd.read_csv('outputs/T12C/predictions.csv',float_precision='round_trip');bridge=bridge.loc[bridge.k.eq(2)&bridge.b.eq(5)&bridge.method.isin(['C0','C2'])].copy();bridge['source_job_id']=bridge.job_id;bridge['job_id']='reference_'+bridge.job_id;bridge['setting']='secondary';bridge['model']=bridge.method;parts.append(bridge)
    for j in secondary:
        f=old.loc[old.job_id.eq(j['T12A_job_id'])&old.FILE_ID.isin(j['test']+j['reverse'])].copy();assert set(f.FILE_ID)==set(j['test']+j['reverse'])
        context={key:j[key] for key in ['k','b','direction','repeat','seed','n_pure','T12A_job_id']}|dict(job_id=f'reference_B0_r{j["repeat"]}_{j["direction"]}',setting='secondary',model='B0',method='B0',n_added=0,n_train=80,source_job_id=j['T12A_job_id'])
        for key,value in context.items():f[key]=value
        parts.append(f)
    result=pd.concat(parts,ignore_index=True);result['score']=result.probability;result['score_origin']='frozen_reference_probability';result.to_csv(OUT/'reference_predictions.csv',index=False);return result


def paired(runs):
    rows=[]
    for setting,direction,contrasts in [('main','main',[('L3','L0'),('L3','L1'),('L3','L2'),('L1','L0'),('L2','L0')])]+[('secondary',direction,[('L3',q) for q in ['C0','C2','B0']]) for direction in ['oneway','twoway']]:
        f=runs.loc[runs.direction.eq(direction)]
        for metric in METRICS+(['reverse_AUC'] if setting=='secondary' else []):
            reverse=metric=='reverse_AUC';name='pooled_AUC' if reverse else metric;values=f.loc[f.reverse.eq(reverse)].pivot(index='repeat',columns='method',values=name)
            for left,right in contrasts:
                rows.extend(dict(setting=setting,direction=direction,contrast=left+'-'+right,metric=metric,repeat=int(r),difference=float(v)) for r,v in (values[left]-values[right]).items())
    raw=pd.DataFrame(rows);assert len(raw)==1950 and raw.difference.notna().all();raw.to_csv(OUT/'paired_repeat_differences.csv',index=False)
    summary=raw.groupby(['setting','direction','contrast','metric']).difference.agg(mean='mean',range_low=lambda s:s.quantile(.025),range_high=lambda s:s.quantile(.975),positive_fraction=lambda s:s.gt(1e-12).mean(),mean_absolute_difference=lambda s:s.abs().mean(),repeats='count').reset_index();assert len(summary)==39 and summary.repeats.eq(50).all();summary.to_csv(OUT/'paired_differences.csv',index=False);return raw,summary


def decision(paired):
    f=paired.loc[paired.setting.eq('main')&paired.metric.eq('within_site_AUC')].set_index('contrast');gain=float(f.loc['L3-L0','mean']);positive=float(f.loc['L3-L0','positive_fraction']);absolute=float(f.loc['L3-L1','mean_absolute_difference']);signed=float(f.loc['L3-L1','mean']);n1=gain>=.01 and positive>=.70;n2=absolute<.005
    table=pd.DataFrame([dict(criterion='N1',point=gain,positive_fraction=positive,passed=n1,details='mean gain>=.01 and positive-repeat fraction>=.70'),dict(criterion='N2',point=absolute,positive_fraction=None,passed=n2,details=f'mean(abs per-repeat L3-L1)<.005;signed mean={signed:.12g}')]);table.to_csv(OUT/'criteria.csv',index=False)
    text='N1不满足：按站点条件化未达到预设准确率增益，方向1作为模型路线停止；其他方向须另行下达。' if not n1 else 'N1满足且N2等价：收益来自按站点条件化，配对仅是一种实现方式，回报讨论定位。' if n2 else f'N1满足、N2不等价；L3-L1有向均值={signed:+.6f}。任务未另给“明显优于”的数值阈值，回报研究审阅，不自行启动后续主干。'
    result=dict(N1=n1,N2_equivalent=n2,L3_minus_L0=gain,positive_fraction=positive,mean_absolute_L3_minus_L1=absolute,signed_mean_L3_minus_L1=signed,decision=text);m.dump('decision.json',result);return table,result


def verify(pred,done):
    cfg=json.loads((OUT/'config.json').read_text());assert cfg['stage']=='training_complete' and (OUT/'exit_code.txt').read_text().strip()=='0'
    for section in ['source_sha256','input_sha256']:
        for path,h in cfg[section].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    assert hashlib.sha256((OUT/'splits.jsonl').read_bytes()).hexdigest()==cfg['splits_sha256']
    jobs=[json.loads(j) for j in (OUT/'splits.jsonl').read_text().splitlines()];assert jobs==m.plans();by={j['job_id']:j for j in jobs};meta=m.subjects().set_index('FILE_ID',drop=False)
    assert len(done)==300 and set(done.job_id)==set(by) and not done.job_id.duplicated().any();assert len(pred)==sum(len(j['test'])+len(j['reverse']) for j in jobs) and not pred.duplicated(['job_id','FILE_ID']).any()
    assert np.isfinite(pred.score).all() and pred.probability.between(0,1).all();np.testing.assert_array_equal(pred.prediction,pred.score.ge(0))
    for job_id,g in pred.groupby('job_id'):
        j=by[job_id];assert set(j['train']).isdisjoint(g.FILE_ID) and set(g.loc[~g.reverse,'FILE_ID'])==set(j['test']) and set(g.loc[g.reverse,'FILE_ID'])==set(j['reverse'])
        for col in m.a.META:pd.testing.assert_series_equal(g[col].reset_index(drop=True),meta.loc[g.FILE_ID,col].reset_index(drop=True),check_names=False,check_dtype=False)
    fits=pd.read_csv(OUT/'solver_fits.csv',float_precision='round_trip');assert len(fits)==7800 and fits.success.all();assert fits.loc[fits.role.eq('selection'),'validation_pairs'].gt(0).all()
    for q in done.itertuples():
        f=fits.loc[fits.job_id.eq(q.job_id)&fits.role.eq('selection')].pivot(index='inner_fold',columns='C',values='AUC').reindex(columns=m.GRID);assert f.shape==(5,5) and q.C==m.GRID[int(np.argmax(f.to_numpy().mean(axis=0)))]
    records=[json.loads(line) for line in (OUT/'inner_splits.jsonl').read_text().splitlines()];assert len(records)==1500
    for s in records:
        j=by[s['job_id']];tr=set(s['train']);va=set(s['validation']);assert tr.isdisjoint(va) and tr|va==set(j['train']) and not (tr|va)&set(j['test']+j['reverse']);assert s['standardizer_fit']==s['train'] and s['site_mean_fit']==s['train'] and not s['validation_target_mean_used']
    final=[json.loads(line) for line in (OUT/'fit_provenance.jsonl').read_text().splitlines()];assert len(final)==300
    for f in final:
        j=by[f['job_id']];assert f['standardizer_fit']==j['train'] and f['site_mean_fit']==j['train'];assert f['unlabelled_test_mean_ids']==(j['test'] if j['model']=='L2' else [])
    result=dict(jobs=300,main_jobs=200,secondary_jobs=100,total_fits=7800,scipy_fits=int(fits.model.isin(['L1','L3']).sum()),sklearn_fits=int(fits.model.isin(['L0','L2']).sum()),predictions=len(pred),inner_splits=len(records),all_memberships_scaling_provenance_and_C_selection_verified=True,warnings=int(done.warning_count.sum()),frozen_sources_cache_and_reference_inputs_unchanged=True);m.dump('verification.json',result);return result


def report():
    pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');done=pd.read_csv(OUT/'completed.csv',float_precision='round_trip');verification=verify(pred,done);pred['score_origin']='decision_function';reference=references();combined=pd.concat([pred,reference],ignore_index=True)
    # 旧汇总函数仅把该列当连续AUC分数；原始概率文件保持不变。
    cr.OUT=OUT;runs,summary=cr.method_metrics(combined.drop(columns='probability').rename(columns={'score':'probability'}));runs['setting']=np.where(runs.direction.eq('main'),'main','secondary');runs['model']=runs.method;runs.to_csv(OUT/'repeat_metrics.csv',index=False);raw,comparison=paired(runs);criteria,result=decision(comparison)
    check=[]
    for metric in METRICS:
        f=runs.loc[runs.direction.eq('main')&~runs.reverse].pivot(index='repeat',columns='method',values=metric);check.append(dict(metric=metric,L0=float(f.L0.mean()),original_F=float(f.F_orig.mean()),mean_difference=float((f.L0-f.F_orig).mean()),repeats=50))
    check=pd.DataFrame(check);check.to_csv(OUT/'F_reference_check.csv',index=False);done.to_csv(OUT/'selected_C.csv',index=False)
    main=summary.loc[summary.direction.eq('main')];secondary=summary.loc[summary.direction.ne('main')&((~summary.reverse&summary.metric.isin(METRICS))|(summary.reverse&summary.metric.eq('pooled_AUC')))];warning=pd.read_csv(OUT/'warnings.csv') if verification['warnings'] else pd.DataFrame(columns=['category','message','count'])
    text='# T13-A 站点内配对模型\n\n'+result['decision']+'\n\n## 判读\n\n'+markdown(criteria)+'\n\n## 主要设置完整指标\n\n'+markdown(main)+'\n\n## 同重复配对差值\n\n'+markdown(comparison)+'\n\n## 次要设置（只报告）\n\n'+markdown(secondary)+'\n\n## 原F核对（不重跑）\n\n'+markdown(check)+'\n\n'
    text+='## 固定实现与限制\n\n- 缓存和T12-A/C名单只读。主要F12训练站点/5测试站点、50重复×四模型；次要仅L3单/双向各50重复。C0/C2/B0和旧F均引用既有逐人预测，不重跑。ABIDE不含BNU四名观察被试，不另构造no_watch。\n'
    text+='- 所有模型每个实际训练部分先拟合StandardScaler。L0/L2保留sklearn原默认优化设置；L1/L3用SciPy L-BFGS-B零初始化、maxiter5000/gtol1e−8/ftol1e−12/maxls50。L1损失按被试平均、每站点一个自由截距（不另设全局截距），L3按所有同站点ASD-TD配对平均；两者惩罚均||w||²/(2*C*N)，N含无配对训练者。\n'
    text+='- L1/L3内部用训练站点中心化作精确数值重参数化，配对差值不变，L1截距恢复到原坐标；不改变w的惩罚。两者预测均用原标准化x上的w·x，不加站点截距或测试站点中心化。L2先训练Scaler、再减站点均值，不二次缩放；内层已知验证站点用内层训练均值，新测试站点用自身全体无标签均值，属直推式处理。L2均值从不读取标签。\n'
    text+='- 统一五C(.001,.01,.1,1,10)、五折站点×诊断分层，种子=重复号。主选参是各合格验证站点（两类各≥2）AUC等权平均；次选参按修订合并全部同站点配对（两类≥1）、平局.5。预检500/500折均有配对：单向每折8对、双向16对；不跳过空折或重复16，五折均值选C，平分取最小列首C。\n'
    text+='- 新模型AUC用原始decision score以免sigmoid饱和造成机器精度下的额外平局；概率另存expit(score)，分类阈值score≥0。参照只能读取原概率，原样作为其连续AUC分数，不反推缺失logits或重新训练。男性SEX=1，沿T12-A每诊断≥3人的站点规则。\n'
    text+='- N2严格为逐重复绝对差的均值mean(abs(L3−L1))，不是abs(mean(L3−L1))。所有范围是50个依赖重复的2.5%–97.5%变异范围，不是置信区间；正比例按用户已确认的>1e−12数值判等，均值、范围及N阈值不改。不新增显著性检验或图。\n'
    text+='- C2反向AUC已在T12-C审阅中被确认含校正伪影；本任务按要求只原样列示，不能把对该项的增益解释为方法优越性。次要设置全部不进入N1/N2。原F的C仅四值、按合并验证AUC选择，与本任务L0框架不同，F核对不是要求逐人相等。\n'
    text+='- 不安装依赖、不读取restricted目录、不上传数据。结论不自动授权接到其他主干或执行方向2/3；本任务结束后停下。\n\n验证：'+json.dumps(verification,ensure_ascii=False)+'\n\n计时：'+json.dumps(json.loads((OUT/'timing.json').read_text()),ensure_ascii=False)+'\n\n警告记录：\n\n'+markdown(warning)+'\n'
    (OUT/'REPORT.md').write_text(text);cfg=json.loads((OUT/'config.json').read_text());cfg['stage']='complete';m.dump('config.json',cfg);print('REPORT_COMPLETE '+result['decision'],flush=True)


if __name__=='__main__':report()
