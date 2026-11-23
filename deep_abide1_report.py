"""T16-C：核对冻结参照，按预设重复汇总 D1、D2。"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import bridge_design_report as bridge
from abide_class_pure_sites_report import markdown
import deep_abide1 as task

OUT = task.FORMAL
CELL = ['k','b','direction']
AUC = ['within_site_AUC','male_within_site_AUC','pooled_AUC','reverse_AUC']
METRICS = AUC+['Acc','Sens','Spec','BA']
CONTRASTS = ['E-CE','E-GDRO','D-ERM','D-GDRO','D-DANN','linear_L3']
CHECK_METHODS = {'D-ERM_old':'D-ERM','E-CE_old':'E-CE','E-L3_old':'E-L3'}


def align_predictions(pred,refs):
    key = CELL+['repeat','method','FILE_ID']
    prior = refs.loc[refs.method.isin(CHECK_METHODS)].copy()
    prior['method'] = prior.method.map(CHECK_METHODS)
    cols = ['site','label','SEX','reverse','score','probability','prediction']
    aligned = prior[key+cols+['source']].merge(pred[key+cols],on=key,
                                               suffixes=('_ref',''),validate='one_to_one')
    assert len(aligned)==len(prior)
    for c in ['site','label','SEX','reverse']:
        np.testing.assert_array_equal(aligned[c],aligned[c+'_ref'])
    aligned['score_difference'] = (aligned.score-aligned.score_ref).abs()
    aligned['probability_difference'] = (aligned.probability-aligned.probability_ref).abs()
    aligned['predictions_identical'] = aligned.prediction.eq(aligned.prediction_ref)
    checks = aligned.groupby(CELL+['repeat','method','source']).agg(
        n=('FILE_ID','size'),max_abs_score_difference=('score_difference','max'),
        max_abs_probability_difference=('probability_difference','max'),
        predictions_identical=('predictions_identical','all')).reset_index()
    assert checks.groupby('method').size().to_dict()=={'D-ERM':40,'E-CE':10,'E-L3':10}
    return checks


def collect(jobs,config,meta):
    done, predictions, probes, weights, logs, online_checks = [],[],[],[],[],[]
    reused = {key:task.OUT/path for path,key in config['reused'].items()}
    assert {p.name for p in (OUT/'jobs').iterdir()}=={j['job_id'] for j in jobs}-reused.keys()
    for j in jobs:
        key = j['job_id']
        folder = reused.get(key,OUT/'jobs'/key)
        frame = pd.read_csv(folder/'predictions.csv',float_precision='round_trip')
        assert set(frame.job_id)=={key}
        assert frame.method.drop_duplicates().tolist()==task.HEADS+task.DEEP
        expect_ids = j['test']+j['reverse']
        for _,g in frame.groupby('method'):
            assert g.FILE_ID.tolist()==expect_ids and g.loc[~g.reverse,'FILE_ID'].tolist()==j['test']
            for c in task.META[1:]:
                np.testing.assert_array_equal(g[c],meta.loc[g.FILE_ID,c])
        assert np.isfinite(frame.score).all() and frame.probability.between(0,1).all()
        expected_prediction = np.where(frame.method.isin(task.HEADS),frame.score.ge(0),frame.probability.ge(.5))
        np.testing.assert_array_equal(frame.prediction,expected_prediction)
        predictions.append(frame.assign(reused=key in reused))
        stage = pd.read_csv(folder/'completed.csv',float_precision='round_trip')
        assert stage.stage.tolist()==task.STAGES and set(stage.job_id)=={key}
        assert (folder/'seconds.txt').is_file()
        done.append(stage.assign(reused=key in reused))
        lr_folds = [json.loads(s) for s in (folder/'head_inner_splits.jsonl').read_text().splitlines()]
        assert len(lr_folds)==15
        for method in task.HEADS:
            folds = [f for f in lr_folds if f['method']==method]
            assert [f['fold'] for f in folds]==list(range(5))
            for f in folds:
                tr,va = set(f['train']),set(f['validation'])
                assert tr.isdisjoint(va) and tr|va==set(j['train']) and f['standardizer_fit']==f['train']
            mean_auc = pd.DataFrame(f['validation_auc'] for f in folds).mean()
            assert float(mean_auc.idxmax())==stage.loc[stage.stage.eq(method),'C'].item()
        p = pd.read_csv(folder/'site_probe_predictions.csv')
        for m,g in p.groupby('method'):
            assert g.FILE_ID.tolist()==j['unlabeled_pool']
            np.testing.assert_array_equal(g.site,meta.loc[g.FILE_ID,'site'])
            assert set(g.predicted_site).issubset(set(g.site))
            np.testing.assert_array_equal(g.correct,g.site.eq(g.predicted_site))
            s = stage.loc[stage.stage.eq(m)].iloc[0]
            np.testing.assert_allclose([s.probe_accuracy,s.probe_chance],
                                       [g.correct.mean(),1/g.site.nunique()],rtol=0,atol=1e-15)
        probes.append(p.assign(**{c:j[c] for c in task.KEY}))
        w = json.loads((folder/'group_weights.json').read_text())
        assert set(w)==set(meta.loc[j['train'],'site'])
        np.testing.assert_allclose(sum(w.values()),1.,rtol=0,atol=1e-12)
        weights.extend({c:j[c] for c in task.KEY}|dict(site=s,weight=v) for s,v in w.items())
        loss = pd.read_csv(folder/'epoch_losses.csv',float_precision='round_trip')
        erm = pd.read_csv(folder/'V-CNN/epoch_losses.csv',float_precision='round_trip')
        loss = pd.concat([loss,erm.rename(columns={'method':'stage','weighted_train_loss':'loss'})])
        assert loss.groupby('stage').size().to_dict()=={'encoder':100,'D-ERM':50,'D-GDRO':50,'D-DANN':50}
        assert np.isfinite(loss.loss).all()
        logs.append(loss.assign(**{c:j[c] for c in task.KEY}))
        if (folder/'consistency.csv').is_file():
            online_checks.append(pd.read_csv(folder/'consistency.csv'))
    predictions = pd.concat(predictions,ignore_index=True)
    done = pd.concat(done,ignore_index=True)
    assert len(done)==1400 and done.reused.sum()==28
    return predictions,done,pd.concat(probes),pd.DataFrame(weights),pd.concat(logs),pd.concat(online_checks)


def aggregate(pred,refs,jobs):
    match = pd.DataFrame([{c:j[c] for c in CELL+['repeat','n_pure','n_added','n_train','T12A_job_id','confirmatory']}
                          for j in jobs])
    frame = pd.concat([pred[task.META+CELL+['repeat','seed','method','reverse','score','prediction']],
                       refs.loc[~refs.method.isin(CHECK_METHODS),task.META+CELL+['repeat','seed','method','reverse','score','prediction']]],
                      ignore_index=True).merge(match,on=CELL+['repeat'],validate='many_to_one')
    frame['job_id'] = (frame.method+':k'+frame.k.astype(str)+':'+frame.direction+':r'+frame.repeat.astype(str))
    frame['probability'] = frame.score
    bridge.OUT = OUT
    runs,_ = bridge.method_metrics(frame)
    index = CELL+['repeat','method']
    test = runs.loc[~runs.reverse,index+AUC[:3]+METRICS[4:]]
    reverse = runs.loc[runs.reverse,index+['pooled_AUC']].rename(columns={'pooled_AUC':'reverse_AUC'})
    values = test.merge(reverse,on=index,validate='one_to_one').merge(
        match[CELL+['repeat','confirmatory']],on=CELL+['repeat'],validate='many_to_one')
    assert len(values)==1700
    values.to_csv(OUT/'repeat_metrics.csv',index=False)
    summaries,raw = [],[]
    for label,group in [('all_repeats',values),('confirmatory',values.loc[values.confirmatory])]:
        mean = group.groupby(CELL+['method'],as_index=False).agg(
            **{c:(c,'mean') for c in METRICS},repeats=('repeat','size'))
        summaries.append(mean.assign(scope=label))
        for (k,b,direction),g in group.groupby(CELL):
            for m in AUC:
                wide = g.pivot(index='repeat',columns='method',values=m)
                for other in CONTRASTS:
                    raw.extend(dict(scope=label,k=k,b=b,direction=direction,metric=m,
                                    contrast=f'E-L3 - {other}',repeat=r,difference=v)
                               for r,v in (wide['E-L3']-wide[other]).items())
    summary,pairs = pd.concat(summaries,ignore_index=True),pd.DataFrame(raw)
    assert pairs.difference.notna().all()
    paired = pairs.groupby(['scope']+CELL+['contrast','metric']).difference.agg(
        mean='mean',range_low=lambda s:s.quantile(.025),range_high=lambda s:s.quantile(.975),
        positive_fraction=lambda s:s.gt(1e-12).mean(),repeats='count').reset_index()
    summary.to_csv(OUT/'main_table.csv',index=False)
    pairs.to_csv(OUT/'paired_repeat_differences.csv',index=False)
    paired.to_csv(OUT/'paired_differences.csv',index=False)
    return summary,paired


def judge(summary):
    values = summary.loc[summary.scope.eq('confirmatory')]
    rev = values.pivot(index=CELL,columns='method',values='reverse_AUC')
    within = values.pivot(index=CELL,columns='method',values='within_site_AUC')
    rows = []
    for (k,b,way),value in (rev['E-L3']-rev['E-CE']).items():
        rows.append(dict(criterion='D1',check=f'k{k} b{b} {way}: reverse E-L3 − E-CE',
                         point=value,threshold=.10,passed=value>=.10))
    test_gap = (within['E-L3']-within['E-CE']).mean()
    rows.append(dict(criterion='D1',check='mean within-site E-L3 − E-CE',
                     point=test_gap,threshold=-.01,passed=test_gap>=-.01))
    averages = rev[['E-L3']+task.DEEP].mean()
    strongest = averages[task.DEEP].idxmax()
    l3_reverse = averages['E-L3']
    erm_gap = (within['E-L3']-within['D-ERM']).mean()
    rows.extend([
        dict(criterion='D2',check='mean E-L3 reverse AUC',
             point=l3_reverse,threshold=.55,passed=l3_reverse>=.55),
        dict(criterion='D2',check=f'mean E-L3 reverse AUC ≥ strongest deep baseline ({strongest})',
             point=l3_reverse,threshold=averages[strongest],passed=l3_reverse>=averages[strongest]),
        dict(criterion='D2',check='mean within-site E-L3 − D-ERM',
             point=erm_gap,threshold=-.01,passed=erm_gap>=-.01),
    ])
    table = pd.DataFrame(rows)
    flags = table.groupby('criterion').passed.all().to_dict()
    if flags['D1'] and flags['D2']:
        text = 'D1、D2 均满足。按预设将深度表示结果纳入论文 §6，进入 ABIDE-II 第二阶段。'
    elif flags['D1']:
        text = f'D1 满足，D2 未满足。按预设报告同一嵌入更换目标函数后的结构结果及深度基线比较，第二阶段待讨论。反向集最高深度基线为 {strongest}。'
    else:
        text = 'D1 未满足。按预设，T16-B 的结构结果未在四格中稳定复现，深度部分只作补充，论文以线性方法为主。'
    return table,flags|dict(decision=text,reverse_means=averages.to_dict())


def report():
    p = OUT/'config.json'
    config = json.loads(p.read_text())
    assert config['stage']=='training_complete' and (OUT/'exit_code.txt').read_text().strip()=='0'
    for f,digest in (config['hashes']|config['reused_hashes']).items():
        assert task.pilot.digest(f)==digest,f
    jobs = [json.loads(s) for s in (task.OUT/'splits.jsonl').read_text().splitlines()]
    meta = task.pilot.table().set_index('FILE_ID')
    pred,done,probe,weights,loss,online = collect(jobs,config,meta)
    refs = pd.read_csv(task.OUT/'reference_predictions.csv',float_precision='round_trip')
    checks = align_predictions(pred,refs)
    checks.to_csv(OUT/'consistency.csv',index=False)
    assert len(online)==len(checks)==60
    assert checks.predictions_identical.all() and checks.max_abs_score_difference.eq(0).all()
    assert checks.max_abs_probability_difference.eq(0).all()
    pred.to_csv(OUT/'predictions.csv',index=False)
    done.to_csv(OUT/'completed.csv',index=False)
    probe.to_csv(OUT/'site_probe_predictions.csv',index=False)
    weights.to_csv(OUT/'group_weights.csv',index=False)
    loss.to_csv(OUT/'epoch_losses.csv',index=False)
    summary,paired = aggregate(pred,refs,jobs)
    criteria,decision = judge(summary)
    criteria.to_csv(OUT/'criteria.csv',index=False)
    (OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    diag_fields = ['seconds','final_loss','domain_accuracy','probe_accuracy','probe_chance','probe_seconds']
    diagnostics = done.groupby(CELL+['stage'],as_index=False)[diag_fields].mean()
    diagnostics.to_csv(OUT/'diagnostics.csv',index=False)
    done.loc[done.stage.isin(task.HEADS)].groupby(CELL+['stage','C']).size().rename('n').reset_index().to_csv(
        OUT/'selected_C_distribution.csv',index=False)
    verification = dict(jobs=len(jobs),new_jobs=196,reused_jobs=4,stages=len(done),predictions=len(pred),
                        probe_predictions=len(probe),epochs=len(loss),inner_folds=3000,
                        consistency_checks=len(checks),max_abs_score_difference=0,max_abs_probability_difference=0,
                        training_input_and_reused_hashes_verified=True)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    confirm = summary.loc[summary.scope.eq('confirmatory')].drop(columns='scope')
    pairs = paired.loc[paired.scope.eq('confirmatory')].drop(columns='scope')
    text = '# T16-C 修订一正式运行\n\n'+decision['decision']
    text += '\n\n## 预设判读\n\n'+markdown(criteria)
    text += '\n\n## 主表（判读重复的均值）\n\n'+markdown(confirm)
    text += '\n\n## E-L3 的逐重复配对差值\n\n'+markdown(pairs)
    text += '\n\n## 训练与站点探针\n\n'+markdown(diagnostics)
    text += f"\n\n新跑 196 次，实测 {config['new_wall_seconds']/3600:.3f} 小时。另复用双进程计时的四个重复 0。\n"
    text += '''\n- k2 双向判读用重复 10–49，其余各格用 0–49。四格均值等权。全重复结果另见主表。
- 站点内 AUC 等权平均测试站点。男性为 SEX=1，仅纳入两类各至少 3 名男性的测试站点。反向集用合并 AUC。原预测阈值沿用。
- 配对范围为跨重复的 2.5%–97.5% 分位数。沿用 T13-B，只在正比例统计中把 ≤1e−12 的正差值当成数值相等。
- E-* 和 D-DANN 使用外部无标签池；D-ERM、D-GDRO 的模型训练只使用有标签训练集。超参数保持预设，没有调整。D-DANN 仅判别器输入加无仿射 LayerNorm。
- 探针使用同一可用池、各模型自己的标准化器和 FC128 表示，按站点分层做默认的 5 折 LR。探针的均匀机会水平为站点数的倒数。D-DANN 训练已接触该池的无标签数据。
- 已逐人复核 60 项旧参照。T12-D 覆盖 k2 每个方向的重复 0–19，T16-B rev1 覆盖 k2 双向重复 0–9 的 E-CE、E-L3，分数和概率最大绝对差均为 0。
- ABIDE-I 中没有 BNU 的观察被试，本任务没有 no_watch 子组。
'''
    (OUT/'REPORT.md').write_text(text)
    p.write_text(json.dumps(config|dict(stage='complete'),indent=2)+'\n')
    print('REPORT_COMPLETE '+json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    report()
