"""T16-B：配对评估、重构诊断和试验门槛。"""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import bridge_design_report as metric
from abide_class_pure_sites_report import markdown
import deep_pair_pilot as pilot

OUT = pilot.OUT
AUC = ['within_site_AUC', 'male_within_site_AUC', 'pooled_AUC', 'reverse_AUC']
STAGES = ['A-train','A-ext']+pilot.METHODS


def read(name):
    return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')


def verify(cfg, jobs, pred, ref, loss, done):
    assert cfg['stage']=='training_complete' and (OUT/'exit_code.txt').read_text().strip()=='0'
    for path, expected in cfg['hashes'].items():
        assert pilot.digest(path)==expected,path
    assert set(zip(done.repeat,done.stage))=={(r,s) for r in range(10) for s in STAGES}
    assert len(done)==70
    assert len(loss)==2500 and np.isfinite(loss.loss).all()
    assert not loss.duplicated(['stage','repeat','epoch']).any()
    for (repeat, stage), frame in loss.groupby(['repeat','stage']):
        epochs = 50 if stage=='B' else 100
        assert set(frame.epoch)==set(range(1,epochs+1))
        last = frame.loc[frame.epoch.eq(epochs),'loss'].iloc[0]
        assert last==done.loc[done.repeat.eq(repeat)&done.stage.eq(stage),'final_loss'].iloc[0]
    assert len(pred)==17755 and not pred.duplicated(['repeat','method','FILE_ID']).any()
    assert set(pred.method)==set(pilot.METHODS)
    np.testing.assert_array_equal(pred.prediction, pred.score.ge(0.))
    assert np.isfinite(pred.score).all() and pred.probability.between(0,1).all()
    subjects = pilot.table()
    meta = subjects.set_index('FILE_ID')
    for repeat, j in jobs.items():
        train, target, _ = pilot.indices(subjects,j)
        for method, source in [(name,pred) for name in pilot.METHODS]+[(name,ref) for name in pilot.REFS]:
            part = source.loc[source.repeat.eq(repeat)&source.method.eq(method)]
            assert len(part)==len(target)
            assert set(part.loc[part.reverse,'FILE_ID'])==set(j['reverse'])
            assert set(part.loc[~part.reverse,'FILE_ID'])==set(j['test'])
            for field in pilot.META[1:]:
                np.testing.assert_array_equal(part[field],meta.loc[part.FILE_ID,field])
        assert len(pilot.pair_indices(subjects,train)[0])==400
    folds = list(map(json.loads,(OUT/'head_inner_splits.jsonl').read_text().splitlines()))
    assert len(folds)==200
    for f in folds:
        j = jobs[f['repeat']]
        train, val = set(f['train']), set(f['validation'])
        assert train.isdisjoint(val) and train|val==set(j['train'])
        assert f['standardizer_fit']==f['train']
    for (repeat, method), folds in pd.DataFrame(folds).groupby(['repeat','method']):
        assert set(folds.fold)==set(range(5))
        values = pd.DataFrame(folds.validation_auc.tolist()).mean()
        assert float(values.idxmax())==done.loc[done.repeat.eq(repeat)&done.stage.eq(method),'C'].iloc[0]
    values = dict(repeats=10, trained_autoencoders=20, trained_pair_networks=10, linear_heads=40,
                  new_predictions=len(pred), existing_reference_predictions=len(ref),
                  epochs=len(loss), head_fold_records=200, all_hashes_splits_references_verified=True)
    pilot.dump('verification.json',values)
    return values


def auc_rows(pred, ref, jobs):
    scores = pd.concat([pred,ref],ignore_index=True)
    scores['job_id'] = scores.method + ':r' + scores.repeat.astype(str)
    scores['probability'] = scores.score
    for field in ['k','b','direction','n_pure','n_added','n_train','T12A_job_id']:
        scores[field] = scores.repeat.map({r:j[field] for r,j in jobs.items()})
    metric.OUT = OUT
    runs, _ = metric.method_metrics(scores,repeats=10)
    test = runs.loc[~runs.reverse, ['repeat','method']+AUC[:3]]
    reverse = runs.loc[runs.reverse, ['repeat','method','pooled_AUC']].rename(columns={'pooled_AUC':'reverse_AUC'})
    values = test.merge(reverse,on=['repeat','method'],validate='one_to_one')
    assert len(values)==80
    return values


def comparisons(values):
    rows = []
    for field in AUC:
        wide = values.pivot(index='repeat',columns='method',values=field)
        contrasts = [(m,ref) for m in pilot.METHODS for ref in ['deep_ERM','linear_L3']]
        contrasts += [('A-train-L3','A-train-CE'),('A-ext-L3','A-ext-CE')]
        for left,right in contrasts:
            for repeat, gap in (wide[left]-wide[right]).items():
                rows.append(dict(metric=field,contrast=f'{left} - {right}',repeat=repeat,difference=gap))
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT/'paired_repeat_differences.csv',index=False)
    summary = raw.groupby(['contrast','metric']).difference.agg(
        mean='mean',range_low=lambda x:x.quantile(.025),range_high=lambda x:x.quantile(.975),
        positive_fraction=lambda x:x.gt(1e-12).mean(),repeats='count').reset_index()
    summary.to_csv(OUT/'paired_differences.csv',index=False)
    return summary


def judge(means):
    rows = []
    baseline = means.loc['deep_ERM','within_site_AUC']
    for method in ['A-train-L3','A-ext-L3','B']:
        auc = means.loc[method,'reverse_AUC']
        gap = means.loc[method,'within_site_AUC']-baseline
        t1 = bool(auc>=.55 and gap>=-.01)
        rows.append(dict(criterion='T1',method=method,reverse_AUC=auc,test_gap_from_ERM=gap,
                         applicable=True,passed=t1))
        if method.startswith('A'):
            diff = auc-means.loc[method.replace('-L3','-CE'),'reverse_AUC']
            rows.append(dict(criterion='T2',method=method,reverse_gap_from_CE=diff,
                             applicable=t1,passed=bool(diff>=.10) if t1 else None))
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'criteria.csv',index=False)
    qualifying = result.loc[(result.criterion=='T1')&result.passed.eq(True),'method'].tolist()
    conclusion = ('至少一项满足 T1。按预设，进入完整 P1b。' if qualifying else
                  '各方法均未满足 T1。按预设，完整 P1b 到此停止，深度试验结果作为补充材料。')
    decision = dict(T1_methods=qualifying,conclusion=conclusion)
    pilot.dump('decision.json',decision)
    return result, decision


def report():
    cfg = json.loads((OUT/'config.json').read_text())
    jobs = {j['repeat']:j for j in map(json.loads,(OUT/'splits.jsonl').read_text().splitlines())}
    pred, ref, loss, done = map(read,['predictions','reference_predictions','epoch_losses','completed'])
    checked = verify(cfg,jobs,pred,ref,loss,done)
    auc = auc_rows(pred,ref,jobs)
    auc.to_csv(OUT/'repeat_AUC.csv',index=False)
    means = auc.groupby('method')[AUC].mean()
    summary = auc.melt(id_vars=['repeat','method'],var_name='metric',value_name='value').groupby(
        ['method','metric']).value.agg(mean='mean',range_low=lambda x:x.quantile(.025),
        range_high=lambda x:x.quantile(.975),repeats='count').reset_index()
    summary.to_csv(OUT/'auc_summary.csv',index=False)
    paired = comparisons(auc)
    criteria, decision = judge(means)
    heads = done.loc[done.C.notna()].groupby(['stage','C']).size().rename('count').reset_index()
    heads.to_csv(OUT/'head_C_distribution.csv',index=False)
    diagnostics = done.groupby('stage').agg(
        mean_seconds=('seconds','mean'),total_seconds=('seconds','sum'),final_loss_mean=('final_loss','mean'),
        final_loss_min=('final_loss','min'),final_loss_max=('final_loss','max')).reset_index()
    diagnostics.to_csv(OUT/'diagnostics.csv',index=False)
    reconstruction = loss.groupby(['stage','repeat']).loss.agg(first='first',last='last',max='max')
    reconstruction.to_csv(OUT/'loss_summary.csv')
    timing = json.loads((OUT/'timing.json').read_text())
    elapsed = (timing['pilot_seconds']+cfg['formal_seconds'])/60
    detail = '# T16-B 修订一：深度配对试验\n\n'+decision['conclusion']+'\n\n## T1 / T2\n\n'+markdown(criteria.fillna('—'))
    detail += '\n\n## 四项 AUC 均值\n\n'+markdown(means.reset_index())
    detail += '\n\n## 主要 AUC 配对差值\n\n'+markdown(paired.loc[paired.metric.isin(['within_site_AUC','reverse_AUC'])])
    detail += '\n\n## 训练损失与计时\n\n'+markdown(diagnostics.fillna('—'))+'\n\n'+markdown(heads)+'\n\n'
    detail += (
        '按指定 T12-C 名单，每次为 100 名训练被试和 4 个训练站点，含 20 名桥接者。'
        '站点内配对共 400 对，B 每轮 29 批，深度 ERM 每轮 8 批。'
        'A-ext 的无标签池由训练者和训练、测试站点之外的被试组成，预训练及边标准化共用该池。'
        'A 的内层只选线性头的 C，固定编码器使用该重复完整的无标签预训练池，嵌入的标准化仍在各头训练折内拟合。'
        '每个样本均匀选择 3,980 条无向边并成对清零，对全部标准化边重构。'
        '解码器和 B 的标量输出层沿用骨干的 fan-in Xavier 初始化。'
        '损失为训练批按被试数或配对数加权的 epoch 均值，最终模型取固定最后一轮。\n\n'
        f'修订一的自编码器 Adam 学习率为 {pilot.AE_LR:g}。十次重复均从头训练。'
        '非有限损失立即终止。有限发散的操作规则为连续三个 epoch 损失均严格递增且均大于首轮两倍。'
        '训练过程只检查训练损失。'
        'A-L3、B 和线性 L3 使用原始线性分数计算 AUC，A-CE 的线性分数与其预测概率排序一致。'
        '差值范围取 10 次重复第 2.5 和 97.5 百分位，正差以高于 1e−12 计。'
        'T2 只判读满足 T1 的 A 方法。结果用于决定下一步试验。\n\n'
        f'点估 {timing["point_hours"]*60:.2f} 分钟，含 25% 余量 {timing["with_25percent_margin_hours"]*60:.2f} 分钟。'
        f'含计时的实际训练耗时 {elapsed:.2f} 分钟。修订版计时重复 0 纳入本次结果。'
        '全部产物在本目录，模型和嵌入保存在运行内存中。验证 '+json.dumps(checked,ensure_ascii=False)+'。\n'
    )
    (OUT/'REPORT.md').write_text(detail,encoding='utf-8')
    cfg.update(stage='complete',decision=decision,active_training_minutes=elapsed)
    pilot.dump('config.json',cfg)
    print('REPORT_COMPLETE '+json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    report()
