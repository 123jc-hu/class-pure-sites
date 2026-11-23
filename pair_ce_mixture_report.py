"""T16-A 的参照配对、取舍曲线和 W1–W3 汇总。"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import bridge_design_report as metrics
from abide_class_pure_sites_report import markdown
import pair_ce_mixture as task

OUT = task.OUT
CELL = ['k', 'b', 'direction']
AUC = ['within_site_AUC', 'male_within_site_AUC', 'pooled_AUC', 'reverse_AUC']


def read(name):
    return pd.read_csv(OUT/f'{name}.csv', float_precision='round_trip')


def verify(cfg, jobs, done, predictions, choices, inner, endpoints):
    assert cfg['stage'] == 'training_complete' and (OUT/'exit_code.txt').read_text().strip() == '0'
    for path, expected in cfg['hashes'].items():
        assert task.digest(path) == expected, path
    assert len(jobs) == len(done) == 600 and done.job_id.is_unique and set(done.job_id) == set(jobs)
    assert len(choices) == 5400 and len(inner) == done.grid_fits.sum() == 84000
    assert len(endpoints) == done.endpoint_reference_fits.sum() == 72
    assert np.isfinite(predictions.score).all() and not predictions.duplicated(
        ['job_id', 'method', 'lambda', 'FILE_ID']).any()
    meta = task.l3.subjects().set_index('FILE_ID')
    selected = choices.loc[choices.method.ne('fixed_lambda')].set_index(['job_id', 'method'])
    scores = inner.groupby(['job_id', 'lambda', 'C'])[['auc_W', 'auc_S']].mean()
    fold_records = list(map(json.loads, (OUT/'inner_splits.jsonl').read_text().splitlines()))
    assert len(fold_records) == 2400
    for f in fold_records:
        j = jobs[f['job_id']]
        train, val = set(f['train']), set(f['validation'])
        assert train.isdisjoint(val) and train | val == set(j['train'])
        assert not (train | val)&set(j['test']+j['reverse'])
        assert f['train'] == f['standardizer_fit']
    for key, f in predictions.groupby('job_id'):
        j = jobs[key]
        assert len(f) == 9*(len(j['test'])+len(j['reverse']))
        assert set(f.loc[f.reverse, 'FILE_ID']) == set(j['reverse'])
        assert set(f.loc[~f.reverse, 'FILE_ID']) == set(j['test'])
        for field in ['site', 'DX_GROUP', 'SEX', 'label']:
            np.testing.assert_array_equal(f[field], meta.loc[f.FILE_ID, field])
        for (_, _), part in f.groupby(['method', 'lambda']):
            assert len(part) == len(j['test'])+len(j['reverse'])
            assert set(part.loc[part.reverse, 'FILE_ID']) == set(j['reverse'])
            assert set(part.loc[~part.reverse, 'FILE_ID']) == set(j['test'])
        for method, metric in [('P1-W', 'auc_W'), ('P1-S', 'auc_S')]:
            q = selected.loc[key, method]
            assert (q['lambda'], q.C) == tuple(scores.loc[key, metric].idxmax())
            fixed = f.loc[f.method.eq('fixed_lambda')&f['lambda'].eq(q['lambda'])&f.C.eq(q.C)]
            assert bool(len(fixed)) == bool(q.reused)
            if q.reused:
                other = f.loc[f.method.eq(method)].set_index('FILE_ID').loc[fixed.FILE_ID]
                np.testing.assert_array_equal(fixed.score, other.score)
        for lam, point in choices.loc[choices.job_id.eq(key)&choices.method.eq('fixed_lambda')].groupby('lambda'):
            assert point.C.iloc[0] == scores.loc[(key, lam), 'auc_W'].idxmax()
    assert choices.loc[choices.method.eq('P1-W'), 'reused'].all()
    solo_s = int((~selected.xs('P1-S', level='method').reused).sum())
    assert done.final_fits.sum() == 4200+solo_s
    result = dict(jobs=600, predictions=len(predictions), inner_folds=2400, grid_fits=84000,
                  final_fits=int(done.final_fits.sum()), added_P1_S_fits=solo_s,
                  endpoint_reference_fits=72, endpoint_warnings=int(endpoints.warning.notna().sum()),
                  input_hashes_membership_and_reuse_verified=True)
    task.save_json('verification.json', result)
    return result


def auc_table(runs):
    identity = ['method']+CELL+['repeat']
    test = runs.loc[~runs.reverse, identity+['within_site_AUC', 'male_within_site_AUC', 'pooled_AUC']]
    reverse = runs.loc[runs.reverse, identity+['pooled_AUC']].rename(columns={'pooled_AUC':'reverse_AUC'})
    return test.merge(reverse, on=identity, validate='one_to_one')


def summarize_pairs(auc):
    rows = []
    for (k, b, direction), cell in auc.groupby(CELL):
        for metric in AUC:
            values = cell.pivot(index='repeat', columns='method', values=metric)
            assert len(values) == 50
            for left, right in [('P1-W','L3'), ('P1-W','C0'), ('P1-S','P1-W')]:
                for repeat, difference in (values[left]-values[right]).items():
                    rows.append(dict(k=k, b=b, direction=direction, repeat=repeat,
                                     metric=metric, contrast=left+' - '+right, difference=difference))
    raw = pd.DataFrame(rows)
    assert len(raw) == 7200 and raw.difference.notna().all()
    raw.to_csv(OUT/'paired_repeat_differences.csv', index=False)
    paired = raw.groupby(CELL+['metric','contrast']).difference.agg(
        mean='mean', range_low=lambda x: x.quantile(.025), range_high=lambda x: x.quantile(.975),
        positive_fraction=lambda x: x.gt(1e-12).mean(), repeats='count').reset_index()
    paired.to_csv(OUT/'paired_differences.csv', index=False)
    return paired


def criteria(auc):
    means = auc.loc[auc.b.eq(5)].groupby(CELL+['method'])[AUC].mean().unstack('method')
    reverse = means['reverse_AUC']
    test = means['within_site_AUC']
    rows = []
    for k, b, direction in means.index:
        key = (k, b, direction)
        for check, value, threshold in [
            ('W1', reverse.loc[key, 'P1-W']-reverse.loc[key, 'L3'], -.02),
            ('W2', test.loc[key, 'P1-W']-max(test.loc[key, 'C0'], test.loc[key, 'L3']), -.005),
        ]:
            rows.append(dict(criterion=check, k=k, b=b, direction=direction, gap=value,
                             threshold=threshold, passed=bool(value >= threshold)))
    value = (reverse['P1-W']-reverse['P1-S']).mean()
    rows.append(dict(criterion='W3', b=5, direction='four-cell mean', gap=value,
                     threshold=.05, passed=bool(value >= .05)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'criteria.csv', index=False)
    passed = frame.groupby('criterion').passed.all().to_dict()
    w1, w2, w3 = (passed[key] for key in ['W1','W2','W3'])
    if w1 and w2 and w3:
        conclusion = '三项均满足。按预设，建议进入 P1b。'
    elif w1 and w3:
        conclusion = 'W1 与 W3 满足，W2 未满足。按预设，保留选参准则的结果，回报讨论投稿方向。'
    elif not w1:
        conclusion = 'W1 未满足。按预设，站点内准则不足以挡住站点捷径，回报讨论。'
    else:
        conclusion = '此判据组合超出任务的判读表，报告 W1–W3 并停下讨论。'
    decision = {key: bool(value) for key, value in passed.items()} | {'conclusion': conclusion}
    task.save_json('decision.json', decision)
    return frame, decision


def report():
    cfg = json.loads((OUT/'config.json').read_text())
    jobs = {j['job_id']: j for j in map(json.loads, (OUT/'splits.jsonl').read_text().splitlines())}
    predictions, choices, done, inner, endpoints = map(read, [
        'predictions','choices','completed','inner_scores','endpoints'])
    verified = verify(cfg, jobs, done, predictions, choices, inner, endpoints)
    predictions['method'] = [f'fixed:{lam:g}' if method=='fixed_lambda' else method
                             for method, lam in zip(predictions.method, predictions['lambda'])]
    keys = predictions.job_id.map(jobs)
    for name in ['n_pure','n_added','n_train','T12A_job_id']:
        predictions[name] = keys.map(lambda job: job[name])
    predictions['job_id'] += ':'+predictions.method
    predictions['probability'] = predictions.score
    metrics.OUT = OUT
    runs, _ = metrics.method_metrics(predictions)
    reference = pd.read_csv(ROOT/'outputs/T13B/repeat_metrics.csv', float_precision='round_trip')
    reference = reference.loc[reference.feature.eq('CC200')&reference.method.isin(['C0','L3','B0'])]
    auc = pd.concat([auc_table(runs), auc_table(reference)], ignore_index=True)
    assert len(auc) == 7200 and not auc.duplicated(['method']+CELL+['repeat']).any()
    auc.to_csv(OUT/'all_repeat_AUC.csv', index=False)
    long = auc.melt(id_vars=['method']+CELL+['repeat'], value_vars=AUC, var_name='metric', value_name='value')
    summary = long.groupby(['method']+CELL+['metric']).value.agg(
        mean='mean', range_low=lambda x:x.quantile(.025), range_high=lambda x:x.quantile(.975),
        repeats='count').reset_index()
    summary.to_csv(OUT/'main_table.csv', index=False)
    combined = pd.concat([runs, reference], ignore_index=True)
    combined.groupby(['method']+CELL+['reverse'])[metrics.METRICS].mean().reset_index().to_csv(
        OUT/'classification_summary.csv', index=False)
    paired = summarize_pairs(auc)
    checks, decision = criteria(auc)
    curve = summary.loc[summary.method.str.startswith('fixed:')].copy()
    curve['lambda'] = curve.method.str.split(':').str[1].astype(float)
    curve.to_csv(OUT/'fixed_lambda_curve.csv', index=False)
    selected = choices.loc[choices.method.isin(['P1-W','P1-S'])]
    for parameter in ['lambda','C']:
        dist = selected.groupby(['method']+CELL+[parameter]).size().rename('count').reset_index()
        dist['fraction'] = dist['count']/50
        dist.to_csv(OUT/f'selected_{parameter}_distribution.csv', index=False)
    objective = endpoints.assign(abs_rel=endpoints.relative_difference.abs()).groupby('reference').agg(
        count=('job_id','size'), max_absolute_relative_difference=('abs_rel','max'),
        min_coefficient_cosine=('coefficient_cosine','min')).reset_index()
    main = summary.loc[summary.b.eq(5)&~summary.method.str.startswith('fixed:')&
                       summary.metric.isin(['within_site_AUC','reverse_AUC'])]
    text = '# T16-A P1a\n\n'+decision['conclusion']+'\n\n## b=5 判据\n\n'+markdown(checks)
    text += '\n\n## b=5 主要 AUC\n\n'+markdown(main)+'\n\n## b=5 配对差值\n\n'
    text += markdown(paired.loc[paired.b.eq(5)&paired.metric.isin(['within_site_AUC','reverse_AUC'])])
    text += '\n\n## 端点核对，只报告\n\n'+markdown(objective)+'\n\n'
    text += ('λ=0 与 T13-A L3 比较，λ=1 与默认 sklearn C0 比较，两侧使用同一训练集及同一 C。'
             'C 取固定 λ 曲线按 W 准则所选值，共 36 个任务、72 项端点核对。'
             '目标相对差按（混合实现−原参照）／原参照计算，系数余弦只比较 w。\n\n')
    text += ('两种准则在同一批内层打分上等权平均，按较小 λ、再按较小 C 解并列。'
             '内折按站点×诊断分层，标准化只用各自训练部分。'
             'λ=0 截距固定为零，其余 λ 的交叉熵含自由截距。'
             'AUC 使用线性分数，ASD 分类阈值为零。测试站点等权，男性每站点每类至少 3 人。'
             '范围为 50 次重复的 2.5%–97.5% 变异范围，正差比例以差值高于 1e−12 为正。'
             'b=2/10 的格子及固定 λ 曲线为描述性结果。ABIDE-I 的 watch 为 False。\n\n')
    text += ('全量主表、配对差值、λ 曲线、选参分布及逐条拟合复用关系见同目录 CSV。'
             f'正式运行 {cfg["formal_seconds"]/60:.2f} 分钟。验证：'+json.dumps(verified, ensure_ascii=False)+'\n')
    (OUT/'REPORT.md').write_text(text)
    cfg.update(stage='complete', decision=decision)
    task.save_json('config.json', cfg)
    print('REPORT_COMPLETE '+json.dumps(decision, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    assert sys.flags.isolated
    report()
