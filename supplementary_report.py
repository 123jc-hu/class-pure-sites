"""T18 复核逐人结果并写四份报告。"""
from pathlib import Path
import sys
import json

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
import supplementary_analyses as task
from abide_class_pure_sites_report import markdown
from bridge_design_report import method_metrics
import bridge_design_report as bridge_report

OUT = task.OUT
CELL = ['k','b','direction']


def read(path):
    return pd.read_csv(path,dtype={'FILE_ID':str},float_precision='round_trip',
                       keep_default_na=False,low_memory=False)


def reduce(frame,keys,field='AUC'):
    return frame.groupby(keys)[field].agg(mean='mean',range_low=lambda x:x.quantile(.025),
               range_high=lambda x:x.quantile(.975),repeats='size').reset_index()


def a1(path):
    frame = read(path/'a1_predictions.csv')
    assert not frame.duplicated(['group','mode','repeat','FILE_ID']).any()
    config = json.loads((path/'config.json').read_text())
    designs = config['designs']
    rows = []
    for (name,mode,r),g in frame.groupby(['group','mode','repeat']):
        spec = designs[name]
        assert len(g)==spec['subjects'] and g.target_site.nunique()==spec['sites']
        score = (roc_auc_score(g.target_site,g.site_prediction.astype(float)) if spec['binary']
                 else balanced_accuracy_score(g.target_site,g.site_prediction))
        rows.append(dict(group=name,mode=mode,repeat=r,value=score))
    full = pd.DataFrame(rows)
    full.to_csv(path/'A1_repeat_metrics.csv',index=False)
    done = pd.read_csv(OUT/'completed.csv',keep_default_na=False,
                       na_values={'C':[''],'warning_count':['']}).query('kind == "A1"')
    assert len(done)==4800 and done.id.is_unique
    fixed = json.loads((path/'fixed_C.json').read_text())
    result = []
    for name,g in full.groupby('group'):
        spec = designs[name]
        observed = g.loc[g['mode'].eq('observed'),'value']
        null = g.loc[g['mode'].eq('null'),'value']
        assert len(observed)==5 and len(null)==(0 if spec['binary'] else 100)
        group_done = done.loc[done.group.eq(name)]
        current = group_done.loc[group_done['mode'].eq('observed')]
        row = dict(group=name,subjects=spec['subjects'],S=spec['sites'],metric='AUC' if spec['binary'] else 'BA',
                   chance=1/spec['sites'],point=observed.mean(),range_low=observed.quantile(.025),
                   range_high=observed.quantile(.975),convergence_warnings=int(group_done.warning_count.sum()))
        if len(null):
            assert float(current.C.mode().min())==fixed[name]
            assert group_done.loc[group_done['mode'].eq('null'),'C'].eq(fixed[name]).all()
            ge = int((null>=observed.mean()).sum())
            row.update(fixed_C=fixed[name],permutation_max=null.max(),null_ge=ge,
                       p_plus_one=(1+ge)/101,p_display='<1/101' if ge==0 else f'{(1+ge)/101:.6f}')
        result.append(row)
    table = pd.DataFrame(result)
    table.to_csv(path/'A1_main_table.csv',index=False)
    return table


def a2(path):
    data = read(path/'a2_predictions.csv')
    counts = pd.read_csv(path/'A2_availability.csv')
    eligible = counts.loc[counts.eligible]
    expected = {(j.job_id,j.label,method) for j in eligible.itertuples()
                for method in [j.method]+(['L3'] if j.b==5 else [])}
    by = ['source_job_id']+CELL+['repeat','method','label']
    rows = []
    for key,g in data.groupby(by):
        n = g.site_role.value_counts()
        assert len(n)==2 and n.min()>=5
        rows.append(dict(zip(by,key))|dict(AUC=roc_auc_score(g.site_role.eq('asd_sites'),g.score),
                    case_sites_n=int(n.asd_sites),control_sites_n=int(n.td_sites)))
    raw = pd.DataFrame(rows)
    assert set(raw[['source_job_id','label','method']].itertuples(index=False,name=None))==expected
    raw.to_csv(path/'A2_repeat_metrics.csv',index=False)
    keys = CELL+['method','label']
    table = reduce(raw,keys).merge(raw.groupby(keys,as_index=False)[['case_sites_n','control_sites_n']].mean(),on=keys)
    table.to_csv(path/'A2_main_table.csv',index=False)
    skipped = counts.loc[~counts.eligible]
    skipped.to_csv(path/'A2_skipped.csv',index=False)
    (path/'REPORT.md').write_text('# 第一部分 站点可识别性\n\n## A1 原始特征\n\n'+markdown(a1(path))
        +'\n\n五个种子按站点分层，每次合并折外预测计算 BA。内层用五折 BA 选参。'
        'ABIDE-II 的 ASD 子集有个别内层验证折缺少 EMC，仍保留五折并记录警告。'
        '置换仅用于九个多站点队列，外层划分种子固定 0，按置换后的站点分层。固定 C 为 25 个观测外折的众数，'
        '并列取较小 C。固定 C 的置换 p 值略偏乐观。零次达到观测值时，表中按修订报告 <1/101，并另存加一值。'
        '三个同机构子集合对报告同一套嵌套流程的 AUC。\n\n## A2 同诊断站点 AUC\n\n'
        +markdown(table)+'\n\n病例站点为正类，使用原模型诊断分数。人数不足的诊断层不计算。'
        '\n\n'+markdown(skipped)+'\n')


def old_normative_scores():
    root = task.ROOT/'outputs'
    base = read(root/'T12A/predictions.csv').query('not no_zero_roi and arm == "P" and n == 40 and k in [2,4]')
    cross = read(root/'T12C/predictions.csv').query('method == "C0" and b == 5')
    opts = [read(root/f'T13B/{source}.csv').query('method == "L3" and feature == "CC200" and b == 5')
            for source in ['predictions','reference_predictions']]
    return pd.concat([base.assign(b=0,direction='none',method='B0',score=base.probability),
                      cross.assign(score=cross.probability),*opts],ignore_index=True)


def normative_metrics(data):
    rows = []
    keys = CELL+['method','repeat','reverse']
    for key,g in data.groupby(keys):
        assert g.FILE_ID.is_unique
        value = roc_auc_score(g.label,g.score) if key[-1] else np.mean(
            [roc_auc_score(s.label,s.score) for _,s in g.groupby('site')])
        rows.append(dict(zip(keys,key))|dict(AUC=value,n=len(g)))
    return pd.DataFrame(rows)


def normative(path):
    frame = read(path/'o_predictions.csv')
    keys = CELL+['repeat','reverse','FILE_ID']
    sample = frame.loc[frame.method.eq('O1'),keys+['site','label']]
    assert len(sample)==len(sample.drop_duplicates(keys))
    old = old_normative_scores()
    ref = old[keys+['method','score']].merge(sample,on=keys,validate='many_to_one')
    assert len(ref)==len(sample)+int(sample.b.eq(5).sum())
    ref.to_csv(path/'reference_predictions.csv',index=False)
    raw = normative_metrics(pd.concat([frame,ref],ignore_index=True))
    raw.to_csv(path/'repeat_metrics.csv',index=False)
    summary = reduce(raw,CELL+['method','reverse'])
    assert summary.repeats.eq(50).all()
    summary.to_csv(path/'main_table.csv',index=False)
    rev = summary.loc[summary.reverse]
    oa = rev.query('b==0 and method=="O1"')['mean'].mean()
    cmp = rev.query('b==5 and direction=="oneway"').groupby('method')['mean'].mean()
    checks = dict(O_a=bool(oa<.5),O_a_O1=oa,O_b=bool(cmp.O3>=.5 and cmp.O3>cmp.O1),O_b_O3=cmp.O3,O_b_O1=cmp.O1)
    (path/'criteria.json').write_text(json.dumps(checks,indent=2)+'\n')
    (path/'REPORT.md').write_text('# 第三部分 单分类与规范模型\n\n'+markdown(summary)+'\n\n'
        +markdown(pd.DataFrame([checks]))+'\n\nPCA 固定 30 个成分。O2 使用 RBF 核和 sklearn 默认 gamma，'
        '以训练 TD 的五折平均异常分数选 ν。各法评估名单相同，测试站点用于校准的 TD 不参与评估。'
        '站点内 AUC 等权平均。O-a 与 O-b 对两个 k 等权判读，各格数值并列。\n')


def institutions(path,first):
    frame = read(path/'s_predictions.csv')
    frame = frame.assign(k=3,b=0,direction='none',seed=frame.repeat,n_pure=78,n_added=0,n_train=78,
                         T12A_job_id='',SEX=frame.FILE_ID.map(task.a.subjects().set_index('FILE_ID').SEX))
    bridge_report.OUT = path
    runs,summary = method_metrics(frame,n_test_sites=5)
    by = runs.loc[runs.reverse].groupby('method').pooled_AUC.mean()
    same = frame.loc[frame.method.eq('P_same')&frame.reverse]
    rows = [dict(site=s,repeat=r,reverse_AUC=roc_auc_score(g.label,g.score))
            for (s,r),g in same.groupby(['site','repeat'])]
    q = pd.DataFrame(rows).groupby('site',as_index=False).reverse_AUC.mean()
    paired = pd.read_csv(first/'A1_main_table.csv').set_index('group')
    q['subset_AUC'] = q.site.map(paired.point)
    direction = q[['subset_AUC','reverse_AUC']].rank().corr().iloc[0,1]
    q.to_csv(path/'institution_comparison.csv',index=False)
    wide = runs.loc[~runs.reverse].pivot(index='repeat',columns='method',values='within_site_AUC')
    cost = (wide.M_same-wide.P_same).rename('M_same_minus_P_same').reset_index()
    gap = runs.loc[runs.reverse].pivot(index='repeat',columns='method',values='pooled_AUC')
    cost['P_same_minus_P_cross_reverse'] = (gap.P_same-gap.P_cross).reindex(cost.repeat).to_numpy()
    cost.to_csv(path/'paired_repeat_differences.csv',index=False)
    result = dict(S_a=bool(by.P_same>by.P_cross),P_same_reverse=by.P_same,P_cross_reverse=by.P_cross,
                  S_b_direction='negative' if direction<0 else 'positive' if direction>0 else 'zero',
                  test_cost=cost.M_same_minus_P_same.mean())
    (path/'criteria.json').write_text(json.dumps(result,indent=2)+'\n')
    (path/'REPORT.md').write_text('# 第四部分 同机构拆分\n\n'+markdown(summary)+'\n\n'
        +markdown(pd.DataFrame([result]))+'\n\n'+markdown(q)+'\n\n'
        'M_same 与 P_same 比较共同的测试站点。P_same 与 P_cross 分别使用各自训练站点的原生反向集。'
        'S-b 只报告三家机构的描述性方向。\n\n'
        '若站点效应相关，方差中的站点项为 Σₛ,ₜ cₛ cₜ Cov(uₛ,uₜ)。'
        '当同机构的两集合分别进入相反诊断角色，cₛ 与 cₜ 异号，正协方差使这对集合的交叉项为负。'
        '在其他协方差项给定时，配对的这项贡献使总方差减小。\n')


def verify_splits():
    counts = {}
    for part,stem,n in [('part1','a1',4800),('part1','a2',500),('part3','o',300),('part4','s',150)]:
        path = OUT/part
        p = read(path/(stem+'_predictions.csv'))
        key,lookup = {
            'a1':(['group','mode','repeat','fold'],('group','mode','repeat','fold')),
            'a2':(['source_job_id','method'],('anchor','method')),
            'o':(['source_job_id'],('anchor',)),
            's':(['job_id'],('design',)),
        }[stem]
        p = p.groupby(key if len(key)>1 else key[0],sort=False).FILE_ID.agg(set).to_dict()
        total = 0
        with (path/(stem+'_splits.jsonl')).open() as f:
            for line in f:
                j = json.loads(line)
                k = tuple(j[name] for name in lookup)
                predicted = p[k[0] if len(k)==1 else k]
                if stem=='a2':
                    test = {i for groups in j['evaluation'].values() if min(map(len,groups.values()))>=5
                            for ids in groups.values() for i in ids}
                else:
                    test = set(j['test'])|set(j.get('reverse',[]))
                tr = set(j['train_td'] if stem=='o' else j['train'])
                assert predicted==set(test) and tr.isdisjoint(test)
                if stem=='o':
                    assert set(j['calibration']).isdisjoint(test|tr)
                for inner in j.get('inner',[]):
                    train,valid = set(inner['train']),set(inner['validation'])
                    assert train.isdisjoint(valid) and train|valid==tr
                    assert inner.get('standardizer_fit',inner['train'])==inner['train']
                total += 1
        assert total==n and len(p)==n
        counts[stem]=total
    (OUT/'verification.json').write_text(json.dumps(dict(job_counts=counts,all_saved_split_memberships_verified=True),
                                                    indent=2)+'\n')


def run():
    assert sys.flags.isolated and (OUT/'exit_code.txt').read_text().strip()=='0'
    status = json.loads((OUT/'formal_status.json').read_text())
    done = pd.read_csv(OUT/'completed.csv')
    assert status['stage']=='training_complete' and len(done)==status['work'] and done.id.is_unique
    verify_splits()
    first = OUT/'part1'
    a2(first)
    normative(OUT/'part3')
    institutions(OUT/'part4',first)
    print('REPORT_COMPLETE',flush=True)


if __name__=='__main__':
    run()
