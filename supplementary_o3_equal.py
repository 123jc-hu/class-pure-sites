"""T18 修订二，只重算等量参照的 O3。"""
from pathlib import Path
import json
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import supplementary_analyses as task
from supplementary_report import CELL, markdown, normative_metrics, read, reduce


def run():
    before = task.OUT/'part3'
    out = task.OUT/'part3_eq'
    table = task.a.subjects()
    assert read(task.a.CACHE/'row_order.csv').FILE_ID.equals(table.FILE_ID)
    features = np.load(task.a.CACHE/'fisher_z.npy',mmap_mode='r')
    anchors = {j['job_id']:j for j in task.anchors() if j['b']==5}
    old_splits = {j['anchor']:j for j in map(json.loads,(before/'o_splits.jsonl').read_text().splitlines())}
    old = read(before/'o_predictions.csv').query('method == "O3"')
    print('START O3-eq 200 jobs',flush=True)
    start = perf_counter()
    frames,splits = [],[]
    for key,job in anchors.items():
        split = task.normative_plan(table,job)
        assert all(split[k]==old_splits[key][k] for k in split)
        split['equal_site_td'] = task.equal_site_reference(table,job['repeat'],split['train_td'])
        frame,_ = task.normative_predict(features,table,job,split)
        frames.append(frame)
        splits.append(dict(anchor=key,seed=job['repeat'])|split)
        print(f'TRAIN_DONE {key}',flush=True)

    predictions = pd.concat(frames,ignore_index=True)
    keys = ['source_job_id','FILE_ID','reverse','site','label']
    paired = old[keys+['score']].merge(predictions[keys+['score']],on=keys,validate='one_to_one',suffixes=('_O3','_eq'))
    assert len(paired)==len(old)==len(predictions)
    unchanged = np.isclose(paired.score_O3,paired.score_eq,rtol=1e-11,atol=1e-11)
    assert unchanged[~paired.reverse].all()
    raw = normative_metrics(predictions)
    reference = read(before/'repeat_metrics.csv').query('method == "O3"')
    paired_auc = raw.merge(reference,on=CELL+['repeat','reverse'],validate='one_to_one',suffixes=('_eq','_O3'))
    assert len(paired_auc)==len(raw)==len(reference)==400
    assert np.allclose(paired_auc.loc[~paired_auc.reverse,'AUC_eq'],paired_auc.loc[~paired_auc.reverse,'AUC_O3'],atol=1e-12,rtol=0)
    paired_auc['difference'] = paired_auc.AUC_eq-paired_auc.AUC_O3
    summary = reduce(raw,CELL+['reverse']).merge(reduce(reference,CELL+['reverse'])[
        CELL+['reverse','mean']].rename(columns={'mean':'O3_mean'}),on=CELL+['reverse'],validate='one_to_one')
    assert summary.repeats.eq(50).all()
    summary['difference'] = summary['mean']-summary.O3_mean
    summary['pass'] = np.where(summary.reverse,summary['mean'].ge(.45),'')
    passed = bool(summary.loc[summary.reverse,'mean'].ge(.45).all())

    out.mkdir(exist_ok=True)
    (out/'config.json').write_text(json.dumps(dict(method='O3-eq',seeds=list(range(50)),pca_components=30,
        reference_TD_per_training_site=5,calibration_source=str(before/'o_splits.jsonl'),threshold=.45,
        all_four_pass=passed,seconds=perf_counter()-start),indent=2)+'\n')
    (out/'o_splits.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in splits))
    predictions.to_csv(out/'o_predictions.csv',index=False)
    paired_auc.to_csv(out/'repeat_metrics.csv',index=False)
    summary.to_csv(out/'main_table.csv',index=False)
    conclusion = '四格均达到 .45，支持预设的参照人数解释。' if passed else '至少一格低于 .45，未达到预设标准。'
    (out/'REPORT.md').write_text('# T18 修订二 O3-eq\n\n'+conclusion+'\n\n'
        +'每个训练站点按重复号抽 5 名训练 TD，仅用于估计本站均值。PCA 30 成分和 Ledoit–Wolf 协方差仍用全部训练 TD。'
        +'逐人核对测试站点校准名单和分数，沿用原 O3 的评估名单。下表范围为重复值的 2.5% 和 97.5% 分位数。\n\n'
        +markdown(summary)+'\n')
    print(f'FINISH O3-eq all_four_pass={passed} seconds={perf_counter()-start:.1f}',flush=True)


if __name__=='__main__':
    run()
