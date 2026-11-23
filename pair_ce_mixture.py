"""T16-A 的拟合、共享内层验证和计时入口。"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from hashlib import file_digest
from itertools import product
from multiprocessing import get_context
from pathlib import Path
import json
import sys
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import within_site_pair_model as l3

OUT = ROOT / 'outputs/T16A'
SOURCE = ROOT / 'outputs/T12C/splits.jsonl'
LAMBDAS = (0., .1, .3, .5, .7, .9, 1.)
GRID = tuple(product(LAMBDAS, l3.GRID))
KEY = ['job_id', 'k', 'b', 'direction', 'repeat', 'seed']
PRED_META = ['FILE_ID', 'site', 'DX_GROUP', 'SEX', 'label']
WORKERS = 4
pairs = l3.pairs


def plans():
    return [j for j in map(json.loads, SOURCE.read_text().splitlines()) if j['method'] == 'C0']


def digest(path):
    with Path(path).open('rb') as f:
        return file_digest(f, 'sha256').hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def fit_weight(z, y, left, right, lam, C):
    n, p = z.shape
    penalty = 1. / (C*n)

    def objective(theta):
        w = theta[:p]
        scores = z @ w
        loss = .5 * penalty * (w @ w)
        residual = np.zeros(n)
        if lam < 1.:
            d = scores[right] - scores[left]
            r = expit(d) * ((1.-lam)/len(left))
            loss += (1.-lam) * np.logaddexp(0., d).mean()
            residual += np.bincount(right, weights=r, minlength=n)
            residual -= np.bincount(left, weights=r, minlength=n)
        if lam > 0.:
            scores += theta[p]
            loss += lam * np.logaddexp(0., (1.-2*y)*scores).mean()
            ce_residual = (expit(scores)-y)*(lam/n)
            residual += ce_residual
        grad = z.T @ residual + penalty*w
        return loss, np.r_[grad, ce_residual.sum()] if lam > 0. else grad

    result = minimize(objective, np.zeros(p + int(lam > 0.)), jac=True,
                      method='L-BFGS-B',
                      options=dict(maxiter=5000, gtol=1e-8, ftol=1e-12, maxls=50))
    assert result.success, (lam, C, result.message)
    theta = result.x if lam > 0. else np.r_[result.x, 0.]
    return theta, float(result.fun), int(result.nit)


def endpoint_reference(z, y, sites, C, lam):
    if lam == 0.:
        ref = l3.fit_weight(z, y, sites, 'L3', C)
        return ref['w'], ref['objective'], ''
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        lr = LogisticRegression(C=C).fit(z, y)
    w = lr.coef_[0]
    objective = np.logaddexp(0., (1-2*y)*lr.decision_function(z)).mean()
    objective += (w @ w)/(2*C*len(y))
    return w, float(objective), ' | '.join(str(item.message) for item in caught)


def fit_job(job, data):
    table, x = data
    index = pd.Index(table.FILE_ID)
    train = index.get_indexer(job['train'])
    target = index.get_indexer(job['test'] + job['reverse'])
    assert min(train.min(), target.min()) >= 0 and not set(train)&set(target)
    assert set(table.loc[train, 'site']).isdisjoint(table.loc[index.get_indexer(job['test']), 'site'])
    y = table.label.to_numpy()
    sites = table.site.to_numpy()
    context = {key: job[key] for key in KEY}
    inner_scores, inner_splits, endpoints = [], [], []
    for fold, (tr, va) in enumerate(l3.inner_folds(table, train, job['seed'], min(5, job['b']))):
        assert not set(tr)&set(va) and set(tr)|set(va) == set(train)
        assert not (set(tr)|set(va))&set(target)
        tr_ids = index[tr].tolist()
        inner_splits.append(context | dict(fold=fold, train=tr_ids, validation=index[va].tolist(),
                                           standardizer_fit=tr_ids))
        scale = StandardScaler().fit(x[tr])
        z = scale.transform(x[tr])
        val = scale.transform(x[va])
        left, right = pairs(y[tr], sites[tr])
        va_left, va_right = pairs(y[va], sites[va])
        assert len(left) and len(va_left)
        for lam, C in GRID:
            theta, objective, iterations = fit_weight(z, y[tr], left, right, lam, C)
            score = val @ theta[:-1] + theta[-1]
            delta = score[va_left]-score[va_right]
            inner_scores.append(context | {'lambda': lam, 'C': C, 'fold': fold,
                'auc_W': np.mean((delta > 0) + .5*(delta == 0)),
                'auc_S': roc_auc_score(y[va], score), 'validation_pairs': len(delta),
                'objective': objective, 'iterations': iterations})
    scores = pd.DataFrame(inner_scores)
    means = scores.groupby(['lambda', 'C'])[['auc_W', 'auc_S']].mean()
    fixed = [(lam, float(means.loc[lam, 'auc_W'].idxmax())) for lam in LAMBDAS]
    choice_w = tuple(means.auc_W.idxmax())
    choice_s = tuple(means.auc_S.idxmax())
    scale = StandardScaler().fit(x[train])
    z = scale.transform(x[train])
    test = scale.transform(x[target])
    left, right = pairs(y[train], sites[train])
    fitted = {}
    picks, predictions = [], []
    for method, (lam, C) in [('fixed_lambda', key) for key in fixed] + [
            ('P1-W', choice_w), ('P1-S', choice_s)]:
        key = (lam, C)
        reused = key in fitted
        if not reused:
            fit_id = f'fixed:{lam:g}' if method == 'fixed_lambda' else method
            theta, objective, iterations = fit_weight(z, y[train], left, right, lam, C)
            fitted[key] = (fit_id, theta, objective, iterations)
        fit_id, theta, objective, iterations = fitted[key]
        values = context | {'method': method, 'lambda': lam, 'C': C, 'source_fit_id': fit_id}
        picks.append(values | dict(reused=reused, objective=objective, iterations=iterations,
                                   auc_W=means.loc[key, 'auc_W'], auc_S=means.loc[key, 'auc_S']))
        frame = table.loc[target, PRED_META].assign(**values)
        frame['reverse'] = np.arange(len(target)) >= len(job['test'])
        frame['score'] = test @ theta[:-1] + theta[-1]
        frame['probability'] = expit(frame.score)
        frame['prediction'] = frame.score.ge(0)
        frame['watch'] = False
        predictions.append(frame)
        if method == 'fixed_lambda' and job['repeat'] < 3 and lam in (0., 1.):
            w, old_objective, warning = endpoint_reference(z, y[train], sites[train], C, lam)
            cosine = (w @ theta[:-1])/(np.linalg.norm(w)*np.linalg.norm(theta[:-1]))
            endpoints.append(values | dict(reference='L3' if lam == 0. else 'C0_default',
                reference_objective=old_objective, mixture_objective=objective,
                relative_difference=(objective-old_objective)/old_objective,
                coefficient_cosine=cosine, warning=warning))
    return dict(inner_scores=scores, inner_splits=inner_splits, choices=pd.DataFrame(picks),
                predictions=pd.concat(predictions, ignore_index=True), endpoints=pd.DataFrame(endpoints),
                done=context | dict(grid_fits=len(scores), final_fits=len(fitted),
                                    endpoint_reference_fits=len(endpoints)))


def load_data():
    global DATA
    assert sys.flags.isolated
    table = l3.subjects()
    assert table.FILE_ID.tolist() == pd.read_csv(l3.a.CACHE/'row_order.csv').FILE_ID.tolist()
    DATA = (table, np.load(l3.a.CACHE/'fisher_z.npy', mmap_mode='r'))


def worker(job):
    tick = perf_counter()
    result = fit_job(job, DATA)
    result['done']['seconds'] = perf_counter()-tick
    return result


def append_csv(name, frame):
    with (OUT / f'{name}.csv').open('a') as dest:
        frame.to_csv(dest, index=False, header=dest.tell() == 0)


def run_jobs(jobs):
    pending_jobs = iter(jobs)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context('spawn'),
                             initializer=load_data) as pool:
        pending = {pool.submit(worker, job) for job in
                   (next(pending_jobs, None) for _ in range(WORKERS)) if job is not None}
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                r = future.result()
                for name in ['predictions', 'inner_scores', 'choices', 'endpoints']:
                    if not r[name].empty:
                        append_csv(name, r[name])
                with (OUT/'inner_splits.jsonl').open('a') as dest:
                    for fold in r['inner_splits']:
                        dest.write(json.dumps(fold) + '\n')
                append_csv('completed', pd.DataFrame([r['done']]))
                print('TRAIN_DONE '+json.dumps(r['done']), flush=True)
                job = next(pending_jobs, None)
                if job is not None:
                    pending.add(pool.submit(worker, job))


def timing():
    from importlib.metadata import version
    import shutil

    OUT.mkdir()
    inputs = [SOURCE, l3.a.CACHE/'fisher_z.npy', l3.a.CACHE/'row_order.csv',
              ROOT/'outputs/T12A/subjects.csv']
    inputs += [ROOT/'outputs/T13B'/name for name in
               ['predictions.csv', 'reference_predictions.csv', 'repeat_metrics.csv', 'selected_C.csv']]
    sources = [ROOT/name for name in
               ['pair_ce_mixture.py', 'within_site_pair_model.py', 'abide_class_pure_sites.py',
                'bridge_design_report.py', 'tests/test_pair_ce_mixture.py', 'AGENTS.md',
                'docs/tasks/T16A_p1a_pair_ce_mixture.md']]
    jobs = plans()
    (OUT/'splits.jsonl').write_text(''.join(json.dumps(job)+'\n' for job in jobs))
    hashes = {str(path): digest(path) for path in sources+inputs+[OUT/'splits.jsonl']}
    save_json('config.json', dict(task='T16A', stage='timing', jobs=len(jobs), grid=GRID,
              seeds=list(range(50)), optimizer=dict(maxiter=5000, gtol=1e-8, ftol=1e-12, maxls=50),
              workers=WORKERS, BLAS_threads=1, python_isolated=True, hashes=hashes,
              versions={name: version(name) for name in ['scikit-learn', 'scipy', 'numpy', 'pandas']},
              selection='same fitted validation scores, mean across folds, smaller lambda then C on ties',
              scaler='StandardScaler fitted separately on each inner training part and full outer train',
              criterion='b5: W1 all reverse gap>=-.02; W2 all test gap from max(C0,L3)>=-.005; W3 mean reverse W-S>=.05'))
    for suffix in ['red', 'task']:
        shutil.copyfile(f'/tmp/T16A_pytest_{suffix}.log', OUT/f'pytest_{suffix}.log')
    pilot = [j for j in jobs if j['repeat'] in (0, 16)]
    print('START T16A timing', flush=True)
    tick = perf_counter()
    run_jobs(pilot)
    wall = perf_counter()-tick
    completed = pd.read_csv(OUT/'completed.csv')
    rates = completed.groupby(['k','b','direction']).seconds.mean()
    remaining = [j for j in jobs if j['job_id'] not in set(completed.job_id)]
    seconds = wall + sum(rates.loc[j['k'],j['b'],j['direction']] for j in remaining)/WORKERS + 180
    result = dict(point_hours=seconds/3600, with_25percent_margin_hours=seconds*1.25/3600,
                  pilot_seconds=wall, pilot_jobs=len(pilot), remaining_jobs=len(remaining),
                  method='outer tasks repeats 0 and 16 per cell, mean worker seconds per cell / 4; 180s report reserve')
    save_json('timing.json', result)
    print('TIMING_COMPLETE '+json.dumps(result), flush=True)


def run():
    cfg = json.loads((OUT/'config.json').read_text())
    assert json.loads((OUT/'timing.json').read_text())['point_hours'] <= 3
    for path, h in cfg['hashes'].items():
        assert digest(path) == h, path
    done = set(pd.read_csv(OUT/'completed.csv').job_id)
    jobs = list(map(json.loads, (OUT/'splits.jsonl').read_text().splitlines()))
    work = [j for j in jobs if j['job_id'] not in done]
    cfg.update(stage='running', started_at=datetime.now().astimezone().isoformat())
    save_json('config.json', cfg)
    print(f'START T16A remaining={len(work)}', flush=True)
    tick = perf_counter()
    run_jobs(work)
    cfg.update(stage='training_complete', formal_seconds=perf_counter()-tick,
               finished_at=datetime.now().astimezone().isoformat())
    save_json('config.json', cfg)
    print('ALL_TRAINING_COMPLETE jobs=600', flush=True)


if __name__ == '__main__':
    assert sys.flags.isolated
    {'timing': timing, 'run': run}[sys.argv[1]]()
