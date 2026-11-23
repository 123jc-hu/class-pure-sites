"""T19 noRev extension with reverse subjects introduced only after training."""
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
import json
import sys
import warnings

sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from published_methods_run import MH, WORKERS, input_data, task, write_json

OUT = task.OUT/'rev3'
BASE = task.OUT/'rev2'
MODELS = ('Kunda','Parisot-HO')
GROUP = ['model','design','k']


def parisot_norev(x, site, sex, train, y, query, query_site, query_sex, n_features=2000):
    selector = task.parisot_selector(x,train,y,n_features)
    selected = selector.transform(x)
    adjacency,sigma = task.parisot_graph(selected,site,sex)
    features,supports = task.parisot_normalize(selected),task.parisot_supports(adjacency)
    model = task.train_parisot(features,supports,train,y)
    fit_scores = task.predict_parisot(model,features,supports)

    # Keep the fitted feature selection, edge bandwidth and weights. Attach reverse
    # nodes together by the same pairwise rule, then recompute graph normalization
    # and Chebyshev supports for the augmented inference graph.
    joined = np.vstack([selected,selector.transform(query)])
    enlarged,_ = task.parisot_graph(joined,np.r_[site,query_site],np.r_[sex,query_sex],sigma)
    assert np.array_equal(enlarged[:len(x),:len(x)],adjacency)
    reverse_scores = task.predict_parisot(model,task.parisot_normalize(joined),task.parisot_supports(enlarged))[len(x):]
    return fit_scores,reverse_scores


def reverse_kernel_scores(ridge, cross, projection, fit_kernel_mean, phenotypes):
    # The rev2 ridge consumes K@W. Subtract the fit-only column mean from the
    # query kernel and translate the intercept by (mean@W) dot beta to preserve it.
    centered = (cross-fit_kernel_mean)@projection
    intercept_shift = (fit_kernel_mean@projection)@ridge.coef_.ravel()[:projection.shape[1]]
    return ridge.decision_function(np.column_stack([centered,phenotypes]))+intercept_shift


def kunda_norev(corr, site, p, train, y, query, query_p):
    transformer = task.fit_tpe(corr,train)
    x = transformer.transform(corr)
    dimensions = [h for h in MH if h<=len(x)]
    ridge,design,projection,width2,mean,params = task.fit_kunda_ridge(x,site,p,train,y,dimensions)
    fit_scores = ridge.decision_function(design)

    # Transform only after training. The TPE reference, RBF width, centering
    # and eigenvectors are frozen; query rows never enter the MIDA fit matrix.
    q = transformer.transform(query)
    cross = np.exp(-cdist(q,x,metric='sqeuclidean')/(2*width2))
    return fit_scores,reverse_kernel_scores(ridge,cross,projection,mean,query_p),params,dimensions


def jobs():
    rows = [j for j in map(json.loads,(BASE/'splits.jsonl').read_text().splitlines())
            if j['kind']=='main' and j['design'] in ('B0','C0') and j['model'] in MODELS]
    assert len(rows)==len({j['job_id'] for j in rows})==400
    return rows


def run_job(job):
    tick = perf_counter()
    ids = pd.Index(TABLE.FILE_ID)
    fit = ids.get_indexer(job['train']+job['test'])
    reverse = ids.get_indexer(job['reverse'])
    n_train = len(job['train'])
    assert min(fit.min(),reverse.min())>=0 and len(set(fit)|set(reverse))==len(fit)+len(reverse)
    table,query = TABLE.iloc[fit],TABLE.iloc[reverse]
    train = np.arange(n_train)
    y = table.label.to_numpy()[:n_train]
    x = DATA[job['model']]
    params,dimensions,notes = {},[],[]
    if job['model']=='Kunda':
        sex,query_sex = table.SEX,query.SEX
        p = np.column_stack([sex==1,sex==2,table.AGE_AT_SCAN])
        query_p = np.column_stack([query_sex==1,query_sex==2,query.AGE_AT_SCAN])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            scores,rev_scores,params,dimensions = kunda_norev(x[fit],table.SITE_ID.to_numpy(),p,train,y,x[reverse],query_p)
        notes = [str(w.message) for w in captured]
        threshold = 0.
    else:
        scores,rev_scores = parisot_norev(x[fit],table.SITE_ID.to_numpy(),table.SEX.to_numpy(),train,y,
                                        x[reverse],query.SITE_ID.to_numpy(),query.SEX.to_numpy())
        threshold = .5
    score = np.r_[scores[n_train:],rev_scores]
    assert np.isfinite(score).all()
    context = {k:job[k] for k in ('job_id','source_job_id','repeat','seed','design','k','kind','fold','model')}
    target = pd.concat([table.iloc[n_train:],query])[['FILE_ID','site','label']]
    frame = target.assign(**context,reverse=np.arange(len(target))>=len(job['test']),score=score,
                          prediction=(score>threshold).astype(int))
    receipt = context|dict(seconds=perf_counter()-tick,n_nodes=len(fit),n_train=n_train,n_test=len(job['test']),
        n_reverse=len(reverse),h_grid=json.dumps(dimensions),warnings=json.dumps(notes))|dict.fromkeys(
        ['mu','h','alpha','inner_accuracy'])|params
    return receipt,frame


def parallel(work):
    with get_context('fork').Pool(WORKERS) as pool:
        for row,frame in pool.imap_unordered(run_job,work):
            for name,output in [('predictions.csv',frame),('completed.csv',pd.DataFrame([row]))]:
                with (OUT/name).open('a') as target:
                    output.to_csv(target,index=False,header=target.tell()==0)
            print(f"TRAIN_DONE {row['job_id']} seconds={row['seconds']:.3f}",flush=True)


def execute(mode):
    global TABLE,DATA
    tick = perf_counter()
    path = OUT/'config.json'
    work = jobs()
    TABLE = task.a.subjects()
    DATA = input_data(TABLE,MODELS)
    setup = perf_counter()-tick
    print(f'START T19 noRev {mode}',flush=True)
    if mode=='timing':
        pilot = [j for j in work if j['repeat']<2]
        parallel(pilot)
        seconds = perf_counter()-tick
        done = pd.read_csv(OUT/'completed.csv')
        assert len(pilot)==len(done)==16 and set(done.job_id)=={j['job_id'] for j in pilot}
        pace = done.groupby(GROUP).seconds.mean()
        estimate = seconds+setup+sum(pace[tuple(j[k] for k in GROUP)] for j in work if j['repeat']>=2)/WORKERS
        prior = json.loads((BASE/'main_config.json').read_text())
        cfg = dict(task='T19',revision=3,repeats=50,seeds=list(range(50)),workers=WORKERS,jobs=len(work),
            source_splits=str(BASE/'splits.jsonl'),source_predictions=str(BASE/'predictions.csv'),
            upstream=json.loads((BASE/'config.json').read_text())['upstream'],
            labeled_fit='train only',unlabeled_fit='test-site participants only',
            kunda_phenotypes=prior['kunda_phenotypes'],kunda_h_grid=prior['kunda_h_grid'],
            kunda_mu_grid=prior['kunda_mu_grid'],kunda_alpha_grid=prior['kunda_alpha_grid'],
            dimension_rule='h is from the original grid and at most len(train + test); reverse never counted',
            kunda_reverse='frozen TPE/width/MIDA; cross kernel minus fit column means with equivalent ridge intercept shift',
            parisot_reverse='frozen RFE, bandwidth and GCN weights; batch attachment and recomputed normalized Chebyshev supports',
            parisot_primary_atlas='HO',pilot_jobs=len(pilot),pilot_seconds=seconds,projected_seconds=estimate,
            estimate_method='pilot wall + second input preparation + sum of pilot group mean seconds for remaining jobs / workers')
        (OUT/'splits.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in work))
        write_json(path,cfg)
        print(f'FINISH T19 noRev timing projected_seconds={estimate:.1f}',flush=True)
    else:
        cfg = json.loads(path.read_text())
        assert cfg['projected_seconds']<=3*3600
        done = set(pd.read_csv(OUT/'completed.csv').job_id)
        parallel([j for j in work if j['job_id'] not in done])
        cfg['main_elapsed_seconds'] = perf_counter()-tick
        write_json(path,cfg)
        print(f"FINISH T19 noRev jobs={len(work)} seconds={cfg['main_elapsed_seconds']:.1f}",flush=True)


if __name__=='__main__':
    execute(sys.argv[1])
