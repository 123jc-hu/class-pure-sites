import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def test_mixture_endpoints_shared_fits_and_training_isolation():
    import pair_ce_mixture as p
    import within_site_pair_model as l3

    y = np.array([1, 1, 0, 0, 0, 0, 1, 0, 1, 0])
    sites = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'B', 'A'])
    ids = [f'id{i}' for i in range(len(y))]
    table = pd.DataFrame(dict(FILE_ID=ids, label=y, site=sites, DX_GROUP=2-y, SEX=1))
    x = np.random.default_rng(0).normal(size=(len(y), 4))
    z = StandardScaler().fit_transform(x[:6])
    left, right = p.pairs(y[:6], sites[:6])
    assert set(zip(left, right)) == {(0, 2), (0, 3), (1, 2), (1, 3)}
    C = .1
    _, pair_objective, _ = p.fit_weight(z, y[:6], left, right, 0., C)
    legacy = l3.fit_weight(z, y[:6], sites[:6], 'L3', C)['objective']
    assert abs(pair_objective / legacy - 1) < 1e-6
    _, ce_objective, _ = p.fit_weight(z, y[:6], left, right, 1., C)
    lr = LogisticRegression(C=C, tol=1e-10, max_iter=5000).fit(z, y[:6])
    eta = lr.decision_function(z)
    reference = np.logaddexp(0, (1-2*y[:6])*eta).mean() + (lr.coef_**2).sum()/(2*C*len(z))
    assert abs(ce_objective / reference - 1) < 1e-6

    job = dict(job_id='synthetic', k=1, b=2, direction='oneway', repeat=3, seed=3,
               train=ids[:6], test=ids[6:8], reverse=ids[8:])
    original = p.fit_job(job, (table, x))
    extreme = x.copy()
    extreme[6:] += 1e6
    changed = p.fit_job(job, (table, extreme))
    pd.testing.assert_frame_equal(original['inner_scores'], changed['inner_scores'])
    assert len(original['inner_scores']) == 70
    assert original['inner_scores'][['auc_W', 'auc_S']].notna().all().all()
    for fold in original['inner_splits']:
        tr, va = set(fold['train']), set(fold['validation'])
        assert tr.isdisjoint(va) and tr | va == set(job['train'])
        assert fold['standardizer_fit'] == fold['train']
    choices = original['choices'].set_index('method')
    for method in ['P1-W', 'P1-S']:
        q = choices.loc[method]
        curve = choices.loc['fixed_lambda']
        same = curve.loc[curve['lambda'].eq(q['lambda']) & curve.C.eq(q.C)]
        assert (len(same) == 1) == q.reused
        assert q.source_fit_id == (same.iloc[0].source_fit_id if q.reused else 'P1-S')
    assert choices.loc['P1-W', 'reused']
