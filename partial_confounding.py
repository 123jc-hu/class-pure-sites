"""T20-A：部分站点–诊断混杂（δ 连续设计）的名单、线性模型与评估；论文 A 补充材料的部分混杂一节。"""
import argparse
import hashlib
import json
import platform
import zlib
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import abide_class_pure_sites as abide
import site_pair_validation as paper_a
import within_site_pair_model as within
from design_rule_baselines import combat_lr, leakage

OUT = Path("outputs/T20A")
DELTAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
METHODS = ("M0", "RW", "L1", "L2", "CB", "L3", "E-blind", "E-cond")
PURE_METHODS = ("M0", "L2", "E-blind", "E-cond")
# 2026-11-24 按 T12-A 人数核对的合格站点（两类都 ≥ 20 人）。抽样仍用实时计数。
EXPECTED_ELIGIBLE = {"NYU", "UM", "UCLA", "USM", "LEUVEN", "PITT", "YALE", "MAX_MUN", "KKI", "TRINITY"}


def eligible_counts(table):
    counts = pd.crosstab(table.site, table.label)
    eligible = counts[counts.min(axis=1) >= 20]
    assert set(eligible.index) == EXPECTED_ELIGIBLE
    return eligible


def make_split(table, seed):
    rng = np.random.default_rng(seed)
    eligible = eligible_counts(table).index.to_numpy()
    selected = rng.choice(eligible, size=6, replace=False)
    roles = dict(zip(rng.permutation(selected).tolist(), [1, 1, 1, 0, 0, 0]))
    test_sites = sorted(set(table.site) - set(roles))
    train = {str(delta): [] for delta in DELTAS}
    reverse = []
    for site, majority in sorted(roles.items()):
        site_rng = np.random.default_rng([seed, zlib.crc32(site.encode("utf-8"))])
        pool = table.loc[table.site.eq(site)]
        orders = {label: site_rng.permutation(pool.loc[pool.label.eq(label), "FILE_ID"]).tolist()
                  for label in (0, 1)}
        for delta in DELTAS:
            n_majority = round(20 * (0.5 + delta))
            train[str(delta)].extend(orders[majority][:n_majority] + orders[1 - majority][:20 - n_majority])
        reverse.extend(orders[1 - majority][10:])
    return dict(repeat=seed, seed=seed, train_roles=roles, test_sites=test_sites,
                train=train, test=table.loc[table.site.isin(test_sites), "FILE_ID"].tolist(), reverse=reverse)


def split_indices(table, split, delta):
    lookup = pd.Index(table.FILE_ID)
    train, test, reverse = (lookup.get_indexer(ids) for ids in
                            (split["train"][delta], split["test"], split["reverse"]))
    assert (np.r_[train, test, reverse] >= 0).all()
    assert len(np.unique(np.r_[train, test, reverse])) == len(train) + len(test) + len(reverse)
    return train, test, reverse


def cell_weights(rows):
    sizes = rows.groupby(["site", "label"])["label"].transform("size").to_numpy()
    weight = 1 / sizes
    return weight * (len(rows) / weight.sum())


def weighted_lr(x, table, train, target, seed):
    y = table.label.to_numpy()
    folds = paper_a.subject_folds(table, train, seed)
    scores = []
    for tr, va in folds:
        w_train = cell_weights(table.loc[tr])
        w_valid = cell_weights(table.loc[va])
        scale = StandardScaler().fit(x[tr])
        z, v = scale.transform(x[tr]), scale.transform(x[va])
        scores.append([
            roc_auc_score(y[va], LogisticRegression(C=c).fit(z, y[tr], sample_weight=w_train)
                          .decision_function(v), sample_weight=w_valid)
            for c in abide.GRID
        ])
    C = abide.GRID[int(np.argmax(np.mean(scores, axis=0)))]
    fitted = make_pipeline(StandardScaler(), LogisticRegression(C=C))
    fitted.fit(x[train], y[train], logisticregression__sample_weight=cell_weights(table.loc[train]))
    return fitted.predict_proba(x[target])[:, 1], C


def erasure_fit(x, table, train, conditional):
    scale = StandardScaler().fit(x[train])
    z = scale.transform(x[train])
    rows = table.loc[train, ["site", "label"]].reset_index(drop=True)
    if conditional:
        offsets = []
        for _, by_label in rows.groupby("label"):
            means = np.array([z[cell.index].mean(axis=0) for _, cell in by_label.groupby("site")])
            offsets.append(means - means.mean(axis=0))
        offsets = np.vstack(offsets)
    else:
        means = np.array([z[cell.index].mean(axis=0) for _, cell in rows.groupby("site")])
        offsets = means - means.mean(axis=0)
    _, singular, right = np.linalg.svd(offsets, full_matrices=False)
    rank = int((singular > max(offsets.shape) * np.finfo(float).eps * singular[0]).sum())
    assert rank <= (10 if conditional else 5)
    return scale, right[:rank].T


def site_model(x, table, train, target, seed, delta, method):
    if delta == 0.5:
        C = 1.0
    else:
        n_splits = min(5, int(table.loc[train].groupby(["site", "label"]).size().min()))
        scores = []
        for tr, va in within.inner_folds(table, train, seed, n_splits):
            val = table.loc[va]
            y, sites = val.label.to_numpy(), val.site.to_numpy()
            scores.append([
                within.pair_auc(y, sites, within.decision(within.fit_model(x, table, tr, method, c), x, table, va))
                for c in within.GRID
            ])
        C = within.GRID[int(np.argmax(np.mean(scores, axis=0)))]
    fit = within.fit_model(x, table, train, method, C)
    score = within.decision(fit, x, table, target)
    probability = expit(score)
    if delta == 0.5:
        assert np.all(probability == 0.5), "L2: pure-site predictions must all equal 0.5"
    return probability, score, C


def fit_predict(x, table, train, target, seed, delta, method):
    if method in ("L1", "L2", "L3"):
        return site_model(x, table, train, target, seed, delta, method)
    if method == "CB":
        result = combat_lr(x, table, train, target, seed, eb=True)
        return result["probability"], result["probability"], result["model"].best_params_["lr__C"]
    if method == "RW":
        probability, C = weighted_lr(x, table, train, target, seed)
        return probability, probability, C

    y = table.label.to_numpy()
    if method == "M0":
        fitted = abide.select_model(x, y, train, seed)
        matrix = x[target]
    else:
        scale, u = erasure_fit(x, table, train, conditional=method == "E-cond")
        projected_train = paper_a.nap_transform(x[train], scale, u)
        fitted = abide.select_model(projected_train, y[train], np.arange(len(train)), seed)
        matrix = paper_a.nap_transform(x[target], scale, u)
    probability = fitted.predict_proba(matrix)[:, 1]
    return probability, probability, fitted.best_params_["lr__C"]


def test_auc(rows, scores):
    values = []
    for site in sorted(rows.site.unique()):
        mask = rows.site.eq(site).to_numpy()
        values.append(roc_auc_score(rows.label.to_numpy()[mask], scores[mask]))
    return float(np.mean(values))


def subject_cv_auc(x, table, train, seed):
    y = table.label.to_numpy()
    values = []
    for tr, va in paper_a.subject_folds(table, train, seed):
        fitted = abide.select_model(x, y, tr, seed)
        values.append(roc_auc_score(y[va], fitted.predict_proba(x[va])[:, 1]))
    return float(np.mean(values))


def leave_site_auc(x, table, train, seed, delta):
    if delta == 0.5:
        return np.nan
    y = table.label.to_numpy()
    sites = table.loc[train, "site"].to_numpy()
    values = []
    for site in np.unique(sites):
        held = sites == site
        tr, va = train[~held], train[held]
        fitted = abide.select_model(x, y, tr, seed)
        values.append(roc_auc_score(y[va], fitted.predict_proba(x[va])[:, 1]))
    return float(np.mean(values))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def config(table, phase, seeds):
    paths = [Path("partial_confounding.py"), Path("partial_confounding_report.py"),
             Path("tests/test_partial_confounding.py")]
    copies = [Path(f"{name}.py") for name in ("abide_class_pure_sites", "bridge_design", "within_site_pair_model",
                                              "pair_estimator_confirmation", "design_rule_baselines",
                                              "site_pair_validation")]
    return dict(
        task="T20-A", revision=1, phase=phase, seeds=seeds, deltas=DELTAS, methods=METHODS,
        pure_methods=PURE_METHODS, k=3, m=20, pooled_C=abide.GRID, within_site_C=within.GRID,
        L2_pure_C=1.0, feature_cache=str(abide.CACHE), no_zero_roi=False,
        eligible_sites=eligible_counts(table).rename(columns={0: "control", 1: "case"}).to_dict("index"),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + copies},
        versions={"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
                                             "neuroCombat", "threadpoolctl")
        }},
        site_seed="SeedSequence([repeat, crc32(site UTF-8)])",
        within_site_selection="site-by-diagnosis folds, pooled within-site validation pair AUC",
        subject_cv_strata="diagnosis",
        subject_cv_aggregation="mean of the five held-out fold AUCs",
        blas_threads=1, BNU_watch_not_applicable=True,
    )


def run(phase):
    from partial_confounding_report import paired_differences

    start = perf_counter()
    timing = phase == "timing"
    out = OUT / "timing" if timing else OUT
    seeds = [0] if timing else list(range(50))
    out.mkdir(parents=True, exist_ok=True)
    table, x = paper_a.data()
    assert table.FILE_ID.is_unique and set(table.label) == {0, 1}
    assert (table.label == table.DX_GROUP.eq(1)).all()
    assert x.shape == (len(table), 19_900) and len(table) == 1035 and table.site.nunique() == 17
    splits = [make_split(table, seed) for seed in seeds]
    print(f"START T20-A {phase} repeats={len(seeds)}", flush=True)
    with (out / "splits.jsonl").open("x") as manifest:
        for split in splits:
            manifest.write(json.dumps(split) + "\n")
    write_json(out / "config.json", config(table, phase, seeds))

    metrics = []
    with (out / "predictions.csv").open("x") as predictions, threadpool_limits(limits=1):
        for split in splits:
            seed = split["seed"]
            for delta in DELTAS:
                train, test, reverse = split_indices(table, split, str(delta))
                design_l = leakage(table, train)
                assert abs(design_l - 8 * delta**2 / 3) < 1e-12
                target = np.r_[test, reverse]
                for method in METHODS if delta < .5 else PURE_METHODS:
                    started = perf_counter()
                    probability, score, C = fit_predict(x, table, train, target, seed, delta, method)
                    context = dict(repeat=seed, delta=delta, method=method)
                    n_test = len(test)
                    row = dict(
                        **context, L=design_l, L_relative=(2 * delta)**2, C=C, n_train=len(train),
                        n_test=n_test, n_reverse=len(reverse),
                        test_auc=test_auc(table.loc[test], score[:n_test]),
                        pooled_auc=roc_auc_score(table.loc[test, "label"], score[:n_test]),
                        reverse_auc=roc_auc_score(table.loc[reverse, "label"], score[n_test:]),
                        seconds=perf_counter() - started, subject_cv_auc=np.nan, site_cv_auc=np.nan,
                        site_cv_seconds=np.nan,
                    )
                    if method == "M0":
                        row["subject_cv_auc"] = subject_cv_auc(x, table, train, seed)
                        site_cv_start = perf_counter()
                        row["site_cv_auc"] = leave_site_auc(x, table, train, seed, delta)
                        row["site_cv_seconds"] = perf_counter() - site_cv_start if delta < 0.5 else np.nan
                    metrics.append(row)
                    pred = table.loc[target, ["FILE_ID", "site", "label"]].assign(
                        **context, is_reverse=np.arange(len(target)) >= n_test, score=score,
                        probability=probability,
                    )
                    pred.to_csv(predictions, index=False, header=len(metrics) == 1)
                    predictions.flush()
                    print(f"TRAIN_DONE repeat={seed} delta={delta:.1f} method={method} "
                          f"seconds={row['seconds']:.2f}", flush=True)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "repeat_metrics.csv", index=False)
    paired_differences(metrics).to_csv(out / "paired_differences.csv", index=False)
    seconds = perf_counter() - start
    if timing:
        write_json(out / "timing.json", dict(
            repeat0_seconds=seconds, estimated_50_seconds=50 * seconds,
            subject_cv_included=True,
        ))
    print(f"COMPLETE T20-A {phase} fits={len(metrics)} seconds={seconds:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("timing", "stage1"))
    run(parser.parse_args().phase)
