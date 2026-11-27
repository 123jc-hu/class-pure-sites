import numpy as np
import pandas as pd
import pytest

import partial_confounding as pc
import site_pair_validation as paper_a


@pytest.fixture(scope="module")
def abide():
    return paper_a.data()


@pytest.fixture(scope="module")
def all_splits(abide):
    table, _ = abide
    return [pc.make_split(table, seed) for seed in range(50)]


@pytest.fixture(scope="module")
def synthetic():
    sites = np.repeat(["A", "B", "C", "D", "E", "F", "U", "V"], 24)
    labels = np.tile(np.repeat([0, 1], 12), 8)
    table = pd.DataFrame({"site": sites, "label": labels})
    rng = np.random.default_rng(42)
    x = rng.normal(size=(len(table), 16)) + labels[:, None] * 0.3
    for site in ("A", "C", "E"):
        x[sites == site] += 1.5
    train = np.concatenate([
        24 * i + np.r_[0:(4 if i % 2 == 0 else 8), 12:(20 if i % 2 == 0 else 16)]
        for i in range(6)
    ])
    return x, table, train


def test_training_counts_and_roles(abide, all_splits):
    table, _ = abide
    by_id = table.set_index("FILE_ID")
    for split in all_splits:
        roles = split["train_roles"]
        assert sorted(roles.values()) == [0, 0, 0, 1, 1, 1]
        previous = {}
        for delta, majority in zip(("0.0", "0.1", "0.2", "0.3", "0.4", "0.5"), (10, 12, 14, 16, 18, 20)):
            selected = by_id.loc[split["train"][delta]]
            assert len(selected) == 120
            assert set(selected.site) == set(roles)
            for site, role in roles.items():
                t = selected[selected.site == site]
                major = set(t.index[t.label == role])
                minor = set(t.index[t.label != role])
                assert (len(major), len(minor)) == (majority, 20 - majority)
                if site in previous:
                    old_major, old_minor = previous[site]
                    assert old_major <= major and minor <= old_minor
                previous[site] = major, minor


def test_fixed_test_and_reverse_sets_without_overlap(abide, all_splits):
    table, _ = abide
    by_id = table.set_index("FILE_ID")
    for split in all_splits:
        roles = split["train_roles"]
        expected_sites = set(table.site) - set(roles)
        assert len(expected_sites) == 11 and set(split["test_sites"]) == expected_sites
        assert set(split["test"]) == set(table.FILE_ID[table.site.isin(expected_sites)])
        # The holdout contains every minority-label subject beyond the balanced prefix.
        reverse_pool = table.loc[table.site.isin(roles) & table.label.ne(table.site.map(roles)), "FILE_ID"]
        assert set(split["reverse"]) == set(reverse_pool) - set(split["train"]["0.0"])
        fixed_test, fixed_reverse = set(split["test"]), set(split["reverse"])
        for delta in split["train"]:
            train, test, reverse = pc.split_indices(table, split, delta)
            assert set(table.loc[test, "FILE_ID"]) == fixed_test
            assert set(table.loc[reverse, "FILE_ID"]) == fixed_reverse
            assert set(train).isdisjoint(test)
            assert set(train).isdisjoint(reverse)
            assert set(test).isdisjoint(reverse)
            assert by_id.loc[split["reverse"], "label"].nunique() == 2


def test_leakage_matches_design(abide, all_splits):
    table, _ = abide
    for split in all_splits:
        for delta in split["train"]:
            train, _, _ = pc.split_indices(table, split, delta)
            assert abs(pc.leakage(table, train) - 8 * float(delta)**2 / 3) < 1e-12


@pytest.mark.parametrize("method", ["M0", "RW", "CB", "L2", "E-blind", "E-cond"])
def test_external_rows_cannot_change_fit(synthetic, method):
    x, table, train = synthetic
    # Known-site and new-site probes are scored together. L2 may use the
    # unlabeled new-site cohort mean, so keep that cohort's features fixed.
    target = np.r_[np.arange(8, 12), np.arange(144, 168)]
    withheld = np.setdiff1d(np.arange(len(x)), np.r_[train, target])
    changed_x = x.copy()
    changed_x[withheld] += np.arange(len(withheld))[:, None] * 2
    changed_x[target[1:4]] += 50
    changed_table = table.copy()
    changed_table.loc[changed_table.index.difference(train), "label"] ^= 1
    baseline = pc.fit_predict(x, table, train, target, seed=0, delta=0.2, method=method)
    altered = pc.fit_predict(changed_x, changed_table, train, target, seed=0, delta=0.2, method=method)
    probes = np.r_[0, np.arange(4, len(target))]
    np.testing.assert_allclose(baseline[0][probes], altered[0][probes], rtol=0, atol=1e-12)
    np.testing.assert_allclose(baseline[1][probes], altered[1][probes], rtol=0, atol=1e-12)
    assert baseline[2] == altered[2]


def test_weights_are_normalized_within_each_fitting_subset():
    table = pd.DataFrame({"site": ["A", "A", "A", "B", "B", "B", "B"],
                          "label": [0, 0, 1, 0, 1, 1, 1]})
    np.testing.assert_allclose(pc.cell_weights(table.iloc[:6]), [.75, .75, 1.5, 1.5, .75, .75])
    np.testing.assert_allclose(pc.cell_weights(table.iloc[[0, 2, 3, 4, 5]]), [1.25, 1.25, 1.25, .625, .625])


@pytest.mark.parametrize(("conditional", "max_rank"), [(False, 5), (True, 10)])
def test_erasure_projection_geometry(synthetic, conditional, max_rank):
    x, table, train = synthetic
    scale, basis = pc.erasure_fit(x, table, train, conditional)
    projector = basis @ basis.T
    np.testing.assert_allclose(scale.mean_, x[train].mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(projector, projector.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(projector @ projector, projector, rtol=0, atol=1e-12)
    assert np.linalg.matrix_rank(projector) <= max_rank


def test_conditional_erasure_matches_nap_on_pure_abide_split(abide, all_splits):
    table, x = abide
    train, _, _ = pc.split_indices(table, all_splits[0], "0.5")
    _, basis = pc.erasure_fit(x, table, train, conditional=True)
    _, reference = paper_a.nap_fit(x, table, train)
    # Compare every matrix entry in blocks to avoid two dense 19,900² arrays.
    max_error = 0.0
    for start in range(0, x.shape[1], 128):
        block = slice(start, start + 128)
        difference = basis[block] @ basis.T - reference[block] @ reference.T
        max_error = max(max_error, np.abs(difference).max())
    assert max_error < 1e-8
