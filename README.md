# Class-pure sites

English | [中文](README.zh-CN.md)

Code and summary results for the manuscript

> **When the site predicts the diagnosis: evaluation, bridging and within-site training for multi-site neuroimaging classifiers**
> Jincheng Hu, Ping Xie, Sha Wang, Jiaxin Li, Guoqian Jiang (Yanshan University)

When every site in a multi-site study contributes participants of a single diagnosis (class-pure sites), the site predicts the label and subject-level cross-validation rewards models that learn it. The study tests three rules on resting-state fMRI from ABIDE-I, ABIDE-II and ADHD-200:

1. **Evaluate:** hold out whole sites (leave-site-pair-out) and check a reverse set.
2. **Design:** recruit a few bridging participants of the missing diagnosis at each site, guided by the design coefficient *L*.
3. **Estimate:** train only on within-site case and control pairs (the within-site pairwise logistic loss L3).

## Contents

| Path | What it holds |
|---|---|
| `*.py` | One script per experiment (see the table below); `*_report.py` writes the summary tables of that experiment |
| `tests/` | Tests for the steps that could fail silently: site roles, subject splits, label mapping. They read the subject rosters that `abide_class_pure_sites.py`, `abide2_prepare.py` and `adhd200_prepare.py` write, so run those first |
| `outputs/` | Summary tables read by `paper_figures.py` and `paper_tables.py` |
| `paper/` | LaTeX sources, figures and PDFs of the manuscript and the Supplementary Material |

Code comments and docstrings are in Chinese; function and variable names are in English.

## Experiments

| ID | Script | Content | Manuscript |
|---|---|---|---|
| T12A | `abide_class_pure_sites.py` | Class-pure (P) and mixed (M) designs on ABIDE-I, cost of purity, reverse set | Sec. 6.1 |
| T12B | `site_pair_validation.py` | Subject-level cross-validation and leave-site-pair-out (LSPO) estimates | Sec. 6.2 |
| T12C | `bridge_design.py` | Bridged designs, pooled model C0, ComBat | Sec. 6.4 |
| T12C2 | `same_class_control.py` | Same-class control S0 | Sec. 6.4 |
| T12D | `abide_robustness.py` | Quality-controlled pool, Harvard-Oxford atlas, BrainNetCNN | Supp. S5 |
| T13A, T13B | `within_site_pair_model.py`, `pair_estimator_confirmation.py` | Within-site pairwise loss L3 and site fixed-effects model L1 | Sec. 6.5 |
| T14A | `design_rule_baselines.py` | Budget allocation and the design coefficient *L* | Sec. 6.4 |
| T14B | `abide2_prepare.py`, `abide2_replication.py` | ABIDE-II replication | Sec. 6.6 |
| T15A | `group_dro.py` | Group DRO with one group per site | Sec. 6.3 |
| T15B | `natural_pure_collections.py` | Natural class-pure collections (KUL_3, WashU) | Sec. 6.2 |
| T16A | `pair_ce_mixture.py` | Mixing the pooled loss into L3 | Sec. 6.5 |
| T16B, T16C | `deep_pair_pilot.py`, `deep_abide1.py` | Frozen self-supervised encoder with linear heads; deep baselines | Supp. S4 |
| T17A | `adhd200_prepare.py`, `adhd200_replication.py` | ADHD-200 replication | Sec. 6.6 |
| T18 | `supplementary_*.py` | Site probes, same-institution designs, corrected resampled tests | Supp. S3, S5 |
| T19 | `published_methods*.py` | Population GCN (Parisot et al.) and TPE-MIDA (Kunda et al.) | Sec. 6.3, Supp. S3, S5 |
| | `theory_a_check.py` | Monte Carlo check of the propositions (synthetic data) | Supp. S1 |

The criteria for each experiment were fixed before it was run (Supplementary Section S2). Some scripts record hashes of internal task descriptions (`docs/tasks/`, `AGENTS.md`) that are not part of this repository; remove those file names from the provenance hashes before re-running.

## Data

All three cohorts are public and are not redistributed here.

* **ABIDE-I:** Preprocessed Connectomes Project release (CPAC, band-pass filtered, no global signal regression), CC200 time series. Expected under `~/datasets_cache/ABIDE_PCP`.
* **ABIDE-II:** DCAN-processed release, Gordon parcellation plus subcortical regions, downloaded by `abide2_prepare.py`. Expected under `~/datasets_cache/ABIDE2_DCAN`.
* **ADHD-200:** Athena release, CC200 time series. Expected under `~/datasets_cache/ADHD200_Athena`.

Derived arrays are cached under `~/datasets_cache/ABIDE_derived` and `~/datasets_cache/ADHD200_derived`. Change the path constants at the top of `abide_class_pure_sites.py`, `abide_robustness.py`, `pair_estimator_confirmation.py`, `abide2_prepare.py`, `abide2_replication.py` and `adhd200_prepare.py` to use other locations.

ABIDE and ADHD-200 are shared under their own data use terms (ABIDE: CC BY-NC-SA 3.0). The two per-subject prediction files in `outputs/` (`T12A/predictions.csv`, `T13B/reference_predictions.csv`) keep only the rows used in the individual-score figure and contain subject IDs, site, label and model score, without phenotypic variables.

## Published methods

`published_methods.py` calls the authors' code, expected under `~/external_code`:

| Method | Repository | Commit |
|---|---|---|
| Population GCN | https://github.com/parisots/population-gcn | `185e2f5` |
| Chebyshev GCN layers | https://github.com/parisots/gcn | `506ef57` |
| TPE-MIDA | https://github.com/kundaMwiza/fMRI-site-adaptation | `0a70cf8` |

## Reproducing the figures

```bash
pip install -r requirements.txt
python paper_figures.py      # writes the result figures to paper/figures/ from outputs/
python paper_tables.py       # writes paper/tables/tests_en.tex
```

To rebuild the PDFs, run in `paper/`: `pdflatex -output-directory=build main`, `bibtex build/main`, the same two steps for `supplement`, then `supplement` once more and `main` twice, so that the cross-references between the two documents resolve.

The experiments themselves need the data above; each script writes its outputs to `outputs/<ID>/`. Several experiments take hours.

## License

Code: MIT (see `LICENSE`). The data belong to the ABIDE and ADHD-200 consortia and follow their terms.
