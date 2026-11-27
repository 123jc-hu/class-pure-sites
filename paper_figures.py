"""Figures of paper A, read from outputs/ (summary tables and saved predictions only); writes vector PDFs to paper/figures/.

Colours come from the dataviz reference palette and were checked with a Python port of its validator
(2026-11-23): case/control = gold/aqua; framework (LSPO, bridging, L3) = blue; standard practice (P, C0,
subject-level CV) = red; references (mixed sites) = grey; design ordinal ramp = three reds (passes ordinal).
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper' / 'figures'
CASE, CTRL, L3C, INK, INK2, GRID = '#eda100', '#1baf7a', '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df'
STD = '#e34948'  # standard practice (pooled model, subject-level CV): red; framework: blue (validated 2026-11-23)
DESIGN = {'pure': '#a32b2b', 'oneway': '#e34948', 'twoway': '#ef8a89'}  # red ramp, ordinal by shortcut (validated)
BLUE_RAMP = {'oneway': '#77a9e8', 'twoway': '#174a8c'}

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 600,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
})


def rows(path):
    return list(csv.DictReader(open(ROOT / 'outputs' / path)))


def fig2():
    """L versus reverse-set AUC of the pooled model C0, all cells."""
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    L_panel(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / 'fig2_L_reverse.pdf')
    plt.close(fig)


def L_panel(ax, fs=None):
    """Scatter of L against the reverse-set AUC of C0 on a given axis (used alone and in fig_bridging)."""
    pts = []
    for r in rows('T14A/leakage_cells.csv'):
        design = 'pure' if r['method'] == 'B0' else r['direction']
        pts.append(('ABIDE-I, CC200' if r['feature'] == 'CC200' else 'ABIDE-I, Harvard-Oxford', design,
                    r['allocation'] == 'concentrated', float(r['mean_L']), float(r['native_reverse_AUC'])))
    for r in rows('T14B/stage2/L_outcomes.csv'):
        design = 'pure' if r['method'] == 'B0' else r['direction']
        pts.append(('ABIDE-II, Gordon', design, False, float(r['L']), float(r['native_reverse_AUC'])))
    shape = {'ABIDE-I, CC200': 'o', 'ABIDE-I, Harvard-Oxford': 's', 'ABIDE-II, Gordon': '^'}
    rho = {'ABIDE-I, CC200': -0.997, 'ABIDE-I, Harvard-Oxford': -0.991, 'ABIDE-II, Gordon': -0.970}
    ax.axhline(0.5, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.text(0.62, 0.512, 'chance level', ha='left', fontsize=6.5, color=INK2)
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    for name, design, conc, L, auc in sorted(pts, key=lambda q: q[2]):
        if conc:  # pre-specified confirmatory cells: hollow, drawn on top
            ax.scatter(L, auc, s=46, marker=shape[name], zorder=4, linewidths=1.3, facecolor='none',
                       edgecolor=DESIGN[design])
        else:
            ax.scatter(L, auc, s=28, marker=shape[name], zorder=3, linewidths=0.7, facecolor=DESIGN[design],
                       edgecolor='white')
    ax.set_xlabel('Design coefficient $L$ (computed before training)')
    ax.set_ylabel('Reverse-set AUC of pooled model C0\n(lower = stronger site shortcut)')
    ax.set_xlim(-0.06, 2.08)
    ax.set_ylim(0, 0.75)
    h1 = [Line2D([], [], ls='', marker=shape[n], color=INK2, mec='white', ms=5.5,
                 label=f'{n} ($\\rho={rho[n]:.3f}$)') for n in shape]
    h2 = [Line2D([], [], ls='', marker='o', color=DESIGN[d], mec='white', ms=5.5, label=l)
          for d, l in (('pure', 'class-pure'), ('oneway', 'one-way bridging'), ('twoway', 'two-way bridging'))]
    h2.append(Line2D([], [], ls='', marker='o', mfc='none', mec=DESIGN['twoway'], mew=1.3, ms=6.5,
                     label='concentrated (confirmatory)'))
    l1 = ax.legend(handles=h1, loc='upper right', frameon=False, handletextpad=0.3, borderaxespad=0.2)
    ax.add_artist(l1)
    ax.legend(handles=h2, loc='lower right', bbox_to_anchor=(1.0, 0.25), frameon=False,
              handletextpad=0.3, borderaxespad=0.2)


def fig3():
    """Trade-off between reverse-set AUC and within-site AUC as bridges are added (ABIDE-I, CC200)."""
    val = {}
    for r in rows('T13B/main_table.csv'):
        if r['feature'] == 'CC200' and r['method'] in ('C0', 'L3') and (
                (r['reverse'] == 'False' and r['metric'] == 'within_site_AUC') or r['reverse'] == 'True'):
            val[(r['method'], r['k'], r['direction'], int(r['b']), r['reverse'] == 'True')] = float(r['mean'])
    b0 = {}  # pure baseline (b = 0) from T12-A, n = 40
    for r in rows('T12A/main_table.csv'):
        if r['arm'] == 'P' and r['n'] == '40' and r['k'] in ('2', '4') and r['no_zero_roi'] == 'False':
            if r['reverse'] == 'False' and r['metric'] == 'within_site_AUC':
                b0[(r['k'], False)] = float(r['mean'])
            if r['reverse'] == 'True' and r['metric'] == 'pooled_AUC':
                b0[(r['k'], True)] = float(r['mean'])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex=True, sharey=True)
    for i, k in enumerate(('2', '4')):
        for j, d in enumerate(('oneway', 'twoway')):
            ax = axes[i, j]
            ax.grid(color=GRID, lw=0.5, zorder=0)
            ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
            for meth, col in (('C0', STD), ('L3', L3C)):
                bs = [2, 5, 10]
                x = [val[(meth, k, d, b, True)] for b in bs]
                y = [val[(meth, k, d, b, False)] for b in bs]
                if meth == 'C0':
                    x, y, bs = [b0[(k, True)]] + x, [b0[(k, False)]] + y, [0] + bs
                ax.plot(x, y, color=col, lw=1.6, zorder=3, solid_capstyle='round')
                ax.scatter(x, y, s=22, color=col, edgecolor='white', linewidths=0.8, zorder=4)
                for xb, yb, b in zip(x, y, bs):
                    off, ha = ((-6, 0), 'right') if meth == 'L3' else ((5, -8), 'left')
                    ax.annotate(f'b={b}' if b in (0, 10) else f'{b}', (xb, yb), xytext=off, va='center',
                                textcoords='offset points', ha=ha, fontsize=6, color=INK2)
            ax.set_title(f"$k={k}$, {'one-way' if d == 'oneway' else 'two-way'} bridging", loc='left')
            if i == 1:
                ax.set_xlabel('Reverse-set AUC (higher = less site shortcut)')
            if j == 0:
                ax.set_ylabel('Within-site AUC on new sites')
    axes[0, 0].set_xlim(0.05, 0.72)
    axes[0, 0].set_ylim(0.54, 0.665)
    hs = [Line2D([], [], color=STD, lw=1.6, marker='o', ms=4, mec='white', label='C0: pooled logistic regression'),
          Line2D([], [], color=L3C, lw=1.6, marker='o', ms=4, mec='white', label='L3: within-site pairs')]
    fig.legend(handles=hs, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / 'fig3_tradeoff.pdf')
    plt.close(fig)


# ---------------------------------------------------------------- result figures added 2026-11-23
# Colour jobs, fixed across all figures: orange/aqua = case/control; blue = components of the
# framework (LSPO, bridging, L3, within-site criterion); dark ink = standard practice (P, C0,
# subject-level CV); light grey = reference (M, mixed sites). Dots = single repeats, large
# markers = means over repeats.
import numpy as np
from collections import defaultdict

REF = '#a3a29d'
TINT = '#fcebea'  # below-chance region of reverse-set axes (light tint of the standard-practice red)
RNG = np.random.default_rng(0)


def f(x):
    return float(x)


def title(ax, s):
    ax.set_title(s, loc='left', fontweight='bold', pad=4)


def chance(ax, axis='y', label=True, pos=None):
    line = ax.axhline if axis == 'y' else ax.axvline
    line(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)


def strip(ax, x, ys, color, w=0.09, s=5, alpha=0.35, horizontal=False):
    j = x + RNG.uniform(-w, w, len(ys))
    if horizontal:
        ax.scatter(ys, j, s=s, color=color, alpha=alpha, lw=0, zorder=2)
    else:
        ax.scatter(j, ys, s=s, color=color, alpha=alpha, lw=0, zorder=2)


def mean_marker(ax, x, y, color, ms=6, marker='o', hollow=False, z=5):
    ax.plot(x, y, marker=marker, ms=ms, color=color, mec='white' if not hollow else color,
            mfc='white' if hollow else color, mew=0.9 if not hollow else 1.2, ls='', zorder=z)


def t12a():
    """repeat-level metrics of T12-A: d[(arm, k, n)] = (within-site AUCs, reverse-set AUCs)."""
    d = defaultdict(lambda: ([], []))
    for r in rows('T12A/repeat_metrics.csv'):
        if r['no_zero_roi'] != 'False' or r['arm'] not in ('P', 'M'):
            continue
        key = (r['arm'], int(r['k']), int(r['n']))
        if r['reverse'] == 'False':
            d[key][0].append(f(r['within_site_AUC']))
        else:
            d[key][1].append(f(r['pooled_AUC']))
    return d


def fig_problem():
    """Cost of class purity and the site shortcut versus k and n (ABIDE-I, CC200)."""
    d = t12a()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
    ks = [1, 2, 4, 6]
    ax = axes[0]
    for arm, col, dx in (('P', STD, -0.13), ('M', REF, 0.13)):
        means = []
        for i, k in enumerate(ks):
            ys = d[(arm, k, 40)][0]
            strip(ax, i + dx, ys, col, w=0.07)
            means.append(np.mean(ys))
        ax.plot(np.arange(4) + dx, means, color=col, lw=1.4, zorder=4)
        for i, m in enumerate(means):
            mean_marker(ax, i + dx, m, col)
    for i, k in enumerate(ks):
        cost = np.mean(d[('M', k, 40)][0]) - np.mean(d[('P', k, 40)][0])
        ax.text(i, 0.79, f'{cost:+.3f}', ha='center', fontsize=6.3,
                color=INK2)
    ax.text(1.5, 0.835, 'cost of purity, M − P', fontsize=6.3, color=INK2, ha='center')
    chance(ax)
    ax.set_xticks(range(4), [str(k) for k in ks])
    ax.set_xlabel('Class-pure sites per class $k$')
    ax.set_ylabel('Within-site AUC on new sites')
    ax.set_ylim(0.24, 0.86)
    ax.legend(handles=[Line2D([], [], color=STD, lw=1.4, marker='o', ms=4.5, mec='white', label='P: class-pure'),
                       Line2D([], [], color=REF, lw=1.4, marker='o', ms=4.5, mec='white', label='M: mixed sites')],
              loc='lower left', frameon=False, fontsize=6.3, handlelength=1.8, borderaxespad=0.1)
    title(ax, 'a  Class purity costs accuracy')

    ax = axes[1]
    ax.axhspan(0, 0.5, color=TINT, zorder=0)
    means = []
    for i, k in enumerate(ks):
        ys = d[('P', k, 40)][1]
        strip(ax, i, ys, STD, w=0.12)
        means.append(np.mean(ys))
    ax.plot(range(4), means, color=STD, lw=1.4, zorder=4)
    for i, m in enumerate(means):
        mean_marker(ax, i, m, STD)
        ax.annotate(f'{m:.2f}', (i, m), xytext=(7, 0), textcoords='offset points', va='center', fontsize=6.3,
                    color=INK)
    chance(ax)
    ax.text(3.45, 0.52, 'chance', ha='right', fontsize=6.3, color=INK2)
    ax.text(3.45, 0.025, 'model ranks by site', ha='right', fontsize=6.3, color=INK2)
    ax.set_xticks(range(4), [str(k) for k in ks])
    ax.set_xlabel('Class-pure sites per class $k$')
    ax.set_ylabel('Reverse-set AUC of P')
    ax.set_ylim(0, 1)
    title(ax, 'b  and teaches a site shortcut')

    ax = axes[2]
    ax.axhspan(0, 0.5, color=TINT, zorder=0)
    ns = [20, 40, 80]
    series = (('M', 0, REF, 'M, within-site'), ('P', 0, STD, 'P, within-site'), ('P', 1, STD, 'P, reverse set'))
    for arm, idx, col, lab in series:
        means = []
        for i, n in enumerate(ns):
            ys = d[(arm, 2, n)][idx]
            if idx == 1:
                strip(ax, i, ys, col, w=0.1)
            means.append(np.mean(ys))
        ax.plot(range(3), means, color=col, lw=1.4, zorder=4, ls='-' if idx == 0 else (0, (4, 2)))
        for i, m in enumerate(means):
            mean_marker(ax, i, m, col, hollow=idx == 1)
        ax.text(2.15, means[-1] + (0.035 if arm == 'M' else (-0.045 if idx == 0 else 0)), lab, fontsize=6.3,
                color=INK2, va='center')
    chance(ax)
    ax.set_xticks(range(3), [str(n) for n in ns])
    ax.set_xlim(-0.3, 3.3)
    ax.set_xlabel('Training subjects per class $n$ ($k=2$)')
    ax.set_ylabel('AUC')
    ax.set_ylim(0, 1)
    title(ax, 'c  More subjects keep the shortcut')
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT / 'fig_problem.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def ranks_by_job(path, filt, score):
    """Percentile rank of each evaluated subject within its model (test sites + reverse set)."""
    jobs = defaultdict(list)
    for r in rows(path):
        if filt(r):
            jobs[r['job_id']].append((f(r[score]), r['reverse'] == 'True', int(r['label'])))
    out = defaultdict(list)
    for v in jobs.values():
        s = np.array([q[0] for q in v])
        pr = (np.argsort(np.argsort(s)) + 0.5) / len(s)
        for (_, rev, lab), p in zip(v, pr):
            out[(rev, lab)].append(p)
    return out


def fig_anatomy():
    """What the shortcut looks like: score ranks of four subject groups for three models (k=2)."""
    from scipy.stats import gaussian_kde
    tab = {}
    for r in rows('T13B/main_table.csv'):
        if r['feature'] == 'CC200' and r['k'] == '2' and r['b'] == '5' and r['direction'] == 'twoway':
            if (r['reverse'] == 'False' and r['metric'] == 'within_site_AUC') or r['reverse'] == 'True':
                tab[(r['method'], r['reverse'] == 'True')] = f(r['mean'])
    for r in rows('T12A/main_table.csv'):
        if r['arm'] == 'P' and r['n'] == '40' and r['k'] == '2' and r['no_zero_roi'] == 'False':
            if (r['reverse'] == 'False' and r['metric'] == 'within_site_AUC') or (
                    r['reverse'] == 'True' and r['metric'] == 'pooled_AUC'):
                tab[('B0', r['reverse'] == 'True')] = f(r['mean'])
    models = [
        ('a  Class-pure, pooled (B0)', 'B0', ranks_by_job(
            'T12A/predictions.csv', lambda r: r['arm'] == 'P' and r['k'] == '2' and r['n'] == '40'
            and r['no_zero_roi'] == 'False', 'probability')),
        ('b  Two-way bridging, pooled (C0)', 'C0', ranks_by_job(
            'T13B/reference_predictions.csv', lambda r: r['feature'] == 'CC200' and r['method'] == 'C0' and r['k'] == '2'
            and r['b'] == '5' and r['direction'] == 'twoway', 'score')),
        ('c  Two-way bridging, within-site (L3)', 'L3', ranks_by_job(
            'T13B/reference_predictions.csv', lambda r: r['feature'] == 'CC200' and r['method'] == 'L3' and r['k'] == '2'
            and r['b'] == '5' and r['direction'] == 'twoway', 'score')),
    ]
    groups = [((False, 1), 'Cases at new sites', CASE, False), ((False, 0), 'Controls at new sites', CTRL, False),
              ((True, 0), 'Reverse set: controls\nat case sites', CTRL, True),
              ((True, 1), 'Reverse set: cases\nat control sites', CASE, True)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    grid = np.linspace(0, 1, 200)
    for ax, (ttl, key, data) in zip(axes, models):
        ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        for gi, (g, lab, col, rev) in enumerate(groups):
            y0 = 3 - gi
            v = np.array(data[g])
            dens = gaussian_kde(np.concatenate([-v, v, 2 - v]), bw_method=0.06)(grid)  # reflected at 0 and 1
            dens = dens / dens.max() * 0.78
            if rev:
                ax.fill_between(grid, y0, y0 + dens, color=col, alpha=0.12, lw=0, zorder=2)
                ax.plot(grid, y0 + dens, color=col, lw=1.0, ls=(0, (2.5, 1.5)), zorder=3)
            else:
                ax.fill_between(grid, y0, y0 + dens, color=col, alpha=0.75, lw=0, zorder=2)
            med = np.median(v)
            ax.plot([med, med], [y0, y0 + 0.78], color=INK, lw=0.9, zorder=4)
        ax.set_yticks([3.35, 2.35, 1.35, 0.35], [g[1] for g in groups])
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 4.0)
        ax.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel(f'within-site AUC {tab[(key, False)]:.3f}, reverse-set AUC {tab[(key, True)]:.2f}',
                      fontsize=6.6, color=INK2, labelpad=3)
        title(ax, ttl)
    fig.supxlabel('Relative score rank within each model, not a probability (0 = most control-like, 1 = most case-like; 50 repeats pooled)',
                  fontsize=7.5, y=0.02)
    fig.tight_layout(pad=0.4, w_pad=0.8, rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / 'fig_anatomy.pdf')
    plt.close(fig)


def fig_evaluation():
    """Subject-level CV versus LSPO against the test-site value; natural class-pure collections."""
    est = defaultdict(lambda: ([], []))
    for r in rows('T12B/performance_estimates.csv'):
        est[(r['method'], int(r['k']))][0].append(f(r['within_site_AUC']))
        est[(r['method'], int(r['k']))][1].append(f(r['estimate']))
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.55, wspace=0.32)
    for i, k in enumerate((2, 4, 6)):
        ax = fig.add_subplot(gs[0, i])
        ax.plot([0.2, 1], [0.2, 1], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.grid(color=GRID, lw=0.5, zorder=0)
        hs = []
        for m, col, lab in (('E_subj', STD, 'subject-level CV'), ('E_LSPO', L3C, 'LSPO')):
            x, y = map(np.array, est[(m, k)])
            ax.scatter(x, y, s=9, color=col, alpha=0.45, lw=0, zorder=2)
            mean_marker(ax, x.mean(), y.mean(), col, ms=7)
            bias = (y - x).mean()
            hs.append(Line2D([], [], ls='', marker='o', ms=5, color=col, mec='white',
                             label=f'{lab}: mean error {bias:+.3f}'))
        ax.legend(handles=hs, loc='lower right', bbox_to_anchor=(1.02, 0.99), frameon=False, fontsize=6.2,
                  handletextpad=0.1, borderaxespad=0.0, labelspacing=0.2)
        ax.set_xlim(0.28, 0.82)
        ax.set_ylim(0.2, 1.03)
        ax.set_aspect('auto')
        ax.set_xlabel('True within-site AUC on new sites')
        if i == 0:
            ax.set_ylabel('Estimate from training data')
        ax.text(0.8, 0.79, '$y=x$', fontsize=6.3, color=INK2, ha='right')
        ax.set_title(f"{'abc'[i]}  $k={k}$", loc='left', fontweight='bold', pad=22)

    ax = fig.add_subplot(gs[1, :])
    nat = []
    a = {r['comparison']: r for r in rows('T15B/addendum1/auc_summary.csv') if r['cohort'] == 'all'}
    nat.append(('KUL_3 (ABIDE-II), cases only', f(a['KUL_3_ASD_vs_mixed_TD']['subject_CV_AUC_mean']),
                f(a['KUL_3_ASD_vs_mixed_TD']['LOSO_AUC']), f(a['mixed_ASD_vs_mixed_TD']['subject_CV_AUC_mean']),
                f(a['mixed_ASD_vs_mixed_TD']['LOSO_AUC']), f(a['DiD']['drop_or_DiD_mean']), 'exploratory'))
    t = {(r['analysis'], r['evaluation'], r['contrast']): f(r['mean_AUC'])
         for r in rows('T17A/stage2/natural_auc_table.csv')}
    did = {r['analysis']: r for r in rows('T17A/stage2/DiD_table.csv')}
    for an, lab, note in (('main', 'WashU (ADHD-200), controls only', 'pre-specified'),
                          ('pittsburgh', 'Pittsburgh (ADHD-200), 92 controls, 4 cases', 'secondary')):
        dd = [r for r in rows('T17A/stage2/DiD_table.csv') if r['analysis'] == an and r['quantity'] == 'DiD'][0]
        nat.append((lab, t[(an, 'subject', 'target_TD')], t[(an, 'LOSO', 'target_TD')],
                    t[(an, 'subject', 'mixed_TD')], t[(an, 'LOSO', 'mixed_TD')], f(dd['mean']), note))
    ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
    yt, yl = [], []
    for i, (lab, ts, tl, ms_, ml, dd, note) in enumerate(nat):
        y = 2 - i
        for (s0, s1, col, dy, who) in ((ts, tl, STD, 0.14, 'class-pure collection'),
                                       (ms_, ml, REF, -0.14, 'mixed-site reference')):
            ax.annotate('', xy=(s1, y + dy), xytext=(s0, y + dy),
                        arrowprops=dict(arrowstyle='-|>', lw=1.3, color=col, shrinkA=3, shrinkB=3), zorder=3)
            mean_marker(ax, s0, y + dy, col, ms=6, hollow=True)
            mean_marker(ax, s1, y + dy, col, ms=6)
        ax.text(0.99, y, f'DiD ${dd:+.2f}$\n({note})', fontsize=6.3, color=INK, va='center', ha='right')
        yt.append(y)
        yl.append(lab)
    ax.set_yticks(yt, yl)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel('AUC: subject-level cross-validation (open) → leave-one-site-out (filled)')
    hs = [Line2D([], [], color=STD, lw=1.3, label='class-pure collection against mixed-site comparison group'),
          Line2D([], [], color=REF, lw=1.3, label='mixed-site subjects against the same comparison group')]
    ax.legend(handles=hs, loc='lower center', frameon=False, fontsize=6.3, ncol=2, bbox_to_anchor=(0.5, 1.0),
              borderaxespad=0.1)
    ax.set_title('d  Real class-pure collections are overstated by 0.3 to 0.5 AUC', loc='left', fontweight='bold',
                 pad=16)
    fig.savefig(OUT / 'fig_evaluation.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def paired(path, contrast, metric, **cond):
    out = defaultdict(list)
    for r in rows(path):
        if r['contrast'] == contrast and r['metric'] == metric and all(r[a] == str(b) for a, b in cond.items()):
            out[int(r['k'])].append(f(r['difference']))
    return out


def fig_pureonly():
    """Methods on class-pure data alone leave the shortcut; Group DRO weights stay uniform."""
    gd = defaultdict(dict)
    for r in rows('T15A/rev1/repeat_metrics.csv'):
        if r['method'] in ('GroupDRO', 'GD_ERM'):
            m = 'within' if r['reverse'] == 'False' else 'reverse'
            gd[(r['method'], int(r['k']), m)][int(r['repeat'])] = f(
                r['within_site_AUC'] if m == 'within' else r['pooled_AUC'])
    gdd = {(k, m): [gd[('GroupDRO', k, m)][i] - gd[('GD_ERM', k, m)][i] for i in gd[('GroupDRO', k, m)]]
           for k in (2, 4) for m in ('within', 'reverse')}
    meths = [
        ('LSPO tuning', {m: paired('T12B/paired_repeat_differences.csv', 'B1-B0', mm) for m, mm in
                         (('within', 'within_site_AUC'), ('reverse', 'reverse_AUC'))}, STD),
        ('NAP', {m: paired('T12B/paired_repeat_differences.csv', 'B2-B0', mm) for m, mm in
                 (('within', 'within_site_AUC'), ('reverse', 'reverse_AUC'))}, STD),
        ('Group DRO', {m: {k: gdd[(k, m)] for k in (2, 4)} for m in ('within', 'reverse')}, STD),
        ('Two-way bridging, $b=5$\n(for comparison)', {m: paired('T12C/paired_repeat_differences.csv', 'C0-B0', mm,
                                                                 b=5, direction='twoway') for m, mm in
                                                       (('within', 'within_site_AUC'), ('reverse', 'reverse_AUC'))},
         L3C),
    ]
    fig = plt.figure(figsize=(7.2, 2.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.15, 0.85, 0.85], wspace=0.55)
    for j, (m, xl, ttl) in enumerate((('within', '$\\Delta$ within-site AUC', 'a  Change in test AUC'),
                                       ('reverse', '$\\Delta$ reverse-set AUC', 'b  Change in shortcut'))):
        ax = fig.add_subplot(gs[0, j])
        ax.axvline(0, color=INK2, lw=0.6, zorder=1)
        for i, (lab, dat, col) in enumerate(meths):
            y0 = 3 - i
            for kk, dy in ((2, 0.14), (4, -0.14)):
                v = dat[m][kk]
                strip(ax, y0 + dy, v, col, w=0.05, s=5, alpha=0.4, horizontal=True)
                mean_marker(ax, np.mean(v), y0 + dy, col, ms=5.5, marker='o' if kk == 2 else 's')
        ax.set_yticks([3, 2, 1, 0], [q[0] for q in meths] if j == 0 else [''] * 4)
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel(xl + ' vs. B0')
        ax.set_xlim(*((-0.12, 0.14) if m == 'within' else (-0.3, 0.75)))
        ax.set_ylim(-0.5, 3.5)
        title(ax, ttl)
    axes0 = fig.axes[0]
    axes0.legend(handles=[Line2D([], [], ls='', marker='o', color=INK2, ms=5, label='$k=2$'),
                          Line2D([], [], ls='', marker='s', color=INK2, ms=5, label='$k=4$')],
                 loc='lower left', frameon=False, fontsize=6.3, handletextpad=0.2, bbox_to_anchor=(-0.02, 0.18))
    diag = [r for r in rows('T15A/rev1/final_diagnostics.csv') if r['method'] == 'GroupDRO' and r['role'] == 'final']
    ax = fig.add_subplot(gs[0, 2])
    for i, k in enumerate((2, 4)):
        q = [x for r in diag if r['k'] == str(k) for x in eval(r['q'])]
        strip(ax, i, q, STD, w=0.15, s=5, alpha=0.35)
        ax.plot([i - 0.28, i + 0.28], [1 / (2 * k)] * 2, color=L3C, lw=1.4, zorder=4)
    ax.text(1.32, 0.125, '1/(2k)', color=L3C, fontsize=6.3, va='center')
    ax.set_xticks([0, 1], ['$k=2$', '$k=4$'])
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0, 0.4)
    ax.set_ylabel('Final group weight')
    title(ax, 'c  Group weights')
    ax = fig.add_subplot(gs[0, 3])
    for i, k in enumerate((2, 4)):
        ls = [x for r in diag if r['k'] == str(k) for x in eval(r['group_losses'])]
        strip(ax, i, ls, STD, w=0.15, s=5, alpha=0.35)
        ax.plot([i - 0.28, i + 0.28], [np.median(ls)] * 2, color=INK, lw=1.4, zorder=4)
    ax.axhline(np.log(2), color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.text(0.5, np.log(2) * 0.75, 'loss of a constant score', fontsize=6.3, color=INK2, ha='center', va='top')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 3)
    ax.set_xticks([0, 1], ['$k=2$', '$k=4$'])
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylabel('Training loss per group (site)')
    title(ax, 'd  Group losses')
    fig.savefig(OUT / 'fig_pureonly.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_bridging():
    """Cross-class bridges versus same-class additions, and even versus concentrated allocation."""
    cells = [(k, b, d) for b in (2, 5) for k in (2, 4) for d in ('oneway', 'twoway')]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.9))
    axes = axs.ravel()
    for j, (mm, xl, ttl) in enumerate((('within_site_AUC', '$\\Delta$ within-site AUC',
                                        'a  Is crossing needed? Test AUC'),
                                       ('reverse_AUC', '$\\Delta$ reverse-set AUC', 'b  Is crossing needed? Shortcut'))):
        ax = axes[j]
        ax.axvline(0, color=INK2, lw=0.6, zorder=1)
        yl = []
        for i, (k, b, d) in enumerate(cells):
            v = paired('T12C2/paired_repeat_differences.csv', 'C0-S0', mm, b=b, direction=d)[k]
            y = len(cells) - 1 - i
            strip(ax, y, v, BLUE_RAMP[d], w=0.12, s=6, alpha=0.55, horizontal=True)
            mean_marker(ax, np.mean(v), y, L3C, ms=6)
            yl.append(f"$k={k}$, $b={b}$, {'one' if d == 'oneway' else 'two'}-way ({len(v)})")
        ax.set_yticks(range(len(cells))[::-1], yl if j == 0 else [''] * len(cells))
        ax.set_ylim(-0.6, len(cells) - 0.4)
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel(xl + ', C0 − S0')
        title(ax, ttl)
    axes[0].set_xlim(-0.15, 0.2)
    axes[1].set_xlim(-0.35, 0.75)
    ax = axes[2]
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    yl = []
    cc = [(k, d) for k in (2, 4) for d in ('oneway', 'twoway')]
    for i, (k, d) in enumerate(cc):
        for feat, dy, mk in (('CC200', 0.13, 'o'), ('HO', -0.13, 's')):
            v = [f(r['difference']) for r in rows('T14A/paired_repeat_differences.csv')
                 if r['contrast'] == 'C0:equal-concentrated' and r['metric'] == 'common_reverse_AUC'
                 and r['k'] == str(k) and r['direction'] == d and r['feature'] == feat]
            y = 3 - i + dy
            strip(ax, y, v, BLUE_RAMP[d], w=0.05, s=6, alpha=0.5,
                  horizontal=True)
            mean_marker(ax, np.mean(v), y, L3C, ms=5.5, marker=mk)
        yl.append(f"$k={k}$, {'one' if d == 'oneway' else 'two'}-way")
    ax.set_yticks([3, 2, 1, 0], yl)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('$\\Delta$ reverse-set AUC,\neven − concentrated allocation')
    ax.legend(handles=[Line2D([], [], ls='', marker='o', color=L3C, ms=5, label='CC200'),
                       Line2D([], [], ls='', marker='s', color=L3C, ms=5, label='Harvard-Oxford')],
              loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6.3, handletextpad=0.2)
    title(ax, 'c  How to spend a budget? Spread evenly')
    L_panel(axes[3])
    title(axes[3], 'd  Does $L$ rank designs before training?')
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.6)
    fig.savefig(OUT / 'fig_bridging.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_lambda():
    """Mixing the pooled cross-entropy into L3: the cliff at lambda = 0, and what selection picks."""
    cur = defaultdict(dict)
    for r in rows('T16A/fixed_lambda_curve.csv'):
        if r['b'] == '5':
            cur[(int(r['k']), r['direction'], r['metric'])][f(r['lambda'])] = f(r['mean'])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    from matplotlib.colors import LinearSegmentedColormap
    lamcol = LinearSegmentedColormap.from_list('lam', [L3C, '#c3c2bd', STD])  # diverging: L3 blue -> C0 red
    styles = {(2, 'oneway'): (REF, '-'), (2, 'twoway'): (INK2, '-'),
              (4, 'oneway'): (REF, (0, (3, 1.5))), (4, 'twoway'): (INK2, (0, (3, 1.5)))}
    for j, (metric, yl, ttl, lim) in enumerate((('reverse_AUC', 'Reverse-set AUC', 'a  A cliff at $\\lambda=0$',
                                                (0.15, 0.72)),
                                               ('within_site_AUC', 'Within-site AUC on new sites',
                                                'b  for little gain in test AUC', (0.57, 0.65)))):
        ax = axes[j]
        if metric == 'reverse_AUC':
            chance(ax)
        ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
        for (k, d), (col, ls) in styles.items():
            c = cur[(k, d, metric)]
            lams = sorted(c)
            ax.plot(lams, [c[x] for x in lams], color=col, ls=ls, lw=1.3, zorder=3)
            if metric == 'reverse_AUC':
                ax.text(1.05, c[1.0], f"$k={k}$, {'one' if d == 'oneway' else 'two'}-way", fontsize=6.2,
                        color=INK2, va='center')
            ax.scatter(lams, [c[x] for x in lams], s=26, c=[lamcol(x) for x in lams], edgecolor='white',
                       linewidths=0.7, zorder=6)
        ax.set_xticks([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1], ['0\nL3', '.1', '.3', '.5', '.7', '.9', '1\nC0'])
        ax.set_xlabel('Weight $\\lambda$ of pooled cross-entropy')
        ax.set_ylabel(yl)
        ax.set_ylim(*lim)
        title(ax, ttl)
    axes[0].set_xlim(-0.05, 1.45)
    ax = axes[2]
    cnt = defaultdict(lambda: defaultdict(int))
    for r in rows('T16A/selected_lambda_distribution.csv'):
        if r['b'] == '5':
            cnt[r['method']][f(r['lambda'])] += int(r['count'])
    lams = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for m, col, dx, lab in (('P1-W', L3C, -0.17, 'within-site pair AUC'), ('P1-S', STD, 0.17, 'subject-level folds')):
        tot = sum(cnt[m].values())
        ax.bar(np.arange(7) + dx, [100 * cnt[m][x] / tot for x in lams], width=0.32, color=col, label=lab,
               zorder=3)
    ax.set_xticks(range(7), ['0', '.1', '.3', '.5', '.7', '.9', '1'])
    ax.set_xlabel('Selected $\\lambda$ ($b=5$ cells pooled)')
    ax.set_ylabel('Repeats (%)')
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.legend(title='inner-fold criterion', title_fontsize=6.3, loc='upper right', frameon=False, fontsize=6.3)
    title(ax, 'c  Tuning often misses $\\lambda=0$')
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(OUT / 'fig_lambda.pdf')
    plt.close(fig)


def fig_replication():
    """The three findings on ABIDE-II (top row) and ADHD-200 (bottom row)."""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for row, (cohort, base, ks) in enumerate((('ABIDE-II', 'T14B/stage2', (1, 2, 4)),
                                              ('ADHD-200', 'T17A/stage2', (1,)))):
        rm = defaultdict(list)
        for r in rows(f'{base}/repeat_metrics.csv'):
            v = f(r['within_site_AUC']) if r['reverse'] == 'False' else f(r['pooled_AUC'])
            rm[(r['method'], int(r['k']), int(r['b']), r['direction'], r['reverse'] == 'True')].append(v)
        ax = axes[row, 0]
        ax.axhspan(0, 0.5, color=TINT, zorder=0)
        for arm, rev, col, dx, lab in (('M', False, REF, 0.18, 'M, within-site'), ('P', False, STD, 0.0,
                                                                                    'P, within-site'),
                                       ('P', True, STD, -0.18, 'P, reverse set')):
            means = []
            for i, k in enumerate(ks):
                v = rm[(arm, k, 0, 'none', rev)]
                strip(ax, i + dx, v, col, w=0.05, s=4)
                means.append(np.mean(v))
                mean_marker(ax, i + dx, means[-1], col, hollow=rev)
            ax.plot(np.arange(len(ks)) + dx, means, color=col, lw=1.3, ls=(0, (4, 2)) if rev else '-', zorder=4,
                    label=lab)
        chance(ax)
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlim(-0.5, len(ks) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Class-pure sites per class $k$')
        ax.set_ylabel(f'{cohort}\nAUC')
        if row == 0:
            ax.legend(loc='upper right', frameon=False, fontsize=6, handlelength=2.0, borderaxespad=0.1)
        title(ax, f"{'ad'[row]}  Cost and shortcut")
        ax = axes[row, 1]
        ax.plot([0.2, 1], [0.2, 1], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.grid(color=GRID, lw=0.5, zorder=0)
        for r_m, col in (('E_subj', STD), ('E_LSPO', L3C)):
            for k, mk in ((1, 'D'), (2, 'o'), (4, 's')):
                pe = [r for r in rows(f'{base}/performance_estimates.csv') if r['method'] == r_m and r['k'] == str(k)]
                if not pe:
                    continue
                x = np.array([f(r['within_site_AUC']) for r in pe])
                y = np.array([f(r['estimate']) for r in pe])
                ax.scatter(x, y, s=8, color=col, alpha=0.4, lw=0, marker=mk, zorder=2)
                mean_marker(ax, x.mean(), y.mean(), col, ms=6.5, marker=mk)
        ax.set_xlim(0.2, 0.9)
        ax.set_ylim(0.2, 1.03)
        ax.set_xlabel('True within-site AUC on new sites')
        ax.set_ylabel('Estimate from training data')
        hs = [Line2D([], [], ls='', marker='o', color=STD, ms=5, label='subject-level CV')]
        if row == 0:
            hs.append(Line2D([], [], ls='', marker='o', color=L3C, ms=5, label='LSPO'))
            hs += [Line2D([], [], ls='', marker=mk, color=INK2, mfc='none', ms=5, label=f'$k={k}$')
                   for k, mk in ((2, 'o'), (4, 's'))]
        else:
            hs.append(Line2D([], [], ls='', marker='D', color=INK2, mfc='none', ms=5, label='$k=1$ (LSPO n.a.)'))
        ax.legend(handles=hs, loc='lower right', frameon=False, fontsize=6, handletextpad=0.2, borderaxespad=0.1)
        title(ax, f"{'be'[row]}  Subject-level CV overstates")
        ax = axes[row, 2]
        chance(ax, axis='x')
        cells = sorted({(k, b, d) for (m, k, b, d, rev) in rm if m == 'L3'})
        yl = []
        for i, (k, b, d) in enumerate(cells):
            y = len(cells) - 1 - i
            c0, l3 = np.mean(rm[('C0', k, b, d, True)]), np.mean(rm[('L3', k, b, d, True)])
            if abs(l3 - c0) > 0.04:
                ax.annotate('', xy=(l3, y), xytext=(c0, y),
                            arrowprops=dict(arrowstyle='-|>', lw=1.2, color=L3C, shrinkA=3, shrinkB=3))
            mean_marker(ax, c0, y, STD, ms=5.5)
            mean_marker(ax, l3, y, L3C, ms=5.5)
            yl.append(f"$k={k}$, $b={b}$, {'one' if d == 'oneway' else 'two'}-way")
        ax.set_yticks(range(len(cells))[::-1], yl)
        ax.tick_params(axis='y', length=0, labelsize=6.3)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(-0.6, len(cells) - 0.4)
        ax.set_xlabel('Reverse-set AUC: C0 (red) → L3 (blue)')
        title(ax, f"{'cf'[row]}  L3 lifts the reverse set")
    fig.tight_layout(pad=0.4, h_pad=1.2, w_pad=0.8)
    fig.savefig(OUT / 'fig_replication.pdf')
    plt.close(fig)


def fig_sitecode():
    """Sites are identifiable from connectivity, and the diagnostic score carries the site unless trained within sites."""
    a1 = defaultdict(lambda: defaultdict(list))
    for r in rows('T18/part1/A1_repeat_metrics.csv'):
        a1[r['group']][r['mode']].append(f(r['value']))
    chance_ = {r['group']: f(r['chance']) for r in rows('T18/part1/A1_main_table.csv')}
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.2], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    cohorts = ['ABIDE-I', 'ABIDE-II', 'ADHD-200']
    subsets = [('all', 'all participants', INK2), ('TD', 'controls only', CTRL), ('case', 'cases only', CASE)]
    for i, c in enumerate(cohorts):
        for j, (s, lab, col) in enumerate(subsets):
            x = i + (j - 1) * 0.25
            g = f'{c}_{s}'
            obs = np.mean(a1[g]['observed'])
            ax.bar(x, obs, width=0.2, color=col, zorder=3)
            ax.scatter(x + RNG.uniform(-0.06, 0.06, len(a1[g]['null'])), a1[g]['null'], s=4, color=INK, alpha=0.35,
                       lw=0, zorder=5)
        ax.plot([i - 0.42, i + 0.42], [chance_[f'{c}_all']] * 2, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
    ax.set_xticks(range(3), cohorts)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Balanced accuracy of site identification')
    ax.legend(handles=[Line2D([], [], ls='', marker='s', ms=6, color=c, label=l) for _, l, c in subsets] +
              [Line2D([], [], ls='', marker='o', ms=4, color=INK, alpha=0.5, label='permuted site labels'),
               Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), label='chance, 1/S')],
              loc='upper left', frameon=False, fontsize=6.2, ncol=2, handletextpad=0.3, columnspacing=0.8)
    title(ax, 'a  Sites are easy to identify')
    ax = fig.add_subplot(gs[0, 1])
    a2 = defaultdict(lambda: defaultdict(list))
    for r in rows('T18/part1/A2_repeat_metrics.csv'):
        a2[(r['method'], r['k'], r['b'], r['direction'])][r['repeat']].append(f(r['AUC']))
    series = [('B0', '0', 'none', 'B0, class-pure', DESIGN['pure']),
              ('C0', '5', 'oneway', 'C0, one-way bridges', DESIGN['oneway']),
              ('C0', '5', 'twoway', 'C0, two-way bridges', DESIGN['twoway']),
              ('L3', '5', 'oneway', 'L3, one-way bridges', BLUE_RAMP['oneway']),
              ('L3', '5', 'twoway', 'L3, two-way bridges', BLUE_RAMP['twoway'])]
    for i, k in enumerate(('2', '4')):
        for j, (m, b, d, lab, col) in enumerate(series):
            x = i + (j - 2) * 0.17
            v = [np.mean(q) for q in a2[(m, k, b, d)].values()]
            strip(ax, x, v, col, w=0.04, s=5, alpha=0.5)
            mean_marker(ax, x, np.mean(v), col, ms=6)
    chance(ax)
    ax.set_xticks([0, 1], ['$k=2$', '$k=4$'])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.15, 1.02)
    ax.set_ylabel('Same-diagnosis site AUC of the diagnostic score')
    ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=5, color=c, mec='white', label=l)
                       for *_, l, c in series], loc='lower left', frameon=False, fontsize=6.2, ncol=2,
              handletextpad=0.2, columnspacing=0.8)
    title(ax, 'b  and the pooled score encodes them')
    fig.savefig(OUT / 'fig_sitecode.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_sameinst():
    """Class-pure collections from the same institution carry much less shortcut than from different institutions."""
    rm = defaultdict(list)
    for r in rows('T18/part4/repeat_metrics.csv'):
        rm[(r['method'], r['reverse'] == 'True')].append(f(r['pooled_AUC'] if r['reverse'] == 'True'
                                                           else r['within_site_AUC']))
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.4)
    designs = [('P_cross', 'cases and controls\nfrom different institutions', STD),
               ('P_same', 'cases and controls from two\ncollections of one institution', L3C),
               ('M_same', 'both classes in\nevery collection', REF)]
    for j, (rev, yl, ttl) in enumerate(((False, 'Within-site AUC on new sites', 'a  Test AUC'),
                                       (True, 'Reverse-set AUC', 'b  Shortcut'))):
        ax = fig.add_subplot(gs[0, j])
        if rev:
            ax.axhspan(0, 0.5, color=TINT, zorder=0)
        chance(ax)
        xs = []
        for i, (m, lab, col) in enumerate(designs):
            v = rm[(m, rev)]
            if not v:
                continue
            strip(ax, i, v, col, w=0.12, s=5, alpha=0.45)
            mean_marker(ax, i, np.mean(v), col, ms=7)
            ax.annotate(f'{np.mean(v):.2f}', (i, np.mean(v)), xytext=(9, 0), textcoords='offset points',
                        va='center', fontsize=6.3, color=INK)
            xs.append(i)
        if rev:
            ax.text(2, 0.06, 'no reverse\nset', ha='center', fontsize=6.2, color=INK2)
        ax.set_xticks(range(3), ['different\ninstitutions', 'same\ninstitution', 'mixed\n(reference)'])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0 if rev else 0.35, 1 if rev else 0.8)
        ax.set_ylabel(yl)
        title(ax, ttl)
    ax = fig.add_subplot(gs[0, 2])
    pts = sorted(rows('T18/part4/institution_comparison.csv'), key=lambda r: f(r['subset_AUC']))
    x = [f(r['subset_AUC']) for r in pts]
    y = [f(r['reverse_AUC']) for r in pts]
    ax.axhspan(0, 0.5, color=TINT, zorder=0)
    chance(ax)
    ax.plot(x, y, color=REF, lw=0.8, zorder=2)
    for r, xi, yi in zip(pts, x, y):
        mean_marker(ax, xi, yi, L3C, ms=7)
        ax.annotate(r['site'], (xi, yi), xytext=(-6, 6), textcoords='offset points', fontsize=6.5, color=INK,
                    ha='right')
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.15, 0.8)
    ax.set_xlabel('AUC for telling the two collections apart')
    ax.set_ylabel('Reverse-set AUC (same institution)')
    title(ax, 'c  The more distinct, the worse')
    fig.savefig(OUT / 'fig_sameinst.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_published():
    """Two published multi-site methods (T19): shortcut under class-pure training, with and without the reverse set
    in their transductive unlabeled pool; bridging; subject-level CV against new sites."""
    d = defaultdict(list)  # (method, design, k, metric, condition) -> per-repeat AUCs
    for r in rows('T19/rev2/repeat_metrics.csv'):
        if r['model'] == 'LR':
            rev = r['reverse'] == 'True'
            d[('LR', r['design'], int(r['k']), rev, 'real')].append(f(r['pooled_AUC'] if rev else r['within_site_AUC']))
        elif r['design'] == 'M' and r['model'] in ('Kunda', 'Parisot-HO'):
            d[(r['model'], 'M', int(r['k']), False, 'real')].append(f(r['within_site_AUC']))
    for r in rows('T19/rev3/repeat_metrics.csv'):
        key = (r['model'], r['design'], int(r['k']), r['reverse'] == 'True')
        d[key + ('real',)].append(f(r['AUC']))
        d[key + ('pool',)].append(f(r['AUC_rev2']))
    methods = [('LR', 'Logistic\nregression'), ('Parisot-HO', 'Parisot\n(population GCN)'), ('Kunda', 'Kunda\n(MIDA)')]
    conds = [('B0', 'real', STD, -0.24, False), ('B0', 'pool', REF, 0.0, True), ('C0', 'real', L3C, 0.24, False)]
    fig = plt.figure(figsize=(7.2, 5.1))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.3)
    panels = [(gs[0, 0], 2, True, 'a  Shortcut, $k=2$'), (gs[0, 1], 4, True, 'b  Shortcut, $k=4$'),
              (gs[1, 0], 2, False, 'c  Test AUC on new sites, $k=2$')]
    for spec, k, rev, ttl in panels:
        ax = fig.add_subplot(spec)
        if rev:
            ax.axhspan(0, 0.5, color=TINT, zorder=0)
        chance(ax)
        for i, (m, _) in enumerate(methods):
            for des, cond, col, dx, hollow in conds:
                v = d.get((m, des, k, rev, cond))
                if not v:
                    continue
                strip(ax, i + dx, v, col, w=0.06, s=4, alpha=0.4)
                mean_marker(ax, i + dx, np.mean(v), col, ms=6.5, hollow=hollow)
                ax.text(i + dx, 0.965 if rev else 0.785, f'{np.mean(v):.2f}', ha='center', va='top', fontsize=5.8,
                        color=INK)
            if not rev:
                mv = d[(m, 'M', k, False, 'real')]
                ax.plot([i - 0.36, i + 0.36], [np.mean(mv)] * 2, color=INK, lw=0.9, zorder=4)
        ax.set_xticks(range(3), [lab for _, lab in methods])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim((0, 1) if rev else (0.35, 0.8))
        ax.set_yticks(np.arange(0, 1.01, 0.2) if rev else np.arange(0.4, 0.71, 0.1))
        ax.set_ylabel('Reverse-set AUC' if rev else 'Within-site AUC on new sites')
        title(ax, ttl)
    ax = fig.add_subplot(gs[1, 1])
    chance(ax)
    es = defaultdict(list)
    for r in rows('T19/rev2/E_subj.csv'):
        es[r['model']].append(f(r['E_subj']))
    for i, (m, _) in enumerate(methods):
        a, b = np.mean(es[m]), np.mean(d[(m, 'B0', 2, False, 'real')])
        strip(ax, i - 0.15, es[m], STD, w=0.05, s=4, alpha=0.4)
        strip(ax, i + 0.15, d[(m, 'B0', 2, False, 'real')], L3C, w=0.05, s=4, alpha=0.4)
        ax.plot([i - 0.15, i + 0.15], [a, b], color=INK2, lw=0.7, zorder=4)
        mean_marker(ax, i - 0.15, a, STD, ms=6.5)
        mean_marker(ax, i + 0.15, b, L3C, ms=6.5)
        ax.text(i, 1.1, f'{a - b:+.2f}', ha='center', va='top', fontsize=6.3, color=INK, fontweight='bold')
    ax.set_xticks(range(3), [lab for _, lab in methods])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('AUC (class-pure B0, $k=2$)')
    title(ax, 'd  Subject-level CV vs. new sites')
    h = [Line2D([], [], marker='o', ls='', color=STD, mec='white', ms=6.5, label='class-pure B0, reverse set not in training pool'),
         Line2D([], [], marker='o', ls='', color=REF, mfc='white', mew=1.2, ms=6.5,
                label='class-pure B0, reverse set in the unlabeled pool (original transductive setting)'),
         Line2D([], [], marker='o', ls='', color=L3C, mec='white', ms=6.5, label='two-way bridging C0, $b=5$'),
         Line2D([], [], color=INK, lw=0.9, label='mixed sites M (panel c)')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, fontsize=6.6,
               handletextpad=0.3, columnspacing=1.2)
    h2 = [Line2D([], [], marker='o', ls='', color=STD, mec='white', ms=6.5, label='subject-level 5-fold CV'),
          Line2D([], [], marker='o', ls='', color=L3C, mec='white', ms=6.5, label='within-site AUC on new sites')]
    ax.legend(handles=h2, loc='lower left', frameon=False, fontsize=6.3, handletextpad=0.2)
    fig.subplots_adjust(top=0.87)
    fig.savefig(OUT / 'fig_published.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_partial():
    """Partial confounding on ABIDE-I (T20-A): case fraction 0.5 + delta at case-leaning sites, k = 3, m = 20."""
    deltas = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
    met = defaultdict(list)
    for r in rows('T20A/repeat_metrics.csv'):
        met[(r['method'], r['delta'])].append(r)
    x = np.arange(len(deltas))
    meths = (('M0', STD, 'o', '-', 'pooled'), ('RW', DESIGN['twoway'], 's', '-', 'site-by-class weights'),
             ('CB', DESIGN['pure'], 'D', '-', 'ComBat'), ('L3', L3C, 'o', '-', 'L3'),
             ('L1', BLUE_RAMP['oneway'], '^', (0, (3, 1.5)), 'L1 (site intercepts)'))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for ax, metric, ttl in ((axes[0], 'reverse_auc', 'a  Reverse-set AUC'),
                            (axes[1], 'test_auc', 'b  Within-site AUC on new sites')):
        ax.grid(color=GRID, lw=0.5, zorder=0)
        if metric == 'reverse_auc':
            ax.axhspan(0, 0.5, color=TINT, zorder=0, lw=0)
            chance(ax)
        for m, col, mk, ls, lab in meths:
            xs = [i for i, d in enumerate(deltas) if met[(m, d)]]
            ys = [np.mean([f(r[metric]) for r in met[(m, deltas[i])]]) for i in xs]
            ax.plot(xs, ys, color=col, lw=1.4, ls=ls, marker=mk, ms=4, mec='white', mew=0.6, zorder=3, label=lab)
        title(ax, ttl)
    axes[0].set_ylim(0.15, 0.72)
    axes[1].set_ylim(0.57, 0.665)
    ax = axes[2]
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.axhline(0, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for col, key, off, lab in ((STD, 'subject_cv_auc', -0.15, 'subject-level CV'),
                               (L3C, 'site_cv_auc', 0.15, 'leave one training site out')):
        xs, ys = [], []
        for i, d in enumerate(deltas):
            e = [f(r[key]) - f(r['test_auc']) for r in met[('M0', d)] if r[key]]
            if e:
                strip(ax, i + off, e, col, w=0.06, s=4)
                xs.append(i + off)
                ys.append(np.mean(e))
        ax.plot(xs, ys, color=col, lw=1.4, marker='o', ms=4, mec='white', mew=0.6, zorder=4, label=lab)
    ax.legend(loc='upper left', frameon=False, fontsize=6.3, handlelength=1.6, borderaxespad=0.2)
    title(ax, 'c  Error of training-data estimates')
    for ax in axes:
        ax.set_xticks(x, ['0', '0.04', '0.16', '0.36', '0.64', '1'])
        ax.set_xlabel(r'$L/L_{\mathrm{pure}}$')
    axes[0].set_ylabel('AUC')
    axes[2].set_ylabel('Estimate $-$ AUC on new sites')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.36, 1.13), fontsize=6.5,
               handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(OUT / 'fig_partial.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    todo = sys.argv[1:] or ['fig2', 'fig3', 'fig_problem', 'fig_anatomy', 'fig_evaluation', 'fig_pureonly',
                            'fig_bridging', 'fig_lambda', 'fig_replication', 'fig_sitecode', 'fig_sameinst', 'fig_published', 'fig_partial']
    for name in todo:
        globals()[name]()
