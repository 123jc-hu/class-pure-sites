"""Appendix tables of paper A built from saved summary CSVs (no recomputation); writes paper/tables/*.tex."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper' / 'tables'
FAMILY = {1: ('Cost of class purity', '类别纯度的代价'), 2: ('Evaluation', '评估'), 3: ('Bridging', '桥接'),
          4: ('Within-site estimation', '站点内估计'), 5: ('Methods on class-pure data', '只用纯类数据的方法'),
          6: ('Replication', '复现')}
CONTRAST = {'M-P': 'M $-$ P', 'E_subj-truth': '$E_{\\mathrm{subj}}-$ truth',
            'abs(E_LSPO-truth)-abs(E_subj-truth)': 'abs.\\ error LSPO $-$ subj.',
            'C0-B0': 'C0 $-$ B0', 'C0-S0': 'C0 $-$ S0', 'L3-C0': 'L3 $-$ C0', 'GroupDRO-B0': 'Group DRO $-$ ERM',
            'LSPO-selected-B0': 'LSPO tuning $-$ B0', 'NAP-B0': 'NAP $-$ B0', 'P_k4-P_k1': 'P, $k$4 $-$ $k$1',
            'gap_k4-gap_k1': 'cost, $k$4 $-$ $k$1'}
METRIC = {'within_site_AUC': ('test', '测试'), 'reverse_AUC': ('reverse', '反向集')}


def tests_table():
    """All corrected resampled t-tests of T18 part 2, with the number of repeats whose difference is positive."""
    d = pd.read_csv(ROOT / 'outputs/T18/part2/corrected_tests.csv')
    r = pd.read_csv(ROOT / 'outputs/T18/part2/paired_repeats.csv')
    key = ['family', 'dataset', 'contrast', 'metric', 'cell']
    pos = r.assign(pos=r.difference > 0).groupby(key).agg(npos=('pos', 'sum'), J2=('pos', 'size')).reset_index()
    d = d.merge(pos, on=key, how='left', validate='one_to_one')
    assert (d.J == d.J2).all()  # paired repeats must match the tested repeats
    for lang in (0, 1):
        head = ('Cohort & Contrast & Metric & Cell & $J$ & Mean & $n_{>0}$ & $p$ & $p_{\\mathrm{Holm}}$\\\\',
                '队列 & 对比 & 指标 & 格子 & $J$ & 均值 & $n_{>0}$ & $p$ & $p_{\\mathrm{Holm}}$\\\\')[lang]
        cap = ('Corrected resampled $t$-tests of the paired contrasts (test: within-site AUC on new sites; '
               'reverse: reverse-set AUC).', '配对对比的校正重抽样 $t$ 检验（测试：新站点上的站点内 AUC；反向集：反向集 AUC）。')[lang]
        lines = ['\\begin{longtable}{lllcrrrrr}', f'\\caption{{{cap}}}\\label{{tab:tests}}\\\\', '\\toprule', head,
                 '\\midrule', '\\endfirsthead', '\\toprule', head, '\\midrule', '\\endhead', '\\bottomrule', '\\endfoot']
        for fam, g in d.groupby('family'):
            lines.append(f'\\multicolumn{{9}}{{l}}{{\\textit{{{fam}. {FAMILY[fam][lang]}}}}}\\\\')
            for _, x in g.iterrows():
                c = CONTRAST[x.contrast]
                if lang:
                    c = c.replace('truth', '真值').replace('abs.\\ error LSPO $-$ subj.', '绝对误差 LSPO $-$ 被试').replace(
                        'LSPO tuning', 'LSPO 选参').replace('cost,', '代价,')
                cell = x.cell.replace('oneway', '1w').replace('twoway', '2w')
                lines.append(f'{x.dataset} & {c} & {METRIC[x.metric][lang]} & {cell} & {x.J} & '
                             f'${x["mean"]:+.3f}$ & {x.npos} & {x.p:.3f} & {x.p_Holm:.3f}\\\\')
        lines.append('\\end{longtable}')
        (OUT / ('tests_en.tex', 'tests_zh.tex')[lang]).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    tests_table()
