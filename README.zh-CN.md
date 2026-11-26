# Class-pure sites

[English](README.md) | 中文

本仓库是以下论文的代码和汇总结果：

> **When the site predicts the diagnosis: evaluation, bridging and within-site training for multi-site neuroimaging classifiers**
> Jincheng Hu, Ping Xie, Jiaxin Li, Sha Wang, Mengchen Liu, Guoqian Jiang（燕山大学）

多站点研究中，如果每个站点只提供一种诊断的被试（纯类站点），站点就能预测标签，按被试的交叉验证会奖励学会站点的模型。本文在 ABIDE-I、ABIDE-II 和 ADHD-200 的静息态 fMRI 上检验了三条规则：

1. **评估：**留出整个站点（站点对留出，LSPO），并用反向集检验模型是否按站点排序。
2. **设计：**在每个站点招募少量缺少的那种诊断的桥接被试，用设计系数 *L* 指导招募方案。
3. **估计：**只用站点内的病例与对照配对训练（站点内配对 logistic 损失 L3）。

## 目录

| 路径 | 内容 |
|---|---|
| `*.py` | 每个实验一个脚本（见下表）；`*_report.py` 生成该实验的汇总表 |
| `tests/` | 针对容易悄悄出错的步骤的测试：站点角色、被试划分、标签对应。测试读取 `abide_class_pure_sites.py`、`abide2_prepare.py` 和 `adhd200_prepare.py` 生成的被试名单，需先运行这三个脚本 |
| `outputs/` | `paper_figures.py` 和 `paper_tables.py` 读取的汇总表 |
| `paper/` | 论文和补充材料的 LaTeX 源文件、图和 PDF |

代码注释和文档字符串是中文，函数名和变量名是英文。

## 实验

| 编号 | 脚本 | 内容 | 论文位置 |
|---|---|---|---|
| T12A | `abide_class_pure_sites.py` | ABIDE-I 上的纯类设计（P）与混合设计（M）、纯类代价、反向集 | 第 6.1 节 |
| T12B | `site_pair_validation.py` | 按被试交叉验证与站点对留出（LSPO）的估计 | 第 6.2 节 |
| T12C | `bridge_design.py` | 桥接设计、合并模型 C0、ComBat | 第 6.4 节 |
| T12C2 | `same_class_control.py` | 同类对照 S0 | 第 6.4 节 |
| T12D | `abide_robustness.py` | 质控后的被试池、Harvard-Oxford 图谱、BrainNetCNN | 补充材料 S5 |
| T13A, T13B | `within_site_pair_model.py`, `pair_estimator_confirmation.py` | 站点内配对损失 L3 与站点固定效应模型 L1 | 第 6.5 节 |
| T14A | `design_rule_baselines.py` | 预算分配与设计系数 *L* | 第 6.4 节 |
| T14B | `abide2_prepare.py`, `abide2_replication.py` | ABIDE-II 复现 | 第 6.6 节 |
| T15A | `group_dro.py` | 每个站点一组的 Group DRO | 第 6.3 节 |
| T15B | `natural_pure_collections.py` | 天然纯类集合（KUL_3、WashU） | 第 6.2 节 |
| T16A | `pair_ce_mixture.py` | 在 L3 中混入合并损失 | 第 6.5 节 |
| T16B, T16C | `deep_pair_pilot.py`, `deep_abide1.py` | 冻结自监督编码器加线性头；深度基线 | 补充材料 S4 |
| T17A | `adhd200_prepare.py`, `adhd200_replication.py` | ADHD-200 复现 | 第 6.6 节 |
| T18 | `supplementary_*.py` | 站点探针、同机构设计、校正重抽样检验 | 补充材料 S3、S5 |
| T19 | `published_methods*.py` | 人群图卷积网络（Parisot 等）与 TPE-MIDA（Kunda 等） | 第 6.3 节，补充材料 S3、S5 |
| | `theory_a_check.py` | 命题的蒙特卡洛检验（合成数据） | 补充材料 S1 |

每个实验的判据都在运行之前确定（补充材料 S2）。部分脚本会对内部任务说明（`docs/tasks/`、`AGENTS.md`）计算溯源哈希，这些文件不在本仓库中；重新运行前，请从溯源哈希的文件列表中删去它们。

## 数据

三个队列都是公开数据，本仓库不再分发。

* **ABIDE-I：**Preprocessed Connectomes Project 发布版（CPAC 流程，带通滤波，不做全局信号回归），CC200 时间序列。默认位置 `~/datasets_cache/ABIDE_PCP`。
* **ABIDE-II：**DCAN 处理版，Gordon 分区加皮层下区域，由 `abide2_prepare.py` 下载。默认位置 `~/datasets_cache/ABIDE2_DCAN`。
* **ADHD-200：**Athena 发布版，CC200 时间序列。默认位置 `~/datasets_cache/ADHD200_Athena`。

派生数组缓存在 `~/datasets_cache/ABIDE_derived` 和 `~/datasets_cache/ADHD200_derived`。如需改用其他位置，请修改 `abide_class_pure_sites.py`、`abide_robustness.py`、`pair_estimator_confirmation.py`、`abide2_prepare.py`、`abide2_replication.py` 和 `adhd200_prepare.py` 开头的路径常量。

ABIDE 与 ADHD-200 按各自的数据使用条款共享（ABIDE 为 CC BY-NC-SA 3.0）。`outputs/` 中的两份逐被试预测文件（`T12A/predictions.csv`、`T13B/reference_predictions.csv`）只保留个体分数图用到的行，只含被试编号、站点、标签和模型分数，不含表型变量。

## 已发表方法

`published_methods.py` 调用原作者的代码，默认位置 `~/external_code`：

| 方法 | 仓库 | 提交 |
|---|---|---|
| 人群图卷积网络 | https://github.com/parisots/population-gcn | `185e2f5` |
| Chebyshev 图卷积层 | https://github.com/parisots/gcn | `506ef57` |
| TPE-MIDA | https://github.com/kundaMwiza/fMRI-site-adaptation | `0a70cf8` |

## 复现图表

```bash
pip install -r requirements.txt
python paper_figures.py      # 由 outputs/ 生成结果图，写入 paper/figures/
python paper_tables.py       # 生成 paper/tables/tests_en.tex
```

重新编译 PDF：在 `paper/` 中依次运行 `pdflatex -output-directory=build main`、`bibtex build/main`，对 `supplement` 做同样两步，再编译一次 `supplement`、两次 `main`，使两份文档之间的交叉引用生效。

实验本身需要上述数据，每个脚本把结果写入 `outputs/<编号>/`。部分实验需要运行数小时。

## 许可证

代码采用 MIT 许可证（见 `LICENSE`）。数据归 ABIDE 和 ADHD-200 联盟所有，遵循其使用条款。
