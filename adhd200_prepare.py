"""T17-A 的 ADHD-200 Athena CC200 数据准备。"""
from hashlib import file_digest
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
RAW = Path.home()/'datasets_cache/ADHD200_Athena'
CACHE = Path.home()/'datasets_cache/ADHD200_derived/T17A'
OUT = REPO/'outputs/T17A'
ID = 'ScanDir ID'
DX = {'0':0,'1':1,'2':1,'3':1}
SCAN = re.compile(r'sfnwmrda(\d+)_session_(\d+)_rest_(\d+)_cc200_TCs\.1D')
# 两个发布的首扫表头一致，标签间有十个空缺。
ROI = tuple(i for i in range(1,201) if i not in {2,5,11,25,34,68,82,95,126,191})
HEADER = ('File','Sub-brick',*(f'Mean_{i}' for i in ROI))
REASONS = ['pending','qc_fail','missing_session_1_rest_1','read_failure']
OFFICIAL_URL = 'https://fcon_1000.projects.nitrc.org/indi/adhd200/general/allSubs_testSet_phenotypic_dx.csv'
KEY_URL = 'https://fcon_1000.projects.nitrc.org/indi/adhd200/general/ADHD-200_PhenotypicKey.pdf'
WIKI = 'https://www.nitrc.org/plugins/mwiki/index.php/neurobureau:AthenaPipeline'
ATHENA_CODE = 'https://github.com/preprocessed-connectomes-project/adhd200_athena_scripts/blob/master/NYU/preproc.sh#L253-L288'
ATHENA_PAPER = 'https://pure.mpg.de/rest/items/item_2353919_6/component/file_3512764/content'


def subjects():
    assert sys.flags.isolated
    integrated = pd.read_csv(RAW/'adhd200_preprocessed_phenotypics.tsv',sep='\t',
                             usecols=[ID,'Site','DX','Age','Gender','QC_Athena'],dtype={ID:str,'DX':str})
    integrated['id'] = integrated[ID].astype(int)
    local = []
    for release in ('train','test'):
        for file in sorted((RAW/release).glob('*/*_phenotypic.csv')):
            site = file.parent
            original = pd.read_csv(file,usecols=lambda c:c in [ID,'ScanDirID','Site','DX'],
                                   dtype={ID:str,'ScanDirID':str,'DX':str}).rename(columns={'ScanDirID':ID})
            original['id'] = original[ID].astype(int)
            dirs = [p.name for p in site.iterdir() if p.is_dir() and p.name.isdecimal()]
            mapping = {int(name):name for name in dirs}
            assert len(mapping)==len(dirs) and original.id.is_unique and set(mapping)==set(original.id),file
            local.append(original.drop(columns=ID).rename(columns={'DX':'original_DX','Site':'original_Site'})
                         .assign(release=release,site=site.name,dir_id=original.id.map(mapping)))
    inventory = pd.concat(local,ignore_index=True)
    overlap = set(inventory.loc[inventory.release.eq('train'),'id']) & set(inventory.loc[inventory.release.eq('test'),'id'])
    assert not overlap and inventory.id.is_unique and set(inventory.id)==set(integrated.id)
    f = integrated.merge(inventory,on='id',validate='one_to_one')
    assert f.Site.eq(f.original_Site).all()
    f['label'] = f.DX.map(DX).astype('Int64')
    candidate = f.DX.isin(DX)&f.QC_Athena.eq(1)
    train = f.loc[candidate&f.release.eq('train')]
    test = f.loc[f.release.eq('test')]
    assert test.original_DX.eq('withheld').all()
    official = pd.read_csv(CACHE/'official_test_phenotypes.csv',usecols=['ID','DX','Site'],
                           dtype={'ID':str,'DX':str}).rename(columns={'DX':'official_DX','Site':'official_Site'})
    official['id'] = official.ID.astype(int)
    assert set(official.id)==set(test.id)
    audited = test.merge(official.drop(columns='ID'),on='id',validate='one_to_one')
    assert audited.Site.eq(audited.official_Site).all()
    train_mismatch = int(train.DX.ne(train.original_DX).sum())
    test_mismatch = int(audited.DX.ne(audited.official_DX).sum())
    assert train_mismatch==test_mismatch==0,(train_mismatch,test_mismatch)
    audit = dict(release_counts=inventory.release.value_counts().to_dict(),directory_matches=len(f),
                 leading_zero_id_matches=int(f[ID].ne(f.dir_id).sum()),duplicate_release_ids=len(overlap),
                 train_dx_checked=len(train),train_dx_mismatches=train_mismatch,
                 test_dx_checked=len(audited),test_dx_eligible=int((audited.DX.isin(DX)&audited.QC_Athena.eq(1)).sum()),
                 test_dx_mismatches=test_mismatch)
    return f,audit


def scans(row):
    folder = RAW/row.release/row.site/row.dir_id
    indexed = {}
    for path in folder.glob('sfnwmrda*_cc200_TCs.1D'):
        raw_id,session,rest = map(int,SCAN.fullmatch(path.name).groups())
        assert raw_id==row.id and (session,rest) not in indexed,path
        indexed[session,rest] = path
    return indexed


def time_series(path):
    with path.open() as stream:
        actual = tuple(value.strip() for value in stream.readline().split('\t'))
        assert actual==HEADER,path
        values = np.loadtxt(stream,delimiter='\t',usecols=range(2,len(HEADER)),ndmin=2)
    if len(values)<2 or not np.isfinite(values).all():
        raise ValueError('需要至少两行有限的 ROI 时间点')
    return values


def site_table(subject_table):
    rows = []
    for site,all_rows in subject_table.groupby('site'):
        g = all_rows.loc[all_rows.include]
        qc = all_rows.loc[all_rows.QC_Athena.eq(1)]
        male = g.loc[g.Gender.eq(1)]
        age,time = g.Age,g.n_timepoints
        td,adhd = int(g.label.eq(0).sum()),int(g.label.eq(1).sum())
        group = 'empty' if not len(g) else 'TD_only' if adhd==0 else 'ADHD_only' if td==0 else 'small_class' if min(td,adhd)<5 else 'mixed'
        rows.append(dict(site=site,raw_n=len(all_rows),qc_TD=int(qc.label.eq(0).sum()),
                         qc_ADHD=int(qc.label.eq(1).sum()),included_n=len(g),TD=td,ADHD=adhd,
                         male_TD=int(male.label.eq(0).sum()),male_ADHD=int(male.label.eq(1).sum()),
                         gender_missing=int(g.Gender.isna().sum()),age_min=age.min(),age_median=age.median(),
                         age_max=age.max(),time_min=time.min(),time_median=time.median(),time_max=time.max(),
                         subjects_with_zero_roi=int(g.n_zero_roi.gt(0).sum()),site_type=group))
    return pd.DataFrame(rows)


def report(f,sites,audit,counts,shape):
    kept = f.loc[f.include]
    male = int(kept.Gender.eq(1).sum())
    lines = ['# T17-A ADHD-200 Athena 数据准备','',
             f"训练发布 776 人，测试发布 197 人，共 {audit['directory_matches']} 人。"
             f"有 {audit['leading_zero_id_matches']} 个整合表 ID 的前导零写法与目录不同，统一按数值 ID 对应。"
             '两个发布的重复 ID 为 0。','',
             f"通过诊断和 QC 筛选的训练发布 {audit['train_dx_checked']} 人，与原站点 DX 逐人一致。"
             f"测试发布原站点表全部写 withheld，依用户确认，以联盟赛后公开表核对全部 {audit['test_dx_checked']} 人"
             f"（其中 {audit['test_dx_eligible']} 人通过诊断和 QC 筛选），DX 不一致为 0。"
             f"标签始终来自整合 TSV。[联盟公开核对表]({OFFICIAL_URL})",'',
             '## 排除和扫描','',
             f"- 依序排除 pending {counts['pending']} 人、QC 不合格 {counts['qc_fail']} 人、"
             f"缺 session_1_rest_1 滤波文件 {counts['missing_session_1_rest_1']} 人、读取失败 {counts['read_failure']} 人。"
             f"最终 {len(kept)} 人，TD {kept.label.eq(0).sum()} 人，ADHD {kept.label.eq(1).sum()} 人。",
             f"- 原发布中 {f.n_rest_scans.gt(1).sum()} 人有多个已发布 rest 扫描，"
             f"{f.n_sessions.gt(1).sum()} 人有多个已发布 session。纳入者相应为 "
             f"{kept.n_rest_scans.gt(1).sum()} 人及 {kept.n_sessions.gt(1).sum()} 人。"
             '每人只取 session_1_rest_1 的 sfnwmrda，忽略通配残留。未追加删帧或头动筛选。',
             f"- Gender 1 为男性，0 为女性。纳入者男性 {male} 人，性别缺失 {kept.Gender.isna().sum()} 人；"
             f"性别缺失者纳入总体，男性小计按 Gender=1。[表型编码说明]({KEY_URL})",'',
             '## QC 和首扫筛选后的站点表','',
             '| 站点 | TD | ADHD | 男 TD | 男 ADHD | 年龄中位数（范围） | 时间点中位数（范围） | 有零方差 ROI 的人数 |',
             '|---|---:|---:|---:|---:|---|---|---:|']
    for r in sites.itertuples():
        age = f'{r.age_median:g}（{r.age_min:g}–{r.age_max:g}）' if r.included_n else '—'
        time = f'{r.time_median:g}（{r.time_min:g}–{r.time_max:g}）' if r.included_n else '—'
        lines.append(f'| {r.site} | {r.TD} | {r.ADHD} | {r.male_TD} | {r.male_ADHD} | {age} | {time} | {r.subjects_with_zero_roi} |')
    lines.extend(['','单类或任一类少于 5 人的非空站点：'])
    for r in sites.loc[sites.site_type.isin(['TD_only','ADHD_only','small_class'])].itertuples():
        lines.append(f'- {r.site}：TD {r.TD} 人、ADHD {r.ADHD} 人。')
    lines.extend(['','Brown 因 pending 无纳入者。精确时间点数频数见 timepoint_counts.csv。','',
                  '## 特征与全局信号回归','',
                  f"AFNI 3dROIstats 输出为制表符分隔。首行为 File、Sub-brick 和 {len(ROI)} 个 Mean_<ROI>，"
                  f"前两列分别是输入 NIfTI 文件名和时间子砖索引。按表头原顺序保留数值列。"
                  f"CC200 文件实际含 {len(ROI)} 个 ROI，Fisher z 上三角为 {shape[1]:,} 维。"
                  '沿用 Fisher z 函数处理零方差 ROI。缓存 fisher_z.npy 的行顺序为 row_order.csv。','',
                  '对该版 Athena 文件，官方流程及实现支持未做全局信号回归的结论。'
                  'NITRC AthenaPipeline wiki 无法打开，以发布论文 §3.1.2 和官方 3dDeconvolve 命令核对。'
                  '该回归使用六项头动、白质、脑脊液和多项式基线，随后做 0.009–0.08 Hz 带通。'
                  '本地两个发布的 extract_tcs.sh 相同，用 3dROIstats 提取区域均值，'
                  f"滤波分支不进入脚本中针对 sn* 的 FSL GLM。[流程论文]({ATHENA_PAPER}) · [官方实现]({ATHENA_CODE})",'',
                  '第一阶段已停止于数据准备。未做分类或计算 AUC。'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def prepare():
    tick = perf_counter()
    f,audit = subjects()
    sys.path.insert(0,str(REPO))
    from abide_class_pure_sites import fisher_features

    features,rows = [],[]
    for row in f.itertuples():
        files = scans(row)
        primary = files.get((1,1))
        record = dict(n_rest_scans=len(files),n_sessions=len({session for session,_ in files}),
                      n_timepoints=None,n_zero_roi=None,exclude_reason='',read_error='')
        if row.DX=='pending':
            record['exclude_reason'] = 'pending'
        elif row.QC_Athena!=1:
            record['exclude_reason'] = 'qc_fail'
        elif primary is None:
            record['exclude_reason'] = 'missing_session_1_rest_1'
        else:
            try:
                x = time_series(primary)
            except (OSError,ValueError) as error:
                record.update(exclude_reason='read_failure',read_error=str(error))
            else:
                v,n_zero = fisher_features(x)
                assert np.isfinite(v).all(),primary
                features.append(v)
                record.update(n_timepoints=len(x),n_zero_roi=n_zero)
        rows.append(record)
    f = pd.concat([f,pd.DataFrame(rows)],axis=1)
    f['include'] = f.exclude_reason.eq('')
    matrix = np.stack(features)
    assert matrix.shape==(f.include.sum(),len(ROI)*(len(ROI)-1)//2)
    sites = site_table(f)
    assert sites.included_n.sum()==len(matrix) and (sites.TD+sites.ADHD).sum()==len(matrix)
    counts = f.exclude_reason.value_counts().reindex(REASONS,fill_value=0).to_dict()
    assert sum(counts.values())+len(matrix)==len(f)
    OUT.mkdir()
    np.save(CACHE/'fisher_z.npy',matrix,allow_pickle=False)
    f.loc[f.include,[ID]].to_csv(CACHE/'row_order.csv',index=False)
    fields = [ID,'site','release','DX','label','Age','Gender','QC_Athena','n_timepoints','n_zero_roi',
              'n_rest_scans','n_sessions','include','exclude_reason','read_error']
    f[fields].to_csv(OUT/'subjects.csv',index=False)
    sites.to_csv(OUT/'sites.csv',index=False)
    f.loc[f.include].groupby(['site','n_timepoints']).size().rename('subjects').reset_index().to_csv(
        OUT/'timepoint_counts.csv',index=False)
    inputs = [RAW/'adhd200_preprocessed_phenotypics.tsv',CACHE/'official_test_phenotypes.csv',
              *sorted(RAW.glob('*/*/*_phenotypic.csv')), *sorted(RAW.glob('*/extract_tcs.sh'))]
    hashes = {}
    for path in inputs:
        with path.open('rb') as stream:
            hashes[str(path)] = file_digest(stream,'sha256').hexdigest()
    config = dict(audit=audit,exclusions=counts,feature_shape=matrix.shape,roi=ROI,header=HEADER,
                  cache=str(CACHE),official_verification_url=OFFICIAL_URL,athena_wiki=WIKI,
                  athena_source=ATHENA_CODE,athena_paper=ATHENA_PAPER,roi_filename=SCAN.pattern,
                  selection='sfnwmrda session 1 rest 1; QC_Athena=1; DX in 0,1,2,3',
                  input_sha256=hashes,python=sys.version,numpy=np.__version__,pandas=pd.__version__,
                  wall_seconds=perf_counter()-tick)
    (OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    report(f,sites,audit,counts,matrix.shape)
    print(json.dumps(dict(included=len(matrix),shape=matrix.shape,exclusions=counts,
                          seconds=round(perf_counter()-tick,2)),ensure_ascii=False))


if __name__=='__main__':
    if sys.argv[1:]==['check_labels']:
        print(json.dumps(subjects()[1]))
    else:
        prepare()
