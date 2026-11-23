"""T14-B第一阶段核对和汇总；不定义或运行分类/性能评估。"""
from pathlib import Path
import sys,json,hashlib,runpy,csv
import numpy as np
import pandas as pd
import nibabel as nib

REPO=Path(__file__).resolve().parent
d=runpy.run_path(str(REPO/'abide2_prepare.py'))
OUT=d['OUT'];CACHE=d['CACHE'];RAW=d['RAW']


def markdown(frame):
    cols=frame.columns.tolist();rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for values in frame.itertuples(index=False,name=None):rows.append('| '+' | '.join('—' if pd.isna(x) else str(x) for x in values)+' |')
    return '\n'.join(rows)


def finish():
    cfg=json.loads((OUT/'config.json').read_text());assert cfg['stage']=='processing_complete' and (OUT/'exit_code.txt').read_text().strip()=='0'
    for name,h in cfg['source_sha256'].items():assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==h,name
    assert hashlib.sha256((OUT/'jobs.jsonl').read_bytes()).hexdigest()==cfg['jobs_sha256'] and hashlib.sha256((CACHE/'parcel_names.json').read_bytes()).hexdigest()==cfg['parcel_names_sha256']
    jobs=[json.loads(line) for line in (OUT/'jobs.jsonl').read_text().splitlines()];completed=pd.read_csv(OUT/'completed.csv',dtype={'participant_id':str,'DCAN_ID':str});assert len(completed)==len(jobs)==cfg['phenotype_rows'] and completed.DCAN_ID.is_unique and set(completed.DCAN_ID)=={j['DCAN_ID'] for j in jobs}
    assert not completed.collection.isin(d['EXCLUDED']).any();sessions=[json.loads(line) for line in (OUT/'sessions.jsonl').read_text().splitlines()];assert len(sessions)==len(completed)
    for s in sessions:assert s['selected']==(min(s['sessions']) if s['sessions'] else '')
    # 独立TSV解析核对自己的集合与ID，避免复用同一pandas对应逻辑。
    source={}
    for collection in sorted(completed.collection.unique()):
        with (RAW/'RawData'/collection/'participants.tsv').open(encoding='latin1',newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):source[collection,row['participant_id']]=row
    for r in completed.itertuples():
        row=source[r.collection,r.participant_id];assert r.DCAN_ID=='sub-'+r.participant_id
        if not pd.isna(r.dx_group):assert float(row['dx_group'])==r.dx_group and r.label==int(r.dx_group==1)
    included=completed.loc[completed.status.eq('included')].sort_values(['collection','DCAN_ID']).reset_index(drop=True);assert included.kept_seconds.ge(240).all() and included.parcels.eq(352).all()
    feature=np.lib.format.open_memmap(CACHE/'fisher_z.npy',mode='w+',dtype=np.float64,shape=(len(included),61776))
    for i,r in enumerate(included.itertuples()):
        values=np.load(r.feature_file,allow_pickle=False);assert values.shape==(61776,) and np.isfinite(values).all();feature[i]=values
    feature.flush();del feature
    order=included[['collection','participant_id','DCAN_ID','label','age','sex','TR','n_kept','kept_seconds','n_zero_parcels']];order.to_csv(CACHE/'row_order.csv',index=False);order.to_csv(OUT/'subjects.csv',index=False)
    feature=np.load(CACHE/'fisher_z.npy',mmap_mode='r');spot=[]
    for _,r in included.groupby('collection',sort=True).head(1).iterrows():
        root=RAW/r.selected_session.removeprefix(d['PREFIX'])/'func';series=sorted(root.glob('*'+d['TS']))[0];mask=sorted(root.glob('*'+d['MASK']))[0];x=d['censor'](np.asarray(nib.load(series).dataobj),d['motion'](mask));assert len(x)==r.n_kept
        zeros=int(np.sum(np.ptp(x,axis=0)==0));assert zeros==r.n_zero_parcels
        for a,b in [(0,1),(0,351),(350,351)]:
            u=x[:,a].astype(float);v=x[:,b].astype(float)
            if np.ptp(u)==0 or np.ptp(v)==0:rho=0.
            else:
                u-=u.mean();v-=v.mean();rho=float(u@v/np.sqrt((u@u)*(v@v)))
            expected=np.arctanh(np.clip(rho,-.999999,.999999));column=a*(2*352-a-1)//2+b-a-1;actual=feature[r.name,column];assert abs(actual-expected)<1e-10
            spot.append(dict(collection=r.collection,DCAN_ID=r.DCAN_ID,parcel_a=a,parcel_b=b,column=column,absolute_error=abs(actual-expected)))
    del feature
    pd.DataFrame(spot).to_csv(OUT/'feature_spotchecks.csv',index=False)
    administrative=pd.read_csv(OUT/'administrative_exclusions.csv');inventory=pd.read_csv(OUT/'collection_inventory.csv');phenotypes=pd.read_csv(OUT/'phenotype_subjects.csv',dtype={'participant_id':str});sites=[]
    for c in inventory.itertuples():
        f=completed.loc[completed.collection.eq(c.collection)];q=included.loc[included.collection.eq(c.collection)];ph=phenotypes.loc[phenotypes.collection.eq(c.collection)];admin=administrative.loc[administrative.collection.eq(c.collection)];asd=int(q.label.eq(1).sum());td=int(q.label.eq(0).sum());male=q.loc[q.sex.eq(1)]
        sites.append(dict(collection=c.collection,raw_subjects=c.raw_subject_directories,phenotype_subjects=len(ph),administrative_reason='' if admin.empty else admin.reason.iloc[0],ASD=asd,TD=td,male_ASD=int(male.label.eq(1).sum()),male_TD=int(male.label.eq(0).sum()),age_min=q.age.min(),age_max=q.age.max(),TR_seconds=';'.join(f'{v:g}' for v in sorted(f.TR.dropna().unique())),mean_kept_seconds=q.kept_seconds.mean(),multiple_session_subjects=int(f.n_sessions.gt(1).sum()),discarded_sessions=int((f.n_sessions-1).clip(lower=0).sum()),phenotype_ASD=int(ph.dx_group.eq(1).sum()),phenotype_TD=int(ph.dx_group.eq(2).sum()),natural_single_class=bool(len(ph)>0 and ph.dx_group.dropna().nunique()==1)))
    sites=pd.DataFrame(sites);sites.to_csv(OUT/'site_table.csv',index=False);sites.loc[sites.natural_single_class].to_csv(OUT/'natural_single_class_sites.csv',index=False)
    counts=completed.status.value_counts();excluded=[dict(reason=reason,n_subjects=int(administrative.loc[administrative.reason.eq(reason),'n_subjects'].sum())) for reason in ['Long_followup','no_public_diagnosis']]
    excluded.extend(dict(reason=reason,n_subjects=int(counts.get(reason,0))) for reason in ['missing_diagnosis','missing_timeseries','length_mismatch','duration_below_240s']);pd.DataFrame(excluded).to_csv(OUT/'exclusions.csv',index=False)
    included.groupby('n_zero_parcels').size().rename('n_subjects').reset_index().to_csv(OUT/'zero_parcel_distribution.csv',index=False)
    manifest=[]
    for path in sorted(RAW.rglob('*')):
        if not path.is_file():continue
        kind='phenotype' if path.name=='participants.tsv' else 'ptseries' if path.name.endswith(d['TS']) else 'motion_mask' if path.name.endswith(d['MASK']) else 'UNAUTHORIZED';assert kind!='UNAUTHORIZED',path
        key=d['PREFIX']+str(path.relative_to(RAW));manifest.append(dict(kind=kind,path=str(path),key=key,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest=pd.DataFrame(manifest);manifest.to_csv(OUT/'download_manifest.csv',index=False)
    events=[json.loads(line) for line in (OUT/'download_events.jsonl').read_text().splitlines()];assert sorted(x['key'] for x in events if x['status']=='started')==sorted(x['key'] for x in events if x['status']=='done')
    pd.DataFrame(columns=['key','reason']).to_csv(OUT/'failed_downloads.csv',index=False)
    payload=pd.read_csv(OUT/'payload_downloads.csv');assert payload.key.is_unique and set(payload.key)==set(manifest.key)
    multi=int(completed.n_sessions.gt(1).sum());discarded=int((completed.n_sessions-1).clip(lower=0).sum());timing=json.loads((OUT/'timing.json').read_text());seconds=cfg['inventory_seconds']+timing['pilot_seconds']+cfg['formal_seconds']+cfg['prior_failed_seconds']
    verification=dict(phenotype_rows=len(completed),included_subjects=len(included),ASD=int(included.label.eq(1).sum()),TD=int(included.label.eq(0).sum()),administrative_excluded=int(administrative.n_subjects.sum()),all_ID_labels_verified_from_own_TSV=True,all_first_sessions_verified=True,all_feature_rows_finite=True,feature_spotchecks=len(spot),max_spotcheck_error=float(pd.DataFrame(spot).absolute_error.max()),parcels=352,feature_dimensions=61776,multiple_session_subjects=multi,discarded_sessions=discarded,files=len(manifest),bytes=int(manifest.bytes.sum()),failed_downloads=0,processing_seconds=seconds,cumulative_file_IO_seconds=float(payload.loc[payload.new_download,'seconds'].sum()),python_isolated=bool(sys.flags.isolated),classification_started=False,AUC_computed=False)
    (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2));hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [CACHE/'fisher_z.npy',CACHE/'row_order.csv',CACHE/'parcel_names.json']};(OUT/'feature_sha256.json').write_text(json.dumps(hashes,indent=2))
    display=sites.loc[sites.administrative_reason.eq(''),['collection','ASD','TD','male_ASD','male_TD','age_min','age_max','TR_seconds','mean_kept_seconds']].copy();display['age_min']=display.age_min.round(2);display['age_max']=display.age_max.round(2);display['mean_kept_seconds']=display.mean_kept_seconds.round(1)
    text=f'''# T14-B 第一阶段完成：ABIDE-II数据准备

完成下载、质控与Gordon特征，未分类、未计算AUC、未启动第二阶段。

- 16个参与集合，表型{len(completed)}人；QC后{len(included)}人：ASD {verification['ASD']}、TD {verification['TD']}。
- 原始文件{len(manifest)}个，{verification['bytes']}字节（{verification['bytes']/2**20:.2f} MiB），失败0。总处理{seconds/60:.2f}分钟（含清单、下载、读取与特征；不包含低频监控等待），新文件传输/IO累计{verification['cumulative_file_IO_seconds']:.2f}秒，各线程有重叠，不能当墙钟。
- 三类行政排除：两个Long随访集合21+17=38人；ETHZ_1公开无诊断37人，与结果无关，不从其他来源补标签。行政计数来自RawData被试目录；参与集合人数来自各自participants.tsv。
- 多session {multi}人，共跳过{discarded}个后续session；仅取字典序首session，未下载后续session。这是丢弃重复session，不是排除这些被试。
- FD=.2 mm、frame_removal=0保留；保留时长=保留帧×CIFTI SeriesAxis TR，严格<240秒排除，不重复应用skip。所有读取下载文件的Python用-I，脚本在仓库。掩码均由固定MAT-v5函数读取；未遇v7.3，无需h5py。
- 352个parcel，61776维；直接调用abide_class_pure_sites.py的fisher_features，零方差相关置0。所有被试parcel名称/顺序与预检模板一致，不做标准化、特征选择、分类或性能评估。

## 站点表

'''
    text+=markdown(display)+'\n\n## 原因分列（被试互斥主原因）\n\n'+markdown(pd.DataFrame(excluded))+'\n\n## 原表型天然单类集合\n\n'+markdown(sites.loc[sites.natural_single_class,['collection','phenotype_ASD','phenotype_TD','ASD','TD']])+'\n\n## 零方差parcel分布（QC纳入者）\n\n'+markdown(pd.read_csv(OUT/'zero_parcel_distribution.csv'))+'\n'
    text+='\n排除主原因顺序：行政排除→缺诊断→缺首session时间序列→长度不一致→不足240秒；completed.csv保留缺诊断/缺时间序列标记以供核查。站点按集合不合并。所用五列表型值均为ASCII，Latin-1逐字节保留GU未用字段中的非UTF-8字节，不改变所用值。未下载dtseries/nii.gz、其他表型来源或额外原始类型；无上传，不读restricted。\n'
    text+='\n仅安装nibabel5.5.0，--no-deps；规定唯一测试1 passed。掩码/阈值、标签所属集合与Long/ETHZ排除测试先失败后通过；未运行已有分类/AUC测试。独立csv解析核对所有有效诊断，逐站点样本用中心化点积复算三条边，误差见verification.json；不做模型训练或AUC。\n'
    text+='\n特征位于'+str(CACHE/'fisher_z.npy')+'及row_order.csv，parcel_names.json记录有序列名。仓库无派生数组。成功合并核对后删除按人临时特征，completed.csv中的feature_file为当时中间文件位置；最终映射以subjects.csv和缓存row_order.csv为准。\n'
    (OUT/'REPORT.md').write_text(text)
    for path in (CACHE/'subject_features').glob('*.npy'):path.unlink()
    (CACHE/'subject_features').rmdir();cfg.update(stage='complete',included_subjects=len(included),classification_started=False,AUC_computed=False,report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('STAGE1_COMPLETE '+json.dumps(verification,ensure_ascii=False),flush=True)


if __name__=='__main__':
    assert sys.flags.isolated
    finish()
