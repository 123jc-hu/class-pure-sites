"""T14-B第一阶段：集合内表型对应及DCAN固定0.2mm掩码。"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

RAW=Path.home()/'datasets_cache/ABIDE2_DCAN'
CACHE=Path.home()/'datasets_cache/ABIDE_derived/T14B'
OUT=Path(__file__).resolve().parent/'outputs/T14B/stage1'
LONG={'ABIDEII-UCLA_Long','ABIDEII-UPSM_Long'}
EXCLUDED=LONG|{'ABIDEII-ETHZ_1'}
COLUMNS=['site_id','participant_id','dx_group','age_at_scan ','sex']


def participants(paths):
    frames=[]
    for path in paths:
        collection=path.parent.name
        if collection in EXCLUDED:continue
        # 所用五列经预检均为ASCII；Latin-1逐字节保留其他列中的非UTF-8字节。
        f=pd.read_csv(path,sep='\t',encoding='latin1',usecols=COLUMNS,dtype={'participant_id':str});f['collection']=collection;f['DCAN_ID']='sub-'+f.participant_id;frames.append(f)
    return pd.concat(frames,ignore_index=True).set_index(['collection','DCAN_ID'],drop=False)


def motion(path):
    # 三人预检：MAT-v5，51个cell，第20项FD=.2；frame_removal=1是删帧。
    item=loadmat(path,simplify_cells=True)['motion_data'][20]
    assert np.isclose(item['FD_threshold'],.2,rtol=0,atol=1e-12)
    assert np.isin(item['frame_removal'],[0,1]).all()
    return item


def censor(x,item):return x[np.asarray(item['frame_removal'])==0]


REPO=Path(__file__).resolve().parent
BASE='https://fcp-indi.s3.amazonaws.com'
PREFIX='data/Projects/ABIDE2/'
TS='_task-rest_bold_atlas-Gordon2014FreeSurferSubcortical_desc-filtered_timeseries.ptseries.nii'
MASK='_task-rest_desc-filtered_motion_mask.mat'


def fetch(url,key):
    import subprocess,tempfile,json,time
    # curl不自行重试；每次使用独立文件，避免失败响应与重试内容拼接。
    with tempfile.TemporaryDirectory(prefix='T14B_http_') as folder:
        path=Path(folder)/'response'
        for attempt in range(1,4):
            result=subprocess.run(['curl','--disable','--retry','0','--fail','--silent','--show-error','--connect-timeout','30','--max-time','120','--output',str(path),url],stderr=subprocess.PIPE)
            if result.returncode==0:return path.read_bytes()
            error=result.stderr.decode('utf-8',errors='replace').strip();delay=2**attempt if attempt<3 else 0
            event=dict(request=key,url=url,attempt=attempt,error=error,exit_code=result.returncode,retry=attempt<3,wait_seconds=delay)
            with (OUT/'network_retry.log').open('a') as f:f.write(json.dumps(event)+'\n')
            if attempt==3:result.check_returncode()
            time.sleep(delay)


def listing(prefix):
    from urllib.parse import urlencode
    import xml.etree.ElementTree as ET
    params={'list-type':'2','prefix':prefix,'delimiter':'/'};dirs=[];objects=[];ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
    while True:
        root=ET.fromstring(fetch(BASE+'/?'+urlencode(params),'LIST '+prefix))
        dirs.extend(x.find('s:Prefix',ns).text for x in root.findall('s:CommonPrefixes',ns));objects.extend(dict(key=x.find('s:Key',ns).text,size=int(x.find('s:Size',ns).text)) for x in root.findall('s:Contents',ns))
        if root.find('s:IsTruncated',ns).text=='false':break
        params['continuation-token']=root.find('s:NextContinuationToken',ns).text
    return dirs,objects


def download(item):
    import json,hashlib
    from urllib.parse import quote
    from time import perf_counter
    key=item['key'];path=RAW/key.removeprefix(PREFIX);assert path.resolve().is_relative_to(RAW.resolve());tick=perf_counter();new=not path.exists();url=BASE+'/'+quote(key,safe='/')
    if new:
        path.parent.mkdir(parents=True,exist_ok=True)
        with (OUT/'download_events.jsonl').open('a') as f:f.write(json.dumps(dict(status='started',key=key,url=url))+'\n')
        data=fetch(url,key)
        assert len(data)==item['size'],key
        path.write_bytes(data)
        with (OUT/'download_events.jsonl').open('a') as f:f.write(json.dumps(dict(status='done',key=key,bytes=len(data)))+'\n')
    assert path.stat().st_size==item['size'],path
    return dict(key=key,path=str(path),url=url,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),new_download=new,seconds=perf_counter()-tick)


def append(name,frame):
    with (OUT/name).open('a') as f:frame.to_csv(f,index=False,header=f.tell()==0)


def prepare():
    import json,hashlib,sys,importlib.metadata
    from datetime import datetime
    from time import perf_counter
    tick=perf_counter();print('START T14B stage1 inventory',flush=True);roots,_=listing(PREFIX+'RawData/');roots=[x for x in roots if not x.endswith('/sidecards/')];assert len(roots)==19
    administrative=[];paths=[];manifests=[];collections=[]
    for prefix in roots:
        collection=prefix.rstrip('/').rsplit('/',1)[1];dirs,objects=listing(prefix);n_raw=sum(x.rstrip('/').rsplit('/',1)[1].startswith('sub-') for x in dirs);collections.append(dict(collection=collection,raw_subject_directories=n_raw,administrative_exclusion=collection in EXCLUDED))
        if collection in EXCLUDED:
            administrative.append(dict(collection=collection,reason='Long_followup' if collection in LONG else 'no_public_diagnosis',n_subjects=n_raw));continue
        target=[x for x in objects if x['key']==prefix+'participants.tsv'];assert len(target)==1,collection;info=download(target[0]);paths.append(Path(info['path']));manifests.append(info)
    pd.DataFrame(collections).to_csv(OUT/'collection_inventory.csv',index=False);pd.DataFrame(administrative).to_csv(OUT/'administrative_exclusions.csv',index=False);assert next(x for x in administrative if x['collection']=='ABIDEII-ETHZ_1')['n_subjects']==37
    table=participants(paths);assert table.index.is_unique and table.DCAN_ID.is_unique and table.dx_group.dropna().isin([1,2]).all()
    assert all(str(x).isascii() for x in table[COLUMNS].fillna('').to_numpy().ravel())
    table=table.sort_index();table.to_csv(OUT/'phenotype_subjects.csv',index=False);append('payload_downloads.csv',pd.DataFrame(manifests));dirs,_=listing(PREFIX+'Derivatives/DCAN/');known={x.rstrip('/').rsplit('/',1)[1] for x in dirs}
    work=table.reset_index(drop=True).to_dict('records')
    for row in work:row['has_DCAN_directory']=row['DCAN_ID'] in known
    (OUT/'jobs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in work))
    import nibabel as nib
    samples=json.loads((OUT/'sample_subjects.json').read_text());names=nib.load(samples[0]['ptseries']).header.get_axis(1).name.tolist();CACHE.mkdir(parents=True,exist_ok=True);(CACHE/'subject_features').mkdir(exist_ok=True);(CACHE/'parcel_names.json').write_text(json.dumps(names));assert len(names)==352
    old=json.loads((OUT/'config.json').read_text());(OUT/'pre_ETHZ_resolution_config.json').write_text(json.dumps(old,ensure_ascii=False,indent=2))
    cfg=dict(task='T14B_stage1',stage='prepared',started_at=datetime.now().astimezone().isoformat(),inventory_seconds=perf_counter()-tick,python_isolated=bool(sys.flags.isolated),raw=str(RAW),cache=str(CACHE),workers=4,download_concurrency_limit=4,CPU_only=True,seeds='not applicable;no stochastic analysis',FD_threshold=.2,duration_threshold_seconds=240,parcels=352,feature_dimensions=61776,diagnosis='each collection own participants.tsv;sub- prefix only;dx1 ASD dx2 TD;missing dx excluded',encoding='latin1 byte-preserving;all required values verified ASCII',session='lexicographic first;others not downloaded',excluded=sorted(EXCLUDED),ETHZ_reason='no public diagnosis,37 raw subject directories;no alternative source',classification_started=False,AUC_computed=False,phenotype_rows=len(table),DCAN_directories=len(known),new_dependency={'nibabel':importlib.metadata.version('nibabel')},h5py_installed=False,source_sha256={name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in ['abide2_prepare.py','abide_class_pure_sites.py','tests/test_abide2_prepare.py','AGENTS.md','docs/tasks/T14B_abide2_replication.md']},jobs_sha256=hashlib.sha256((OUT/'jobs.jsonl').read_bytes()).hexdigest(),parcel_names_sha256=hashlib.sha256((CACHE/'parcel_names.json').read_bytes()).hexdigest())
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('INVENTORY_COMPLETE '+json.dumps(dict(subjects=len(table),DCAN_present=sum(row['has_DCAN_directory'] for row in work),administrative=administrative)),flush=True)


def process(row,names):
    import nibabel as nib
    import runpy,json
    from time import perf_counter
    tick=perf_counter();sid=row['DCAN_ID'];files=[];sessions=[];selected='';ts=[];masks=[]
    if row['has_DCAN_directory']:
        dirs,_=listing(PREFIX+'Derivatives/DCAN/'+sid+'/');sessions=sorted(x for x in dirs if x.rstrip('/').rsplit('/',1)[1].startswith('ses-'))
        if sessions:
            selected=sessions[0];_,objects=listing(selected+'func/');ts=[x for x in objects if x['key'].endswith(TS)];masks=[x for x in objects if x['key'].endswith(MASK)]
    info=dict(collection=row['collection'],participant_id=row['participant_id'],DCAN_ID=sid,dx_group=row['dx_group'],label=np.nan if pd.isna(row['dx_group']) else int(row['dx_group']==1),age=row['age_at_scan '],sex=row['sex'],n_sessions=len(sessions),selected_session=selected,missing_dx=pd.isna(row['dx_group']),missing_timeseries=len(ts)==0,status='',TR=np.nan,n_frames=np.nan,mask_length=np.nan,n_kept=np.nan,kept_seconds=np.nan,parcels=np.nan,n_zero_parcels=np.nan,feature_file='',seconds=0.)
    if info['missing_dx']:info['status']='missing_diagnosis'
    elif not ts:info['status']='missing_timeseries'
    else:
        assert len(ts)==len(masks)==1,sid
        assert ts[0]['key'].rsplit('/',1)[1].startswith(sid+'_'),sid
        files=[download(item) for item in ts+masks];image=nib.load(files[0]['path']);assert image.header.get_axis(0).unit=='SECOND';assert np.array_equal(image.header.get_axis(1).name,names),sid
        x=np.asarray(image.dataobj);item=motion(files[1]['path']);tr=float(image.header.get_axis(0).step);assert np.isclose(tr,item['epi_TR'],rtol=1e-6,atol=1e-8),sid
        removal=np.asarray(item['frame_removal']);info.update(TR=tr,n_frames=len(x),mask_length=len(removal),parcels=x.shape[1])
        if len(removal)!=len(x):info['status']='length_mismatch'
        else:
            kept=censor(x,item);assert len(kept)==item['remaining_frame_count'],sid
            info.update(n_kept=len(kept),kept_seconds=len(kept)*tr)
            if len(kept)*tr<240:info['status']='duration_below_240s'
            else:
                fisher=runpy.run_path(str(REPO/'abide_class_pure_sites.py'))['fisher_features'];values,n_zero=fisher(kept);assert len(values)==61776 and np.isfinite(values).all(),sid
                path=CACHE/'subject_features'/(sid+'.npy');np.save(path,values);info.update(status='included',n_zero_parcels=n_zero,feature_file=str(path))
    info['seconds']=perf_counter()-tick
    return info,files,dict(collection=row['collection'],DCAN_ID=sid,sessions=sessions,selected=selected)


def parallel(work):
    import json
    from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
    names=json.loads((CACHE/'parcel_names.json').read_text());remaining=iter(work[4:])
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(process,row,names) for row in work[:4]}
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future);info,files,session=future.result();append('completed.csv',pd.DataFrame([info]))
                if files:append('payload_downloads.csv',pd.DataFrame(files))
                with (OUT/'sessions.jsonl').open('a') as f:f.write(json.dumps(session)+'\n')
                print('SUBJECT_DONE '+json.dumps({k:info[k] for k in ['collection','DCAN_ID','status','seconds']}),flush=True)
                row=next(remaining,None)
                if row is not None:pending.add(pool.submit(process,row,names))


def timing():
    import json
    from time import perf_counter
    prepare();work=[json.loads(line) for line in (OUT/'jobs.jsonl').read_text().splitlines()];sample_ids={x['DCAN_ID'] for x in json.loads((OUT/'sample_subjects.json').read_text())};pilot=[]
    for collection in sorted({row['collection'] for row in work}):pilot.extend([row for row in work if row['collection']==collection and row['has_DCAN_directory'] and not pd.isna(row['dx_group']) and row['DCAN_ID'] not in sample_ids][:2])
    print('START T14B timing subjects='+str(len(pilot)),flush=True);tick=perf_counter();parallel(pilot);wall=perf_counter()-tick;done=pd.read_csv(OUT/'completed.csv');active=sum(row['has_DCAN_directory'] for row in work);point=wall+done.seconds.mean()*max(0,active-len(pilot))/4+60
    estimate=dict(point_minutes=point/60,with_25percent_margin_minutes=point*1.25/60,pilot_seconds=wall,pilot_subjects=len(pilot),remaining_subjects=len(work)-len(pilot),pilot_reused=True,formula='mean actual first-two-per-site new-subject seconds*remaining DCAN subjects/4+pilot wall+60s;missing DCAN subjects cheap')
    (OUT/'timing.json').write_text(json.dumps(estimate,indent=2));cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='timed');(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('TIMING_COMPLETE '+json.dumps(estimate),flush=True)


def run():
    import json
    from datetime import datetime
    from time import perf_counter
    cfg=json.loads((OUT/'config.json').read_text());cfg.update(stage='running',formal_started_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));done=set(pd.read_csv(OUT/'completed.csv',dtype={'DCAN_ID':str}).DCAN_ID);work=[json.loads(line) for line in (OUT/'jobs.jsonl').read_text().splitlines()];work=[row for row in work if row['DCAN_ID'] not in done];print('START T14B remaining_subjects='+str(len(work)),flush=True);tick=perf_counter();parallel(work);cfg.update(stage='processing_complete',formal_seconds=perf_counter()-tick,formal_finished_at=datetime.now().astimezone().isoformat());(OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2));print('ALL_PROCESSING_COMPLETE',flush=True)


if __name__=='__main__':
    import sys
    assert sys.flags.isolated
    {'timing':timing,'run':run}[sys.argv[1]]()
