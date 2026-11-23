from pathlib import Path
import runpy
import numpy as np
import pandas as pd
from scipy.io import savemat


def test_collection_label_identity_censor_count_and_long_exclusion(tmp_path):
    d=runpy.run_path(str(Path(__file__).resolve().parents[1]/'abide2_prepare.py'))
    paths=[]
    for collection,diagnosis in [('ABIDEII-A',1),('ABIDEII-B',2),('ABIDEII-UCLA_Long',1),('ABIDEII-UPSM_Long',2),('ABIDEII-ETHZ_1',1)]:
        path=tmp_path/collection/'participants.tsv';path.parent.mkdir();pd.DataFrame({'site_id':[collection],'participant_id':['100'],'dx_group':[diagnosis],'age_at_scan ':[12.],'sex':[1]}).to_csv(path,sep='\t',index=False);paths.append(path)
    f=d['participants'](paths)
    assert f.loc[('ABIDEII-A','sub-100'),'dx_group']==1
    assert f.loc[('ABIDEII-B','sub-100'),'dx_group']==2
    assert not f.collection.isin(d['LONG']).any() and not f.collection.eq('ABIDEII-ETHZ_1').any() and len(f)==2
    removal=np.array([1,0,0,1,0],dtype=np.uint8);cells=np.empty((1,51),dtype=object)
    for i in range(51):cells[0,i]=dict(FD_threshold=i/100,frame_removal=removal,remaining_frame_count=3)
    path=tmp_path/'mask.mat';savemat(path,{'motion_data':cells});motion=d['motion'](path);x=np.arange(20).reshape(5,4);kept=d['censor'](x,motion)
    assert len(kept)==motion['remaining_frame_count']==np.sum(removal==0)
    np.testing.assert_array_equal(kept,x[[1,2,4]])
