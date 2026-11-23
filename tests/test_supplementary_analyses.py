def test_supplementary_evaluation_and_calibration_do_not_leak():
    import supplementary_analyses as t

    table = t.a.subjects()
    meta = table.set_index('FILE_ID')
    for j in t.anchors():
        train = set(j['train'])
        for label,groups in t.same_diagnosis(table,j).items():
            for site_role,ids in groups.items():
                assert set(ids).isdisjoint(train)
                assert meta.loc[ids,'label'].eq(label).all()
                assert set(meta.loc[ids,'site'])<=set(j[site_role])
        split = t.normative_plan(table,j)
        td,reference,test,reverse = (set(split[key]) for key in ('train_td','calibration','test','reverse'))
        assert td=={i for i in train if meta.at[i,'label']==0}
        assert test|reference==set(j['test']) and test.isdisjoint(reference)
        assert meta.loc[list(reference),'label'].eq(0).all()
        assert (test|reverse|reference).isdisjoint(train)
        assert reverse==set(j['reverse']) and reference.isdisjoint(reverse)
        for _,group in meta.loc[j['test']].groupby('site'):
            assert len(reference.intersection(group.index))==int(group.label.eq(0).sum())//2
        if j['b']==5:
            equal = t.equal_site_reference(table,j['repeat'],split['train_td'])
            assert set(equal)<=td and len(set(equal))==len(equal)
            counts = meta.loc[equal].groupby('site').size()
            assert set(counts.index)==set(meta.loc[list(td),'site']) and counts.eq(5).all()

    for repeat in range(50):
        designs = t.same_design(table,repeat)
        same = next(j for j in designs if j['method']=='P_same')
        mixed = next(j for j in designs if j['method']=='M_same')
        for j in designs:
            assert not set(meta.loc[j['test'],'site'])&{'LEUVEN','UCLA','UM'}
            assert set(j['train']).isdisjoint(j['test'])
        fitted = meta.loc[same['train']]
        assert fitted.SITE_ID.nunique()==6
        assert fitted.groupby('SITE_ID').label.nunique().eq(1).all()
        assert fitted.groupby('SITE_ID').size().eq(13).all()
        assert fitted.label.value_counts().to_dict()=={0:39,1:39}
        fitted = meta.loc[mixed['train']]
        counts = fitted.groupby(['SITE_ID','label']).size()
        for site in ['LEUVEN','UCLA','UM']:
            assert list(counts.loc[f'{site}_1'])==[7-repeat%2]*2
            assert list(counts.loc[f'{site}_2'])==[6+repeat%2]*2
