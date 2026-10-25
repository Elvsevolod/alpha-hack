import joblib
import numpy as np
import pandas as pd
import pytest

from automl_agent.group_context import GroupContext


def test_context_statistics_are_fit_only_and_unknown_groups_use_global_fallback(tmp_path):
    fit=pd.DataFrame({'category':['A','A','B','B'],'value':[1.,3.,10.,14.],
        'target':[100.,200.,300.,400.]})
    transform=GroupContext(['value'],['category'],min_support=2).fit(fit)
    query=pd.DataFrame({'category':['A','new'],'value':[4.,100.],'target':[-1.,-2.]})
    actual=transform.transform(query)
    assert actual.ctx_frequency__category.tolist()==[.5,0]
    assert transform.tables['category'][1].loc['V:A','value']==2
    assert transform.global_median['value']==6.5
    np.testing.assert_allclose(actual.ctx_center__category__value,
        [(4-2)/transform.global_scale['value'],(100-6.5)/transform.global_scale['value']])
    altered=transform.transform(query.assign(target=999))
    pd.testing.assert_frame_equal(actual.filter(regex='^ctx_'),altered.filter(regex='^ctx_'))
    joblib.dump(transform,tmp_path/'context.joblib')
    pd.testing.assert_frame_equal(joblib.load(tmp_path/'context.joblib').transform(query),actual)
    assert not any('target' in c for c in actual.filter(regex='^ctx_'))


def test_missing_category_is_distinct_from_text_and_rare_groups_are_bounded():
    fit=pd.DataFrame({'group':[None,None,'M:','M:','rare'],'number':[1,1,2,2,100]})
    t=GroupContext(['number'],['group'],min_support=2,max_groups=1).fit(fit)
    assert len(t.tables['group'][0])==1
    assert GroupContext.keys(fit.group).iloc[0]!=GroupContext.keys(fit.group).iloc[2]
    q=t.transform(pd.DataFrame({'group':['new',None],'number':[np.inf,2.]}))
    assert not np.isinf(q.filter(regex='^ctx_').to_numpy()).any()


def test_query_batch_does_not_change_context():
    fit=pd.DataFrame({'c':['A']*4,'x':[1.,2.,3.,4.]})
    t=GroupContext(['x'],['c'],min_support=2).fit(fit)
    point=pd.DataFrame({'c':['A'],'x':[3.]})
    batch=pd.concat([point,pd.DataFrame({'c':['A']*20,'x':[1e9]*20})],ignore_index=True)
    pd.testing.assert_series_equal(t.transform(point).iloc[0],t.transform(batch).iloc[0])


@pytest.mark.parametrize('support',[0,-1])
def test_invalid_support_rejected(support):
    with pytest.raises(ValueError):
        GroupContext(['x'],['c'],min_support=support).fit(pd.DataFrame({'c':['A'],'x':[1]}))
