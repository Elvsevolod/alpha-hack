import numpy as np
import pandas as pd
import pytest

from tools.mega_research_audit import aggregate_rmse, exact_group_facts


def test_conflict_bound_is_empirical_and_test_has_no_labels():
    train = pd.DataFrame({'id':[1,2,3,4], 'x':[1.,1.,2.,np.nan], 'class':['A','B','B','A']})
    test = pd.DataFrame({'id':[5,6,7], 'x':[1.,3.,np.nan]})
    facts = exact_group_facts(train,test,'class')
    assert facts['conflicting_groups']==1 and facts['rows_in_conflicting_groups']==2
    assert facts['feature_only_deterministic_training_accuracy_upper_bound']==.75
    assert facts['test_rows_with_train_feature_key']==2
    # IDs and class names do not define feature equality or change the bound.
    altered=train.assign(id=[20,30,40,50], **{'class':['X','Y','Y','X']})
    assert exact_group_facts(altered,test,'class')==facts


def test_different_rmse_aggregation_answers_different_questions():
    facts=aggregate_rmse([2.,4.],[1,3])
    assert facts['equal_period_mean_rmse']==3.
    assert facts['equal_period_root_mean_mse']==pytest.approx(np.sqrt(10))
    assert facts['pooled_row_rmse']==pytest.approx(np.sqrt(13))


@pytest.mark.parametrize('scores,counts',[([1],[0]),([1,2],[1]),([np.nan],[1]),([-1],[1])])
def test_invalid_aggregation_is_explicit(scores,counts):
    with pytest.raises(ValueError):aggregate_rmse(scores,counts)
