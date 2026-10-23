# feature_engine.selection.DropCorrelatedFeatures

Version: 1.8.3

feature_engine.selection.DropCorrelatedFeatures(variables: Union[NoneType, int, str, List[Union[str, int]]] = None, method: str = 'pearson', threshold: float = 0.8, missing_values: str = 'ignore', confirm_variables: bool = False)

DropCorrelatedFeatures() finds and removes correlated features. Correlation is
calculated with `pandas.corr()`. Features are removed on first found, first removed
basis, without any further insight.

DropCorrelatedFeatures() works only with numerical variables. Categorical variables
will need to be encoded to numerical or will be excluded from the analysis.

To make the selector deterministic, features are sorted alphabetically before
examining correlation.

More details in the :ref:`User Guide <drop_correlated>`.

Parameters
----------
variables: str or list, default=None
    The list of variables to evaluate. If None, the transformer will evaluate
    all numerical features in the dataset.

method: string or callable, default='pearson'
    Can take 'pearson', 'spearman', 'kendall' or callable. It refers to the
    correlation method to be used to identify the correlated features.

    - 'pearson': standard correlation coefficient
    - 'kendall': Kendall Tau correlation coefficient
    - 'spearman': Spearman rank correlation
    - callable: callable with input two 1d ndarrays and returning a float.

    For more details on this parameter visit the  `pandas.corr()` documentation.

threshold: float, default=0.8
    The correlation threshold above which a feature will be deemed correlated with
    another one and removed from the dataset.

missing_values: str, default=ignore
    Whether the missing values should be raised as error or ignored when
    determining correlation. Takes values 'raise' and 'ignore'.

confirm_variables: bool, default=False
        If set to True, variables that are not present in the input dataframe will
        be removed from the list of variables. Only used when passing a variable
        list to the parameter `variables`. See parameter variables for more details.

Attributes
----------
features_to_drop_:
    Set with the correlated features that will be dropped.

correlated_feature_sets_:
    Groups of correlated features. Each list is a group of correlated features.

correlated_feature_dict_: dict
    Dictionary containing the correlated feature groups. The key is the feature
    against which all other features were evaluated. The values are the features
    correlated with the key. Key + values should be the same as the set found in
    `correlated_feature_groups`. We introduced this attribute in version 1.17.0
    because from the set, it is not easy to see which feature will be retained and
    which ones will be removed. The key is retained, the values will be dropped.

variables_:
    The variables that will be considered for the feature selection procedure.

feature_names_in_:
    List with the names of features seen during `fit`.

n_features_in_:
    The number of features in the train set used in fit.

Methods
-------
fit:
    Find correlated features.

fit_transform:
    Fit to data, then transform it.

get_feature_names_out:
    Get output feature names for transformation.

get_params:
    Get parameters for this estimator.

set_params:
    Set the parameters of this estimator.

get_support:
    Get a mask, or integer index, of the features selected.

transform:
    Remove correlated features.

Notes
-----
If you want to select from each group of correlated features those that are perhaps
more predictive or more complete, check Feature-engine's SmartCorrelationSelection.

See Also
--------
pandas.corr
feature_engine.selection.SmartCorrelationSelection

Examples
--------

>>> import pandas as pd
>>> from feature_engine.selection import DropCorrelatedFeatures
>>> X = pd.DataFrame(dict(x1 = [1,2,1,1], x2 = [2,4,3,1], x3 = [1, 0, 0, 1]))
>>> dcf = DropCorrelatedFeatures(threshold=0.7)
>>> dcf.fit_transform(X)
    x1  x3
0   1   1
1   2   0
2   1   0
3   1   1

