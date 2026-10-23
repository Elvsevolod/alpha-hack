# feature_engine.selection.DropDuplicateFeatures

Version: 1.8.3

feature_engine.selection.DropDuplicateFeatures(variables: Union[NoneType, int, str, List[Union[str, int]]] = None, missing_values: str = 'ignore', confirm_variables: bool = False)

DropDuplicateFeatures() finds and removes duplicated features in a dataframe.

Duplicated features are identical features, regardless of the variable or column
name. If they show the same values for every observation, then they are considered
duplicated.

This transformer works with numerical and categorical variables. The user can
indicate a list of variables to examine. Alternatively, the transformer will
evaluate all the variables in the dataset.

The transformer will first identify and store the duplicated variables. Next, the
transformer will drop these variables from a dataframe.

More details in the :ref:`User Guide <drop_duplicate>`.

Parameters
----------
variables: list, default=None
    The list of variables to evaluate. If None, the transformer will evaluate all
    variables in the dataset.

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
    Set with the duplicated features that will be dropped.

duplicated_feature_sets_:
    Groups of duplicated features. Each list is a group of duplicated features.

variables_:
    The variables that will be considered for the feature selection procedure.

feature_names_in_:
    List with the names of features seen during `fit`.

n_features_in_:
    The number of features in the train set used in fit.

Methods
-------
fit:
    Find duplicated features.

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
    Remove duplicated features.

Examples
--------

>>> import pandas as pd
>>> from feature_engine.selection import DropDuplicateFeatures
>>> X = pd.DataFrame(dict(x1 = [1,1,1,1],
>>>                     x2 = [1,1,1,1],
>>>                     x3 = [True, False, False, False]))
>>> ddf = DropDuplicateFeatures()
>>> ddf.fit_transform(X)
    x1     x3
0   1   True
1   1  False
2   1  False
3   1  False

