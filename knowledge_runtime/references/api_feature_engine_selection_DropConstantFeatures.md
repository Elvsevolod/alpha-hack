# feature_engine.selection.DropConstantFeatures

Version: 1.8.3

feature_engine.selection.DropConstantFeatures(variables: Union[NoneType, int, str, List[Union[str, int]]] = None, tol: float = 1, missing_values: str = 'raise', confirm_variables: bool = False)

DropConstantFeatures() drops constant and quasi-constant variables from a dataframe.
Constant variables show the same value in all the observations in the dataset.
Quasi-constant variables show the same value in almost all the observations in the
dataset.

This transformer works with numerical and categorical variables. The user can
indicate a list of variables to examine. Alternatively, the transformer will
evaluate all the variables in the dataset.

The transformer will first identify and store the constant and quasi-constant
variables. Next, the transformer will drop these variables from a dataframe.

More details in the :ref:`User Guide <drop_constant>`.

Parameters
----------
variables: list, default=None
    The list of variables to evaluate. If None, the transformer will evaluate all
    variables in the dataset.

tol: float,int,  default=1
    Threshold to detect constant/quasi-constant features. Variables showing the
    same value in a percentage of observations greater than tol will be considered
    constant / quasi-constant and dropped. If tol=1, the transformer removes
    constant variables. Else, it will remove quasi-constant variables. For example,
    if tol=0.98, the transformer will remove variables that show the same value in
    98% of the observations.

missing_values: str, default=raises
    Whether the missing values should be raised as error, ignored or included as an
    additional value of the variable. Takes values 'raise', 'ignore', 'include'.

confirm_variables: bool, default=False
        If set to True, variables that are not present in the input dataframe will
        be removed from the list of variables. Only used when passing a variable
        list to the parameter `variables`. See parameter variables for more details.

Attributes
----------
features_to_drop_:
    List with constant and quasi-constant features.

variables_:
    The variables that will be considered for the feature selection procedure.:

feature_names_in_:
    List with the names of features seen during `fit`.

n_features_in_:
    The number of features in the train set used in fit.

Methods
-------
fit:
    Find constant and quasi-constant features.

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
    Remove constant and quasi-constant features.

Notes
-----
This transformer is a similar concept to the VarianceThreshold from Scikit-learn,
but it evaluates number of unique values instead of variance.

See Also
--------
sklearn.feature_selection.VarianceThreshold

Examples
--------

>>> import pandas as pd
>>> from feature_engine.selection import DropConstantFeatures
>>> X = pd.DataFrame(dict(x1 = [1,1,1,1],
>>>                     x2 = ["a", "a", "b", "c"],
>>>                     x3 = [True, False, False, True]))
>>> dcf = DropConstantFeatures()
>>> dcf.fit_transform(X)
    x2     x3
0  a   True
1  a  False
2  b  False
3  c   True

Additionally, you can set the Threshold for quasi-constant features:

>>> X = pd.DataFrame(dict(x1 = [1,1,1,1],
>>>                      x2 = ["a", "a", "b", "c"],
>>>                      x3 = [True, False, False, False]))
>>> dcf = DropConstantFeatures(tol = 0.75)
>>> dcf.fit_transform(X)
    x2
0  a
1  a
2  b
3  c

