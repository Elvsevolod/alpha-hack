# feature_engine.creation.RelativeFeatures

Version: 1.8.3

feature_engine.creation.RelativeFeatures(variables: List[Union[str, int]], reference: List[Union[str, int]], func: List[str], fill_value: Union[int, float, NoneType] = None, missing_values: str = 'ignore', drop_original: bool = False) -> None

RelativeFeatures() applies basic mathematical operations between a group
of variables and one or more reference features. It adds the resulting features
to the dataframe.

In other words, RelativeFeatures() adds, subtracts, multiplies, performs the
division, true division, floor division, module or exponentiation of a group of
features to / by a group of reference variables. The features resulting from these
functions are added to the dataframe.

This transformer works only with numerical variables. It uses the pandas methods
`pd.DataFrme.add`, `pd.DataFrme.sub`, `pd.DataFrme.mul`, `pd.DataFrme.div`,
`pd.DataFrme.truediv`, `pd.DataFrme.floordiv`, `pd.DataFrme.mod` and
`pd.DataFrme.pow`.
Find out more in `pandas documentation
<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.add.html>`_.

More details in the :ref:`User Guide <relative_features>`.

Parameters
----------
variables: list
    The list of numerical variables to combine with the reference variables.

reference: list
    The list of reference variables that will be added, subtracted, multiplied,
    used as denominator for division and module, or exponent for the exponentiation.

func: list
    The list of functions to be used in the transformation. The list can contain
    one or more of the following strings: 'add', 'mul','sub', 'div', truediv,
    'floordiv', 'mod', 'pow'.

fill_value: int, float, default=None
    When dividing by zero, this value is used in place of infinity. If None,
    then an error will be raised when dividing by zero.

missing_values: string, default='raise'
    Indicates if missing values should be ignored or raised. If `'raise'` the
    transformer will return an error if the the datasets to `fit` or `transform`
    contain missing values. If `'ignore'`, missing data will be ignored when
    learning parameters or performing the transformation.

drop_original: bool, default=False
    If True, the original variables to transform will be dropped from the dataframe.

Attributes
----------
variables_:
    The group of variables that will be transformed.

feature_names_in_:
    List with the names of features seen during `fit`.

n_features_in_:
    The number of features in the train set used in fit.

Methods
-------
fit:
    This transformer does not learn parameters.

fit_transform:
    Fit to data, then transform it.

get_feature_names_out:
    Get output feature names for transformation.

get_params:
    Get parameters for this estimator.

set_params:
    Set the parameters of this estimator.

transform:
    Create new features.

Notes
-----
Although the transformer allows us to combine any feature with any function, we
recommend its use to create domain knowledge variables. Typical examples within the
financial sector are:

- Ratio between income and debt to create the debt_to_income_ratio.
- Subtraction of rent from income to obtain the disposable_income.

Examples
--------

>>> import pandas as pd
>>> from feature_engine.creation import RelativeFeatures
>>> X = pd.DataFrame(dict(x1 = [1,2,3], x2 = [4,5,6], x3 = [3,4,5]))
>>> rf = RelativeFeatures(variables = ["x1","x2"],
>>>                     reference = ["x3"],
>>>                     func = ["div"])
>>> rf.fit(X)
>>> rf.transform(X)
   x1  x2  x3  x1_div_x3  x2_div_x3
0   1   4   3   0.333333   1.333333
1   2   5   4   0.500000   1.250000
2   3   6   5   0.600000   1.200000

