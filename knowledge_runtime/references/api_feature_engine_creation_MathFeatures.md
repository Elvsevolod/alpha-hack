# feature_engine.creation.MathFeatures

Version: 1.8.3

feature_engine.creation.MathFeatures(variables: List[Union[str, int]], func: Any, new_variables_names: Optional[List[str]] = None, missing_values: str = 'raise', drop_original: bool = False) -> None

MathFeatures(() applies functions across multiple features returning one or more
additional features as a result. It uses `pandas.agg()` to create the features,
setting `axis=1`.

For supported aggregation functions, see `pandas documentation
<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html>`_.

Note that if some of the variables have missing data and `missing_values='ignore'`,
the value will be ignored in the computation. To be clear, if variables A, B and C,
have values 10, 20 and NA, and we perform the sum, the result will be A + B = 30.

More details in the :ref:`User Guide <math_features>`.

Parameters
----------
variables: list
    The list of input variables. Variables must be numerical and there must be at
    least 2 different variables in the list.

func: function, string, list
    Functions to use for aggregating the data. Same functionality as parameter
    `func` in `pandas.agg()`. If a function, it must either work when passed a
    DataFrame or when passed to DataFrame.apply. Accepted combinations are:

    - function
    - string function name
    - list of functions and/or function names, e.g. [np.sum, 'mean']

    Each function will result in a new variable that will be added to the
    transformed dataset.

new_variables_names: list, default=None
    Names of the new variables. If passing a list with names (recommended), enter
    one name per function. If None, the transformer will assign arbitrary names,
    starting with the function and followed by the variables separated by _.

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
Although the transformer allows us to combine any features with any functions, we
recommend using it to create features based on domain knowledge. Typical examples
in finance are:

- Sum debt across financial products, i.e., credit cards, to obtain the total debt.
- Take the average payments to various financial products.
- Find the minimum payment done at any one month.

In insurance, we can sum the damage to various parts of a car to obtain the
total damage.

Examples
--------

>>> import pandas as pd
>>> from feature_engine.creation import MathFeatures
>>> X = pd.DataFrame(dict(x1 = [1,2,3], x2 = [4,5,6]))
>>> mf = MathFeatures(variables = ["x1","x2"], func = "sum")
>>> mf.fit(X)
>>> mf.transform(X)
   x1  x2  sum_x1_x2
0   1   4          5
1   2   5          7
2   3   6          9

>>> mf = MathFeatures(variables = ["x1","x2"], func = "prod")
>>> mf.fit(X)
>>> mf.transform(X)
   x1  x2  prod_x1_x2
0   1   4           4
1   2   5          10
2   3   6          18

>>> mf = MathFeatures(variables = ["x1","x2"], func = "mean")
>>> mf.fit(X)
>>> mf.transform(X))
   x1  x2  mean_x1_x2
0   1   4         2.5
1   2   5         3.5
2   3   6         4.5

