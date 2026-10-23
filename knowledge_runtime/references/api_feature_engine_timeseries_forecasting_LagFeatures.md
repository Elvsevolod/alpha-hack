# feature_engine.timeseries.forecasting.LagFeatures

Version: 1.8.3

feature_engine.timeseries.forecasting.LagFeatures(variables: Union[NoneType, int, str, List[Union[str, int]]] = None, periods: Union[int, List[int]] = 1, freq: Union[str, List[str], NoneType] = None, fill_value: collections.abc.Hashable = None, sort_index: bool = True, missing_values: str = 'raise', drop_original: bool = False, drop_na: bool = False) -> None

LagFeatures adds lag features to the dataframe. A lag feature is a feature with
information about a prior time step.

LagFeatures has the same functionality as pandas `shift()` with the exception that
only one of `periods` or `freq` can be indicated at a time. LagFeatures builds on
top of pandas `shift()` in that multiple lags can be created at the same time and
the features with names will be concatenated to the original dataframe.

To be compatible with LagFeatures, the dataframe's index must have unique values
and no NaN.

LagFeatures works only with numerical variables. You can pass a list of variables
to lag. Alternatively, LagFeatures will automatically select and lag all numerical
variables found in the training set.

More details in the :ref:`User Guide <lag_features>`.

Parameters
----------
variables: list, default=None
    The list of numerical variables to transform. If None, the transformer will
    automatically find and select all numerical variables.

periods: int, list of ints, default=1
    Number of periods to shift. Can be a positive integer or list of positive
    integers. If list, features will be created for each one of the periods in the
    list. If the parameter `freq` is specified, `periods` will be ignored.

freq: str, list of str, default=None
    Offset to use from the tseries module or time rule. See parameter `freq` in
    pandas `shift()`. It is the same functionality. If freq is a list, lag features
    will be created for each one of the frequency values in the list. If freq is not
    None, then this parameter overrides the parameter `periods`.

fill_value: object, optional
    The scalar value to use for newly introduced missing values. The default
    depends on the dtype of the variable. For numeric data, np.nan is used. For
    datetime, timedelta, or period data, NaT is used. For extension dtypes,
    self.dtype.na_value is used.

sort_index: bool, default=True
    Whether to order the index of the dataframe before creating the lag features.

missing_values: string, default='raise'
    Indicates if missing values should be ignored or raised. If `'raise'` the
    transformer will return an error if the the datasets to `fit` or `transform`
    contain missing values. If `'ignore'`, missing data will be ignored when
    learning parameters or performing the transformation.

drop_original: bool, default=False
    If True, the original variables to transform will be dropped from the dataframe.

drop_na: bool, default=False.
    Whether the NAN introduced in the lag features should be removed.

Attributes
----------
variables_:
    The group of variables that will be lagged.

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
    Add lag features.

transform_x_y:
    Remove rows with missing data from X and y.

See Also
--------
pandas.shift

Examples
--------

>>> import pandas as pd
>>> from feature_engine.timeseries.forecasting import LagFeatures
>>> X = pd.DataFrame(dict(date = ["2022-09-18",
>>>                               "2022-09-19",
>>>                               "2022-09-20",
>>>                               "2022-09-21",
>>>                               "2022-09-22"],
>>>                       x1 = [1,2,3,4,5],
>>>                       x2 = [6,7,8,9,10]
>>>                     ))
>>> lf = LagFeatures(periods=[1,2])
>>> lf.fit_transform(X)
            date  x1  x2  x1_lag_1  x2_lag_1  x1_lag_2  x2_lag_2
0  2022-09-18   1   6       NaN       NaN       NaN       NaN
1  2022-09-19   2   7       1.0       6.0       NaN       NaN
2  2022-09-20   3   8       2.0       7.0       1.0       6.0
3  2022-09-21   4   9       3.0       8.0       2.0       7.0
4  2022-09-22   5  10       4.0       9.0       3.0       8.0

