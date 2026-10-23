# feature_engine.datetime.DatetimeFeatures

Version: 1.8.3

feature_engine.datetime.DatetimeFeatures(variables: Union[NoneType, int, str, List[Union[str, int]]] = None, features_to_extract: Union[NoneType, str, List[str]] = None, drop_original: bool = True, missing_values: str = 'raise', dayfirst: bool = False, yearfirst: bool = False, utc: Optional[bool] = None, format: Optional[str] = None) -> None

DatetimeFeatures extracts date and time features from datetime variables, adding
new columns to the dataset. DatetimeFeatures can extract datetime information from
existing datetime or object-like variables or from the dataframe index.

DatetimeFeatures uses `pandas.to_datetime` to convert object variables to datetime
and pandas.dt to extract the features from datetime.

The transformer supports the extraction of the following features:

- "month"
- "quarter"
- "semester"
- "year"
- "week"
- "day_of_week"
- "day_of_month"
- "day_of_year"
- "weekend"
- "month_start"
- "month_end"
- "quarter_start"
- "quarter_end"
- "year_start"
- "year_end"
- "leap_year"
- "days_in_month"
- "hour"
- "minute"
- "second"

More details in the :ref:`User Guide <datetime_features>`.

Parameters
----------
variables: str, list, default=None
    List with the variables from which date and time information will be extracted.
    If None, the transformer will find and select all datetime variables,
    including variables of type object that can be converted to datetime.
    If "index", the transformer will extract datetime features from the
    index of the dataframe.

features_to_extract: list, default=None
    The list of date features to extract. If None, the following features will be
    extracted: "month", "year", "day_of_week", "day_of_month", "hour",
    "minute" and "second". If "all", all supported features will be extracted.
    Alternatively, you can pass a list with the names of the features you want to
    extract.

drop_original: bool, default="True"
    If True, the original datetime variables will be dropped from the dataframe.

missing_values: string, default='raise'
    Indicates if missing values should be ignored or raised. If 'raise' the
    transformer will return an error if the the datasets to `fit` or `transform`
    contain missing values. If 'ignore', missing data will be ignored when
    performing the feature extraction. Missing data is only evaluated in the
    variables that will be used to derive the date and time features. If features
    are derived from the dataframe index, missing data will be checked in the
    index.

dayfirst: bool, default="False"
    Specify a date parse order if arg is str or is list-like. If True, parses
    dates with the day first, e.g. 10/11/12 is parsed as 2012-11-10. Same as in
    `pandas.to_datetime`.

yearfirst: bool, default="False"
    Specify a date parse order if arg is str or is list-like.
    Same as in `pandas.to_datetime`.

    - If True parses dates with the year first, e.g. 10/11/12 is parsed as
      2010-11-12.
    - If both dayfirst and yearfirst are True, yearfirst is preceded.

utc: bool, default=None
    Return UTC DatetimeIndex if True (converting any tz-aware datetime.datetime
    objects as well). Same as in `pandas.to_datetime`.

format: str, default None
    The strftime to parse time, e.g. "%d/%m/%Y". Check pandas `to_datetime()` for
    more information on choices. If you have variables with different formats pass
    “mixed”, to infer the format for each element individually. This is risky,
    and you should probably use it along with dayfirst, according to pandas'
    documentation.


Attributes
----------
variables_:
    List of variables from which date and time features will be extracted. If None,
    features will be extracted from the dataframe index.

features_to_extract_:
    The date and time features that will be extracted from each variable or the
    index.

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
    Add the date and time features.

See also
--------
pandas.to_datetime
pandas.dt

Examples
--------

>>> import pandas as pd
>>> from feature_engine.datetime import DatetimeFeatures
>>> X = pd.DataFrame(dict(date = ["2022-09-18", "2022-10-27", "2022-12-24"]))
>>> dtf = DatetimeFeatures(features_to_extract = ["year", "month", "day_of_month"])
>>> dtf.fit(X)
>>> dtf.transform(X)
    date_year  date_month  date_day_of_month
0       2022           9                 18
1       2022          10                 27
2       2022          12                 24

