# statsmodels.tsa.holtwinters.ExponentialSmoothing

Version: 0.14.0

statsmodels.tsa.holtwinters.ExponentialSmoothing(endog, trend=None, damped_trend=False, seasonal=None, *, seasonal_periods=None, initialization_method='estimated', initial_level=None, initial_trend=None, initial_seasonal=None, use_boxcox=False, bounds=None, dates=None, freq=None, missing='none')

Holt Winter's Exponential Smoothing

Parameters
----------
endog : array_like
    The time series to model.
trend : {"add", "mul", "additive", "multiplicative", None}, optional
    Type of trend component.
damped_trend : bool, optional
    Should the trend component be damped.
seasonal : {"add", "mul", "additive", "multiplicative", None}, optional
    Type of seasonal component.
seasonal_periods : int, optional
    The number of periods in a complete seasonal cycle, e.g., 4 for
    quarterly data or 7 for daily data with a weekly cycle.
initialization_method : str, optional
    Method for initialize the recursions. One of:

    * None
    * 'estimated'
    * 'heuristic'
    * 'legacy-heuristic'
    * 'known'

    None defaults to the pre-0.12 behavior where initial values
    are passed as part of ``fit``. If any of the other values are
    passed, then the initial values must also be set when constructing
    the model. If 'known' initialization is used, then `initial_level`
    must be passed, as well as `initial_trend` and `initial_seasonal` if
    applicable. Default is 'estimated'. "legacy-heuristic" uses the same
    values that were used in statsmodels 0.11 and earlier.
initial_level : float, optional
    The initial level component. Required if estimation method is "known".
    If set using either "estimated" or "heuristic" this value is used.
    This allows one or more of the initial values to be set while
    deferring to the heuristic for others or estimating the unset
    parameters.
initial_trend : float, optional
    The initial trend component. Required if estimation method is "known".
    If set using either "estimated" or "heuristic" this value is used.
    This allows one or more of the initial values to be set while
    deferring to the heuristic for others or estimating the unset
    parameters.
initial_seasonal : array_like, optional
    The initial seasonal component. An array of length `seasonal`
    or length `seasonal - 1` (in which case the last initial value
    is computed to make the average effect zero). Only used if
    initialization is 'known'. Required if estimation method is "known".
    If set using either "estimated" or "heuristic" this value is used.
    This allows one or more of the initial values to be set while
    deferring to the heuristic for others or estimating the unset
    parameters.
use_boxcox : {True, False, 'log', float}, optional
    Should the Box-Cox transform be applied to the data first? If 'log'
    then apply the log. If float then use the value as lambda.
bounds : dict[str, tuple[float, float]], optional
    An dictionary containing bounds for the parameters in the model,
    excluding the initial values if estimated. The keys of the dictionary
    are the variable names, e.g., smoothing_level or initial_slope.
    The initial seasonal variables are labeled initial_seasonal.<j>
    for j=0,...,m-1 where m is the number of period in a full season.
    Use None to indicate a non-binding constraint, e.g., (0, None)
    constrains a parameter to be non-negative.
dates : array_like of datetime, optional
    An array-like object of datetime objects. If a Pandas object is given
    for endog, it is assumed to have a DateIndex.
freq : str, optional
    The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
    'M', 'A', or 'Q'. This is optional if dates are given.
missing : str
    Available options are 'none', 'drop', and 'raise'. If 'none', no nan
    checking is done. If 'drop', any observations with nans are dropped.
    If 'raise', an error is raised. Default is 'none'.

Notes
-----
This is a full implementation of the holt winters exponential smoothing as
per [1]_. This includes all the unstable methods as well as the stable
methods. The implementation of the library covers the functionality of the
R library as much as possible whilst still being Pythonic.

See the notebook `Exponential Smoothing
<../examples/notebooks/generated/exponential_smoothing.html>`__
for an overview.

References
----------
.. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
    and practice. OTexts, 2014.

