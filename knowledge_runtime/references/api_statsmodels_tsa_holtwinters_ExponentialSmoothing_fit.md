# statsmodels.tsa.holtwinters.ExponentialSmoothing.fit
Version: 0.14.0

## statsmodels.tsa.holtwinters.ExponentialSmoothing.fit(self, smoothing_level=None, smoothing_trend=None, smoothing_seasonal=None, damping_trend=None, *, optimized=True, remove_bias=False, start_params=None, method=None, minimize_kwargs=None, use_brute=True, use_boxcox=None, use_basinhopping=None, initial_level=None, initial_trend=None)

Fit the model

Parameters
----------
smoothing_level : float, optional
    The alpha value of the simple exponential smoothing, if the value
    is set then this value will be used as the value.
smoothing_trend :  float, optional
    The beta value of the Holt's trend method, if the value is
    set then this value will be used as the value.
smoothing_seasonal : float, optional
    The gamma value of the holt winters seasonal method, if the value
    is set then this value will be used as the value.
damping_trend : float, optional
    The phi value of the damped method, if the value is
    set then this value will be used as the value.
optimized : bool, optional
    Estimate model parameters by maximizing the log-likelihood.
remove_bias : bool, optional
    Remove bias from forecast values and fitted values by enforcing
    that the average residual is equal to zero.
start_params : array_like, optional
    Starting values to used when optimizing the fit.  If not provided,
    starting values are determined using a combination of grid search
    and reasonable values based on the initial values of the data. See
    the notes for the structure of the model parameters.
method : str, default "L-BFGS-B"
    The minimizer used. Valid options are "L-BFGS-B" , "TNC",
    "SLSQP" (default), "Powell", "trust-constr", "basinhopping" (also
    "bh") and "least_squares" (also "ls"). basinhopping tries multiple
    starting values in an attempt to find a global minimizer in
    non-convex problems, and so is slower than the others.
minimize_kwargs : dict[str, Any]
    A dictionary of keyword arguments passed to SciPy's minimize
    function if method is one of "L-BFGS-B", "TNC",
    "SLSQP", "Powell", or "trust-constr", or SciPy's basinhopping
    or least_squares functions. The valid keywords are optimizer
    specific. Consult SciPy's documentation for the full set of
    options.
use_brute : bool, optional
    Search for good starting values using a brute force (grid)
    optimizer. If False, a naive set of starting values is used.
use_boxcox : {True, False, 'log', float}, optional
    Should the Box-Cox transform be applied to the data first? If 'log'
    then apply the log. If float then use the value as lambda.

    .. deprecated:: 0.12

       Set use_boxcox when constructing the model

use_basinhopping : bool, optional
    Deprecated. Using Basin Hopping optimizer to find optimal values.
    Use ``method`` instead.

    .. deprecated:: 0.12

       Use ``method`` instead.

initial_level : float, optional
    Value to use when initializing the fitted level.

    .. deprecated:: 0.12

       Set initial_level when constructing the model

initial_trend : float, optional
    Value to use when initializing the fitted trend.

    .. deprecated:: 0.12

       Set initial_trend when constructing the model
       or set initialization_method.

Returns
-------
HoltWintersResults
    See statsmodels.tsa.holtwinters.HoltWintersResults.

Notes
-----
This is a full implementation of the holt winters exponential smoothing
as per [1]. This includes all the unstable methods as well as the
stable methods. The implementation of the library covers the
functionality of the R library as much as possible whilst still
being Pythonic.

The parameters are ordered

[alpha, beta, gamma, initial_level, initial_trend, phi]

which are then followed by m seasonal values if the model contains
a seasonal smoother. Any parameter not relevant for the model is
omitted. For example, a model that has a level and a seasonal
component, but no trend and is not damped, would have starting
values

[alpha, gamma, initial_level, s0, s1, ..., s<m-1>]

where sj is the initial value for seasonal component j.

References
----------
[1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
    and practice. OTexts, 2014.
