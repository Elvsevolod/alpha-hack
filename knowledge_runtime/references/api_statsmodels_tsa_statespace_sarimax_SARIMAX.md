# statsmodels.tsa.statespace.sarimax.SARIMAX

Version: 0.14.0

statsmodels.tsa.statespace.sarimax.SARIMAX(endog, exog=None, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None, measurement_error=False, time_varying_regression=False, mle_regression=True, simple_differencing=False, enforce_stationarity=True, enforce_invertibility=True, hamilton_representation=False, concentrate_scale=False, trend_offset=1, use_exact_diffuse=False, dates=None, freq=None, missing='none', validate_specification=True, **kwargs)

Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors
model

Parameters
----------
endog : array_like
    The observed time-series process :math:`y`
exog : array_like, optional
    Array of exogenous regressors, shaped nobs x k.
order : iterable or iterable of iterables, optional
    The (p,d,q) order of the model for the number of AR parameters,
    differences, and MA parameters. `d` must be an integer
    indicating the integration order of the process, while
    `p` and `q` may either be an integers indicating the AR and MA
    orders (so that all lags up to those orders are included) or else
    iterables giving specific AR and / or MA lags to include. Default is
    an AR(1) model: (1,0,0).
seasonal_order : iterable, optional
    The (P,D,Q,s) order of the seasonal component of the model for the
    AR parameters, differences, MA parameters, and periodicity.
    `D` must be an integer indicating the integration order of the process,
    while `P` and `Q` may either be an integers indicating the AR and MA
    orders (so that all lags up to those orders are included) or else
    iterables giving specific AR and / or MA lags to include. `s` is an
    integer giving the periodicity (number of periods in season), often it
    is 4 for quarterly data or 12 for monthly data. Default is no seasonal
    effect.
trend : str{'n','c','t','ct'} or iterable, optional
    Parameter controlling the deterministic trend polynomial :math:`A(t)`.
    Can be specified as a string where 'c' indicates a constant (i.e. a
    degree zero component of the trend polynomial), 't' indicates a
    linear trend with time, and 'ct' is both. Can also be specified as an
    iterable defining the non-zero polynomial exponents to include, in
    increasing order. For example, `[1,1,0,1]` denotes
    :math:`a + bt + ct^3`. Default is to not include a trend component.
measurement_error : bool, optional
    Whether or not to assume the endogenous observations `endog` were
    measured with error. Default is False.
time_varying_regression : bool, optional
    Used when an explanatory variables, `exog`, are provided
    to select whether or not coefficients on the exogenous regressors are
    allowed to vary over time. Default is False.
mle_regression : bool, optional
    Whether or not to use estimate the regression coefficients for the
    exogenous variables as part of maximum likelihood estimation or through
    the Kalman filter (i.e. recursive least squares). If
    `time_varying_regression` is True, this must be set to False. Default
    is True.
simple_differencing : bool, optional
    Whether or not to use partially conditional maximum likelihood
    estimation. If True, differencing is performed prior to estimation,
    which discards the first :math:`s D + d` initial rows but results in a
    smaller state-space formulation. See the Notes section for important
    details about interpreting results when this option is used. If False,
    the full SARIMAX model is put in state-space form so that all
    datapoints can be used in estimation. Default is False.
enforce_stationarity : bool, optional
    Whether or not to transform the AR parameters to enforce stationarity
    in the autoregressive component of the model. Default is True.
enforce_invertibility : bool, optional
    Whether or not to transform the MA parameters to enforce invertibility
    in the moving average component of the model. Default is True.
hamilton_representation : bool, optional
    Whether or not to use the Hamilton representation of an ARMA process
    (if True) or the Harvey representation (if False). Default is False.
concentrate_scale : bool, optional
    Whether or not to concentrate the scale (variance of the error term)
    out of the likelihood. This reduces the number of parameters estimated
    by maximum likelihood by one, but standard errors will then not
    be available for the scale parameter.
trend_offset : int, optional
    The offset at which to start time trend values. Default is 1, so that
    if `trend='t'` the trend is equal to 1, 2, ..., nobs. Typically is only
    set when the model created by extending a previous dataset.
use_exact_diffuse : bool, optional
    Whether or not to use exact diffuse initialization for non-stationary
    states. Default is False (in which case approximate diffuse
    initialization is used).
**kwargs
    Keyword arguments may be used to provide default values for state space
    matrices or for Kalman filtering options. See `Representation`, and
    `KalmanFilter` for more details.

Attributes
----------
measurement_error : bool
    Whether or not to assume the endogenous
    observations `endog` were measured with error.
state_error : bool
    Whether or not the transition equation has an error component.
mle_regression : bool
    Whether or not the regression coefficients for
    the exogenous variables were estimated via maximum
    likelihood estimation.
state_regression : bool
    Whether or not the regression coefficients for
    the exogenous variables are included as elements
    of the state space and estimated via the Kalman
    filter.
time_varying_regression : bool
    Whether or not coefficients on the exogenous
    regressors are allowed to vary over time.
simple_differencing : bool
    Whether or not to use partially conditional maximum likelihood
    estimation.
enforce_stationarity : bool
    Whether or not to transform the AR parameters
    to enforce stationarity in the autoregressive
    component of the model.
enforce_invertibility : bool
    Whether or not to transform the MA parameters
    to enforce invertibility in the moving average
    component of the model.
hamilton_representation : bool
    Whether or not to use the Hamilton representation of an ARMA process.
trend : str{'n','c','t','ct'} or iterable
    Parameter controlling the deterministic
    trend polynomial :math:`A(t)`. See the class
    parameter documentation for more information.
polynomial_ar : ndarray
    Array containing autoregressive lag polynomial lags, ordered from
    lowest degree to highest. The polynomial begins with lag 0.
    Initialized with ones, unless a coefficient is constrained to be
    zero (in which case it is zero).
polynomial_ma : ndarray
    Array containing moving average lag polynomial lags, ordered from
    lowest degree to highest. Initialized with ones, unless a coefficient
    is constrained to be zero (in which case it is zero).
polynomial_seasonal_ar : ndarray
    Array containing seasonal moving average lag
    polynomial lags, ordered from lowest degree
    to highest. Initialized with ones, unless a
    coefficient is constrained to be zero (in which
    case it is zero).
polynomial_seasonal_ma : ndarray
    Array containing seasonal moving average lag
    polynomial lags, ordered from lowest degree
    to highest. Initialized with ones, unless a
    coefficient is constrained to be zero (in which
    case it is zero).
polynomial_trend : ndarray
    Array containing trend polynomial coefficients,
    ordered from lowest degree to highest. Initialized
    with ones, unless a coefficient is constrained to be
    zero (in which case it is zero).
k_ar : int
    Highest autoregressive order in the model, zero-indexed.
k_ar_params : int
    Number of autoregressive parameters to be estimated.
k_diff : int
    Order of integration.
k_ma : int
    Highest moving average order in the model, zero-indexed.
k_ma_params : int
    Number of moving average parameters to be estimated.
seasonal_periods : int
    Number of periods in a season.
k_seasonal_ar : int
    Highest seasonal autoregressive order in the model, zero-indexed.
k_seasonal_ar_params : int
    Number of seasonal autoregressive parameters to be estimated.
k_seasonal_diff : int
    Order of seasonal integration.
k_seasonal_ma : int
    Highest seasonal moving average order in the model, zero-indexed.
k_seasonal_ma_params : int
    Number of seasonal moving average parameters to be estimated.
k_trend : int
    Order of the trend polynomial plus one (i.e. the constant polynomial
    would have `k_trend=1`).
k_exog : int
    Number of exogenous regressors.

Notes
-----
The SARIMA model is specified :math:`(p, d, q) \times (P, D, Q)_s`.

.. math::

    \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D y_t = A(t) +
        \theta_q (L) \tilde \theta_Q (L^s) \zeta_t

In terms of a univariate structural model, this can be represented as

.. math::

    y_t & = u_t + \eta_t \\
    \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D u_t & = A(t) +
        \theta_q (L) \tilde \theta_Q (L^s) \zeta_t

where :math:`\eta_t` is only applicable in the case of measurement error
(although it is also used in the case of a pure regression model, i.e. if
p=q=0).

In terms of this model, regression with SARIMA errors can be represented
easily as

.. math::

    y_t & = \beta_t x_t + u_t \\
    \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D u_t & = A(t) +
        \theta_q (L) \tilde \theta_Q (L^s) \zeta_t

this model is the one used when exogenous regressors are provided.

Note that the reduced form lag polynomials will be written as:

.. math::

    \Phi (L) \equiv \phi_p (L) \tilde \phi_P (L^s) \\
    \Theta (L) \equiv \theta_q (L) \tilde \theta_Q (L^s)

If `mle_regression` is True, regression coefficients are treated as
additional parameters to be estimated via maximum likelihood. Otherwise
they are included as part of the state with a diffuse initialization.
In this case, however, with approximate diffuse initialization, results
can be sensitive to the initial variance.

This class allows two different underlying representations of ARMA models
as state space models: that of Hamilton and that of Harvey. Both are
equivalent in the sense that they are analytical representations of the
ARMA model, but the state vectors of each have different meanings. For
this reason, maximum likelihood does not result in identical parameter
estimates and even the same set of parameters will result in different
loglikelihoods.

The Harvey representation is convenient because it allows integrating
differencing into the state vector to allow using all observations for
estimation.

In this implementation of differenced models, the Hamilton representation
is not able to accommodate differencing in the state vector, so
`simple_differencing` (which performs differencing prior to estimation so
that the first d + sD observations are lost) must be used.

Many other packages use the Hamilton representation, so that tests against
Stata and R require using it along with simple differencing (as Stata
does).

If `filter_concentrated = True` is used, then the scale of the model is
concentrated out of the likelihood. A benefit of this is that there the
dimension of the parameter vector is reduced so that numerical maximization
of the log-likelihood function may be faster and more stable. If this
option in a model with measurement error, it is important to note that the
estimated measurement error parameter will be relative to the scale, and
is named "snr.measurement_error" instead of "var.measurement_error". To
compute the variance of the measurement error in this case one would
multiply `snr.measurement_error` parameter by the scale.

If `simple_differencing = True` is used, then the `endog` and `exog` data
are differenced prior to putting the model in state-space form. This has
the same effect as if the user differenced the data prior to constructing
the model, which has implications for using the results:

- Forecasts and predictions will be about the *differenced* data, not about
  the original data. (while if `simple_differencing = False` is used, then
  forecasts and predictions will be about the original data).
- If the original data has an Int64Index, a new RangeIndex will be created
  for the differenced data that starts from one, and forecasts and
  predictions will use this new index.

Detailed information about state space models can be found in [1]_. Some
specific references are:

- Chapter 3.4 describes ARMA and ARIMA models in state space form (using
  the Harvey representation), and gives references for basic seasonal
  models and models with a multiplicative form (for example the airline
  model). It also shows a state space model for a full ARIMA process (this
  is what is done here if `simple_differencing=False`).
- Chapter 3.6 describes estimating regression effects via the Kalman filter
  (this is performed if `mle_regression` is False), regression with
  time-varying coefficients, and regression with ARMA errors (recall from
  above that if regression effects are present, the model estimated by this
  class is regression with SARIMA errors).
- Chapter 8.4 describes the application of an ARMA model to an example
  dataset. A replication of this section is available in an example
  IPython notebook in the documentation.

References
----------
.. [1] Durbin, James, and Siem Jan Koopman. 2012.
   Time Series Analysis by State Space Methods: Second Edition.
   Oxford University Press.

