# statsmodels.tsa.statespace.sarimax.SARIMAX.fit
Version: 0.14.0

## statsmodels.tsa.statespace.sarimax.SARIMAX.fit(self, start_params=None, transformed=True, includes_fixed=False, cov_type=None, cov_kwds=None, method='lbfgs', maxiter=50, full_output=1, disp=5, callback=None, return_params=False, optim_score=None, optim_complex_step=None, optim_hessian=None, flags=None, low_memory=False, **kwargs)

Fits the model by maximum likelihood via Kalman filter.

Parameters
----------
start_params : array_like, optional
    Initial guess of the solution for the loglikelihood maximization.
    If None, the default is given by Model.start_params.
transformed : bool, optional
    Whether or not `start_params` is already transformed. Default is
    True.
includes_fixed : bool, optional
    If parameters were previously fixed with the `fix_params` method,
    this argument describes whether or not `start_params` also includes
    the fixed parameters, in addition to the free parameters. Default
    is False.
cov_type : str, optional
    The `cov_type` keyword governs the method for calculating the
    covariance matrix of parameter estimates. Can be one of:

    - 'opg' for the outer product of gradient estimator
    - 'oim' for the observed information matrix estimator, calculated
      using the method of Harvey (1989)
    - 'approx' for the observed information matrix estimator,
      calculated using a numerical approximation of the Hessian matrix.
    - 'robust' for an approximate (quasi-maximum likelihood) covariance
      matrix that may be valid even in the presence of some
      misspecifications. Intermediate calculations use the 'oim'
      method.
    - 'robust_approx' is the same as 'robust' except that the
      intermediate calculations use the 'approx' method.
    - 'none' for no covariance matrix calculation.

    Default is 'opg' unless memory conservation is used to avoid
    computing the loglikelihood values for each observation, in which
    case the default is 'approx'.
cov_kwds : dict or None, optional
    A dictionary of arguments affecting covariance matrix computation.

    **opg, oim, approx, robust, robust_approx**

    - 'approx_complex_step' : bool, optional - If True, numerical
      approximations are computed using complex-step methods. If False,
      numerical approximations are computed using finite difference
      methods. Default is True.
    - 'approx_centered' : bool, optional - If True, numerical
      approximations computed using finite difference methods use a
      centered approximation. Default is False.
method : str, optional
    The `method` determines which solver from `scipy.optimize`
    is used, and it can be chosen from among the following strings:

    - 'newton' for Newton-Raphson
    - 'nm' for Nelder-Mead
    - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
    - 'lbfgs' for limited-memory BFGS with optional box constraints
    - 'powell' for modified Powell's method
    - 'cg' for conjugate gradient
    - 'ncg' for Newton-conjugate gradient
    - 'basinhopping' for global basin-hopping solver

    The explicit arguments in `fit` are passed to the solver,
    with the exception of the basin-hopping solver. Each
    solver has several optional arguments that are not the same across
    solvers. See the notes section below (or scipy.optimize) for the
    available arguments and for the list of explicit arguments that the
    basin-hopping solver supports.
maxiter : int, optional
    The maximum number of iterations to perform.
full_output : bool, optional
    Set to True to have all available output in the Results object's
    mle_retvals attribute. The output is dependent on the solver.
    See LikelihoodModelResults notes section for more information.
disp : bool, optional
    Set to True to print convergence messages.
callback : callable callback(xk), optional
    Called after each iteration, as callback(xk), where xk is the
    current parameter vector.
return_params : bool, optional
    Whether or not to return only the array of maximizing parameters.
    Default is False.
optim_score : {'harvey', 'approx'} or None, optional
    The method by which the score vector is calculated. 'harvey' uses
    the method from Harvey (1989), 'approx' uses either finite
    difference or complex step differentiation depending upon the
    value of `optim_complex_step`, and None uses the built-in gradient
    approximation of the optimizer. Default is None. This keyword is
    only relevant if the optimization method uses the score.
optim_complex_step : bool, optional
    Whether or not to use complex step differentiation when
    approximating the score; if False, finite difference approximation
    is used. Default is True. This keyword is only relevant if
    `optim_score` is set to 'harvey' or 'approx'.
optim_hessian : {'opg','oim','approx'}, optional
    The method by which the Hessian is numerically approximated. 'opg'
    uses outer product of gradients, 'oim' uses the information
    matrix formula from Harvey (1989), and 'approx' uses numerical
    approximation. This keyword is only relevant if the
    optimization method uses the Hessian matrix.
low_memory : bool, optional
    If set to True, techniques are applied to substantially reduce
    memory usage. If used, some features of the results object will
    not be available (including smoothed results and in-sample
    prediction), although out-of-sample forecasting is possible.
    Default is False.
**kwargs
    Additional keyword arguments to pass to the optimizer.

Returns
-------
results
    Results object holding results from fitting a state space model.

See Also
--------
statsmodels.base.model.LikelihoodModel.fit
statsmodels.tsa.statespace.mlemodel.MLEResults
statsmodels.tsa.statespace.structural.UnobservedComponentsResults
