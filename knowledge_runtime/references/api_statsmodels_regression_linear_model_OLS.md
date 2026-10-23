# statsmodels.regression.linear_model.OLS

Version: 0.14.0

statsmodels.regression.linear_model.OLS(endog, exog=None, missing='none', hasconst=None, **kwargs)

Ordinary Least Squares

Parameters
----------
endog : array_like
    A 1-d endogenous response variable. The dependent variable.
exog : array_like
    A nobs x k array where `nobs` is the number of observations and `k`
    is the number of regressors. An intercept is not included by default
    and should be added by the user. See
    :func:`statsmodels.tools.add_constant`.
missing : str
    Available options are 'none', 'drop', and 'raise'. If 'none', no nan
    checking is done. If 'drop', any observations with nans are dropped.
    If 'raise', an error is raised. Default is 'none'.
hasconst : None or bool
    Indicates whether the RHS includes a user-supplied constant. If True,
    a constant is not checked for and k_constant is set to 1 and all
    result statistics are calculated as if a constant is present. If
    False, a constant is not checked for and k_constant is set to 0.
**kwargs
    Extra arguments that are used to set model properties when using the
    formula interface.

Attributes
----------
weights : scalar
    Has an attribute weights = array(1.0) due to inheritance from WLS.

See Also
--------
WLS : Fit a linear model using Weighted Least Squares.
GLS : Fit a linear model using Generalized Least Squares.

Notes
-----
No constant is added by the model unless you are using formulas.

Examples
--------
>>> import statsmodels.api as sm
>>> import numpy as np
>>> duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
>>> Y = duncan_prestige.data['income']
>>> X = duncan_prestige.data['education']
>>> X = sm.add_constant(X)
>>> model = sm.OLS(Y,X)
>>> results = model.fit()
>>> results.params
const        10.603498
education     0.594859
dtype: float64

>>> results.tvalues
const        2.039813
education    6.892802
dtype: float64

>>> print(results.t_test([1, 0]))
                             Test for Constraints
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
c0            10.6035      5.198      2.040      0.048       0.120      21.087
==============================================================================

>>> print(results.f_test(np.identity(2)))
<F test: F=array([[159.63031026]]), p=1.2607168903696672e-20,
 df_denom=43, df_num=2>

