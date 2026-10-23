# sklearn.linear_model.ElasticNet

Version: 1.5.2

sklearn.linear_model.ElasticNet(alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')

Linear regression with combined L1 and L2 priors as regularizer.

Minimizes the objective function::

        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

If you are interested in controlling the L1 and L2 penalty
separately, keep in mind that this is equivalent to::

        a * ||w||_1 + 0.5 * b * ||w||_2^2

where::

        alpha = a + b and l1_ratio = a / (a + b)

The parameter l1_ratio corresponds to alpha in the glmnet R package while
alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
= 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
unless you supply your own sequence of alpha.

Read more in the :ref:`User Guide <elastic_net>`.

Parameters
----------
alpha : float, default=1.0
    Constant that multiplies the penalty terms. Defaults to 1.0.
    See the notes for the exact mathematical meaning of this
    parameter. ``alpha = 0`` is equivalent to an ordinary least square,
    solved by the :class:`LinearRegression` object. For numerical
    reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
    Given this, you should use the :class:`LinearRegression` object.

l1_ratio : float, default=0.5
    The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
    ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
    is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
    combination of L1 and L2.

fit_intercept : bool, default=True
    Whether the intercept should be estimated or not. If ``False``, the
    data is assumed to be already centered.

precompute : bool or array-like of shape (n_features, n_features),                 default=False
    Whether to use a precomputed Gram matrix to speed up
    calculations. The Gram matrix can also be passed as argument.
    For sparse input this option is always ``False`` to preserve sparsity.
    Check :ref:`an example on how to use a precomputed Gram Matrix in ElasticNet
    <sphx_glr_auto_examples_linear_model_plot_elastic_net_precomputed_gram_matrix_with_weighted_samples.py>`
    for details.

max_iter : int, default=1000
    The maximum number of iterations.

copy_X : bool, default=True
    If ``True``, X will be copied; else, it may be overwritten.

tol : float, default=1e-4
    The tolerance for the optimization: if the updates are
    smaller than ``tol``, the optimization code checks the
    dual gap for optimality and continues until it is smaller
    than ``tol``, see Notes below.

warm_start : bool, default=False
    When set to ``True``, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

positive : bool, default=False
    When set to ``True``, forces the coefficients to be positive.

random_state : int, RandomState instance, default=None
    The seed of the pseudo random number generator that selects a random
    feature to update. Used when ``selection`` == 'random'.
    Pass an int for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

selection : {'cyclic', 'random'}, default='cyclic'
    If set to 'random', a random coefficient is updated every iteration
    rather than looping over features sequentially by default. This
    (setting to 'random') often leads to significantly faster convergence
    especially when tol is higher than 1e-4.

Attributes
----------
coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
    Parameter vector (w in the cost function formula).

sparse_coef_ : sparse matrix of shape (n_features,) or             (n_targets, n_features)
    Sparse representation of the `coef_`.

intercept_ : float or ndarray of shape (n_targets,)
    Independent term in decision function.

n_iter_ : list of int
    Number of iterations run by the coordinate descent solver to reach
    the specified tolerance.

dual_gap_ : float or ndarray of shape (n_targets,)
    Given param alpha, the dual gaps at the end of the optimization,
    same shape as each observation of y.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

See Also
--------
ElasticNetCV : Elastic net model with best model selection by
    cross-validation.
SGDRegressor : Implements elastic net regression with incremental training.
SGDClassifier : Implements logistic regression with elastic net penalty
    (``SGDClassifier(loss="log_loss", penalty="elasticnet")``).

Notes
-----
To avoid unnecessary memory duplication the X argument of the fit method
should be directly passed as a Fortran-contiguous numpy array.

The precise stopping criteria based on `tol` are the following: First, check that
that maximum coordinate update, i.e. :math:`\max_j |w_j^{new} - w_j^{old}|`
is smaller than `tol` times the maximum absolute coefficient, :math:`\max_j |w_j|`.
If so, then additionally check whether the dual gap is smaller than `tol` times
:math:`||y||_2^2 / n_{      ext{samples}}`.

Examples
--------
>>> from sklearn.linear_model import ElasticNet
>>> from sklearn.datasets import make_regression

>>> X, y = make_regression(n_features=2, random_state=0)
>>> regr = ElasticNet(random_state=0)
>>> regr.fit(X, y)
ElasticNet(random_state=0)
>>> print(regr.coef_)
[18.83816048 64.55968825]
>>> print(regr.intercept_)
1.451...
>>> print(regr.predict([[0, 0]]))
[1.451...]

