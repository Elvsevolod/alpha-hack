# sklearn.preprocessing.SplineTransformer

Version: 1.5.2

sklearn.preprocessing.SplineTransformer(n_knots=5, degree=3, *, knots='uniform', extrapolation='constant', include_bias=True, order='C', sparse_output=False)

Generate univariate B-spline bases for features.

Generate a new feature matrix consisting of
`n_splines=n_knots + degree - 1` (`n_knots - 1` for
`extrapolation="periodic"`) spline basis functions
(B-splines) of polynomial order=`degree` for each feature.

In order to learn more about the SplineTransformer class go to:
:ref:`sphx_glr_auto_examples_applications_plot_cyclical_feature_engineering.py`

Read more in the :ref:`User Guide <spline_transformer>`.

.. versionadded:: 1.0

Parameters
----------
n_knots : int, default=5
    Number of knots of the splines if `knots` equals one of
    {'uniform', 'quantile'}. Must be larger or equal 2. Ignored if `knots`
    is array-like.

degree : int, default=3
    The polynomial degree of the spline basis. Must be a non-negative
    integer.

knots : {'uniform', 'quantile'} or array-like of shape         (n_knots, n_features), default='uniform'
    Set knot positions such that first knot <= features <= last knot.

    - If 'uniform', `n_knots` number of knots are distributed uniformly
      from min to max values of the features.
    - If 'quantile', they are distributed uniformly along the quantiles of
      the features.
    - If an array-like is given, it directly specifies the sorted knot
      positions including the boundary knots. Note that, internally,
      `degree` number of knots are added before the first knot, the same
      after the last knot.

extrapolation : {'error', 'constant', 'linear', 'continue', 'periodic'},         default='constant'
    If 'error', values outside the min and max values of the training
    features raises a `ValueError`. If 'constant', the value of the
    splines at minimum and maximum value of the features is used as
    constant extrapolation. If 'linear', a linear extrapolation is used.
    If 'continue', the splines are extrapolated as is, i.e. option
    `extrapolate=True` in :class:`scipy.interpolate.BSpline`. If
    'periodic', periodic splines with a periodicity equal to the distance
    between the first and last knot are used. Periodic splines enforce
    equal function values and derivatives at the first and last knot.
    For example, this makes it possible to avoid introducing an arbitrary
    jump between Dec 31st and Jan 1st in spline features derived from a
    naturally periodic "day-of-year" input feature. In this case it is
    recommended to manually set the knot values to control the period.

include_bias : bool, default=True
    If False, then the last spline element inside the data range
    of a feature is dropped. As B-splines sum to one over the spline basis
    functions for each data point, they implicitly include a bias term,
    i.e. a column of ones. It acts as an intercept term in a linear models.

order : {'C', 'F'}, default='C'
    Order of output array in the dense case. `'F'` order is faster to compute, but
    may slow down subsequent estimators.

sparse_output : bool, default=False
    Will return sparse CSR matrix if set True else will return an array. This
    option is only available with `scipy>=1.8`.

    .. versionadded:: 1.2

Attributes
----------
bsplines_ : list of shape (n_features,)
    List of BSplines objects, one for each feature.

n_features_in_ : int
    The total number of input features.

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

n_features_out_ : int
    The total number of output features, which is computed as
    `n_features * n_splines`, where `n_splines` is
    the number of bases elements of the B-splines,
    `n_knots + degree - 1` for non-periodic splines and
    `n_knots - 1` for periodic ones.
    If `include_bias=False`, then it is only
    `n_features * (n_splines - 1)`.

See Also
--------
KBinsDiscretizer : Transformer that bins continuous data into intervals.

PolynomialFeatures : Transformer that generates polynomial and interaction
    features.

Notes
-----
High degrees and a high number of knots can cause overfitting.

See :ref:`examples/linear_model/plot_polynomial_interpolation.py
<sphx_glr_auto_examples_linear_model_plot_polynomial_interpolation.py>`.

Examples
--------
>>> import numpy as np
>>> from sklearn.preprocessing import SplineTransformer
>>> X = np.arange(6).reshape(6, 1)
>>> spline = SplineTransformer(degree=2, n_knots=3)
>>> spline.fit_transform(X)
array([[0.5 , 0.5 , 0.  , 0.  ],
       [0.18, 0.74, 0.08, 0.  ],
       [0.02, 0.66, 0.32, 0.  ],
       [0.  , 0.32, 0.66, 0.02],
       [0.  , 0.08, 0.74, 0.18],
       [0.  , 0.  , 0.5 , 0.5 ]])

