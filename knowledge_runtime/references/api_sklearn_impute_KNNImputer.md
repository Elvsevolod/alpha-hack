# sklearn.impute.KNNImputer

Version: 1.5.2

sklearn.impute.KNNImputer(*, missing_values=nan, n_neighbors=5, weights='uniform', metric='nan_euclidean', copy=True, add_indicator=False, keep_empty_features=False)

Imputation for completing missing values using k-Nearest Neighbors.

Each sample's missing values are imputed using the mean value from
`n_neighbors` nearest neighbors found in the training set. Two samples are
close if the features that neither is missing are close.

Read more in the :ref:`User Guide <knnimpute>`.

.. versionadded:: 0.22

Parameters
----------
missing_values : int, float, str, np.nan or None, default=np.nan
    The placeholder for the missing values. All occurrences of
    `missing_values` will be imputed. For pandas' dataframes with
    nullable integer dtypes with missing values, `missing_values`
    should be set to np.nan, since `pd.NA` will be converted to np.nan.

n_neighbors : int, default=5
    Number of neighboring samples to use for imputation.

weights : {'uniform', 'distance'} or callable, default='uniform'
    Weight function used in prediction.  Possible values:

    - 'uniform' : uniform weights. All points in each neighborhood are
      weighted equally.
    - 'distance' : weight points by the inverse of their distance.
      in this case, closer neighbors of a query point will have a
      greater influence than neighbors which are further away.
    - callable : a user-defined function which accepts an
      array of distances, and returns an array of the same shape
      containing the weights.

metric : {'nan_euclidean'} or callable, default='nan_euclidean'
    Distance metric for searching neighbors. Possible values:

    - 'nan_euclidean'
    - callable : a user-defined function which conforms to the definition
      of ``func_metric(x, y, *, missing_values=np.nan)``. `x` and `y`
      corresponds to a row (i.e. 1-D arrays) of `X` and `Y`, respectively.
      The callable should returns a scalar distance value.

copy : bool, default=True
    If True, a copy of X will be created. If False, imputation will
    be done in-place whenever possible.

add_indicator : bool, default=False
    If True, a :class:`MissingIndicator` transform will stack onto the
    output of the imputer's transform. This allows a predictive estimator
    to account for missingness despite imputation. If a feature has no
    missing values at fit/train time, the feature won't appear on the
    missing indicator even if there are missing values at transform/test
    time.

keep_empty_features : bool, default=False
    If True, features that consist exclusively of missing values when
    `fit` is called are returned in results when `transform` is called.
    The imputed value is always `0`.

    .. versionadded:: 1.2

Attributes
----------
indicator_ : :class:`~sklearn.impute.MissingIndicator`
    Indicator used to add binary indicators for missing values.
    ``None`` if add_indicator is False.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

See Also
--------
SimpleImputer : Univariate imputer for completing missing values
    with simple strategies.
IterativeImputer : Multivariate imputer that estimates values to impute for
    each feature with missing values from all the others.

References
----------
* `Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor
  Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing
  value estimation methods for DNA microarrays, BIOINFORMATICS Vol. 17
  no. 6, 2001 Pages 520-525.
  <https://academic.oup.com/bioinformatics/article/17/6/520/272365>`_

Examples
--------
>>> import numpy as np
>>> from sklearn.impute import KNNImputer
>>> X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
>>> imputer = KNNImputer(n_neighbors=2)
>>> imputer.fit_transform(X)
array([[1. , 2. , 4. ],
       [3. , 4. , 3. ],
       [5.5, 6. , 5. ],
       [8. , 8. , 7. ]])

For a more detailed example see
:ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.

