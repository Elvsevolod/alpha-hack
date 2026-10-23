# sklearn.preprocessing.RobustScaler

Version: 1.5.2

sklearn.preprocessing.RobustScaler(*, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, unit_variance=False)

Scale features using statistics that are robust to outliers.

This Scaler removes the median and scales the data according to
the quantile range (defaults to IQR: Interquartile Range).
The IQR is the range between the 1st quartile (25th quantile)
and the 3rd quartile (75th quantile).

Centering and scaling happen independently on each feature by
computing the relevant statistics on the samples in the training
set. Median and interquartile range are then stored to be used on
later data using the :meth:`transform` method.

Standardization of a dataset is a common preprocessing for many machine
learning estimators. Typically this is done by removing the mean and
scaling to unit variance. However, outliers can often influence the sample
mean / variance in a negative way. In such cases, using the median and the
interquartile range often give better results. For an example visualization
and comparison to other scalers, refer to :ref:`Compare RobustScaler with
other scalers <plot_all_scaling_robust_scaler_section>`.

.. versionadded:: 0.17

Read more in the :ref:`User Guide <preprocessing_scaler>`.

Parameters
----------
with_centering : bool, default=True
    If `True`, center the data before scaling.
    This will cause :meth:`transform` to raise an exception when attempted
    on sparse matrices, because centering them entails building a dense
    matrix which in common use cases is likely to be too large to fit in
    memory.

with_scaling : bool, default=True
    If `True`, scale the data to interquartile range.

quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0,         default=(25.0, 75.0)
    Quantile range used to calculate `scale_`. By default this is equal to
    the IQR, i.e., `q_min` is the first quantile and `q_max` is the third
    quantile.

    .. versionadded:: 0.18

copy : bool, default=True
    If `False`, try to avoid a copy and do inplace scaling instead.
    This is not guaranteed to always work inplace; e.g. if the data is
    not a NumPy array or scipy.sparse CSR matrix, a copy may still be
    returned.

unit_variance : bool, default=False
    If `True`, scale data so that normally distributed features have a
    variance of 1. In general, if the difference between the x-values of
    `q_max` and `q_min` for a standard normal distribution is greater
    than 1, the dataset will be scaled down. If less than 1, the dataset
    will be scaled up.

    .. versionadded:: 0.24

Attributes
----------
center_ : array of floats
    The median value for each feature in the training set.

scale_ : array of floats
    The (scaled) interquartile range for each feature in the training set.

    .. versionadded:: 0.17
       *scale_* attribute.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

See Also
--------
robust_scale : Equivalent function without the estimator API.
sklearn.decomposition.PCA : Further removes the linear correlation across
    features with 'whiten=True'.

Notes
-----

https://en.wikipedia.org/wiki/Median
https://en.wikipedia.org/wiki/Interquartile_range

Examples
--------
>>> from sklearn.preprocessing import RobustScaler
>>> X = [[ 1., -2.,  2.],
...      [ -2.,  1.,  3.],
...      [ 4.,  1., -2.]]
>>> transformer = RobustScaler().fit(X)
>>> transformer
RobustScaler()
>>> transformer.transform(X)
array([[ 0. , -2. ,  0. ],
       [-1. ,  0. ,  0.4],
       [ 1. ,  0. , -1.6]])

