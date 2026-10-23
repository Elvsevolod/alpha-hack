# sklearn.preprocessing.StandardScaler

Version: 1.5.2

sklearn.preprocessing.StandardScaler(*, copy=True, with_mean=True, with_std=True)

Standardize features by removing the mean and scaling to unit variance.

The standard score of a sample `x` is calculated as:

    z = (x - u) / s

where `u` is the mean of the training samples or zero if `with_mean=False`,
and `s` is the standard deviation of the training samples or one if
`with_std=False`.

Centering and scaling happen independently on each feature by computing
the relevant statistics on the samples in the training set. Mean and
standard deviation are then stored to be used on later data using
:meth:`transform`.

Standardization of a dataset is a common requirement for many
machine learning estimators: they might behave badly if the
individual features do not more or less look like standard normally
distributed data (e.g. Gaussian with 0 mean and unit variance).

For instance many elements used in the objective function of
a learning algorithm (such as the RBF kernel of Support Vector
Machines or the L1 and L2 regularizers of linear models) assume that
all features are centered around 0 and have variance in the same
order. If a feature has a variance that is orders of magnitude larger
than others, it might dominate the objective function and make the
estimator unable to learn from other features correctly as expected.

`StandardScaler` is sensitive to outliers, and the features may scale
differently from each other in the presence of outliers. For an example
visualization, refer to :ref:`Compare StandardScaler with other scalers
<plot_all_scaling_standard_scaler_section>`.

This scaler can also be applied to sparse CSR or CSC matrices by passing
`with_mean=False` to avoid breaking the sparsity structure of the data.

Read more in the :ref:`User Guide <preprocessing_scaler>`.

Parameters
----------
copy : bool, default=True
    If False, try to avoid a copy and do inplace scaling instead.
    This is not guaranteed to always work inplace; e.g. if the data is
    not a NumPy array or scipy.sparse CSR matrix, a copy may still be
    returned.

with_mean : bool, default=True
    If True, center the data before scaling.
    This does not work (and will raise an exception) when attempted on
    sparse matrices, because centering them entails building a dense
    matrix which in common use cases is likely to be too large to fit in
    memory.

with_std : bool, default=True
    If True, scale the data to unit variance (or equivalently,
    unit standard deviation).

Attributes
----------
scale_ : ndarray of shape (n_features,) or None
    Per feature relative scaling of the data to achieve zero mean and unit
    variance. Generally this is calculated using `np.sqrt(var_)`. If a
    variance is zero, we can't achieve unit variance, and the data is left
    as-is, giving a scaling factor of 1. `scale_` is equal to `None`
    when `with_std=False`.

    .. versionadded:: 0.17
       *scale_*

mean_ : ndarray of shape (n_features,) or None
    The mean value for each feature in the training set.
    Equal to ``None`` when ``with_mean=False`` and ``with_std=False``.

var_ : ndarray of shape (n_features,) or None
    The variance for each feature in the training set. Used to compute
    `scale_`. Equal to ``None`` when ``with_mean=False`` and
    ``with_std=False``.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

n_samples_seen_ : int or ndarray of shape (n_features,)
    The number of samples processed by the estimator for each feature.
    If there are no missing samples, the ``n_samples_seen`` will be an
    integer, otherwise it will be an array of dtype int. If
    `sample_weights` are used it will be a float (if no missing data)
    or an array of dtype float that sums the weights seen so far.
    Will be reset on new calls to fit, but increments across
    ``partial_fit`` calls.

See Also
--------
scale : Equivalent function without the estimator API.

:class:`~sklearn.decomposition.PCA` : Further removes the linear
    correlation across features with 'whiten=True'.

Notes
-----
NaNs are treated as missing values: disregarded in fit, and maintained in
transform.

We use a biased estimator for the standard deviation, equivalent to
`numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
affect model performance.

Examples
--------
>>> from sklearn.preprocessing import StandardScaler
>>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
>>> scaler = StandardScaler()
>>> print(scaler.fit(data))
StandardScaler()
>>> print(scaler.mean_)
[0.5 0.5]
>>> print(scaler.transform(data))
[[-1. -1.]
 [-1. -1.]
 [ 1.  1.]
 [ 1.  1.]]
>>> print(scaler.transform([[2, 2]]))
[[3. 3.]]

