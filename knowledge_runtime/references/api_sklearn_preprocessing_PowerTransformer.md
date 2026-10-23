# sklearn.preprocessing.PowerTransformer

Version: 1.5.2

sklearn.preprocessing.PowerTransformer(method='yeo-johnson', *, standardize=True, copy=True)

Apply a power transform featurewise to make data more Gaussian-like.

Power transforms are a family of parametric, monotonic transformations
that are applied to make data more Gaussian-like. This is useful for
modeling issues related to heteroscedasticity (non-constant variance),
or other situations where normality is desired.

Currently, PowerTransformer supports the Box-Cox transform and the
Yeo-Johnson transform. The optimal parameter for stabilizing variance and
minimizing skewness is estimated through maximum likelihood.

Box-Cox requires input data to be strictly positive, while Yeo-Johnson
supports both positive or negative data.

By default, zero-mean, unit-variance normalization is applied to the
transformed data.

For an example visualization, refer to :ref:`Compare PowerTransformer with
other scalers <plot_all_scaling_power_transformer_section>`. To see the
effect of Box-Cox and Yeo-Johnson transformations on different
distributions, see:
:ref:`sphx_glr_auto_examples_preprocessing_plot_map_data_to_normal.py`.

Read more in the :ref:`User Guide <preprocessing_transformer>`.

.. versionadded:: 0.20

Parameters
----------
method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
    The power transform method. Available methods are:

    - 'yeo-johnson' [1]_, works with positive and negative values
    - 'box-cox' [2]_, only works with strictly positive values

standardize : bool, default=True
    Set to True to apply zero-mean, unit-variance normalization to the
    transformed output.

copy : bool, default=True
    Set to False to perform inplace computation during transformation.

Attributes
----------
lambdas_ : ndarray of float of shape (n_features,)
    The parameters of the power transformation for the selected features.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

See Also
--------
power_transform : Equivalent function without the estimator API.

QuantileTransformer : Maps data to a standard normal distribution with
    the parameter `output_distribution='normal'`.

Notes
-----
NaNs are treated as missing values: disregarded in ``fit``, and maintained
in ``transform``.

References
----------

.. [1] :doi:`I.K. Yeo and R.A. Johnson, "A new family of power
       transformations to improve normality or symmetry." Biometrika,
       87(4), pp.954-959, (2000). <10.1093/biomet/87.4.954>`

.. [2] :doi:`G.E.P. Box and D.R. Cox, "An Analysis of Transformations",
       Journal of the Royal Statistical Society B, 26, 211-252 (1964).
       <10.1111/j.2517-6161.1964.tb00553.x>`

Examples
--------
>>> import numpy as np
>>> from sklearn.preprocessing import PowerTransformer
>>> pt = PowerTransformer()
>>> data = [[1, 2], [3, 2], [4, 5]]
>>> print(pt.fit(data))
PowerTransformer()
>>> print(pt.lambdas_)
[ 1.386... -3.100...]
>>> print(pt.transform(data))
[[-1.316... -0.707...]
 [ 0.209... -0.707...]
 [ 1.106...  1.414...]]

