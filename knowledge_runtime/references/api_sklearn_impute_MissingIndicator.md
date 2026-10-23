# sklearn.impute.MissingIndicator

Version: 1.5.2

sklearn.impute.MissingIndicator(*, missing_values=nan, features='missing-only', sparse='auto', error_on_new=True)

Binary indicators for missing values.

Note that this component typically should not be used in a vanilla
:class:`~sklearn.pipeline.Pipeline` consisting of transformers and a
classifier, but rather could be added using a
:class:`~sklearn.pipeline.FeatureUnion` or
:class:`~sklearn.compose.ColumnTransformer`.

Read more in the :ref:`User Guide <impute>`.

.. versionadded:: 0.20

Parameters
----------
missing_values : int, float, str, np.nan or None, default=np.nan
    The placeholder for the missing values. All occurrences of
    `missing_values` will be imputed. For pandas' dataframes with
    nullable integer dtypes with missing values, `missing_values`
    should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

features : {'missing-only', 'all'}, default='missing-only'
    Whether the imputer mask should represent all or a subset of
    features.

    - If `'missing-only'` (default), the imputer mask will only represent
      features containing missing values during fit time.
    - If `'all'`, the imputer mask will represent all features.

sparse : bool or 'auto', default='auto'
    Whether the imputer mask format should be sparse or dense.

    - If `'auto'` (default), the imputer mask will be of same type as
      input.
    - If `True`, the imputer mask will be a sparse matrix.
    - If `False`, the imputer mask will be a numpy array.

error_on_new : bool, default=True
    If `True`, :meth:`transform` will raise an error when there are
    features with missing values that have no missing values in
    :meth:`fit`. This is applicable only when `features='missing-only'`.

Attributes
----------
features_ : ndarray of shape (n_missing_features,) or (n_features,)
    The features indices which will be returned when calling
    :meth:`transform`. They are computed during :meth:`fit`. If
    `features='all'`, `features_` is equal to `range(n_features)`.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

See Also
--------
SimpleImputer : Univariate imputation of missing values.
IterativeImputer : Multivariate imputation of missing values.

Examples
--------
>>> import numpy as np
>>> from sklearn.impute import MissingIndicator
>>> X1 = np.array([[np.nan, 1, 3],
...                [4, 0, np.nan],
...                [8, 1, 0]])
>>> X2 = np.array([[5, 1, np.nan],
...                [np.nan, 2, 3],
...                [2, 4, 0]])
>>> indicator = MissingIndicator()
>>> indicator.fit(X1)
MissingIndicator()
>>> X2_tr = indicator.transform(X2)
>>> X2_tr
array([[False,  True],
       [ True, False],
       [False, False]])

