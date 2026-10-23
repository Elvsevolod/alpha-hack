# sklearn.preprocessing.OrdinalEncoder

Version: 1.5.2

sklearn.preprocessing.OrdinalEncoder(*, categories='auto', dtype=<class 'numpy.float64'>, handle_unknown='error', unknown_value=None, encoded_missing_value=nan, min_frequency=None, max_categories=None)

Encode categorical features as an integer array.

The input to this transformer should be an array-like of integers or
strings, denoting the values taken on by categorical (discrete) features.
The features are converted to ordinal integers. This results in
a single column of integers (0 to n_categories - 1) per feature.

Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
For a comparison of different encoders, refer to:
:ref:`sphx_glr_auto_examples_preprocessing_plot_target_encoder.py`.

.. versionadded:: 0.20

Parameters
----------
categories : 'auto' or a list of array-like, default='auto'
    Categories (unique values) per feature:

    - 'auto' : Determine categories automatically from the training data.
    - list : ``categories[i]`` holds the categories expected in the ith
      column. The passed categories should not mix strings and numeric
      values, and should be sorted in case of numeric values.

    The used categories can be found in the ``categories_`` attribute.

dtype : number type, default=np.float64
    Desired dtype of output.

handle_unknown : {'error', 'use_encoded_value'}, default='error'
    When set to 'error' an error will be raised in case an unknown
    categorical feature is present during transform. When set to
    'use_encoded_value', the encoded value of unknown categories will be
    set to the value given for the parameter `unknown_value`. In
    :meth:`inverse_transform`, an unknown category will be denoted as None.

    .. versionadded:: 0.24

unknown_value : int or np.nan, default=None
    When the parameter handle_unknown is set to 'use_encoded_value', this
    parameter is required and will set the encoded value of unknown
    categories. It has to be distinct from the values used to encode any of
    the categories in `fit`. If set to np.nan, the `dtype` parameter must
    be a float dtype.

    .. versionadded:: 0.24

encoded_missing_value : int or np.nan, default=np.nan
    Encoded value of missing categories. If set to `np.nan`, then the `dtype`
    parameter must be a float dtype.

    .. versionadded:: 1.1

min_frequency : int or float, default=None
    Specifies the minimum frequency below which a category will be
    considered infrequent.

    - If `int`, categories with a smaller cardinality will be considered
      infrequent.

    - If `float`, categories with a smaller cardinality than
      `min_frequency * n_samples`  will be considered infrequent.

    .. versionadded:: 1.3
        Read more in the :ref:`User Guide <encoder_infrequent_categories>`.

max_categories : int, default=None
    Specifies an upper limit to the number of output categories for each input
    feature when considering infrequent categories. If there are infrequent
    categories, `max_categories` includes the category representing the
    infrequent categories along with the frequent categories. If `None`,
    there is no limit to the number of output features.

    `max_categories` do **not** take into account missing or unknown
    categories. Setting `unknown_value` or `encoded_missing_value` to an
    integer will increase the number of unique integer codes by one each.
    This can result in up to `max_categories + 2` integer codes.

    .. versionadded:: 1.3
        Read more in the :ref:`User Guide <encoder_infrequent_categories>`.

Attributes
----------
categories_ : list of arrays
    The categories of each feature determined during ``fit`` (in order of
    the features in X and corresponding with the output of ``transform``).
    This does not include categories that weren't seen during ``fit``.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 1.0

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

infrequent_categories_ : list of ndarray
    Defined only if infrequent categories are enabled by setting
    `min_frequency` or `max_categories` to a non-default value.
    `infrequent_categories_[i]` are the infrequent categories for feature
    `i`. If the feature `i` has no infrequent categories
    `infrequent_categories_[i]` is None.

    .. versionadded:: 1.3

See Also
--------
OneHotEncoder : Performs a one-hot encoding of categorical features. This encoding
    is suitable for low to medium cardinality categorical variables, both in
    supervised and unsupervised settings.
TargetEncoder : Encodes categorical features using supervised signal
    in a classification or regression pipeline. This encoding is typically
    suitable for high cardinality categorical variables.
LabelEncoder : Encodes target labels with values between 0 and
    ``n_classes-1``.

Notes
-----
With a high proportion of `nan` values, inferring categories becomes slow with
Python versions before 3.10. The handling of `nan` values was improved
from Python 3.10 onwards, (c.f.
`bpo-43475 <https://github.com/python/cpython/issues/87641>`_).

Examples
--------
Given a dataset with two features, we let the encoder find the unique
values per feature and transform the data to an ordinal encoding.

>>> from sklearn.preprocessing import OrdinalEncoder
>>> enc = OrdinalEncoder()
>>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
>>> enc.fit(X)
OrdinalEncoder()
>>> enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> enc.transform([['Female', 3], ['Male', 1]])
array([[0., 2.],
       [1., 0.]])

>>> enc.inverse_transform([[1, 0], [0, 1]])
array([['Male', 1],
       ['Female', 2]], dtype=object)

By default, :class:`OrdinalEncoder` is lenient towards missing values by
propagating them.

>>> import numpy as np
>>> X = [['Male', 1], ['Female', 3], ['Female', np.nan]]
>>> enc.fit_transform(X)
array([[ 1.,  0.],
       [ 0.,  1.],
       [ 0., nan]])

You can use the parameter `encoded_missing_value` to encode missing values.

>>> enc.set_params(encoded_missing_value=-1).fit_transform(X)
array([[ 1.,  0.],
       [ 0.,  1.],
       [ 0., -1.]])

Infrequent categories are enabled by setting `max_categories` or `min_frequency`.
In the following example, "a" and "d" are considered infrequent and grouped
together into a single category, "b" and "c" are their own categories, unknown
values are encoded as 3 and missing values are encoded as 4.

>>> X_train = np.array(
...     [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3 + [np.nan]],
...     dtype=object).T
>>> enc = OrdinalEncoder(
...     handle_unknown="use_encoded_value", unknown_value=3,
...     max_categories=3, encoded_missing_value=4)
>>> _ = enc.fit(X_train)
>>> X_test = np.array([["a"], ["b"], ["c"], ["d"], ["e"], [np.nan]], dtype=object)
>>> enc.transform(X_test)
array([[2.],
       [0.],
       [1.],
       [2.],
       [3.],
       [4.]])

