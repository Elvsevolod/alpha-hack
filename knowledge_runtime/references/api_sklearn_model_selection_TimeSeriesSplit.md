# sklearn.model_selection.TimeSeriesSplit

Version: 1.5.2

sklearn.model_selection.TimeSeriesSplit(n_splits=5, *, max_train_size=None, test_size=None, gap=0)

Time Series cross-validator.

Provides train/test indices to split time series data samples
that are observed at fixed time intervals, in train/test sets.
In each split, test indices must be higher than before, and thus shuffling
in cross validator is inappropriate.

This cross-validation object is a variation of :class:`KFold`.
In the kth split, it returns first k folds as train set and the
(k+1)th fold as test set.

Note that unlike standard cross-validation methods, successive
training sets are supersets of those that come before them.

Read more in the :ref:`User Guide <time_series_split>`.

For visualisation of cross-validation behaviour and
comparison between common scikit-learn split methods
refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

.. versionadded:: 0.18

Parameters
----------
n_splits : int, default=5
    Number of splits. Must be at least 2.

    .. versionchanged:: 0.22
        ``n_splits`` default value changed from 3 to 5.

max_train_size : int, default=None
    Maximum size for a single training set.

test_size : int, default=None
    Used to limit the size of the test set. Defaults to
    ``n_samples // (n_splits + 1)``, which is the maximum allowed value
    with ``gap=0``.

    .. versionadded:: 0.24

gap : int, default=0
    Number of samples to exclude from the end of each train set before
    the test set.

    .. versionadded:: 0.24

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import TimeSeriesSplit
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> tscv = TimeSeriesSplit()
>>> print(tscv)
TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
>>> for i, (train_index, test_index) in enumerate(tscv.split(X)):
...     print(f"Fold {i}:")
...     print(f"  Train: index={train_index}")
...     print(f"  Test:  index={test_index}")
Fold 0:
  Train: index=[0]
  Test:  index=[1]
Fold 1:
  Train: index=[0 1]
  Test:  index=[2]
Fold 2:
  Train: index=[0 1 2]
  Test:  index=[3]
Fold 3:
  Train: index=[0 1 2 3]
  Test:  index=[4]
Fold 4:
  Train: index=[0 1 2 3 4]
  Test:  index=[5]
>>> # Fix test_size to 2 with 12 samples
>>> X = np.random.randn(12, 2)
>>> y = np.random.randint(0, 2, 12)
>>> tscv = TimeSeriesSplit(n_splits=3, test_size=2)
>>> for i, (train_index, test_index) in enumerate(tscv.split(X)):
...     print(f"Fold {i}:")
...     print(f"  Train: index={train_index}")
...     print(f"  Test:  index={test_index}")
Fold 0:
  Train: index=[0 1 2 3 4 5]
  Test:  index=[6 7]
Fold 1:
  Train: index=[0 1 2 3 4 5 6 7]
  Test:  index=[8 9]
Fold 2:
  Train: index=[0 1 2 3 4 5 6 7 8 9]
  Test:  index=[10 11]
>>> # Add in a 2 period gap
>>> tscv = TimeSeriesSplit(n_splits=3, test_size=2, gap=2)
>>> for i, (train_index, test_index) in enumerate(tscv.split(X)):
...     print(f"Fold {i}:")
...     print(f"  Train: index={train_index}")
...     print(f"  Test:  index={test_index}")
Fold 0:
  Train: index=[0 1 2 3]
  Test:  index=[6 7]
Fold 1:
  Train: index=[0 1 2 3 4 5]
  Test:  index=[8 9]
Fold 2:
  Train: index=[0 1 2 3 4 5 6 7]
  Test:  index=[10 11]

For a more extended example see
:ref:`sphx_glr_auto_examples_applications_plot_cyclical_feature_engineering.py`.

Notes
-----
The training set has size ``i * n_samples // (n_splits + 1)
+ n_samples % (n_splits + 1)`` in the ``i`` th split,
with a test set of size ``n_samples//(n_splits + 1)`` by default,
where ``n_samples`` is the number of samples. Note that this
formula is only valid when ``test_size`` and ``max_train_size`` are
left to their default values.
