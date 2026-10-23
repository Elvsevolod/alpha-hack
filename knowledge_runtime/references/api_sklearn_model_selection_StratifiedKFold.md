# sklearn.model_selection.StratifiedKFold

Version: 1.5.2

sklearn.model_selection.StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)

Stratified K-Fold cross-validator.

Provides train/test indices to split data in train/test sets.

This cross-validation object is a variation of KFold that returns
stratified folds. The folds are made by preserving the percentage of
samples for each class.

Read more in the :ref:`User Guide <stratified_k_fold>`.

For visualisation of cross-validation behaviour and
comparison between common scikit-learn split methods
refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

Parameters
----------
n_splits : int, default=5
    Number of folds. Must be at least 2.

    .. versionchanged:: 0.22
        ``n_splits`` default value changed from 3 to 5.

shuffle : bool, default=False
    Whether to shuffle each class's samples before splitting into batches.
    Note that the samples within each split will not be shuffled.

random_state : int, RandomState instance or None, default=None
    When `shuffle` is True, `random_state` affects the ordering of the
    indices, which controls the randomness of each fold for each class.
    Otherwise, leave `random_state` as `None`.
    Pass an int for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import StratifiedKFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([0, 0, 1, 1])
>>> skf = StratifiedKFold(n_splits=2)
>>> skf.get_n_splits(X, y)
2
>>> print(skf)
StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
>>> for i, (train_index, test_index) in enumerate(skf.split(X, y)):
...     print(f"Fold {i}:")
...     print(f"  Train: index={train_index}")
...     print(f"  Test:  index={test_index}")
Fold 0:
  Train: index=[1 3]
  Test:  index=[0 2]
Fold 1:
  Train: index=[0 2]
  Test:  index=[1 3]

Notes
-----
The implementation is designed to:

* Generate test sets such that all contain the same distribution of
  classes, or as close as possible.
* Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
  ``y = [1, 0]`` should not change the indices generated.
* Preserve order dependencies in the dataset ordering, when
  ``shuffle=False``: all samples from class k in some test set were
  contiguous in y, or separated in y by samples from classes other than k.
* Generate test sets where the smallest and largest differ by at most one
  sample.

.. versionchanged:: 0.22
    The previous implementation did not follow the last constraint.

See Also
--------
RepeatedStratifiedKFold : Repeats Stratified K-Fold n times.
