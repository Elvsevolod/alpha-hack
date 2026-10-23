# imblearn.over_sampling.SMOTE

Version: 0.12.4

imblearn.over_sampling.SMOTE(*, sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=None)

Class to perform over-sampling using SMOTE.

This object is an implementation of SMOTE - Synthetic Minority
Over-sampling Technique as presented in [1]_.

Read more in the :ref:`User Guide <smote_adasyn>`.

Parameters
----------
sampling_strategy : float, str, dict or callable, default='auto'
    Sampling information to resample the data set.

    - When ``float``, it corresponds to the desired ratio of the number of
      samples in the minority class over the number of samples in the
      majority class after resampling. Therefore, the ratio is expressed as
      :math:`\alpha_{os} = N_{rm} / N_{M}` where :math:`N_{rm}` is the
      number of samples in the minority class after resampling and
      :math:`N_{M}` is the number of samples in the majority class.

        .. warning::
           ``float`` is only available for **binary** classification. An
           error is raised for multi-class classification.

    - When ``str``, specify the class targeted by the resampling. The
      number of samples in the different classes will be equalized.
      Possible choices are:

        ``'minority'``: resample only the minority class;

        ``'not minority'``: resample all classes but the minority class;

        ``'not majority'``: resample all classes but the majority class;

        ``'all'``: resample all classes;

        ``'auto'``: equivalent to ``'not majority'``.

    - When ``dict``, the keys correspond to the targeted classes. The
      values correspond to the desired number of samples for each targeted
      class.

    - When callable, function taking ``y`` and returns a ``dict``. The keys
      correspond to the targeted classes. The values correspond to the
      desired number of samples for each class.

random_state : int, RandomState instance, default=None
    Control the randomization of the algorithm.

    - If int, ``random_state`` is the seed used by the random number
      generator;
    - If ``RandomState`` instance, random_state is the random number
      generator;
    - If ``None``, the random number generator is the ``RandomState``
      instance used by ``np.random``.

k_neighbors : int or object, default=5
    The nearest neighbors used to define the neighborhood of samples to use
    to generate the synthetic samples. You can pass:

    - an `int` corresponding to the number of neighbors to use. A
      `~sklearn.neighbors.NearestNeighbors` instance will be fitted in this
      case.
    - an instance of a compatible nearest neighbors algorithm that should
      implement both methods `kneighbors` and `kneighbors_graph`. For
      instance, it could correspond to a
      :class:`~sklearn.neighbors.NearestNeighbors` but could be extended to
      any compatible class.

n_jobs : int, default=None
    Number of CPU cores used during the cross-validation loop.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See
    `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
    for more details.

    .. deprecated:: 0.10
       `n_jobs` has been deprecated in 0.10 and will be removed in 0.12.
       It was previously used to set `n_jobs` of nearest neighbors
       algorithm. From now on, you can pass an estimator where `n_jobs` is
       already set instead.

Attributes
----------
sampling_strategy_ : dict
    Dictionary containing the information to sample the dataset. The keys
    corresponds to the class labels from which to sample and the values
    are the number of samples to sample.

nn_k_ : estimator object
    Validated k-nearest neighbours created from the `k_neighbors` parameter.

n_features_in_ : int
    Number of features in the input dataset.

    .. versionadded:: 0.9

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during `fit`. Defined only when `X` has feature
    names that are all strings.

    .. versionadded:: 0.10

See Also
--------
SMOTENC : Over-sample using SMOTE for continuous and categorical features.

SMOTEN : Over-sample using the SMOTE variant specifically for categorical
    features only.

BorderlineSMOTE : Over-sample using the borderline-SMOTE variant.

SVMSMOTE : Over-sample using the SVM-SMOTE variant.

ADASYN : Over-sample using ADASYN.

KMeansSMOTE : Over-sample applying a clustering before to oversample using
    SMOTE.

Notes
-----
See the original papers: [1]_ for more details.

Supports multi-class resampling. A one-vs.-rest scheme is used as
originally proposed in [1]_.

References
----------
.. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
   synthetic minority over-sampling technique," Journal of artificial
   intelligence research, 321-357, 2002.

Examples
--------
>>> from collections import Counter
>>> from sklearn.datasets import make_classification
>>> from imblearn.over_sampling import SMOTE
>>> X, y = make_classification(n_classes=2, class_sep=2,
... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
>>> print('Original dataset shape %s' % Counter(y))
Original dataset shape Counter({1: 900, 0: 100})
>>> sm = SMOTE(random_state=42)
>>> X_res, y_res = sm.fit_resample(X, y)
>>> print('Resampled dataset shape %s' % Counter(y_res))
Resampled dataset shape Counter({0: 900, 1: 900})

