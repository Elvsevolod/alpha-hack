# sklearn.ensemble.HistGradientBoostingClassifier

Version: 1.5.2

sklearn.ensemble.HistGradientBoostingClassifier(loss='log_loss', *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_features=1.0, max_bins=255, categorical_features='warn', monotonic_cst=None, interaction_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None, class_weight=None)

Histogram-based Gradient Boosting Classification Tree.

This estimator is much faster than
:class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`
for big datasets (n_samples >= 10 000).

This estimator has native support for missing values (NaNs). During
training, the tree grower learns at each split point whether samples
with missing values should go to the left or right child, based on the
potential gain. When predicting, samples with missing values are
assigned to the left or right child consequently. If no missing values
were encountered for a given feature during training, then samples with
missing values are mapped to whichever child has the most samples.

This implementation is inspired by
`LightGBM <https://github.com/Microsoft/LightGBM>`_.

Read more in the :ref:`User Guide <histogram_based_gradient_boosting>`.

.. versionadded:: 0.21

Parameters
----------
loss : {'log_loss'}, default='log_loss'
    The loss function to use in the boosting process.

    For binary classification problems, 'log_loss' is also known as logistic loss,
    binomial deviance or binary crossentropy. Internally, the model fits one tree
    per boosting iteration and uses the logistic sigmoid function (expit) as
    inverse link function to compute the predicted positive class probability.

    For multiclass classification problems, 'log_loss' is also known as multinomial
    deviance or categorical crossentropy. Internally, the model fits one tree per
    boosting iteration and per class and uses the softmax function as inverse link
    function to compute the predicted probabilities of the classes.

learning_rate : float, default=0.1
    The learning rate, also known as *shrinkage*. This is used as a
    multiplicative factor for the leaves values. Use ``1`` for no
    shrinkage.
max_iter : int, default=100
    The maximum number of iterations of the boosting process, i.e. the
    maximum number of trees for binary classification. For multiclass
    classification, `n_classes` trees per iteration are built.
max_leaf_nodes : int or None, default=31
    The maximum number of leaves for each tree. Must be strictly greater
    than 1. If None, there is no maximum limit.
max_depth : int or None, default=None
    The maximum depth of each tree. The depth of a tree is the number of
    edges to go from the root to the deepest leaf.
    Depth isn't constrained by default.
min_samples_leaf : int, default=20
    The minimum number of samples per leaf. For small datasets with less
    than a few hundred samples, it is recommended to lower this value
    since only very shallow trees would be built.
l2_regularization : float, default=0
    The L2 regularization parameter penalizing leaves with small hessians.
    Use ``0`` for no regularization (default).
max_features : float, default=1.0
    Proportion of randomly chosen features in each and every node split.
    This is a form of regularization, smaller values make the trees weaker
    learners and might prevent overfitting.
    If interaction constraints from `interaction_cst` are present, only allowed
    features are taken into account for the subsampling.

    .. versionadded:: 1.4

max_bins : int, default=255
    The maximum number of bins to use for non-missing values. Before
    training, each feature of the input array `X` is binned into
    integer-valued bins, which allows for a much faster training stage.
    Features with a small number of unique values may use less than
    ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
    is always reserved for missing values. Must be no larger than 255.
categorical_features : array-like of {bool, int, str} of shape (n_features)             or shape (n_categorical_features,), default=None
    Indicates the categorical features.

    - None : no feature will be considered categorical.
    - boolean array-like : boolean mask indicating categorical features.
    - integer array-like : integer indices indicating categorical
      features.
    - str array-like: names of categorical features (assuming the training
      data has feature names).
    - `"from_dtype"`: dataframe columns with dtype "category" are
      considered to be categorical features. The input must be an object
      exposing a ``__dataframe__`` method such as pandas or polars
      DataFrames to use this feature.

    For each categorical feature, there must be at most `max_bins` unique
    categories. Negative values for categorical features encoded as numeric
    dtypes are treated as missing values. All categorical values are
    converted to floating point numbers. This means that categorical values
    of 1.0 and 1 are treated as the same category.

    Read more in the :ref:`User Guide <categorical_support_gbdt>`.

    .. versionadded:: 0.24

    .. versionchanged:: 1.2
       Added support for feature names.

    .. versionchanged:: 1.4
       Added `"from_dtype"` option. The default will change to `"from_dtype"` in
       v1.6.

monotonic_cst : array-like of int of shape (n_features) or dict, default=None
    Monotonic constraint to enforce on each feature are specified using the
    following integer values:

    - 1: monotonic increase
    - 0: no constraint
    - -1: monotonic decrease

    If a dict with str keys, map feature to monotonic constraints by name.
    If an array, the features are mapped to constraints by position. See
    :ref:`monotonic_cst_features_names` for a usage example.

    The constraints are only valid for binary classifications and hold
    over the probability of the positive class.
    Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.

    .. versionadded:: 0.23

    .. versionchanged:: 1.2
       Accept dict of constraints with feature names as keys.

interaction_cst : {"pairwise", "no_interactions"} or sequence of lists/tuples/sets             of int, default=None
    Specify interaction constraints, the sets of features which can
    interact with each other in child node splits.

    Each item specifies the set of feature indices that are allowed
    to interact with each other. If there are more features than
    specified in these constraints, they are treated as if they were
    specified as an additional set.

    The strings "pairwise" and "no_interactions" are shorthands for
    allowing only pairwise or no interactions, respectively.

    For instance, with 5 features in total, `interaction_cst=[{0, 1}]`
    is equivalent to `interaction_cst=[{0, 1}, {2, 3, 4}]`,
    and specifies that each branch of a tree will either only split
    on features 0 and 1 or only split on features 2, 3 and 4.

    .. versionadded:: 1.2

warm_start : bool, default=False
    When set to ``True``, reuse the solution of the previous call to fit
    and add more estimators to the ensemble. For results to be valid, the
    estimator should be re-trained on the same data only.
    See :term:`the Glossary <warm_start>`.
early_stopping : 'auto' or bool, default='auto'
    If 'auto', early stopping is enabled if the sample size is larger than
    10000. If True, early stopping is enabled, otherwise early stopping is
    disabled.

    .. versionadded:: 0.23

scoring : str or callable or None, default='loss'
    Scoring parameter to use for early stopping. It can be a single
    string (see :ref:`scoring_parameter`) or a callable (see
    :ref:`scoring`). If None, the estimator's default scorer
    is used. If ``scoring='loss'``, early stopping is checked
    w.r.t the loss value. Only used if early stopping is performed.
validation_fraction : int or float or None, default=0.1
    Proportion (or absolute size) of training data to set aside as
    validation data for early stopping. If None, early stopping is done on
    the training data. Only used if early stopping is performed.
n_iter_no_change : int, default=10
    Used to determine when to "early stop". The fitting process is
    stopped when none of the last ``n_iter_no_change`` scores are better
    than the ``n_iter_no_change - 1`` -th-to-last one, up to some
    tolerance. Only used if early stopping is performed.
tol : float, default=1e-7
    The absolute tolerance to use when comparing scores. The higher the
    tolerance, the more likely we are to early stop: higher tolerance
    means that it will be harder for subsequent iterations to be
    considered an improvement upon the reference score.
verbose : int, default=0
    The verbosity level. If not zero, print some information about the
    fitting process.
random_state : int, RandomState instance or None, default=None
    Pseudo-random number generator to control the subsampling in the
    binning process, and the train/validation data split if early stopping
    is enabled.
    Pass an int for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.
class_weight : dict or 'balanced', default=None
    Weights associated with classes in the form `{class_label: weight}`.
    If not given, all classes are supposed to have weight one.
    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as `n_samples / (n_classes * np.bincount(y))`.
    Note that these weights will be multiplied with sample_weight (passed
    through the fit method) if `sample_weight` is specified.

    .. versionadded:: 1.2

Attributes
----------
classes_ : array, shape = (n_classes,)
    Class labels.
do_early_stopping_ : bool
    Indicates whether early stopping is used during training.
n_iter_ : int
    The number of iterations as selected by early stopping, depending on
    the `early_stopping` parameter. Otherwise it corresponds to max_iter.
n_trees_per_iteration_ : int
    The number of tree that are built at each iteration. This is equal to 1
    for binary classification, and to ``n_classes`` for multiclass
    classification.
train_score_ : ndarray, shape (n_iter_+1,)
    The scores at each iteration on the training data. The first entry
    is the score of the ensemble before the first iteration. Scores are
    computed according to the ``scoring`` parameter. If ``scoring`` is
    not 'loss', scores are computed on a subset of at most 10 000
    samples. Empty if no early stopping.
validation_score_ : ndarray, shape (n_iter_+1,)
    The scores at each iteration on the held-out validation data. The
    first entry is the score of the ensemble before the first iteration.
    Scores are computed according to the ``scoring`` parameter. Empty if
    no early stopping or if ``validation_fraction`` is None.
is_categorical_ : ndarray, shape (n_features, ) or None
    Boolean mask for the categorical features. ``None`` if there are no
    categorical features.
n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24
feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

See Also
--------
GradientBoostingClassifier : Exact gradient boosting method that does not
    scale as good on datasets with a large number of samples.
sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
RandomForestClassifier : A meta-estimator that fits a number of decision
    tree classifiers on various sub-samples of the dataset and uses
    averaging to improve the predictive accuracy and control over-fitting.
AdaBoostClassifier : A meta-estimator that begins by fitting a classifier
    on the original dataset and then fits additional copies of the
    classifier on the same dataset where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers
    focus more on difficult cases.

Examples
--------
>>> from sklearn.ensemble import HistGradientBoostingClassifier
>>> from sklearn.datasets import load_iris
>>> X, y = load_iris(return_X_y=True)
>>> clf = HistGradientBoostingClassifier().fit(X, y)
>>> clf.score(X, y)
1.0

