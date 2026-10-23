# sklearn.model_selection.TunedThresholdClassifierCV

Version: 1.5.2

sklearn.model_selection.TunedThresholdClassifierCV(estimator, *, scoring='balanced_accuracy', response_method='auto', thresholds=100, cv=None, refit=True, n_jobs=None, random_state=None, store_cv_results=False)

Classifier that post-tunes the decision threshold using cross-validation.

This estimator post-tunes the decision threshold (cut-off point) that is
used for converting posterior probability estimates (i.e. output of
`predict_proba`) or decision scores (i.e. output of `decision_function`)
into a class label. The tuning is done by optimizing a binary metric,
potentially constrained by a another metric.

Read more in the :ref:`User Guide <TunedThresholdClassifierCV>`.

.. versionadded:: 1.5

Parameters
----------
estimator : estimator instance
    The classifier, fitted or not, for which we want to optimize
    the decision threshold used during `predict`.

scoring : str or callable, default="balanced_accuracy"
    The objective metric to be optimized. Can be one of:

    * a string associated to a scoring function for binary classification
      (see model evaluation documentation);
    * a scorer callable object created with :func:`~sklearn.metrics.make_scorer`;

response_method : {"auto", "decision_function", "predict_proba"}, default="auto"
    Methods by the classifier `estimator` corresponding to the
    decision function for which we want to find a threshold. It can be:

    * if `"auto"`, it will try to invoke, for each classifier,
      `"predict_proba"` or `"decision_function"` in that order.
    * otherwise, one of `"predict_proba"` or `"decision_function"`.
      If the method is not implemented by the classifier, it will raise an
      error.

thresholds : int or array-like, default=100
    The number of decision threshold to use when discretizing the output of the
    classifier `method`. Pass an array-like to manually specify the thresholds
    to use.

cv : int, float, cross-validation generator, iterable or "prefit", default=None
    Determines the cross-validation splitting strategy to train classifier.
    Possible inputs for cv are:

    * `None`, to use the default 5-fold stratified K-fold cross validation;
    * An integer number, to specify the number of folds in a stratified k-fold;
    * A float number, to specify a single shuffle split. The floating number should
      be in (0, 1) and represent the size of the validation set;
    * An object to be used as a cross-validation generator;
    * An iterable yielding train, test splits;
    * `"prefit"`, to bypass the cross-validation.

    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    .. warning::
        Using `cv="prefit"` and passing the same dataset for fitting `estimator`
        and tuning the cut-off point is subject to undesired overfitting. You can
        refer to :ref:`TunedThresholdClassifierCV_no_cv` for an example.

        This option should only be used when the set used to fit `estimator` is
        different from the one used to tune the cut-off point (by calling
        :meth:`TunedThresholdClassifierCV.fit`).

refit : bool, default=True
    Whether or not to refit the classifier on the entire training set once
    the decision threshold has been found.
    Note that forcing `refit=False` on cross-validation having more
    than a single split will raise an error. Similarly, `refit=True` in
    conjunction with `cv="prefit"` will raise an error.

n_jobs : int, default=None
    The number of jobs to run in parallel. When `cv` represents a
    cross-validation strategy, the fitting and scoring on each data split
    is done in parallel. ``None`` means 1 unless in a
    :obj:`joblib.parallel_backend` context. ``-1`` means using all
    processors. See :term:`Glossary <n_jobs>` for more details.

random_state : int, RandomState instance or None, default=None
    Controls the randomness of cross-validation when `cv` is a float.
    See :term:`Glossary <random_state>`.

store_cv_results : bool, default=False
    Whether to store all scores and thresholds computed during the cross-validation
    process.

Attributes
----------
estimator_ : estimator instance
    The fitted classifier used when predicting.

best_threshold_ : float
    The new decision threshold.

best_score_ : float or None
    The optimal score of the objective metric, evaluated at `best_threshold_`.

cv_results_ : dict or None
    A dictionary containing the scores and thresholds computed during the
    cross-validation process. Only exist if `store_cv_results=True`. The
    keys are `"thresholds"` and `"scores"`.

classes_ : ndarray of shape (n_classes,)
    The class labels.

n_features_in_ : int
    Number of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.

See Also
--------
sklearn.model_selection.FixedThresholdClassifier : Classifier that uses a
    constant threshold.
sklearn.calibration.CalibratedClassifierCV : Estimator that calibrates
    probabilities.

Examples
--------
>>> from sklearn.datasets import make_classification
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.metrics import classification_report
>>> from sklearn.model_selection import TunedThresholdClassifierCV, train_test_split
>>> X, y = make_classification(
...     n_samples=1_000, weights=[0.9, 0.1], class_sep=0.8, random_state=42
... )
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, stratify=y, random_state=42
... )
>>> classifier = RandomForestClassifier(random_state=0).fit(X_train, y_train)
>>> print(classification_report(y_test, classifier.predict(X_test)))
              precision    recall  f1-score   support
<BLANKLINE>
           0       0.94      0.99      0.96       224
           1       0.80      0.46      0.59        26
<BLANKLINE>
    accuracy                           0.93       250
   macro avg       0.87      0.72      0.77       250
weighted avg       0.93      0.93      0.92       250
<BLANKLINE>
>>> classifier_tuned = TunedThresholdClassifierCV(
...     classifier, scoring="balanced_accuracy"
... ).fit(X_train, y_train)
>>> print(
...     f"Cut-off point found at {classifier_tuned.best_threshold_:.3f}"
... )
Cut-off point found at 0.342
>>> print(classification_report(y_test, classifier_tuned.predict(X_test)))
              precision    recall  f1-score   support
<BLANKLINE>
           0       0.96      0.95      0.96       224
           1       0.61      0.65      0.63        26
<BLANKLINE>
    accuracy                           0.92       250
   macro avg       0.78      0.80      0.79       250
weighted avg       0.92      0.92      0.92       250
<BLANKLINE>

