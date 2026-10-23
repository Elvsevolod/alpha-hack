# sklearn.calibration.CalibratedClassifierCV

Version: 1.5.2

sklearn.calibration.CalibratedClassifierCV(estimator=None, *, method='sigmoid', cv=None, n_jobs=None, ensemble=True)

Probability calibration with isotonic regression or logistic regression.

This class uses cross-validation to both estimate the parameters of a
classifier and subsequently calibrate a classifier. With default
`ensemble=True`, for each cv split it
fits a copy of the base estimator to the training subset, and calibrates it
using the testing subset. For prediction, predicted probabilities are
averaged across these individual calibrated classifiers. When
`ensemble=False`, cross-validation is used to obtain unbiased predictions,
via :func:`~sklearn.model_selection.cross_val_predict`, which are then
used for calibration. For prediction, the base estimator, trained using all
the data, is used. This is the prediction method implemented when
`probabilities=True` for :class:`~sklearn.svm.SVC` and :class:`~sklearn.svm.NuSVC`
estimators (see :ref:`User Guide <scores_probabilities>` for details).

Already fitted classifiers can be calibrated via the parameter
`cv="prefit"`. In this case, no cross-validation is used and all provided
data is used for calibration. The user has to take care manually that data
for model fitting and calibration are disjoint.

The calibration is based on the :term:`decision_function` method of the
`estimator` if it exists, else on :term:`predict_proba`.

Read more in the :ref:`User Guide <calibration>`.
In order to learn more on the CalibratedClassifierCV class, see the
following calibration examples:
:ref:`sphx_glr_auto_examples_calibration_plot_calibration.py`,
:ref:`sphx_glr_auto_examples_calibration_plot_calibration_curve.py`, and
:ref:`sphx_glr_auto_examples_calibration_plot_calibration_multiclass.py`.

Parameters
----------
estimator : estimator instance, default=None
    The classifier whose output need to be calibrated to provide more
    accurate `predict_proba` outputs. The default classifier is
    a :class:`~sklearn.svm.LinearSVC`.

    .. versionadded:: 1.2

method : {'sigmoid', 'isotonic'}, default='sigmoid'
    The method to use for calibration. Can be 'sigmoid' which
    corresponds to Platt's method (i.e. a logistic regression model) or
    'isotonic' which is a non-parametric approach. It is not advised to
    use isotonic calibration with too few calibration samples
    ``(<<1000)`` since it tends to overfit.

cv : int, cross-validation generator, iterable or "prefit",             default=None
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:

    - None, to use the default 5-fold cross-validation,
    - integer, to specify the number of folds.
    - :term:`CV splitter`,
    - An iterable yielding (train, test) splits as arrays of indices.

    For integer/None inputs, if ``y`` is binary or multiclass,
    :class:`~sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
    neither binary nor multiclass, :class:`~sklearn.model_selection.KFold`
    is used.

    Refer to the :ref:`User Guide <cross_validation>` for the various
    cross-validation strategies that can be used here.

    If "prefit" is passed, it is assumed that `estimator` has been
    fitted already and all data is used for calibration.

    .. versionchanged:: 0.22
        ``cv`` default value if None changed from 3-fold to 5-fold.

n_jobs : int, default=None
    Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors.

    Base estimator clones are fitted in parallel across cross-validation
    iterations. Therefore parallelism happens only when `cv != "prefit"`.

    See :term:`Glossary <n_jobs>` for more details.

    .. versionadded:: 0.24

ensemble : bool, default=True
    Determines how the calibrator is fitted when `cv` is not `'prefit'`.
    Ignored if `cv='prefit'`.

    If `True`, the `estimator` is fitted using training data, and
    calibrated using testing data, for each `cv` fold. The final estimator
    is an ensemble of `n_cv` fitted classifier and calibrator pairs, where
    `n_cv` is the number of cross-validation folds. The output is the
    average predicted probabilities of all pairs.

    If `False`, `cv` is used to compute unbiased predictions, via
    :func:`~sklearn.model_selection.cross_val_predict`, which are then
    used for calibration. At prediction time, the classifier used is the
    `estimator` trained on all the data.
    Note that this method is also internally implemented  in
    :mod:`sklearn.svm` estimators with the `probabilities=True` parameter.

    .. versionadded:: 0.24

Attributes
----------
classes_ : ndarray of shape (n_classes,)
    The class labels.

n_features_in_ : int
    Number of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Only defined if the
    underlying estimator exposes such an attribute when fit.

    .. versionadded:: 1.0

calibrated_classifiers_ : list (len() equal to cv or 1 if `cv="prefit"`             or `ensemble=False`)
    The list of classifier and calibrator pairs.

    - When `cv="prefit"`, the fitted `estimator` and fitted
      calibrator.
    - When `cv` is not "prefit" and `ensemble=True`, `n_cv` fitted
      `estimator` and calibrator pairs. `n_cv` is the number of
      cross-validation folds.
    - When `cv` is not "prefit" and `ensemble=False`, the `estimator`,
      fitted on all the data, and fitted calibrator.

    .. versionchanged:: 0.24
        Single calibrated classifier case when `ensemble=False`.

See Also
--------
calibration_curve : Compute true and predicted probabilities
    for a calibration curve.

References
----------
.. [1] Obtaining calibrated probability estimates from decision trees
       and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

.. [2] Transforming Classifier Scores into Accurate Multiclass
       Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

.. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
       Regularized Likelihood Methods, J. Platt, (1999)

.. [4] Predicting Good Probabilities with Supervised Learning,
       A. Niculescu-Mizil & R. Caruana, ICML 2005

Examples
--------
>>> from sklearn.datasets import make_classification
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.calibration import CalibratedClassifierCV
>>> X, y = make_classification(n_samples=100, n_features=2,
...                            n_redundant=0, random_state=42)
>>> base_clf = GaussianNB()
>>> calibrated_clf = CalibratedClassifierCV(base_clf, cv=3)
>>> calibrated_clf.fit(X, y)
CalibratedClassifierCV(...)
>>> len(calibrated_clf.calibrated_classifiers_)
3
>>> calibrated_clf.predict_proba(X)[:5, :]
array([[0.110..., 0.889...],
       [0.072..., 0.927...],
       [0.928..., 0.071...],
       [0.928..., 0.071...],
       [0.071..., 0.928...]])
>>> from sklearn.model_selection import train_test_split
>>> X, y = make_classification(n_samples=100, n_features=2,
...                            n_redundant=0, random_state=42)
>>> X_train, X_calib, y_train, y_calib = train_test_split(
...        X, y, random_state=42
... )
>>> base_clf = GaussianNB()
>>> base_clf.fit(X_train, y_train)
GaussianNB()
>>> calibrated_clf = CalibratedClassifierCV(base_clf, cv="prefit")
>>> calibrated_clf.fit(X_calib, y_calib)
CalibratedClassifierCV(...)
>>> len(calibrated_clf.calibrated_classifiers_)
1
>>> calibrated_clf.predict_proba([[-0.5, 0.5]])
array([[0.936..., 0.063...]])

