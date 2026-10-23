# imblearn.pipeline.Pipeline

Version: 0.12.4

imblearn.pipeline.Pipeline(steps, *, memory=None, verbose=False)

Pipeline of transforms and resamples with a final estimator.

Sequentially apply a list of transforms, sampling, and a final estimator.
Intermediate steps of the pipeline must be transformers or resamplers,
that is, they must implement fit, transform and sample methods.
The samplers are only applied during fit.
The final estimator only needs to implement fit.
The transformers and samplers in the pipeline can be cached using
``memory`` argument.

The purpose of the pipeline is to assemble several steps that can be
cross-validated together while setting different parameters.
For this, it enables setting parameters of the various steps using their
names and the parameter name separated by a '__', as in the example below.
A step's estimator may be replaced entirely by setting the parameter
with its name to another estimator, or a transformer removed by setting
it to 'passthrough' or ``None``.

Parameters
----------
steps : list
    List of (name, transform) tuples (implementing
    fit/transform/fit_resample) that are chained, in the order in which
    they are chained, with the last object an estimator.

memory : Instance of joblib.Memory or str, default=None
    Used to cache the fitted transformers of the pipeline. By default,
    no caching is performed. If a string is given, it is the path to
    the caching directory. Enabling caching triggers a clone of
    the transformers before fitting. Therefore, the transformer
    instance given to the pipeline cannot be inspected
    directly. Use the attribute ``named_steps`` or ``steps`` to
    inspect estimators within the pipeline. Caching the
    transformers is advantageous when fitting is time consuming.

verbose : bool, default=False
    If True, the time elapsed while fitting each step will be printed as it
    is completed.

Attributes
----------
named_steps : :class:`~sklearn.utils.Bunch`
    Read-only attribute to access any step parameter by user given name.
    Keys are step names and values are steps parameters.

classes_ : ndarray of shape (n_classes,)
    The classes labels.

n_features_in_ : int
    Number of features seen during first step `fit` method.

See Also
--------
make_pipeline : Helper function to make pipeline.

Notes
-----
See :ref:`sphx_glr_auto_examples_pipeline_plot_pipeline_classification.py`

.. warning::
   A surprising behaviour of the `imbalanced-learn` pipeline is that it
   breaks the `scikit-learn` contract where one expects
   `estimmator.fit_transform(X, y)` to be equivalent to
   `estimator.fit(X, y).transform(X)`.

   The semantic of `fit_resample` is to be applied only during the fit
   stage. Therefore, resampling will happen when calling `fit_transform`
   while it will only happen on the `fit` stage when calling `fit` and
   `transform` separately. Practically, `fit_transform` will lead to a
   resampled dataset while `fit` and `transform` will not.

Examples
--------
>>> from collections import Counter
>>> from sklearn.datasets import make_classification
>>> from sklearn.model_selection import train_test_split as tts
>>> from sklearn.decomposition import PCA
>>> from sklearn.neighbors import KNeighborsClassifier as KNN
>>> from sklearn.metrics import classification_report
>>> from imblearn.over_sampling import SMOTE
>>> from imblearn.pipeline import Pipeline
>>> X, y = make_classification(n_classes=2, class_sep=2,
... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
>>> print(f'Original dataset shape {Counter(y)}')
Original dataset shape Counter({1: 900, 0: 100})
>>> pca = PCA()
>>> smt = SMOTE(random_state=42)
>>> knn = KNN()
>>> pipeline = Pipeline([('smt', smt), ('pca', pca), ('knn', knn)])
>>> X_train, X_test, y_train, y_test = tts(X, y, random_state=42)
>>> pipeline.fit(X_train, y_train)
Pipeline(...)
>>> y_hat = pipeline.predict(X_test)
>>> print(classification_report(y_test, y_hat))
              precision    recall  f1-score   support
<BLANKLINE>
           0       0.87      1.00      0.93        26
           1       1.00      0.98      0.99       224
<BLANKLINE>
    accuracy                           0.98       250
   macro avg       0.93      0.99      0.96       250
weighted avg       0.99      0.98      0.98       250
<BLANKLINE>

