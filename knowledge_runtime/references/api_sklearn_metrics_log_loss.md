# sklearn.metrics.log_loss

Version: 1.5.2

sklearn.metrics.log_loss(y_true, y_pred, *, normalize=True, sample_weight=None, labels=None)

Log loss, aka logistic loss or cross-entropy loss.

This is the loss function used in (multinomial) logistic regression
and extensions of it such as neural networks, defined as the negative
log-likelihood of a logistic model that returns ``y_pred`` probabilities
for its training data ``y_true``.
The log loss is only defined for two or more labels.
For a single sample with true label :math:`y \in \{0,1\}` and
a probability estimate :math:`p = \operatorname{Pr}(y = 1)`, the log
loss is:

.. math::
    L_{\log}(y, p) = -(y \log (p) + (1 - y) \log (1 - p))

Read more in the :ref:`User Guide <log_loss>`.

Parameters
----------
y_true : array-like or label indicator matrix
    Ground truth (correct) labels for n_samples samples.

y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
    Predicted probabilities, as returned by a classifier's
    predict_proba method. If ``y_pred.shape = (n_samples,)``
    the probabilities provided are assumed to be that of the
    positive class. The labels in ``y_pred`` are assumed to be
    ordered alphabetically, as done by
    :class:`~sklearn.preprocessing.LabelBinarizer`.

    `y_pred` values are clipped to `[eps, 1-eps]` where `eps` is the machine
    precision for `y_pred`'s dtype.

normalize : bool, default=True
    If true, return the mean loss per sample.
    Otherwise, return the sum of the per-sample losses.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

labels : array-like, default=None
    If not provided, labels will be inferred from y_true. If ``labels``
    is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
    assumed to be binary and are inferred from ``y_true``.

    .. versionadded:: 0.18

Returns
-------
loss : float
    Log loss, aka logistic loss or cross-entropy loss.

Notes
-----
The logarithm used is the natural logarithm (base-e).

References
----------
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
p. 209.

Examples
--------
>>> from sklearn.metrics import log_loss
>>> log_loss(["spam", "ham", "ham", "spam"],
...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
0.21616...
