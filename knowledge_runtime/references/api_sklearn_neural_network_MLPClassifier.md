# sklearn.neural_network.MLPClassifier

Version: 1.5.2

sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

Multi-layer Perceptron classifier.

This model optimizes the log-loss function using LBFGS or stochastic
gradient descent.

.. versionadded:: 0.18

Parameters
----------
hidden_layer_sizes : array-like of shape(n_layers - 2,), default=(100,)
    The ith element represents the number of neurons in the ith
    hidden layer.

activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
    Activation function for the hidden layer.

    - 'identity', no-op activation, useful to implement linear bottleneck,
      returns f(x) = x

    - 'logistic', the logistic sigmoid function,
      returns f(x) = 1 / (1 + exp(-x)).

    - 'tanh', the hyperbolic tan function,
      returns f(x) = tanh(x).

    - 'relu', the rectified linear unit function,
      returns f(x) = max(0, x)

solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
    The solver for weight optimization.

    - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

    - 'sgd' refers to stochastic gradient descent.

    - 'adam' refers to a stochastic gradient-based optimizer proposed
      by Kingma, Diederik, and Jimmy Ba

    For a comparison between Adam optimizer and SGD, see
    :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_training_curves.py`.

    Note: The default solver 'adam' works pretty well on relatively
    large datasets (with thousands of training samples or more) in terms of
    both training time and validation score.
    For small datasets, however, 'lbfgs' can converge faster and perform
    better.

alpha : float, default=0.0001
    Strength of the L2 regularization term. The L2 regularization term
    is divided by the sample size when added to the loss.

    For an example usage and visualization of varying regularization, see
    :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_alpha.py`.

batch_size : int, default='auto'
    Size of minibatches for stochastic optimizers.
    If the solver is 'lbfgs', the classifier will not use minibatch.
    When set to "auto", `batch_size=min(200, n_samples)`.

learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
    Learning rate schedule for weight updates.

    - 'constant' is a constant learning rate given by
      'learning_rate_init'.

    - 'invscaling' gradually decreases the learning rate at each
      time step 't' using an inverse scaling exponent of 'power_t'.
      effective_learning_rate = learning_rate_init / pow(t, power_t)

    - 'adaptive' keeps the learning rate constant to
      'learning_rate_init' as long as training loss keeps decreasing.
      Each time two consecutive epochs fail to decrease training loss by at
      least tol, or fail to increase validation score by at least tol if
      'early_stopping' is on, the current learning rate is divided by 5.

    Only used when ``solver='sgd'``.

learning_rate_init : float, default=0.001
    The initial learning rate used. It controls the step-size
    in updating the weights. Only used when solver='sgd' or 'adam'.

power_t : float, default=0.5
    The exponent for inverse scaling learning rate.
    It is used in updating effective learning rate when the learning_rate
    is set to 'invscaling'. Only used when solver='sgd'.

max_iter : int, default=200
    Maximum number of iterations. The solver iterates until convergence
    (determined by 'tol') or this number of iterations. For stochastic
    solvers ('sgd', 'adam'), note that this determines the number of epochs
    (how many times each data point will be used), not the number of
    gradient steps.

shuffle : bool, default=True
    Whether to shuffle samples in each iteration. Only used when
    solver='sgd' or 'adam'.

random_state : int, RandomState instance, default=None
    Determines random number generation for weights and bias
    initialization, train-test split if early stopping is used, and batch
    sampling when solver='sgd' or 'adam'.
    Pass an int for reproducible results across multiple function calls.
    See :term:`Glossary <random_state>`.

tol : float, default=1e-4
    Tolerance for the optimization. When the loss or score is not improving
    by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
    unless ``learning_rate`` is set to 'adaptive', convergence is
    considered to be reached and training stops.

verbose : bool, default=False
    Whether to print progress messages to stdout.

warm_start : bool, default=False
    When set to True, reuse the solution of the previous
    call to fit as initialization, otherwise, just erase the
    previous solution. See :term:`the Glossary <warm_start>`.

momentum : float, default=0.9
    Momentum for gradient descent update. Should be between 0 and 1. Only
    used when solver='sgd'.

nesterovs_momentum : bool, default=True
    Whether to use Nesterov's momentum. Only used when solver='sgd' and
    momentum > 0.

early_stopping : bool, default=False
    Whether to use early stopping to terminate training when validation
    score is not improving. If set to true, it will automatically set
    aside 10% of training data as validation and terminate training when
    validation score is not improving by at least ``tol`` for
    ``n_iter_no_change`` consecutive epochs. The split is stratified,
    except in a multilabel setting.
    If early stopping is False, then the training stops when the training
    loss does not improve by more than tol for n_iter_no_change consecutive
    passes over the training set.
    Only effective when solver='sgd' or 'adam'.

validation_fraction : float, default=0.1
    The proportion of training data to set aside as validation set for
    early stopping. Must be between 0 and 1.
    Only used if early_stopping is True.

beta_1 : float, default=0.9
    Exponential decay rate for estimates of first moment vector in adam,
    should be in [0, 1). Only used when solver='adam'.

beta_2 : float, default=0.999
    Exponential decay rate for estimates of second moment vector in adam,
    should be in [0, 1). Only used when solver='adam'.

epsilon : float, default=1e-8
    Value for numerical stability in adam. Only used when solver='adam'.

n_iter_no_change : int, default=10
    Maximum number of epochs to not meet ``tol`` improvement.
    Only effective when solver='sgd' or 'adam'.

    .. versionadded:: 0.20

max_fun : int, default=15000
    Only used when solver='lbfgs'. Maximum number of loss function calls.
    The solver iterates until convergence (determined by 'tol'), number
    of iterations reaches max_iter, or this number of loss function calls.
    Note that number of loss function calls will be greater than or equal
    to the number of iterations for the `MLPClassifier`.

    .. versionadded:: 0.22

Attributes
----------
classes_ : ndarray or list of ndarray of shape (n_classes,)
    Class labels for each output.

loss_ : float
    The current loss computed with the loss function.

best_loss_ : float or None
    The minimum loss reached by the solver throughout fitting.
    If `early_stopping=True`, this attribute is set to `None`. Refer to
    the `best_validation_score_` fitted attribute instead.

loss_curve_ : list of shape (`n_iter_`,)
    The ith element in the list represents the loss at the ith iteration.

validation_scores_ : list of shape (`n_iter_`,) or None
    The score at each iteration on a held-out validation set. The score
    reported is the accuracy score. Only available if `early_stopping=True`,
    otherwise the attribute is set to `None`.

best_validation_score_ : float or None
    The best validation score (i.e. accuracy score) that triggered the
    early stopping. Only available if `early_stopping=True`, otherwise the
    attribute is set to `None`.

t_ : int
    The number of training samples seen by the solver during fitting.

coefs_ : list of shape (n_layers - 1,)
    The ith element in the list represents the weight matrix corresponding
    to layer i.

intercepts_ : list of shape (n_layers - 1,)
    The ith element in the list represents the bias vector corresponding to
    layer i + 1.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

n_iter_ : int
    The number of iterations the solver has run.

n_layers_ : int
    Number of layers.

n_outputs_ : int
    Number of outputs.

out_activation_ : str
    Name of the output activation function.

See Also
--------
MLPRegressor : Multi-layer Perceptron regressor.
BernoulliRBM : Bernoulli Restricted Boltzmann Machine (RBM).

Notes
-----
MLPClassifier trains iteratively since at each time step
the partial derivatives of the loss function with respect to the model
parameters are computed to update the parameters.

It can also have a regularization term added to the loss function
that shrinks model parameters to prevent overfitting.

This implementation works with data represented as dense numpy arrays or
sparse scipy arrays of floating point values.

References
----------
Hinton, Geoffrey E. "Connectionist learning procedures."
Artificial intelligence 40.1 (1989): 185-234.

Glorot, Xavier, and Yoshua Bengio.
"Understanding the difficulty of training deep feedforward neural networks."
International Conference on Artificial Intelligence and Statistics. 2010.

:arxiv:`He, Kaiming, et al (2015). "Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification." <1502.01852>`

:arxiv:`Kingma, Diederik, and Jimmy Ba (2014)
"Adam: A method for stochastic optimization." <1412.6980>`

Examples
--------
>>> from sklearn.neural_network import MLPClassifier
>>> from sklearn.datasets import make_classification
>>> from sklearn.model_selection import train_test_split
>>> X, y = make_classification(n_samples=100, random_state=1)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
...                                                     random_state=1)
>>> clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
>>> clf.predict_proba(X_test[:1])
array([[0.038..., 0.961...]])
>>> clf.predict(X_test[:5, :])
array([1, 0, 1, 0, 1])
>>> clf.score(X_test, y_test)
0.8...

