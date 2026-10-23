# optuna.trial.Trial.suggest_float
Version: 4.2.1

## optuna.trial.Trial.suggest_float(self, name: 'str', low: 'float', high: 'float', *, step: 'float | None' = None, log: 'bool' = False) -> 'float'

Suggest a value for the floating point parameter.

Example:

    Suggest a momentum, learning rate and scaling factor of learning rate
    for neural network training.

    .. testcode::

        import numpy as np
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier

        import optuna

        X, y = load_iris(return_X_y=True)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)


        def objective(trial):
            momentum = trial.suggest_float("momentum", 0.0, 1.0)
            learning_rate_init = trial.suggest_float(
                "learning_rate_init", 1e-5, 1e-3, log=True
            )
            power_t = trial.suggest_float("power_t", 0.2, 0.8, step=0.1)
            clf = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                momentum=momentum,
                learning_rate_init=learning_rate_init,
                solver="sgd",
                random_state=0,
                power_t=power_t,
            )
            clf.fit(X_train, y_train)

            return clf.score(X_valid, y_valid)


        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3)

Args:
    name:
        A parameter name.
    low:
        Lower endpoint of the range of suggested values. ``low`` is included in the range.
        ``low`` must be less than or equal to ``high``. If ``log`` is :obj:`True`,
        ``low`` must be larger than 0.
    high:
        Upper endpoint of the range of suggested values. ``high`` is included in the range.
        ``high`` must be greater than or equal to ``low``.
    step:
        A step of discretization.

        .. note::
            The ``step`` and ``log`` arguments cannot be used at the same time. To set
            the ``step`` argument to a float number, set the ``log`` argument to
            :obj:`False`.
    log:
        A flag to sample the value from the log domain or not.
        If ``log`` is true, the value is sampled from the range in the log domain.
        Otherwise, the value is sampled from the range in the linear domain.

        .. note::
            The ``step`` and ``log`` arguments cannot be used at the same time. To set
            the ``log`` argument to :obj:`True`, set the ``step`` argument to :obj:`None`.

Returns:
    A suggested float value.

.. seealso::
    :ref:`configurations` tutorial describes more details and flexible usages.
