# optuna.trial.Trial.suggest_categorical
Version: 4.2.1

## optuna.trial.Trial.suggest_categorical(self, name: 'str', choices: 'Sequence[CategoricalChoiceType]') -> 'CategoricalChoiceType'

Suggest a value for the categorical parameter.

The value is sampled from ``choices``.

Example:

    Suggest a kernel function of `SVC <https://scikit-learn.org/stable/modules/generated/
    sklearn.svm.SVC.html>`__.

    .. testcode::

        import numpy as np
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC

        import optuna

        X, y = load_iris(return_X_y=True)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y)


        def objective(trial):
            kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
            clf = SVC(kernel=kernel, gamma="scale", random_state=0)
            clf.fit(X_train, y_train)
            return clf.score(X_valid, y_valid)


        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3)


Args:
    name:
        A parameter name.
    choices:
        Parameter value candidates.

.. seealso::
    :class:`~optuna.distributions.CategoricalDistribution`.

Returns:
    A suggested value.

.. seealso::
    :ref:`configurations` tutorial describes more details and flexible usages.
