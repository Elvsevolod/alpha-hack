# optuna.trial.Trial.suggest_int
Version: 4.2.1

## optuna.trial.Trial.suggest_int(self, name: 'str', low: 'int', high: 'int', *, step: 'int' = 1, log: 'bool' = False) -> 'int'

Suggest a value for the integer parameter.

The value is sampled from the integers in :math:`[\mathsf{low}, \mathsf{high}]`.

Example:

    Suggest the number of trees in `RandomForestClassifier <https://scikit-learn.org/
    stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`__.

    .. testcode::

        import numpy as np
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        import optuna

        X, y = load_iris(return_X_y=True)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y)


        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 400)
            clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
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
            Note that :math:`\mathsf{high}` is modified if the range is not divisible by
            :math:`\mathsf{step}`. Please check the warning messages to find the changed
            values.

        .. note::
            The method returns one of the values in the sequence
            :math:`\mathsf{low}, \mathsf{low} + \mathsf{step}, \mathsf{low} + 2 *
            \mathsf{step}, \dots, \mathsf{low} + k * \mathsf{step} \le
            \mathsf{high}`, where :math:`k` denotes an integer.

        .. note::
            The ``step != 1`` and ``log`` arguments cannot be used at the same time.
            To set the ``step`` argument :math:`\mathsf{step} \ge 2`, set the
            ``log`` argument to :obj:`False`.
    log:
        A flag to sample the value from the log domain or not.

        .. note::
            If ``log`` is true, at first, the range of suggested values is divided into
            grid points of width 1. The range of suggested values is then converted to
            a log domain, from which a value is sampled. The uniformly sampled
            value is re-converted to the original domain and rounded to the nearest grid
            point that we just split, and the suggested value is determined.
            For example, if `low = 2` and `high = 8`, then the range of suggested values is
            `[2, 3, 4, 5, 6, 7, 8]` and lower values tend to be more sampled than higher
            values.

        .. note::
            The ``step != 1`` and ``log`` arguments cannot be used at the same time.
            To set the ``log`` argument to :obj:`True`, set the ``step`` argument to 1.

.. seealso::
    :ref:`configurations` tutorial describes more details and flexible usages.
