# optuna.create_study

Version: 4.2.1

optuna.create_study(*, storage: 'str | storages.BaseStorage | None' = None, sampler: "'samplers.BaseSampler' | None" = None, pruner: 'pruners.BasePruner | None' = None, study_name: 'str | None' = None, direction: 'str | StudyDirection | None' = None, load_if_exists: 'bool' = False, directions: 'Sequence[str | StudyDirection] | None' = None) -> 'Study'

Create a new :class:`~optuna.study.Study`.

Example:

    .. testcode::

        import optuna


        def objective(trial):
            x = trial.suggest_float("x", 0, 10)
            return x**2


        study = optuna.create_study()
        study.optimize(objective, n_trials=3)

Args:
    storage:
        Database URL. If this argument is set to None,
        :class:`~optuna.storages.InMemoryStorage` is used, and the
        :class:`~optuna.study.Study` will not be persistent.

        .. note::
            When a database URL is passed, Optuna internally uses `SQLAlchemy`_ to handle
            the database. Please refer to `SQLAlchemy's document`_ for further details.
            If you want to specify non-default options to `SQLAlchemy Engine`_, you can
            instantiate :class:`~optuna.storages.RDBStorage` with your desired options and
            pass it to the ``storage`` argument instead of a URL.

         .. _SQLAlchemy: https://www.sqlalchemy.org/
         .. _SQLAlchemy's document:
             https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
         .. _SQLAlchemy Engine: https://docs.sqlalchemy.org/en/latest/core/engines.html

    sampler:
        A sampler object that implements background algorithm for value suggestion.
        If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used during
        single-objective optimization and :class:`~optuna.samplers.NSGAIISampler` during
        multi-objective optimization. See also :class:`~optuna.samplers`.
    pruner:
        A pruner object that decides early stopping of unpromising trials. If :obj:`None`
        is specified, :class:`~optuna.pruners.MedianPruner` is used as the default. See
        also :class:`~optuna.pruners`.
    study_name:
        Study's name. If this argument is set to None, a unique name is generated
        automatically.
    direction:
        Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for
        maximization. You can also pass the corresponding :class:`~optuna.study.StudyDirection`
        object. ``direction`` and ``directions`` must not be specified at the same time.

        .. note::
            If none of `direction` and `directions` are specified, the direction of the study
            is set to "minimize".
    load_if_exists:
        Flag to control the behavior to handle a conflict of study names.
        In the case where a study named ``study_name`` already exists in the ``storage``,
        a :class:`~optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
        set to :obj:`False`.
        Otherwise, the creation of the study is skipped, and the existing one is returned.
    directions:
        A sequence of directions during multi-objective optimization.
        ``direction`` and ``directions`` must not be specified at the same time.

Returns:
    A :class:`~optuna.study.Study` object.

See also:
    :func:`optuna.create_study` is an alias of :func:`optuna.study.create_study`.

See also:
    The :ref:`rdb` tutorial provides concrete examples to save and resume optimization using
    RDB.
