# optuna.study.Study.optimize
Version: 4.2.1

## optuna.study.Study.optimize(self, func: 'ObjectiveFuncType', n_trials: 'int | None' = None, timeout: 'float | None' = None, n_jobs: 'int' = 1, catch: 'Iterable[type[Exception]] | type[Exception]' = (), callbacks: 'Iterable[Callable[[Study, FrozenTrial], None]] | None' = None, gc_after_trial: 'bool' = False, show_progress_bar: 'bool' = False) -> 'None'

Optimize an objective function.

Optimization is done by choosing a suitable set of hyperparameter values from a given
range. Uses a sampler which implements the task of value suggestion based on a specified
distribution. The sampler is specified in :func:`~optuna.study.create_study` and the
default choice for the sampler is TPE.
See also :class:`~optuna.samplers.TPESampler` for more details on 'TPE'.

Optimization will be stopped when receiving a termination signal such as SIGINT and
SIGTERM. Unlike other signals, a trial is automatically and cleanly failed when receiving
SIGINT (Ctrl+C). If ``n_jobs`` is greater than one or if another signal than SIGINT
is used, the interrupted trial state won't be properly updated.

Example:

    .. testcode::

        import optuna


        def objective(trial):
            x = trial.suggest_float("x", -1, 1)
            return x**2


        study = optuna.create_study()
        study.optimize(objective, n_trials=3)

Args:
    func:
        A callable that implements objective function.
    n_trials:
        The number of trials for each process. :obj:`None` represents no limit in terms of
        the number of trials. The study continues to create trials until the number of
        trials reaches ``n_trials``, ``timeout`` period elapses,
        :func:`~optuna.study.Study.stop` is called, or a termination signal such as
        SIGTERM or Ctrl+C is received.

        .. seealso::
            :class:`optuna.study.MaxTrialsCallback` can ensure how many times trials
            will be performed across all processes.
    timeout:
        Stop study after the given number of second(s). :obj:`None` represents no limit in
        terms of elapsed time. The study continues to create trials until the number of
        trials reaches ``n_trials``, ``timeout`` period elapses,
        :func:`~optuna.study.Study.stop` is called or, a termination signal such as
        SIGTERM or Ctrl+C is received.
    n_jobs:
        The number of parallel jobs. If this argument is set to ``-1``, the number is
        set to CPU count.

        .. note::
            ``n_jobs`` allows parallelization using :obj:`threading` and may suffer from
            `Python's GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`__.
            It is recommended to use :ref:`process-based parallelization<distributed>`
            if ``func`` is CPU bound.

    catch:
        A study continues to run even when a trial raises one of the exceptions specified
        in this argument. Default is an empty tuple, i.e. the study will stop for any
        exception except for :class:`~optuna.exceptions.TrialPruned`.
    callbacks:
        List of callback functions that are invoked at the end of each trial. Each function
        must accept two parameters with the following types in this order:
        :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial`.

        .. seealso::

            See the tutorial of :ref:`optuna_callback` for how to use and implement
            callback functions.

    gc_after_trial:
        Flag to determine whether to automatically run garbage collection after each trial.
        Set to :obj:`True` to run the garbage collection, :obj:`False` otherwise.
        When it runs, it runs a full collection by internally calling :func:`gc.collect`.
        If you see an increase in memory consumption over several trials, try setting this
        flag to :obj:`True`.

        .. seealso::

            :ref:`out-of-memory-gc-collect`

    show_progress_bar:
        Flag to show progress bars or not. To show progress bar, set this :obj:`True`.
        Note that it is disabled when ``n_trials`` is :obj:`None`,
        ``timeout`` is not :obj:`None`, and ``n_jobs`` :math:`\ne 1`.

Raises:
    RuntimeError:
        If nested invocation of this method occurs.
