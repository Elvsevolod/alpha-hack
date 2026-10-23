# optuna.trial.Trial

Version: 4.2.1

optuna.trial.Trial(study: "'optuna.study.Study'", trial_id: 'int') -> 'None'

A trial is a process of evaluating an objective function.

This object is passed to an objective function and provides interfaces to get parameter
suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

Note that the direct use of this constructor is not recommended.
This object is seamlessly instantiated and passed to the objective function behind
the :func:`optuna.study.Study.optimize()` method; hence library users do not care about
instantiation of this object.

Args:
    study:
        A :class:`~optuna.study.Study` object.
    trial_id:
        A trial ID that is automatically generated.

