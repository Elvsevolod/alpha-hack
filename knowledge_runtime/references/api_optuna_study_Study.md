# optuna.study.Study

Version: 4.2.1

optuna.study.Study(study_name: 'str', storage: 'str | storages.BaseStorage', sampler: "'samplers.BaseSampler' | None" = None, pruner: 'pruners.BasePruner | None' = None) -> 'None'

A study corresponds to an optimization task, i.e., a set of trials.

This object provides interfaces to run a new :class:`~optuna.trial.Trial`, access trials'
history, set/get user-defined attributes of the study itself.

Note that the direct use of this constructor is not recommended.
To create and load a study, please refer to the documentation of
:func:`~optuna.study.create_study` and :func:`~optuna.study.load_study` respectively.

