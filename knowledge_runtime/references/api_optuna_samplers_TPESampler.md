# optuna.samplers.TPESampler

Version: 4.2.1

optuna.samplers.TPESampler(consider_prior: 'bool' = True, prior_weight: 'float' = 1.0, consider_magic_clip: 'bool' = True, consider_endpoints: 'bool' = False, n_startup_trials: 'int' = 10, n_ei_candidates: 'int' = 24, gamma: 'Callable[[int], int]' = <function default_gamma>, weights: 'Callable[[int], np.ndarray]' = <function default_weights>, seed: 'int | None' = None, *, multivariate: 'bool' = False, group: 'bool' = False, warn_independent_sampling: 'bool' = True, constant_liar: 'bool' = False, constraints_func: 'Callable[[FrozenTrial], Sequence[float]] | None' = None, categorical_distance_func: 'dict[str, Callable[[CategoricalChoiceType, CategoricalChoiceType], float]] | None' = None) -> 'None'

Sampler using TPE (Tree-structured Parzen Estimator) algorithm.

On each trial, for each parameter, TPE fits one Gaussian Mixture Model (GMM) ``l(x)`` to
the set of parameter values associated with the best objective values, and another GMM
``g(x)`` to the remaining parameter values. It chooses the parameter value ``x`` that
maximizes the ratio ``l(x)/g(x)``.

For further information about TPE algorithm, please refer to the following papers:

- `Algorithms for Hyper-Parameter Optimization
  <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__
- `Making a Science of Model Search: Hyperparameter Optimization in Hundreds of
  Dimensions for Vision Architectures <http://proceedings.mlr.press/v28/bergstra13.pdf>`__
- `Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for
  Better Empirical Performance <https://arxiv.org/abs/2304.11127>`__

For multi-objective TPE (MOTPE), please refer to the following papers:

- `Multiobjective Tree-Structured Parzen Estimator for Computationally Expensive Optimization
  Problems <https://doi.org/10.1145/3377930.3389817>`__
- `Multiobjective Tree-Structured Parzen Estimator <https://doi.org/10.1613/jair.1.13188>`__

Please also check our articles:

- `Significant Speed Up of Multi-Objective TPESampler in Optuna v4.0.0
  <https://medium.com/optuna/significant-speed-up-of-multi-objective-tpesampler-in-optuna-v4-0-0-2bacdcd1d99b>`__
- `Multivariate TPE Makes Optuna Even More Powerful
  <https://medium.com/optuna/multivariate-tpe-makes-optuna-even-more-powerful-63c4bfbaebe2>`__

Example:
    An example of a single-objective optimization is as follows:

    .. testcode::

        import optuna
        from optuna.samplers import TPESampler


        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x**2


        study = optuna.create_study(sampler=TPESampler())
        study.optimize(objective, n_trials=10)

.. note::
    :class:`~optuna.samplers.TPESampler`, which became much faster in v4.0.0, c.f. `our article
    <https://medium.com/optuna/significant-speed-up-of-multi-objective-tpesampler-in-optuna-v4-0-0-2bacdcd1d99b>`__,
    can handle multi-objective optimization with many trials as well.
    Please note that :class:`~optuna.samplers.NSGAIISampler` will be used by default for
    multi-objective optimization, so if users would like to use
    :class:`~optuna.samplers.TPESampler` for multi-objective optimization, ``sampler`` must be
    explicitly specified when study is created.

Args:
    consider_prior:
        Enhance the stability of Parzen estimator by imposing a Gaussian prior when
        :obj:`True`. The prior is only effective if the sampling distribution is
        either :class:`~optuna.distributions.FloatDistribution`,
        or :class:`~optuna.distributions.IntDistribution`.
    prior_weight:
        The weight of the prior. This argument is used in
        :class:`~optuna.distributions.FloatDistribution`,
        :class:`~optuna.distributions.IntDistribution`, and
        :class:`~optuna.distributions.CategoricalDistribution`.
    consider_magic_clip:
        Enable a heuristic to limit the smallest variances of Gaussians used in
        the Parzen estimator.
    consider_endpoints:
        Take endpoints of domains into account when calculating variances of Gaussians
        in Parzen estimator. See the original paper for details on the heuristics
        to calculate the variances.
    n_startup_trials:
        The random sampling is used instead of the TPE algorithm until the given number
        of trials finish in the same study.
    n_ei_candidates:
        Number of candidate samples used to calculate the expected improvement.
    gamma:
        A function that takes the number of finished trials and returns the number
        of trials to form a density function for samples with low grains.
        See the original paper for more details.
    weights:
        A function that takes the number of finished trials and returns a weight for them.
        See `Making a Science of Model Search: Hyperparameter Optimization in Hundreds of
        Dimensions for Vision Architectures
        <http://proceedings.mlr.press/v28/bergstra13.pdf>`__ for more details.

        .. note::
            In the multi-objective case, this argument is only used to compute the weights of
            bad trials, i.e., trials to construct `g(x)` in the `paper
            <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__
            ). The weights of good trials, i.e., trials to construct `l(x)`, are computed by a
            rule based on the hypervolume contribution proposed in the `paper of MOTPE
            <https://doi.org/10.1613/jair.1.13188>`__.
    seed:
        Seed for random number generator.
    multivariate:
        If this is :obj:`True`, the multivariate TPE is used when suggesting parameters.
        The multivariate TPE is reported to outperform the independent TPE. See `BOHB: Robust
        and Efficient Hyperparameter Optimization at Scale
        <http://proceedings.mlr.press/v80/falkner18a.html>`__ and `our article
        <https://medium.com/optuna/multivariate-tpe-makes-optuna-even-more-powerful-63c4bfbaebe2>`__
        for more details.

        .. note::
            Added in v2.2.0 as an experimental feature. The interface may change in newer
            versions without prior notice. See
            https://github.com/optuna/optuna/releases/tag/v2.2.0.
    group:
        If this and ``multivariate`` are :obj:`True`, the multivariate TPE with the group
        decomposed search space is used when suggesting parameters.
        The sampling algorithm decomposes the search space based on past trials and samples
        from the joint distribution in each decomposed subspace.
        The decomposed subspaces are a partition of the whole search space. Each subspace
        is a maximal subset of the whole search space, which satisfies the following:
        for a trial in completed trials, the intersection of the subspace and the search space
        of the trial becomes subspace itself or an empty set.
        Sampling from the joint distribution on the subspace is realized by multivariate TPE.
        If ``group`` is :obj:`True`, ``multivariate`` must be :obj:`True` as well.

        .. note::
            Added in v2.8.0 as an experimental feature. The interface may change in newer
            versions without prior notice. See
            https://github.com/optuna/optuna/releases/tag/v2.8.0.

        Example:

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_categorical("x", ["A", "B"])
                if x == "A":
                    return trial.suggest_float("y", -10, 10)
                else:
                    return trial.suggest_int("z", -10, 10)


            sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)
    warn_independent_sampling:
        If this is :obj:`True` and ``multivariate=True``, a warning message is emitted when
        the value of a parameter is sampled by using an independent sampler.
        If ``multivariate=False``, this flag has no effect.
    constant_liar:
        If :obj:`True`, penalize running trials to avoid suggesting parameter configurations
        nearby.

        .. note::
            Abnormally terminated trials often leave behind a record with a state of
            ``RUNNING`` in the storage.
            Such "zombie" trial parameters will be avoided by the constant liar algorithm
            during subsequent sampling.
            When using an :class:`~optuna.storages.RDBStorage`, it is possible to enable the
            ``heartbeat_interval`` to change the records for abnormally terminated trials to
            ``FAIL``.

        .. note::
            It is recommended to set this value to :obj:`True` during distributed
            optimization to avoid having multiple workers evaluating similar parameter
            configurations. In particular, if each objective function evaluation is costly
            and the durations of the running states are significant, and/or the number of
            workers is high.

        .. note::
            Added in v2.8.0 as an experimental feature. The interface may change in newer
            versions without prior notice. See
            https://github.com/optuna/optuna/releases/tag/v2.8.0.
    constraints_func:
        An optional function that computes the objective constraints. It must take a
        :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
        be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
        constraints is violated. A value equal to or smaller than 0 is considered feasible.
        If ``constraints_func`` returns more than one value for a trial, that trial is
        considered feasible if and only if all values are equal to 0 or smaller.

        The ``constraints_func`` will be evaluated after each successful trial.
        The function won't be called when trials fail or they are pruned, but this behavior is
        subject to change in the future releases.

        .. note::
            Added in v3.0.0 as an experimental feature. The interface may change in newer
            versions without prior notice.
            See https://github.com/optuna/optuna/releases/tag/v3.0.0.
    categorical_distance_func:
        A dictionary of distance functions for categorical parameters. The key is the name of
        the categorical parameter and the value is a distance function that takes two
        :class:`~optuna.distributions.CategoricalChoiceType` s and returns a :obj:`float`
        value. The distance function must return a non-negative value.

        While categorical choices are handled equally by default, this option allows users to
        specify prior knowledge on the structure of categorical parameters. When specified,
        categorical choices closer to current best choices are more likely to be sampled.

        .. note::
            Added in v3.4.0 as an experimental feature. The interface may change in newer
            versions without prior notice.
            See https://github.com/optuna/optuna/releases/tag/v3.4.0.
