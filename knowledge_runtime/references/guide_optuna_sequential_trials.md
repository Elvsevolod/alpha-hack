# sequential_trials
Source: https://optuna.readthedocs.io/en/v4.2.1/reference/generated/optuna.study.Study.html



  

  
  optuna.study.Study — Optuna 4.2.1 documentation

      

      

  

      

      

      

    

     

    

            

                    Key Features

                

                    Code Examples

                

                    Installation

                

                    Dashboard

                

                    OptunaHub

                

                    Blog

                

                    Videos

                

                    Paper

            

    

  

    

      

          

      

          
API Reference

optuna.study

optuna.study.Study

  

          

             

optuna.study.Study

class optuna.study.Study(study_name, storage, sampler=None, pruner=None)[source]

A study corresponds to an optimization task, i.e., a set of trials.

This object provides interfaces to run a new Trial, access trials’
history, set/get user-defined attributes of the study itself.

Note that the direct use of this constructor is not recommended.
To create and load a study, please refer to the documentation of
create_study() and load_study() respectively.

Methods

 | 
add_trial(trial)

 | 
Add trial to study.

 | 
add_trials(trials)

 | 
Add trials to study.

 | 
ask([fixed_distributions])

 | 
Create a new trial from which hyperparameters can be suggested.

 | 
enqueue_trial(params[, user_attrs, ...])

 | 
Enqueue a trial with given parameter values.

 | 
get_trials([deepcopy, states])

 | 
Return all trials in the study.

 | 
optimize(func[, n_trials, timeout, n_jobs, ...])

 | 
Optimize an objective function.

 | 
set_metric_names(metric_names)

 | 
Set metric names.

 | 
set_system_attr(key, value)

 | 
Set a system attribute to the study.

 | 
set_user_attr(key, value)

 | 
Set a user attribute to the study.

 | 
stop()

 | 
Exit from the current optimization loop after the running trials finish.

 | 
tell(trial[, values, state, skip_if_finished])

 | 
Finish a trial created with ask().

 | 
trials_dataframe([attrs, multi_index])

 | 
Export trials as a pandas DataFrame.

Attributes

 | 
best_params

 | 
Return parameters of the best trial in the study.

 | 
best_trial

 | 
Return the best trial in the study.

 | 
best_trials

 | 
Return trials located at the Pareto front in the study.

 | 
best_value

 | 
Return the best objective value in the study.

 | 
direction

 | 
Return the direction of the study.

 | 
directions

 | 
Return the directions of the study.

 | 
metric_names

 | 
Return metric names.

 | 
system_attrs

 | 
Return system attributes.

 | 
trials

 | 
Return all trials in the study.

 | 
user_attrs

 | 
Return user attributes.

Parameters:

study_name (str)

storage (str | storages.BaseStorage)

sampler ('samplers.BaseSampler' | None)

pruner (pruners.BasePruner | None)

add_trial(trial)[source]

Add trial to study.

The trial is validated before being added.

Example

import optuna
from optuna.distributions import FloatDistribution

def objective(trial):
    x = trial.suggest_float("x", 0, 10)
    return x**2

study = optuna.create_study()
assert len(study.trials) == 0

trial = optuna.trial.create_trial(
    params={"x": 2.0},
    distributions={"x": FloatDistribution(0, 10)},
    value=4.0,
)

study.add_trial(trial)
assert len(study.trials) == 1

study.optimize(objective, n_trials=3)
assert len(study.trials) == 4

other_study = optuna.create_study()

for trial in study.trials:
    other_study.add_trial(trial)
assert len(other_study.trials) == len(study.trials)

other_study.optimize(objective, n_trials=2)
assert len(other_study.trials) == len(study.trials) + 2

See also

This method should in general be used to add already evaluated trials
(trial.state.is_finished() == True). To queue trials for evaluation,
please refer to enqueue_trial().

See also

See create_trial() for how to create trials.

See also

Please refer to Second scenario: Have Optuna utilize already evaluated hyperparameters for the tutorial of specifying
hyperparameters with the evaluated value manually.

Parameters:

trial (FrozenTrial) – Trial to add.

Return type:

None

add_trials(trials)[source]

Add trials to study.

The trials are validated before being added.

Example

import optuna

def objective(trial):
    x = trial.suggest_float("x", 0, 10)
    return x**2

study = optuna.create_study()
study.optimize(objective, n_trials=3)
assert len(study.trials) == 3

other_study = optuna.create_study()
other_study.add_trials(study.trials)
assert len(other_study.trials) == len(study.trials)

other_study.optimize(objective, n_trials=2)
assert len(other_study.trials) == len(study.trials) + 2

See also

See add_trial() for addition of each trial.

Parameters:

trials (Iterable[FrozenTrial]) – Trials to add.

Return type:

None

ask(fixed_distributions=None)[source]

Create a new trial from which hyperparameters can be suggested.

This method is part of an alternative to optimize() that allows
controlling the lifetime of a trial outside the scope of func. Each call to this
method should be followed by a call to tell() to finish the
created trial.

See also

The Ask-and-Tell Interface tutorial provides use-cases with examples.

Example

Getting the trial object with the ask() method.

import optuna

study = optuna.create_study()

trial = study.ask()

x = trial.suggest_float("x", -1, 1)

study.tell(trial, x**2)

Example

Passing previously defined distributions to the ask()
method.

import optuna

study = optuna.create_study()

distributions = {
    "optimizer": optuna.distributions.CategoricalDistribution(["adam", "sgd"]),
    "lr": optuna.distributions.FloatDistribution(0.0001, 0.1, log=True),
}

# You can pass the distributions previously defined.
trial = study.ask(fixed_distributions=distributions)

# `optimizer` and `lr` are already suggested and accessible with `trial.params`.
assert "optimizer" in trial.params
assert "lr" in trial.params

Parameters:

fixed_distributions (dict[str, BaseDistribution] | None) – A dictionary containing the parameter names and parameter’s distributions. Each
parameter in this dictionary is automatically suggested for the returned trial,
even when the suggest method is not explicitly invoked by the user. If this
argument is set to None, no parameter is automatically suggested.

Returns:

A Trial.

Return type:

Trial

property best_params: dict[str, Any]

Return parameters of the best trial in the study.

Note

This feature can only be used for single-objective optimization.

Returns:

A dictionary containing parameters of the best trial.

property best_trial: FrozenTrial

Return the best trial in the study.

Note

This feature can only be used for single-objective optimization.
If your study is multi-objective,
use best_trials instead.

Returns:

A FrozenTrial object of the best trial.

See also

The Re-use the best trial tutorial provides a detailed example of how to use this
method.

property best_trials: list[FrozenTrial]

Return trials located at the Pareto front in the study.

A trial is located at the Pareto front if there are no trials that dominate the trial.
It’s called that a trial t0 dominates another trial t1 if
all(v0 <= v1) for v0, v1 in zip(t0.values, t1.values) and
any(v0 < v1) for v0, v1 in zip(t0.values, t1.values) are held.

Returns:

A list of FrozenTrial objects.

property best_value: float

Return the best objective value in the study.

Note

This feature can only be used for single-objective optimization.

Returns:

A float representing the best objective value.

property direction: StudyDirection

Return the direction of the study.

Note

This feature can only be used for single-objective optimization.
If your study is multi-objective,
use directions instead.

Returns:

A StudyDirection object.

property directions: list[StudyDirection]

Return the directions of the study.

Returns:

A list of StudyDirection objects.

enqueue_trial(params, user_attrs=None, skip_if_exists=False)[source]

Enqueue a trial with given parameter values.

You can fix the next sampling parameters which will be evaluated in your
objective function.

Example

import optuna

def objective(trial):
    x = trial.suggest_float("x", 0, 10)
    return x**2

study = optuna.create_study()
study.enqueue_trial({"x": 5})
study.enqueue_trial({"x": 0}, user_attrs={"memo": "optimal"})
study.optimize(objective, n_trials=2)

assert study.trials[0].params == {"x": 5}
assert study.trials[1].params == {"x": 0}
assert study.trials[1].user_attrs == {"memo": "optimal"}

Parameters:

params (dict[str, Any]) – Parameter values to pass your objective function.

user_attrs (dict[str, Any] | None) – A dictionary of user-specific attributes other than params.

skip_if_exists (bool) – 
When True, prevents duplicate trials from being enqueued again.

Note

This method might produce duplicated trials if called simultaneously
by multiple processes at the same time with same params dict.

Return type:

None

See also

Please refer to First Scenario: Have Optuna evaluate your hyperparameters for the tutorial of specifying
hyperparameters manually.

get_trials(deepcopy=True, states=None)[source]

Return all trials in the study.

The returned trials are ordered by trial number.

See also

See trials for related property.

Example

import optuna

def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    return x**2

study = optuna.create_study()
study.optimize(objective, n_trials=3)

trials = study.get_trials()
assert len(trials) == 3

Parameters:

deepcopy (bool) – Flag to control whether to apply copy.deepcopy() to the trials.
Note that if you set the flag to False, you shouldn’t mutate
any fields of the returned trial. Otherwise the internal state of
the study may corrupt and unexpected behavior may happen.

states (Container[TrialState] | None) – Trial states to filter on. If None, include all states.

Returns:

A list of FrozenTrial objects.

Return type:

list[FrozenTrial]

property metric_names: list[str] | None

Return metric names.

Note

Use set_metric_names() to set the metric names first.

Returns:

A list with names for each dimension of the returned values of the objective function.

optimize(func, n_trials=None, timeout=None, n_jobs=1, catch=(), callbacks=None, gc_after_trial=False, show_progress_bar=False)[source]

Optimize an objective function.

Optimization is done by choosing a suitable set of hyperparameter values from a given
range. Uses a sampler which implements the task of value suggestion based on a specified
distribution. The sampler is specified in create_study() and the
default choice for the sampler is TPE.
See also TPESampler for more details on ‘TPE’.

Optimization will be stopped when receiving a termination signal such as SIGINT and
SIGTERM. Unlike other signals, a trial is automatically and cleanly failed when receiving
SIGINT (Ctrl+C). If n_jobs is greater than one or if another signal than SIGINT
is used, the interrupted trial state won’t be properly updated.

Example

import optuna

def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    return x**2

study = optuna.create_study()
study.optimize(objective, n_trials=3)

Parameters:

func (ObjectiveFuncType) – A callable that implements objective function.

n_trials (int | None) – 
The number of trials for each process. None represents no limit in terms of
the number of trials. The study continues to create trials until the number of
trials reaches n_trials, timeout period elapses,
stop() is called, or a termination signal such as
SIGTERM or Ctrl+C is received.

See also

optuna.study.MaxTrialsCallback can ensure how many times trials
will be performed across all processes.

timeout (float | None) – Stop study after the given number of second(s). None represents no limit in
terms of elapsed time. The study continues to create trials until the number of
trials reaches n_trials, timeout period elapses,
stop() is called or, a termination signal such as
SIGTERM or Ctrl+C is received.

n_jobs (int) – 
The number of parallel jobs. If this argument is set to -1, the number is
set to CPU count.

Note

n_jobs allows parallelization using threading and may suffer from
Python’s GIL.
It is recommended to use process-based parallelization
if func is CPU bound.

catch (Iterable[type[Exception]] | type[Exception]) – A study continues to run even when a trial raises one of the exceptions specified
in this argument. Default is an empty tuple, i.e. the study will stop for any
exception except for TrialPruned.

callbacks (Iterable[Callable[[Study, FrozenTrial], None]] | None) – 
List of callback functions that are invoked at the end of each trial. Each function
must accept two parameters with the following types in this order:
Study and FrozenTrial.

See also

See the tutorial of Callback for Study.optimize for how to use and implement
callback functions.

gc_after_trial (bool) – 
Flag to determine whether to automatically run garbage collection after each trial.
Set to True to run the garbage collection, False otherwise.
When it runs, it runs a full collection by internally calling gc.collect().
If you see an increase in memory consumption over several trials, try setting this
flag to True.

See also

How do I avoid running out of memory (OOM) when optimizing studies?

show_progress_bar (bool) – Flag to show progress bars or not. To show progress bar, set this True.
Note that it is disabled when n_trials is None,
timeout is not None, and n_jobs \(\ne 1\).

Raises:

RuntimeError – If nested invocation of this method occurs.

Return type:

None

set_metric_names(metric_names)[source]

Set metric names.

This method names each dimension of the returned values of the objective function.
It is particularly useful in multi-objective optimization. The metric names are
mainly referenced by the visualization functions.

Example

import optuna
import pandas

def objective(trial):
    x = trial.suggest_float("x", 0, 10)
    return x**2, x + 1

study = optuna.create_study(directions=["minimize", "minimize"])
study.set_metric_names(["x**2", "x+1"])
study.optimize(objective, n_trials=3)

df = study.trials_dataframe(multi_index=True)
assert isinstance(df, pandas.DataFrame)
assert list(df.get("values").keys()) == ["x**2", "x+1"]

See also

The names set by this method are used in trials_dataframe()
and plot_pareto_front().

Parameters:

metric_names (list[str]) – A list of metric names for the objective function.

Return type:

None

Note

Added in v3.2.0 as an experimental feature. The interface may change in newer versions
without prior notice. See https://github.com/optuna/optuna/releases/tag/v3.2.0.

set_system_attr(key, value)[source]

Set a system attribute to the study.

Note that Optuna internally uses this method to save system messages. Please use
set_user_attr() to set users’ attributes.

Parameters:

key (str) – A key string of the attribute.

value (Any) – A value of the attribute. The value should be JSON serializable.

Return type:

None

Warning

Deprecated in v3.1.0. This feature will be removed in the future. The removal of this
feature is currently scheduled for v5.0.0, but this schedule is subject to change.
See https://github.com/optuna/optuna/releases/tag/v3.1.0.

set_user_attr(key, value)[source]

Set a user attribute to the study.

See also

See user_attrs for related attribute.

See also

See the recipe on User Attributes.

Example

import optuna

def objective(trial):
    x = trial.suggest_float("x", 0, 1)
    y = trial.suggest_float("y", 0, 1)
    return x**2 + y**2

study = optuna.create_study()

study.set_user_attr("objective function", "quadratic function")
study.set_user_attr("dimensions", 2)
study.set_user_attr("contributors", ["Akiba", "Sano"])

assert study.user_attrs == {
    "objective function": "quadratic function",
    "dimensions": 2,
    "contributors": ["Akiba", "Sano"],
}

Parameters:

key (str) – A key string of the attribute.

value (Any) – A value of the attribute. The value should be JSON serializable.

Return type:

None

stop()[source]

Exit from the current optimization loop after the running trials finish.

This method lets the running optimize() method return
immediately after all trials which the optimize() method
spawned finishes.
This method does not affect any behaviors of parallel or successive study processes.
This method only works when it is called inside an objective function or callback.

Example

import optuna

def objective(trial):
    if trial.number == 4:
        trial.study.stop()
    x = trial.suggest_float("x", 0, 10)
    return x**2

study = optuna.create_study()
study.optimize(objective, n_trials=10)
assert len(study.trials) == 5

Return type:

None

property system_attrs: dict[str, Any]

Return system attributes.

Returns:

A dictionary containing all system attributes.

Warning

Deprecated in v3.1.0. This feature will be removed in the future. The removal of this
feature is currently scheduled for v5.0.0, but this schedule is subject to change.
See https://github.com/optuna/optuna/releases/tag/v3.1.0.

tell(trial, values=None, state=None, skip_if_finished=False)[source]

Finish a trial created with ask().

See also

The Ask-and-Tell Interface tutorial provides use-cases with examples.

Example

import optuna
from optuna.trial import TrialState

def f(x):
    return (x - 2) ** 2

def df(x):
    return 2 * x - 4

study = optuna.create_study()

n_trials = 30

for _ in range(n_trials):
    trial = study.ask()

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # Iterative gradient descent objective function.
    x = 3  # Initial value.
    for step in range(128):
        y = f(x)

        trial.report(y, step=step)

        if trial.should_prune():
            # Finish the trial with the pruned state.
            study.tell(trial, state=TrialState.PRUNED)
            break

        gy = df(x)
        x -= gy * lr
    else:
        # Finish the trial with the final value after all iterations.
        study.tell(trial, y)

Parameters:

trial (Trial | int) – A Trial object or a trial number.

values (float | Sequence[float] | None) – Optional objective value or a sequence of such values in case the study is used
for multi-objective optimization. Argument must be provided if state is
COMPLETE and should be None if state
is FAIL or
PRUNED.

state (TrialState | None) – State to be reported. Must be None,
COMPLETE,
FAIL or
PRUNED.
If state is None,
it will be updated to COMPLETE
or FAIL depending on whether
validation for values reported succeed or not.

skip_if_finished (bool) – Flag to control whether exception should be raised when values for already
finished trial are told. If True, tell is skipped without any error
when the trial is already finished.

Returns:

A FrozenTrial representing the resulting trial.
A returned trial is deep copied thus user can modify it as needed.

Return type:

FrozenTrial

property trials: list[FrozenTrial]

Return all trials in the study.

The returned trials are ordered by trial number.

This is a short form of self.get_trials(deepcopy=True, states=None).

Returns:

A list of FrozenTrial objects.

See also

See get_trials() for related method.

trials_dataframe(attrs=('number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'params', 'user_attrs', 'system_attrs', 'state'), multi_index=False)[source]

Export trials as a pandas DataFrame.

The DataFrame provides various features to analyze studies. It is also useful to draw a
histogram of objective values and to export trials as a CSV file.
If there are no trials, an empty DataFrame is returned.

Example

import optuna
import pandas

def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    return x**2

study = optuna.create_study()
study.optimize(objective, n_trials=3)

# Create a dataframe from the study.
df = study.trials_dataframe()
assert isinstance(df, pandas.DataFrame)
assert df.shape[0] == 3  # n_trials.

Parameters:

attrs (tuple[str, ...]) – Specifies field names of FrozenTrial to include them to a
DataFrame of trials.

multi_index (bool) – Specifies whether the returned DataFrame employs MultiIndex or not. Columns that
are hierarchical by nature such as (params, x) will be flattened to
params_x when set to False.

Returns:

A pandas DataFrame of trials in the Study.

Return type:

pd.DataFrame

Note

If value is in attrs during multi-objective optimization, it is implicitly
replaced with values.

Note

If set_metric_names() is called, the value or values
is implicitly replaced with the dictionary with the objective name as key and the
objective value as value.

property user_attrs: dict[str, Any]

Return user attributes.

See also

See set_user_attr() for related method.

Example

import optuna

def objective(trial):
    x = trial.suggest_float("x", 0, 1)
    y = trial.suggest_float("y", 0, 1)
    return x**2 + y**2

study = optuna.create_study()

study.set_user_attr("objective function", "quadratic function")
study.set_user_attr("dimensions", 2)
study.set_user_attr("contributors", ["Akiba", "Sano"])

assert study.user_attrs == {
    "objective function": "quadratic function",
    "dimensions": 2,
    "contributors": ["Akiba", "Sano"],
}

Returns:

A dictionary containing all user attributes.

           

          

      

  

