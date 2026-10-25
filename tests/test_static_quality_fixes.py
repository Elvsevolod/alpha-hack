"""Behavioral acceptance for A01–A28 of the v076–v091 quality audit."""
import copy
import importlib.util
import json
import time
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import EnsembleArgs, Hypothesis, PipelineSpec, TrainingSummary, Transformation, TrainArgs
from automl_agent.cost_model import phase_forecast
from automl_agent.decision import decide, fit_decision, frozen_decision_valid
from automl_agent.finalists import choose_control, component_audit, select_finalists
from automl_agent.pipelines import DirectPipeline, build_pipeline, refit_iterations
from automl_agent.protocols import configure, initial_protocol, paired_comparison
from automl_agent.recipe_patch import inherit_parent
from automl_agent.research_evaluation import oof_predictions
from automl_agent.validation import choose_validation, pilot_population, selection_for
from conftest import write_task


def context(task, tmp_path, strategy="random", seconds=60):
    return {"cpu": 1, "memory_mb": 2048, "budget_seconds": seconds, "strategy": strategy,
        "protocol_hash": "fixed", "data_hash": "fixture", "split_id": "validation",
        "workdir": str(tmp_path), "task": task.model_dump(mode="json"),
        "phase_estimates": {"selected_fit_seconds": .2, "prediction_seconds": .1, "publication_seconds": .1}}


def hypothesis():
    return Hypothesis(observation="Offline regression", expected_effect="Preserve the complete recipe",
        changed_component="audit repair", acceptance="Same permitted validation and complete refit")


class Journal:
    def __init__(self, values=None):
        self.values, self.events = values or {}, []

    def get(self, key, default=None):
        return self.values.get(key, default)

    def put(self, key, value):
        self.values[key] = value

    def event(self, name, payload):
        self.events.append((name, payload))


def classification(X, y, task):
    values = y.iloc[:, 0].to_numpy()
    labels = np.where(values < -.5, "low", np.where(values > .5, "high", "middle"))
    task = task.model_copy(update={"kind": "classification", "classes": {"answer": ["low", "middle", "high"]},
        "metric": task.metric.model_copy(update={"name": "f1", "maximize": True, "average": "macro"}),
        "output": task.output.model_copy(update={"prediction_type": "labels"})})
    return X, pd.DataFrame({"answer": labels}, index=y.index), task


@pytest.mark.skipif(importlib.util.find_spec('autogluon') is None,
    reason='Real AutoGluon integration requires organizer dependency environment')
def test_A01_automl_decision_freeze_audit_and_refit(tabular, tmp_path):
    X, y, task = classification(*tabular)
    # Real engine, OOF decision, serialization, finalist audit and larger refit.
    spec = PipelineSpec(engine="autogluon", model="library_search", learners=["GBM"],
        autogluon_configurations={"GBM": [{"n_estimators": 4, "min_child_samples": 1}]},
        prediction_policy="class_offsets", postprocess_fit_policy="pooled_oof", decision_method="regularized")
    model = build_pipeline(spec, task, context(task, tmp_path))
    model.fit(X.iloc[:140], y.iloc[:140], context(task, tmp_path))
    frozen = model.frozen_spec()
    assert frozen_decision_valid(frozen, task)
    record = SimpleNamespace(pipeline=spec, hash=spec.hash, parent_id=None)
    c = SimpleNamespace(task=task, plan=SimpleNamespace(hash="fixed"), journal=Journal(), lookup=lambda _: (record, None))
    assert not component_audit(c, ["candidate"], {"candidate": {"frozen_pipeline": frozen.model_dump(), "pipeline_hash": frozen.hash}})["warnings"]
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    restored.refit(X, y, context(task, tmp_path))
    assert restored.state == model.state and restored.summary.rows == len(X)
    assert set(restored.predict(X.iloc[:8], context(task, tmp_path)).values["answer"]) <= set(task.classes["answer"])
    invalid = frozen.model_copy(update={"class_offsets": None, "learned_decision": {"answer": {"classes": ["wrong"], "offsets": [0]}}})
    assert not frozen_decision_valid(invalid, task)


@pytest.mark.parametrize("profile", ["bagging", "stacking"])
@pytest.mark.skipif(importlib.util.find_spec('autogluon') is None,
    reason='Real AutoGluon integration requires organizer dependency environment')
def test_A02_A16_explicit_graph_and_exact_groups_replay(tabular, tmp_path, profile):
    X, y, task = tabular
    X = pd.DataFrame({"number": np.repeat(np.arange(30, dtype=float), 4)})
    X.index = [f"train:{i}" for i in range(len(X))]
    y = pd.DataFrame({"answer": X.number * 2}, index=X.index)
    task = task.model_copy(update={"feature_columns": ["number"], "categorical_columns": []})
    ctx = context(task, tmp_path, "exact_group", 90)
    spec = PipelineSpec(engine="autogluon", model="library_search", learners=["GBM"], bag_folds=2,
        autogluon_profile=profile, autogluon_configurations={"GBM": [{"n_estimators": 4, "num_leaves": 7},
            {"n_estimators": 6, "num_leaves": 15}]})
    model = build_pipeline(spec, task, ctx)
    model.fit(X.iloc[:100], y.iloc[:100], ctx)
    configuration = model.profile_configs["answer"]
    required, pending = set(),list(configuration['weights'])
    while pending:
        name=pending.pop()
        if name not in required:
            required.add(name); pending.extend(configuration['graph'][name])
    assert set(configuration['graph']) == required
    assert all("explicit" in name for name in configuration["weights"])
    assert all(parameters["ag_args"]["name"] for level in configuration["hyperparameters"].values()
        for family in level.values() for parameters in family)
    frame = model._autogluon_frame(X, y, "answer", ctx)
    grouped = model._bounded_group_codes(frame)
    assert grouped.groupby("number")[model.autogluon_group_column].nunique().max() == 1
    assert grouped[model.autogluon_group_column].nunique() == 2
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    restored.refit(X, y, ctx)
    assert restored.selected["answer"]["full_refit_rows"] == len(X)
    assert np.isfinite(restored.predict(X.iloc[-8:], ctx).values["answer"]).all()


def test_A03_failed_challenger_does_not_cancel_prepared_baseline(tmp_path, monkeypatch):
    from automl_agent.controller import Controller
    from automl_agent.contracts import ToolCommand
    c = Controller(write_task(tmp_path), Limits(seconds=240, cpu=1))
    try:
        c.no_llm = True
        c.prepare()
        c.dispatch(ToolCommand(operation_id="other_family", tool="train_model", arguments=TrainArgs(
            pipeline=PipelineSpec(model="ridge"), hypothesis=hypothesis(), budget_seconds=30).model_dump()))
        c.freeze("Offline per-finalist recovery test")
        baseline = c.journal.get("baseline_id")
        plan = c.journal.get("finalization_plan")
        other = next(identifier for identifier in plan["finalists"] if identifier != baseline)
        original = c._prepare_finalist
        def prepare(identifier, rows, part, seconds):
            if identifier == other and part == "control":
                raise RuntimeError("Synthetic challenger OOM before control")
            return original(identifier, rows, part, seconds)
        monkeypatch.setattr(c, "_prepare_finalist", prepare)
        c.finalize()
        assert c.journal.get("control_opened")["finalists"] == [baseline]
        assert c.journal.get("control_winner") == baseline
        assert c.journal.get("refit_status") == "completed"
        assert c.journal.get("publication")["scope"] == "full_train_refit"
        assert any(row["id"] == other and row["reason"] == "preparation_failed"
            for row in c.journal.get("finalization_plan")["excluded_before_control"])
    finally:
        c.close()


def test_A04_no_paid_critic_when_it_would_consume_selected_fit(tabular, tmp_path):
    from automl_agent.critic import review_candidate
    X, y, task = tabular
    spec = SimpleNamespace(pipeline=PipelineSpec(), budget_seconds=300, experiment_id="candidate",
        task_hash=task.hash, data_version="fixture", validation_hash="fixed", library_versions={}, parent_id=None,
        model_dump=lambda **kw: {"budget_seconds": 300, "pipeline": PipelineSpec().model_dump()})
    c = SimpleNamespace(task=task, plan=SimpleNamespace(strategy="random", hash="fixed", fit=list(range(len(X)))),
        journal=Journal(), records=lambda: [], workdir=tmp_path,
        dataset=SimpleNamespace(profile=lambda _: SimpleNamespace(columns={})),
        budget=SimpleNamespace(remaining=1000, final_reserve=lambda _: 100),
        qwen=SimpleNamespace(critique=lambda *a, **k: pytest.fail("No room for review plus fit")))
    costs = SimpleNamespace(forecast={"candidate_fit_seconds": 150})
    result = review_candidate(c, spec, costs, candidate_deadline=time.time()+155)
    assert result["status"] == "insufficient_review_and_fit_budget"
    assert not list(tmp_path.glob("hypothesis_reviews/*"))


def test_A04_critic_uses_local_window_after_reserving_fit(tabular, tmp_path):
    from automl_agent.qwen import QwenClient
    client = QwenClient.__new__(QwenClient)
    client.effort, client.reasoning_reason, client.reserve_seconds = "high", "research", 0
    client.journal = SimpleNamespace(directory=tmp_path / 'a' / 'b')
    client.select_reasoning = lambda effort, reason: None
    seen = []
    client._request = lambda *a, **kw: seen.append(kw["max_seconds"]) or {"content": "{}"}
    client.critique({"resources": {"critic_max_seconds": 17}}, 100)
    assert seen == [17] and client.reserve_seconds == 0


def test_A05_research_review_is_not_repeated_startup():
    summary = {"rows": 100, "seconds": 10, "budget": {"selected_fit_actual_seconds": 8, "selected_fit_rows": 100},
        "execution_phases": {"fit_seconds": 10, "load_and_construct_seconds": 1, "save_load_seconds": 1,
            "predict_and_verify_seconds": 1}}
    baseline = phase_forecast(summary, 14, 200, 100)
    for extra in (20, 180):
        assert phase_forecast({**summary, "research_review_seconds": extra}, 14+extra, 200, 100) == baseline
    sparse = {"rows": 100, "seconds": 10, "budget": summary["budget"]}
    assert phase_forecast({**sparse, "research_review_seconds": 180}, 194, 200, 100) == phase_forecast(sparse, 14, 200, 100)


def test_A06_A07_initial_transfer_aligns_all_three_boundaries(tmp_path):
    from automl_agent.data import Dataset
    from automl_agent.identity import feature_groups
    dataset = Dataset(write_task(tmp_path, kind="classification"))
    task = dataset.activate(dataset.explicit())
    dataset.train["value"] = np.arange(len(dataset.train)) // 6
    dataset.train["code"] = "same"
    dataset.test["value"] = 100 + np.arange(len(dataset.test))
    dataset.test["code"] = "same"
    original = choose_validation(dataset)
    revised = initial_protocol(dataset, original)
    assert revised.evaluation_mode == "exact_group"
    groups = feature_groups(dataset.train[task.feature_columns])
    sets = [set(groups[rows]) for rows in (revised.fit, revised.validation, revised.control)]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert revised.observations[-1]["uses_control_feedback"] is False
    assert any(row.get("all_boundaries_chosen_before_baseline") for row in revised.observations)
    with pytest.raises(ValueError, match="Closed control transfer"):
        configure(dataset, original, "exact_group")


def test_A07_low_overlap_control_matches_target_feature_population(tmp_path):
    from automl_agent.data import Dataset
    dataset=Dataset(write_task(tmp_path,kind='classification'))
    dataset.activate(dataset.explicit())
    dataset.train['value']=np.arange(len(dataset.train))//6
    dataset.train['code']='same'
    dataset.test['code']='same'
    dataset.test['value']=200+np.arange(len(dataset.test))
    dataset.test.loc[:4,'value']=np.arange(5)
    plan=initial_protocol(dataset,choose_validation(dataset))
    assert plan.evaluation_mode == 'low_overlap'
    detail=next(row for row in plan.observations if row.get('type') == 'exact_X_protocol')
    assert detail['desired_feature_overlap'] == .2
    assert abs(detail['actual_feature_overlap']-.2) < .08
    assert abs(detail['control_feature_overlap']-.2) < .08
    assert len(set(plan.fit+plan.validation+plan.control)) == len(dataset.train)


@pytest.mark.parametrize("periods", [2, 3])
def test_A08_A23_temporal_oof_uses_label_free_warmup(tabular, tmp_path, periods):
    X, y, task = tabular
    X = X.assign(when=np.arange(len(X)) * periods // len(X))
    task = task.model_copy(update={"time_column": "when", "feature_columns": list(X)})
    spec = PipelineSpec(model="lightgbm", hyperparameters={"n_estimators": 5}, early_stopping_rounds=2)
    report = oof_predictions(spec, task, X, y, context(task, tmp_path, "temporal"))
    assert report["folds"][0]["warmup_policy"].startswith("declared_fixed_iteration_cap")
    for fold in report["folds"]:
        assert not set(fold["fit_keys"]) & set(fold["evaluation_keys"])
        assert X.loc[fold["fit_keys"], "when"].max() < X.loc[fold["evaluation_keys"], "when"].min()
    assert report["coverage"] < 1
    if periods == 2:
        assert report["support_status"] == "limited_temporal_history"


def test_A09_partial_common_stability_is_used():
    rows = [{"id": name, "identity": name, "score": score, "windows": windows,
        "windows_hash": "same", "protocol": "fixed", "cost": 1, "created_index": i}
        for i, (name, score, windows) in enumerate([
            ("baseline", 100, {"late": 100}), ("leader", 99, {"late": 99}),
            ("stable", 101, {"late": 90}), ("unstable", 100.5, {"late": 200})])]
    policy = selection_for(SimpleNamespace(metric=SimpleNamespace(maximize=False)))
    selected, audit = select_finalists(rows, "baseline", "leader", policy, ["early", "late"])
    assert selected == ["baseline", "leader", "stable"]
    assert audit["mode"] == "partial_common_window" and audit["windows"] == ["late"]
    assert audit["coverage"]["baseline"] == ["late"]


def test_A09_controller_retains_limited_measured_windows(tabular, tmp_path, monkeypatch):
    from automl_agent.controller import Controller
    X,y,task=tabular
    records=[]
    for i,(name,score,late) in enumerate([('baseline',100,100),('leader',99,99),('stable',101,90)]):
        pipeline=PipelineSpec(model='ridge',hyperparameters={'alpha':i+1})
        manifest=tmp_path/(name+'.json'); manifest.write_text(json.dumps({'frozen_pipeline':pipeline.model_dump()}))
        spec=SimpleNamespace(pipeline=pipeline,experiment_id=name,validation_hash='fixed',library_versions={})
        result=SimpleNamespace(status='completed',scope='fit',score=score,created_index=i,
            artifacts={'worker_manifest':str(manifest)},training_summary={'temporal_stability':{
                'status':'limited','scores':{'late':late},'windows_hash':'same'}})
        records.append((spec,result))
    c=SimpleNamespace(records=lambda:records,journal=Journal({'baseline_id':'baseline','leader_id':'leader'}),
        plan=SimpleNamespace(hash='fixed',fit=[0,1],validation=[2],selection_policy=selection_for(task)),
        dataset=SimpleNamespace(train=X),stability_windows=lambda:[{'id':'early'},{'id':'late'}])
    monkeypatch.setattr('automl_agent.completion.estimate',lambda *a,**kw:{'seconds':1})
    selected,audit=Controller.prospective_finalists(c)
    assert selected == ['baseline','leader','stable'] and audit['windows'] == ['late']


def test_A10_small_measured_gain_is_not_a_statistical_tie(tabular):
    X, y, task = classification(*tabular)
    policy = selection_for(task)
    selected, audit = choose_control({"baseline": .600, "challenger": .604}, "baseline", policy,
        {"baseline": 1, "challenger": 10}, {"baseline": 0, "challenger": 1})
    assert selected == "challenger"
    report = paired_comparison(task, y, y.answer.to_numpy(), y.answer.to_numpy(), np.arange(len(y)), repeats=30)
    assert report["delta"] == 0 and not report["min_delta_is_significance"]
    assert report["status"] == "descriptive_group_bootstrap"


def test_A10_exhausted_diagnostic_budget_does_not_scan_every_group(tabular):
    _,_,task=tabular
    n=100000
    actual=pd.DataFrame({'answer':np.arange(n,dtype=float)})
    started=time.monotonic()
    report=paired_comparison(task,actual,actual.answer.to_numpy()+1,actual.answer.to_numpy(),
        np.arange(n),max_seconds=0)
    assert report['independent_units'] == n and report['interval'] is None
    assert report['delta'] > 0 and time.monotonic()-started < 2


@pytest.mark.parametrize("model,parameter,good,bad", [("lightgbm", "num_leaves", 2, 1), ("catboost", "depth", 2, 17)])
def test_A11_good_hpo_trial_survives_library_failure(tabular, tmp_path, monkeypatch, model, parameter, good, bad):
    import optuna
    X, y, task = tabular
    original = optuna.create_study
    def study(*args, **kwargs):
        value = original(*args, **kwargs)
        value.enqueue_trial({parameter: good}); value.enqueue_trial({parameter: bad})
        return value
    monkeypatch.setattr(optuna, "create_study", study)
    rounds = "iterations" if model == "catboost" else "n_estimators"
    spec = PipelineSpec(engine="optuna", model=model, n_trials=2, hyperparameters={rounds: 3, parameter: good},
        search_space={parameter: {"type": "categorical", "choices": [good, bad]}})
    pipeline = build_pipeline(spec, task, context(task, tmp_path))
    summary = pipeline.fit(X, y, context(task, tmp_path))
    assert [row["state"] for row in summary.inner_scores["trials"]] == ["completed", "failed"]
    assert pipeline.frozen_spec().hyperparameters[parameter] == good
    assert np.isfinite(pipeline.predict(X.iloc[:5], context(task, tmp_path)).values["answer"]).all()


def test_A12_inactive_conditional_parameter_does_not_reappear(tabular, tmp_path, monkeypatch):
    import optuna
    X, y, task = tabular
    original = optuna.create_study
    def study(*args, **kwargs):
        value = original(*args, **kwargs); value.enqueue_trial({"bootstrap_type": "Bayesian"}); return value
    monkeypatch.setattr(optuna, "create_study", study)
    spec = PipelineSpec(engine="optuna", model="catboost", n_trials=1,
        hyperparameters={"iterations": 3, "depth": 2, "bootstrap_type": "Bernoulli", "subsample": .8},
        search_space={"bootstrap_type": {"type": "categorical", "choices": ["Bernoulli", "Bayesian"]},
            "subsample": {"type": "float", "low": .6, "high": 1., "when": {"parameter": "bootstrap_type", "in": ["Bernoulli"]}}})
    pipeline = build_pipeline(spec, task, context(task, tmp_path))
    summary = pipeline.fit(X, y, context(task, tmp_path))
    assert pipeline.frozen_spec().hyperparameters["bootstrap_type"] == "Bayesian"
    assert "subsample" not in pipeline.frozen_spec().hyperparameters
    assert "subsample" not in summary.inner_scores["trials"][0]["parameters"]


def test_A11_resource_failure_is_not_swallowed_as_a_bad_trial(tabular,tmp_path,monkeypatch):
    X,y,task=tabular
    def exhausted(*args,**kwargs): raise MemoryError('Synthetic process memory limit')
    monkeypatch.setattr(DirectPipeline,'fit',exhausted)
    spec=PipelineSpec(engine='optuna',model='lightgbm',n_trials=2,
        search_space={'num_leaves':{'type':'categorical','choices':[3,7]}})
    ctx=context(task,tmp_path)
    with pytest.raises(MemoryError,match='process memory limit'):
        build_pipeline(spec,task,ctx).fit(X,y,ctx)


def test_A13_iteration_growth_is_explicit_and_forecast():
    fixed = PipelineSpec(model="lightgbm", iteration_scale_rows=100, frozen_iterations=100)
    scaled = fixed.model_copy(update={"refit_iteration_policy": "sqrt_rows"})
    assert refit_iterations(fixed, 100, 400, "refit") == 100
    assert refit_iterations(scaled, 100, 400, "refit") == 200
    summary = {"rows": 100, "seconds": 10, "budget": {"selected_fit_actual_seconds": 10, "selected_fit_rows": 100},
        "execution_phases": {"fit_seconds": 10, "load_and_construct_seconds": 0,
            "save_load_seconds": 0, "predict_and_verify_seconds": 0}}
    assert phase_forecast(summary, 12, 400, 100, pipeline=scaled) - phase_forecast(summary, 12, 400, 100, pipeline=fixed) == pytest.approx(50)


def test_A14_domain_fits_are_repeatable_costs(tabular, tmp_path):
    from automl_agent.shift import ShiftPipeline
    X, y, task = tabular
    model = ShiftPipeline(PipelineSpec(model="extra_trees", hyperparameters={"n_estimators": 3}, shift_strategy="importance_weighting"), task)
    ctx = {**context(task, tmp_path), "unlabelled_features": X.iloc[-40:].copy()}
    summary = model.fit(X.iloc[:140], y.iloc[:140], ctx).model_dump()
    repeated = summary["budget"]["repeatable_auxiliary_fit_seconds"]
    assert repeated > 0 and summary["budget"]["repeatable_auxiliary_fits"] == 3
    assert summary["auxiliary_fits"] >= 3
    without = copy.deepcopy(summary); without["budget"]["repeatable_auxiliary_fit_seconds"] = 0
    assert phase_forecast(summary, summary["seconds"], 280, 140) - phase_forecast(without, summary["seconds"], 280, 140) == pytest.approx(repeated*2*1.25)


def test_A15_transform_order_is_preserved_for_fit_and_inference(tabular, tmp_path):
    from automl_agent.preprocessing import FeatureMatrix
    X, y, task = tabular
    spec = PipelineSpec(model="ridge", transformations=[Transformation(name="target_encode", parameters={"columns": ["category"]}),
        Transformation(name="aggregate", parameters={"columns": ["__te_category_answer", "second"], "prefix": "after_encoding"})])
    matrix = FeatureMatrix(spec, task)
    trained = matrix.fit_transform(X, y)
    restored = cloudpickle.loads(cloudpickle.dumps(matrix))
    frame = restored.feature_frame(X.iloc[:6])
    assert np.allclose(frame.after_encoding__mean, frame[["__te_category_answer", "second"]].mean(axis=1))
    assert trained.shape[1] == restored.transform(X.iloc[:6]).shape[1]


def test_A17_no_llm_blocks_every_qwen_request():
    from automl_agent.qwen import QwenClient, QwenUnavailable
    client = QwenClient.__new__(QwenClient); client.disabled = True
    with pytest.raises(QwenUnavailable, match="disabled"):
        client._request([])


def test_A17_preparation_challenger_failure_retains_full_baseline(tmp_path, monkeypatch):
    from automl_agent.controller import Controller
    from automl_agent.qwen import QwenClient, QwenUnavailable
    path = write_task(tmp_path, kind='classification')
    train = pd.read_csv(tmp_path/'train.csv')
    train['answer'] = np.where(train.value < -.5, 'low', np.where(train.value > .5, 'high', 'middle'))
    train.to_csv(tmp_path/'train.csv', index=False)
    contract = json.loads((tmp_path/'task_spec.json').read_text())
    contract['classes'] = {'answer':['low','middle','high']}
    (tmp_path/'task_spec.json').write_text(json.dumps(contract))
    monkeypatch.setattr(QwenClient, '_request', lambda *a, **kw: pytest.fail('no_llm must block all API channels'))
    original = Controller.execute_experiment
    def execute(self, spec, *args, **kwargs):
        if spec.experiment_id.startswith('baseline_decision'):
            raise QwenUnavailable('Synthetic optional preparation review failure')
        return original(self, spec, *args, **kwargs)
    monkeypatch.setattr(Controller, 'execute_experiment', execute)
    c = Controller(path, Limits(seconds=240, cpu=1))
    try:
        report = c.run(no_llm=True)
        assert report['state'] == 'completed' and report['refit_status'] == 'completed'
        assert c.journal.get('publication')['scope'] == 'full_train_refit'
        assert c.journal.get('bootstrap_decision_id')
        assert c.qwen.disabled
    finally:
        c.close()


def test_A18_typed_tool_error_reaches_knowledge_query(tabular):
    from automl_agent.knowledge import AgentKnowledge
    X, y, task = tabular
    error = "LightGBMError num_leaves must be greater than 1"
    c = SimpleNamespace(task=task, plan=SimpleNamespace(hash="fixed", strategy="random"),
        journal=Journal({"task_confirmation": True, "last_tool_result": {"status": "technical_failure", "data": {"error": error}}}),
        dataset=SimpleNamespace(source_hash="fixture"), executor=SimpleNamespace(cpu=1, memory_mb=2048),
        capabilities=SimpleNamespace(versions={}, summary=lambda: {}),
        context_memory=SimpleNamespace(store=SimpleNamespace(active_note=lambda: None)))
    knowledge = AgentKnowledge.__new__(AgentKnowledge); knowledge.c = c
    selection = knowledge.selection_context({"rows": {"fit": len(X)}, "feature_count": len(X.columns), "columns_with_missing": []})
    assert selection.last_error == error and error in selection.query()


def test_A19_baseline_does_not_prove_search_exhaustion(tabular, tmp_path):
    from automl_agent.directions import direction_map, finish_readiness
    X, y, task = tabular
    p = PipelineSpec(model="lightgbm")
    spec = SimpleNamespace(pipeline=p, validation_hash="fixed", experiment_id="baseline")
    result = SimpleNamespace(status="completed", scope="fit", score=1., seconds=1.)
    c = SimpleNamespace(task=task, plan=SimpleNamespace(hash="fixed", fit=list(range(len(X))), strategy="random"),
        records=lambda: [(spec, result)], dataset=SimpleNamespace(test=X.iloc[:10]),
        budget=SimpleNamespace(remaining=1000, final_reserve=lambda _: 10), finalist_costs=lambda: [])
    mapping = direction_map(c)
    assert {"catboost", "xgboost", "ridge"} <= {row["model"] for row in mapping["affordable_unattempted"]}
    assert not mapping["catalog_exhaustion_is_research_exhaustion"]
    readiness = finish_readiness(c, SimpleNamespace(skipped_directions={}, reason="One baseline is insufficient evidence"))
    assert readiness["ready"] and readiness["remaining_actions_are_advisory"]
    assert not readiness["optimality_established"]
    assert readiness["available_actions"]


def test_A20_ensemble_default_preserves_the_need_for_joint_decision():
    args = EnsembleArgs(members=["a", "b"], hypothesis=hypothesis())
    assert args.final_decision == "auto"
    assert EnsembleArgs(members=["a", "b"], hypothesis=hypothesis(), final_decision="argmax").final_decision == "argmax"


def test_A20_default_macro_f1_ensemble_routes_to_joint_oof(tabular, monkeypatch):
    from automl_agent.controller import Controller
    X, y, task = classification(*tabular)
    member = PipelineSpec(prediction_policy='class_offsets', postprocess_fit_policy='pooled_oof')
    spec = SimpleNamespace(pipeline=member, validation_hash='fixed')
    result = SimpleNamespace(status='completed', scope='fit')
    c = SimpleNamespace(task=task, plan=SimpleNamespace(hash='fixed'), journal=Journal(),
        lookup=lambda _: (spec,result), validate_evidence=lambda _:None, finalist_costs=lambda:[],
        budget=SimpleNamespace(remaining=1000, final_reserve=lambda _:10))
    class ReachedJointOOF(Exception): pass
    def forecast(*args, **kwargs): raise ReachedJointOOF()
    monkeypatch.setattr('automl_agent.learned_blend.oof_forecast', forecast)
    with pytest.raises(ReachedJointOOF):
        Controller.build_ensemble(c, EnsembleArgs(members=['a','b'], hypothesis=hypothesis()), SimpleNamespace(operation_id='blend'))
    assert c.journal.events[-1][1]['resolved'] == 'regularized'


def test_A21_default_hpo_reaches_adaptive_trials(tabular, tmp_path):
    X, y, task = tabular
    spec = PipelineSpec(engine="optuna", model="ridge", search_space={"alpha": {"type": "float", "low": .01, "high": 10., "log": True}})
    assert spec.n_trials == 12
    pipeline = build_pipeline(spec, task, context(task, tmp_path))
    summary = pipeline.fit(X, y, context(task, tmp_path))
    assert summary.budget["sampler"] == "tpe"
    assert len(summary.inner_scores["trials"]) > summary.budget["n_startup_trials"]


def test_A22_default_autogluon_portfolio_is_explicit_and_bounded(tabular):
    from automl_agent.search import autogluon_portfolio
    _, _, task = tabular
    spec = PipelineSpec(engine="autogluon", model="library_search")
    portfolio = autogluon_portfolio(spec, task, ["GBM", "CAT", "XGB"])
    assert sum(map(len, portfolio.values())) == 6
    assert spec.autogluon_profile == "selection"
    assert all(len(values) == 2 and values[0] != values[1] for values in portfolio.values())


def test_A24_offsets_schedule_uses_final_output_checkpoints(tabular, tmp_path):
    X, y, task = classification(*tabular)
    spec = PipelineSpec(model="lightgbm", hyperparameters={"n_estimators": 8}, early_stopping_rounds=2,
        prediction_policy="class_offsets", postprocess_fit_policy="inner_holdout", decision_method="regularized")
    model = DirectPipeline(spec, task)
    summary = model.fit(X, y, context(task, tmp_path))
    schedule = summary.learning_curve["answer"]
    assert schedule["scope"] == "fit_inner_final_output_selection"
    assert {row["iterations"] for row in schedule["curve"]} == {2, 4, 8}
    assert frozen_decision_valid(model.frozen_spec(), task)


def test_A23_temporal_hpo_uses_supported_multiple_transitions(tabular, tmp_path):
    X,y,task=tabular
    X=X.assign(when=np.arange(len(X))*3//len(X))
    task=task.model_copy(update={'time_column':'when','feature_columns':list(X)})
    spec=PipelineSpec(engine='optuna',model='ridge',n_trials=2,hyperparameters={'alpha':1},
        search_space={'alpha':{'type':'categorical','choices':[1,2]}})
    ctx=context(task,tmp_path,'temporal')
    model=build_pipeline(spec,task,ctx); summary=model.fit(X,y,ctx)
    assert len(summary.budget['temporal_transitions']) == 2 and summary.rows == len(X)
    assert not summary.budget['temporal_transfer_quality_proven']


def test_A23_classification_hpo_keeps_metric_supported_holdout(tabular,tmp_path):
    from automl_agent.contracts import MetricSpec,OutputSpec
    X,y,task=tabular
    X=X.assign(when=np.arange(len(X))//60)
    labels=np.where(np.arange(len(X))%2,'positive','negative')
    labels[60:120]='negative'
    y=pd.DataFrame({'answer':labels},index=X.index)
    task=task.model_copy(update={'kind':'classification','classes':{'answer':['negative','positive']},
        'feature_columns':list(X),'time_column':'when','metric':MetricSpec(name='roc_auc',positive_label='positive'),
        'output':OutputSpec(columns=['id','p0','p1'],prediction_type='probabilities',
            probability_columns={'p0':('answer','negative'),'p1':('answer','positive')})})
    spec=PipelineSpec(engine='optuna',model='logistic',n_trials=1,
        search_space={'C':{'type':'categorical','choices':[1.]}})
    ctx=context(task,tmp_path,'temporal')
    model=build_pipeline(spec,task,ctx); summary=model.fit(X,y,ctx)
    assert summary.inner_scores['trials'][0]['state'] == 'completed'
    assert len(summary.budget['temporal_transitions']) == 1


def test_A25_pilot_preserves_structural_support(tabular):
    X, y, task = classification(*tabular)
    X = X.assign(entity=np.r_[np.zeros(150, dtype=int), np.arange(1, 31)])
    task = task.model_copy(update={"group_column": "entity", "feature_columns": list(X)})
    spec = PipelineSpec(model="lightgbm", hyperparameters={"n_estimators": 3}, early_stopping_rounds=2)
    rows, support = pilot_population(X, y, task, "group", 80, 42, spec)
    assert X.iloc[rows].entity.nunique() >= 6
    assert support["failure_on_small_sample_is_recipe_failure"] is False
    X2 = X.drop(columns="entity"); task2 = task.model_copy(update={"group_column":None, "feature_columns":list(X2)})
    rows, support = pilot_population(X2, y, task2, "random", 20, 42, spec)
    assert y.iloc[rows].answer.value_counts().min() >= 12


def test_A26_full_parent_delta_preserves_companions_and_clears_state():
    base = PipelineSpec(model="catboost", representation="native", hyperparameters={"iterations": 15, "depth": 4},
        target_transform="log1p", log_inverse="oof_smearing", early_stopping_rounds=3,
        include_features=["x"], learned_inverse={"answer": {"factor": 1.2}})
    c = SimpleNamespace(lookup=lambda _: (SimpleNamespace(pipeline=base), None))
    args = TrainArgs(parent_id="parent", pipeline={"target_transform":"log1p"}, hypothesis=hypothesis())
    inherit_parent(c, args)
    assert args.pipeline.research_recipe() == base.research_recipe()
    assert not args.pipeline.learned_inverse and args.pipeline.log_inverse == "oof_smearing"
    changed = TrainArgs(parent_id="parent", pipeline={"hyperparameters":{"iterations":20}, "imputation":{"indicators":False}}, hypothesis=hypothesis())
    inherit_parent(c, changed)
    assert changed.pipeline.hyperparameters == {"iterations":20, "depth":4}
    assert changed.pipeline.imputation.indicators is False
    boosting=PipelineSpec(model='lightgbm',hyperparameters={'subsample':.8,'subsample_freq':1})
    args=TrainArgs(parent_id='parent',pipeline={'hyperparameters':{'bagging_fraction':.9}},hypothesis=hypothesis())
    inherit_parent(SimpleNamespace(lookup=lambda _: (SimpleNamespace(pipeline=boosting),None)),args)
    assert args.pipeline.hyperparameters == {'subsample':.9,'subsample_freq':1}


def test_A27_pseudo_selection_ignores_file_prefix_and_balances_classes():
    from automl_agent.shift import pseudo_indices
    classes = ["a", "b", "rare"]
    X = pd.DataFrame({"x": np.arange(30)}, index=[f"r:{i}" for i in range(30)])
    p = np.full((30, 3), .005)
    for i in range(30): p[i, i//10] = .99
    selected = pseudo_indices(p, X, classes, classes, 6)
    assert np.bincount(p[selected].argmax(axis=1), minlength=3).tolist() == [2, 2, 2]
    order = np.random.default_rng(41).permutation(30)
    reordered = pseudo_indices(p[order], X.iloc[order], classes, classes, 6)
    assert set(X.index[selected]) == set(X.iloc[order].index[reordered])


def test_A27_student_gate_is_disjoint_and_its_fits_are_counted(tabular, tmp_path, monkeypatch):
    from automl_agent.api import prediction_bundle
    from automl_agent.shift import ShiftPipeline
    X, y, task = classification(*tabular)
    X = pd.DataFrame({'x':np.tile(np.arange(3),60)}, index=X.index)
    classes = task.classes['answer']
    y = pd.DataFrame({'answer':np.asarray(classes)[X.x]}, index=X.index)
    task = task.model_copy(update={'feature_columns':['x'], 'categorical_columns':[]})
    fits, gate = [], []
    class FakeDirect:
        def __init__(self, spec, task): self.spec, self.task = spec, task
        def fit(self, frame, labels, ctx):
            self.keys = set(frame.index); self.student = 'sample_weights' in ctx
            fits.append(self.keys)
            self.summary = TrainingSummary(rows=len(frame), seconds=.01,
                budget={'selected_fit_actual_seconds':.01,'selected_fit_rows':len(frame),'auxiliary_fit_actual_seconds':0})
            return self.summary
        def predict(self, frame, ctx):
            probabilities = np.full((len(frame),3), .005)
            probabilities[np.arange(len(frame)),frame.x.to_numpy()] = .99
            labels = np.asarray(classes)[frame.x.to_numpy()].copy()
            if ctx['split_id'] == 'student_gate':
                assert not self.keys & set(frame.index)
                gate.append(set(frame.index))
                if not self.student: labels[0] = classes[(frame.x.iloc[0]+1)%3]
            return prediction_bundle({'answer':labels.tolist()}, frame, ctx, 'labels', task.classes,
                raw_probabilities={'answer':probabilities.tolist()})
        def frozen_spec(self): return self.spec
    monkeypatch.setattr('automl_agent.pipelines.DirectPipeline', FakeDirect)
    model = ShiftPipeline(PipelineSpec(model='extra_trees', shift_strategy='pseudo_label'), task)
    future = X.iloc[:30].copy(); future.index = [f'future:{i}' for i in range(30)]
    summary = model.fit(X, y, {**context(task,tmp_path), 'unlabelled_features':future})
    assert len(fits) == 4 and len(gate) == 2 and gate[0] == gate[1]
    assert summary.budget['repeatable_auxiliary_fits'] == 3
    assert model.evidence['student_gate']['gain'] > 0
    assert not model.evidence['student_gate']['evaluation_labels_used_for_training']
    assert not model.evidence['student_gate']['target_domain_precision_known']


def test_A28_legacy_decision_matches_v076_math(tabular):
    _, y, task = classification(*tabular)
    rng = np.random.default_rng(7)
    p = rng.dirichlet([2., 2., .5], size=len(y))
    classes = task.classes["answer"]
    logits = np.log(np.clip(p, 1e-12, 1))
    offsets = np.zeros(len(classes))
    from automl_agent.evaluation import score_predictions
    def objective(values):
        labels = np.asarray(classes)[np.argmax(logits+values, axis=1)]
        return score_predictions(task, y, {"answer":labels}, "labels") - .001*float(values@values)
    best = objective(offsets)
    for _ in range(2):
        for column in range(len(classes)):
            selected = offsets.copy()
            for value in (-.5, -.25, 0., .25, .5):
                trial = offsets.copy(); trial[column] = value
                score = objective(trial)
                if score > best+1e-12: best, selected = score, trial
            offsets = selected
    state = fit_decision(task, y, p, y.index, method="legacy")
    assert state["offsets"] == offsets.tolist()
    assert np.array_equal(decide(p, classes, state), np.asarray(classes)[np.argmax(logits+offsets, axis=1)])
