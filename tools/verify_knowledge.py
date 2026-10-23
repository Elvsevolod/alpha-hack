"""Offline acceptance queries for the immutable reference bundle."""
from pathlib import Path
import json
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_agent.contracts import ReferenceArgs
from automl_agent.references import ReferenceStore
from automl_agent.util import atomic_json

ROOT=Path(__file__).resolve().parents[1]
QUERIES=[
    ('sklearn.impute.SimpleImputer','api_sklearn_impute_SimpleImputer'),
    ('sklearn.impute.IterativeImputer','api_sklearn_impute_IterativeImputer'),
    ('sklearn.preprocessing.OneHotEncoder','api_sklearn_preprocessing_OneHotEncoder'),
    ('sklearn.preprocessing.TargetEncoder','api_sklearn_preprocessing_TargetEncoder'),
    ('category_encoders.TargetEncoder','api_category_encoders_TargetEncoder'),
    ('sklearn.preprocessing.SplineTransformer','api_sklearn_preprocessing_SplineTransformer'),
    ('sklearn.feature_extraction.text.TfidfVectorizer','api_sklearn_feature_extraction_text_TfidfVectorizer'),
    ('sklearn.neural_network.MLPClassifier','api_sklearn_neural_network_MLPClassifier'),
    ('sklearn.linear_model.Ridge','api_sklearn_linear_model_Ridge'),
    ('sklearn.ensemble.ExtraTreesRegressor','api_sklearn_ensemble_ExtraTreesRegressor'),
    ('lightgbm.LGBMRanker.fit','api_lightgbm_LGBMRanker_fit'),
    ('catboost.CatBoostClassifier.fit','api_catboost_CatBoostClassifier_fit'),
    ('xgboost.XGBRegressor.predict','api_xgboost_XGBRegressor_predict'),
    ('flaml.AutoML.retrain_from_log','api_flaml_AutoML_retrain_from_log'),
    ('autogluon.tabular.TabularPredictor.refit_full','api_autogluon_tabular_TabularPredictor_refit_full'),
    ('optuna.study.Study.optimize','api_optuna_study_Study_optimize'),
    ('feature_engine.creation.RelativeFeatures','api_feature_engine_creation_RelativeFeatures'),
    ('imblearn.over_sampling.SMOTE','api_imblearn_over_sampling_SMOTE'),
    ('pandas.read_csv','api_pandas_read_csv'),
    ('statsmodels.tsa.statespace.sarimax.SARIMAX','api_statsmodels_tsa_statespace_sarimax_SARIMAX'),
    ('missing values median imputation','api_sklearn_impute_SimpleImputer'),
    ('iterative imputation BayesianRidge','api_sklearn_impute_IterativeImputer'),
    ('unknown categories OneHotEncoder handle_unknown','api_sklearn_preprocessing_OneHotEncoder'),
    ('time ordered temporal cross validation TimeSeriesSplit','api_sklearn_model_selection_TimeSeriesSplit'),
    ('non overlapping groups GroupShuffleSplit','api_sklearn_model_selection_GroupShuffleSplit'),
    ('text term frequency inverse document frequency','api_sklearn_feature_extraction_text_TfidfVectorizer'),
    ('MLPClassifier sample_weight early_stopping','api_sklearn_neural_network_MLPClassifier_fit'),
    ('LGBMRanker query groups ranking','api_lightgbm_LGBMRanker_fit'),
    ('AutoGluon tuning_data bagging stacking','api_autogluon_tabular_TabularPredictor_fit'),
    ('Optuna sequential timeout n_jobs','api_optuna_study_Study_optimize'),
    ('пропуски SimpleImputer медиана','api_sklearn_impute_SimpleImputer'),
    ('модельная импутация IterativeImputer','api_sklearn_impute_IterativeImputer'),
    ('категории OneHotEncoder неизвестные','api_sklearn_preprocessing_OneHotEncoder'),
    ('временные периоды TimeSeriesSplit','api_sklearn_model_selection_TimeSeriesSplit'),
    ('группы GroupShuffleSplit','api_sklearn_model_selection_GroupShuffleSplit'),
    ('текст TfidfVectorizer','api_sklearn_feature_extraction_text_TfidfVectorizer'),
    ('нейросеть MLPClassifier','api_sklearn_neural_network_MLPClassifier'),
    ('сплайн SplineTransformer гладкие зависимости','api_sklearn_preprocessing_SplineTransformer'),
    ('сохранение CatBoostClassifier save_model','api_catboost_CatBoostClassifier_save_model'),
    ('вероятности LogisticRegression predict_proba','api_sklearn_linear_model_LogisticRegression_predict_proba'),
]


def verify():
    store=ReferenceStore(ROOT/'knowledge_runtime')
    store.verify(ROOT/'data_for_participants/requirements.txt')
    results=[]
    for query,expected in QUERIES:
        started=time.perf_counter()
        response=store.query(ReferenceArgs(mode='search',query=query,max_chars=6000))
        actual=[entry['record']['doc_id'] for entry in response.get('entries',[])]
        results.append({'query':query,'expected':expected,'actual':actual,'hit':expected in actual,'seconds':time.perf_counter()-started})
    critical=[]
    for symbol in ('sklearn.preprocessing.TargetEncoder','category_encoders.TargetEncoder'):
        response=store.query(ReferenceArgs(mode='api',query=symbol))
        critical.append({'check':symbol,'passed':response['entries'][0]['record']['api_symbols']==[symbol]})
    response=store.query(ReferenceArgs(mode='api',query='sklearn.impute.SimpleImputer',library='scikit-learn',version='999.0'))
    critical.append({'check':'missing_version','passed':response['status']=='missing_version'})
    response=store.query(ReferenceArgs(mode='read',reference_id='not_a_real_document'))
    critical.append({'check':'unknown_document','passed':response['status']=='not_found'})
    report={'ordinary_queries':len(results),'recall_at_5':sum(x['hit'] for x in results)/len(results),
        'critical':critical,'all_critical_passed':all(x['passed'] for x in critical),'results':results}
    atomic_json(ROOT/'outputs/knowledge_search_acceptance.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'results','critical'}}))
    return report


if __name__=='__main__':
    report=verify()
    raise SystemExit(0 if report['recall_at_5']>=.9 and report['all_critical_passed'] else 1)
