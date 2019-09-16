import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from evalml import AutoClassifier, demos
from evalml.objectives import (
    FraudCost,
    Precision,
    PrecisionMicro,
    get_objectives
)
from evalml.pipelines import PipelineBase, get_pipelines
from evalml.problem_types import ProblemTypes


def test_init(X_y):
    X, y = X_y

    clf = AutoClassifier(multiclass=False)

    # check loads all pipelines
    assert get_pipelines(problem_type=ProblemTypes.BINARY) == clf.possible_pipelines

    clf.fit(X, y)

    assert isinstance(clf.rankings, pd.DataFrame)

    assert isinstance(clf.best_pipeline, PipelineBase)

    assert isinstance(clf.best_pipeline.feature_importances, pd.DataFrame)

    # test with datafarmes
    clf.fit(pd.DataFrame(X), pd.Series(y))

    assert isinstance(clf.rankings, pd.DataFrame)

    assert isinstance(clf.best_pipeline, PipelineBase)

    assert isinstance(clf.get_pipeline(0), PipelineBase)

    clf.describe_pipeline(0)


def test_cv(X_y):
    X, y = X_y
    cv_folds = 5
    clf = AutoClassifier(cv=StratifiedKFold(cv_folds), max_pipelines=1)

    clf.fit(X, y)

    assert isinstance(clf.rankings, pd.DataFrame)

    assert len(clf.results[0]["scores"]) == cv_folds

    clf = AutoClassifier(cv=TimeSeriesSplit(cv_folds), max_pipelines=1)

    clf.fit(X, y)

    assert isinstance(clf.rankings, pd.DataFrame)

    assert len(clf.results[0]["scores"]) == cv_folds


def test_init_select_model_types():
    model_types = ["random_forest"]
    clf = AutoClassifier(model_types=model_types)

    assert get_pipelines(problem_type=ProblemTypes.BINARY, model_types=model_types) == clf.possible_pipelines
    assert model_types == clf.possible_model_types


def test_max_pipelines(X_y):
    X, y = X_y
    max_pipelines = 6
    clf = AutoClassifier(max_pipelines=max_pipelines)

    clf.fit(X, y)

    assert len(clf.rankings) == max_pipelines


def test_best_pipeline(X_y):
    X, y = X_y
    max_pipelines = 3
    clf = AutoClassifier(max_pipelines=max_pipelines)

    clf.fit(X, y)

    assert len(clf.rankings) == max_pipelines


def test_specify_objective(X_y):
    X, y = X_y
    clf = AutoClassifier(objective=Precision(), max_pipelines=1)
    clf.fit(X, y)


def test_binary_auto(X_y):
    X, y = X_y
    clf = AutoClassifier(objective="recall", multiclass=False)
    clf.fit(X, y)
    y_pred = clf.best_pipeline.predict(X)
    assert len(np.unique(y_pred)) == 2


def test_multi_auto(X_y_multi):
    X, y = X_y_multi
    clf = AutoClassifier(objective="recall_micro", multiclass=True)
    clf.fit(X, y)
    y_pred = clf.best_pipeline.predict(X)
    assert len(np.unique(y_pred)) == 3

    objective = PrecisionMicro()
    clf = AutoClassifier(objective=objective, multiclass=True)
    clf.fit(X, y)
    y_pred = clf.best_pipeline.predict(X)
    assert len(np.unique(y_pred)) == 3

    assert clf.default_objectives == get_objectives('multiclass')


def test_random_state(X_y):
    X, y = X_y

    fc = FraudCost(
        retry_percentage=.5,
        interchange_fee=.02,
        fraud_payout_percentage=.75,
        amount_col=10
    )

    clf = AutoClassifier(objective=Precision(), max_pipelines=5, random_state=0)
    clf.fit(X, y)

    clf_1 = AutoClassifier(objective=Precision(), max_pipelines=5, random_state=0)
    clf_1.fit(X, y)
    assert clf.rankings.equals(clf_1.rankings)

    # test an objective that requires fitting
    clf = AutoClassifier(objective=fc, max_pipelines=5, random_state=30)
    clf.fit(X, y)

    clf_1 = AutoClassifier(objective=fc, max_pipelines=5, random_state=30)
    clf_1.fit(X, y)

    assert clf.rankings.equals(clf_1.rankings)


def test_callback(X_y):
    X, y = X_y

    counts = {
        "start_iteration_callback": 0,
        "add_result_callback": 0,
    }

    def start_iteration_callback(pipeline_class, parameters, counts=counts):
        counts["start_iteration_callback"] += 1

    def add_result_callback(results, trained_pipeline, counts=counts):
        counts["add_result_callback"] += 1

    max_pipelines = 3
    clf = AutoClassifier(objective=Precision(), max_pipelines=max_pipelines,
                         start_iteration_callback=start_iteration_callback,
                         add_result_callback=add_result_callback)
    clf.fit(X, y)

    assert counts["start_iteration_callback"] == max_pipelines
    assert counts["add_result_callback"] == max_pipelines


def test_select_scores():
    X, y = demos.load_breast_cancer()

    clf = AutoClassifier(objective="f1", max_pipelines=5)

    clf.fit(X, y)

    ret_dict = clf.describe_pipeline(0, show_objectives=['F1', 'AUC'], return_dict=True)
    dict_keys = ret_dict['all_objective_scores'][0].keys()
    assert 'F1' in dict_keys
    assert 'AUC' in dict_keys
    assert '# Training' in dict_keys
    assert '# Testing' in dict_keys

    # Make sure that show_objectives only filters output and return_dict
    ret_dict_2 = clf.describe_pipeline(0, return_dict=True)
    dict_keys_2 = ret_dict_2['all_objective_scores'][0].keys()
    assert 'Precision' in dict_keys_2

    error_msg = "{'fraud_objective'} not found in pipeline scores."
    with pytest.raises(Exception, match=error_msg):
        ret_dict = clf.describe_pipeline(0, show_objectives=['F1', 'fraud_objective'])

# def test_serialization(trained_model)
