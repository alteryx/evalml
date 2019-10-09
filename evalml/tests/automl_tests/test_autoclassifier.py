import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from evalml import AutoClassifier, demos
from evalml.model_types import ModelTypes
from evalml.objectives import (
    FraudCost,
    Precision,
    PrecisionMicro,
    get_objective,
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
    model_types = [ModelTypes.RANDOM_FOREST]
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

    expected_additional_objectives = get_objectives('multiclass')
    objective_in_additional_objectives = next((obj for obj in expected_additional_objectives if obj.name == objective.name), None)
    expected_additional_objectives.remove(objective_in_additional_objectives)
    assert clf.additional_objectives == expected_additional_objectives


def test_categorical_classification(X_y_categorical_classification):
    X, y = X_y_categorical_classification
    clf = AutoClassifier(objective="recall", max_pipelines=5, multiclass=False)
    clf.fit(X, y)
    assert not clf.rankings['score'].isnull().all()
    assert not clf.get_pipeline(0).feature_importances.isnull().all().all()


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


def test_additional_objectives(X_y):
    X, y = X_y

    objective = FraudCost(
        retry_percentage=.5,
        interchange_fee=.02,
        fraud_payout_percentage=.75,
        amount_col=10
    )

    clf = AutoClassifier(objective='F1', max_pipelines=2, additional_objectives=[objective])

    clf.fit(X, y)

    results = clf.describe_pipeline(0, return_dict=True)
    assert 'Fraud Cost' in list(results['all_objective_scores'][0].keys())


def test_describe_pipeline_objective_ordered(X_y, capsys):
    X, y = X_y
    clf = AutoClassifier(objective='AUC', max_pipelines=2)
    clf.fit(X, y)

    clf.describe_pipeline(0)
    out, err = capsys.readouterr()
    out_stripped = " ".join(out.split())

    objectives = [get_objective(obj) for obj in clf.additional_objectives]
    objectives_names = [clf.objective.name] + [obj.name for obj in objectives]
    expected_objective_order = " ".join(objectives_names)

    assert err == ''
    assert expected_objective_order in out_stripped


def test_model_types_as_list():
    with pytest.raises(TypeError, match="model_types parameter is not a list."):
        AutoClassifier(objective='AUC', model_types='linear_model', max_pipelines=2)


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
