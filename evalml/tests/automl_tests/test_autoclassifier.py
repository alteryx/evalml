import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from evalml import AutoClassifier
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

    clf = AutoClassifier(multiclass=False, max_pipelines=1)

    # check loads all pipelines
    assert get_pipelines(problem_type=ProblemTypes.BINARY) == clf.possible_pipelines

    clf.fit(X, y, raise_errors=True)

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
    clf.fit(X, y, raise_errors=True)

    assert isinstance(clf.rankings, pd.DataFrame)
    assert len(clf.results['pipeline_results'][0]["cv_data"]) == cv_folds

    clf = AutoClassifier(cv=TimeSeriesSplit(cv_folds), max_pipelines=1)
    clf.fit(X, y, raise_errors=True)

    assert isinstance(clf.rankings, pd.DataFrame)
    assert len(clf.results['pipeline_results'][0]["cv_data"]) == cv_folds


def test_init_select_model_types():
    model_types = [ModelTypes.RANDOM_FOREST]
    clf = AutoClassifier(model_types=model_types)

    assert get_pipelines(problem_type=ProblemTypes.BINARY, model_types=model_types) == clf.possible_pipelines
    assert model_types == clf.possible_model_types


def test_max_pipelines(X_y):
    X, y = X_y
    max_pipelines = 5
    clf = AutoClassifier(max_pipelines=max_pipelines)
    clf.fit(X, y, raise_errors=True)

    assert len(clf.rankings) == max_pipelines


def test_best_pipeline(X_y):
    X, y = X_y
    max_pipelines = 5
    clf = AutoClassifier(max_pipelines=max_pipelines)
    clf.fit(X, y, raise_errors=True)

    assert len(clf.rankings) == max_pipelines


def test_specify_objective(X_y):
    X, y = X_y
    clf = AutoClassifier(objective=Precision(), max_pipelines=1)
    clf.fit(X, y, raise_errors=True)


def test_binary_auto(X_y):
    X, y = X_y
    clf = AutoClassifier(objective="recall", multiclass=False, max_pipelines=5)
    clf.fit(X, y, raise_errors=True)
    y_pred = clf.best_pipeline.predict(X)

    assert len(np.unique(y_pred)) == 2


def test_multi_error(X_y_multi):
    X, y = X_y_multi
    error_clfs = [AutoClassifier(objective='recall'), AutoClassifier(objective='recall_micro', additional_objectives=['recall'], multiclass=True)]
    error_msg = 'not compatible with a multiclass problem.'
    for clf in error_clfs:
        with pytest.raises(ValueError, match=error_msg):
            clf.fit(X, y)


def test_multi_auto(X_y_multi):
    X, y = X_y_multi
    clf = AutoClassifier(objective="recall_micro", multiclass=True, max_pipelines=5)
    clf.fit(X, y, raise_errors=True)
    y_pred = clf.best_pipeline.predict(X)
    assert len(np.unique(y_pred)) == 3

    objective = PrecisionMicro()
    clf = AutoClassifier(objective=objective, multiclass=True, max_pipelines=5)
    clf.fit(X, y, raise_errors=True)
    y_pred = clf.best_pipeline.predict(X)
    assert len(np.unique(y_pred)) == 3

    expected_additional_objectives = get_objectives('multiclass')
    objective_in_additional_objectives = next((obj for obj in expected_additional_objectives if obj.name == objective.name), None)
    expected_additional_objectives.remove(objective_in_additional_objectives)
    for expected, additional in zip(expected_additional_objectives, clf.additional_objectives):
        assert type(additional) is type(expected)


def test_multi_objective(X_y_multi):
    error_msg = 'Given objective Recall is not compatible with a multiclass problem'
    with pytest.raises(ValueError, match=error_msg):
        clf = AutoClassifier(objective="recall", multiclass=True)

    clf = AutoClassifier(objective="log_loss")
    assert clf.problem_type == ProblemTypes.BINARY

    clf = AutoClassifier(objective='recall_micro')
    assert clf.problem_type == ProblemTypes.MULTICLASS

    clf = AutoClassifier(objective='recall')
    assert clf.problem_type == ProblemTypes.BINARY

    clf = AutoClassifier(multiclass=True)
    assert clf.problem_type == ProblemTypes.MULTICLASS

    clf = AutoClassifier()
    assert clf.problem_type == ProblemTypes.BINARY


def test_categorical_classification(X_y_categorical_classification):
    X, y = X_y_categorical_classification
    clf = AutoClassifier(objective="recall", max_pipelines=5, multiclass=False)
    clf.fit(X, y, raise_errors=True)
    assert not clf.rankings['score'].isnull().all()
    assert not clf.get_pipeline(0).feature_importances.isnull().all().all()


def test_random_state(X_y):
    X, y = X_y

    fc = FraudCost(retry_percentage=.5,
                   interchange_fee=.02,
                   fraud_payout_percentage=.75,
                   amount_col=10)

    clf = AutoClassifier(objective=Precision(), max_pipelines=5, random_state=0)
    clf.fit(X, y, raise_errors=True)

    clf_1 = AutoClassifier(objective=Precision(), max_pipelines=5, random_state=0)
    clf_1.fit(X, y, raise_errors=True)
    assert clf.rankings.equals(clf_1.rankings)

    # test an objective that requires fitting
    clf = AutoClassifier(objective=fc, max_pipelines=5, random_state=30)
    clf.fit(X, y, raise_errors=True)

    clf_1 = AutoClassifier(objective=fc, max_pipelines=5, random_state=30)
    clf_1.fit(X, y, raise_errors=True)

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
    clf.fit(X, y, raise_errors=True)

    assert counts["start_iteration_callback"] == max_pipelines
    assert counts["add_result_callback"] == max_pipelines


def test_additional_objectives(X_y):
    X, y = X_y

    objective = FraudCost(retry_percentage=.5,
                          interchange_fee=.02,
                          fraud_payout_percentage=.75,
                          amount_col=10)
    clf = AutoClassifier(objective='F1', max_pipelines=2, additional_objectives=[objective])
    clf.fit(X, y, raise_errors=True)

    results = clf.describe_pipeline(0, return_dict=True)
    assert 'Fraud Cost' in list(results["cv_data"][0]["all_objective_scores"].keys())


def test_describe_pipeline_objective_ordered(X_y, capsys):
    X, y = X_y
    clf = AutoClassifier(objective='AUC', max_pipelines=2)
    clf.fit(X, y, raise_errors=True)

    clf.describe_pipeline(0)
    out, err = capsys.readouterr()
    out_stripped = " ".join(out.split())

    objectives = [get_objective(obj) for obj in clf.additional_objectives]
    objectives_names = [clf.objective.name] + [obj.name for obj in objectives if obj.name not in ["ROC", "Confusion Matrix"]]
    expected_objective_order = " ".join(objectives_names)

    assert err == ''
    assert expected_objective_order in out_stripped


def test_model_types_as_list():
    with pytest.raises(TypeError, match="model_types parameter is not a list."):
        AutoClassifier(objective='AUC', model_types='linear_model', max_pipelines=2)


def test_max_time_units():
    str_max_time = AutoClassifier(objective='F1', max_time='60 seconds')
    assert str_max_time.max_time == 60

    hour_max_time = AutoClassifier(objective='F1', max_time='1 hour')
    assert hour_max_time.max_time == 3600

    min_max_time = AutoClassifier(objective='F1', max_time='30 mins')
    assert min_max_time.max_time == 1800

    min_max_time = AutoClassifier(objective='F1', max_time='30 s')
    assert min_max_time.max_time == 30

    with pytest.raises(AssertionError, match="Invalid unit. Units must be hours, mins, or seconds. Received 'year'"):
        AutoClassifier(objective='F1', max_time='30 years')

    with pytest.raises(TypeError, match="max_time must be a float, int, or string. Received a <class 'tuple'>."):
        AutoClassifier(objective='F1', max_time=(30, 'minutes'))


def test_early_stopping(capsys, X_y):
    X, y = X_y

    with pytest.raises(ValueError, match='patience value must be a positive integer.'):
        clf = AutoClassifier(objective='AUC', max_pipelines=5, model_types=['linear_model'], patience=-1, random_state=0)

    with pytest.raises(ValueError, match='tolerance value must be'):
        clf = AutoClassifier(objective='AUC', max_pipelines=5, model_types=['linear_model'], patience=1, tolerance=1.5, random_state=0)

    clf = AutoClassifier(objective='AUC', max_pipelines=10, model_types=['linear_model'], patience=2, random_state=0)
    clf.fit(X, y, raise_errors=True)
    out, _ = capsys.readouterr()
    assert "2 iterations without improvement. Stopping search early." in out

    num_without_improvement = 0
    best_score = None
    for id in clf.results['search_order']:
        score = clf.results['pipeline_results'][id]['score']
        if best_score is None:
            best_score = score
            continue
        if score > best_score:
            num_without_improvement = 0
            best_score = score
        else:
            num_without_improvement += 1
    assert num_without_improvement == 2
