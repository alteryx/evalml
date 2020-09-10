import os
from itertools import product
from unittest.mock import MagicMock, patch

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from evalml import AutoMLSearch
from evalml.automl import TrainingValidationSplit
from evalml.data_checks import (
    DataCheck,
    DataCheckError,
    DataChecks,
    DataCheckWarning
)
from evalml.demos import load_breast_cancer, load_wine
from evalml.exceptions import AutoMLSearchException, PipelineNotFoundError
from evalml.model_family import ModelFamily
from evalml.objectives import FraudCost
from evalml.objectives.utils import _all_objectives_dict
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline
)
from evalml.pipelines.components.utils import get_estimators
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types import ProblemTypes
from evalml.tuners import NoParamsException, RandomSearchTuner
from evalml.utils.gen_utils import (
    categorical_dtypes,
    numeric_and_boolean_dtypes
)


@pytest.mark.parametrize("automl_type", [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_search_results(X_y_regression, X_y_binary, X_y_multi, automl_type):
    expected_cv_data_keys = {'all_objective_scores', 'score', 'binary_classification_threshold'}
    automl = AutoMLSearch(problem_type=automl_type, max_pipelines=2)
    if automl_type == ProblemTypes.REGRESSION:
        expected_pipeline_class = RegressionPipeline
        X, y = X_y_regression
    elif automl_type == ProblemTypes.BINARY:
        expected_pipeline_class = BinaryClassificationPipeline
        X, y = X_y_binary
    elif automl_type == ProblemTypes.MULTICLASS:
        expected_pipeline_class = MulticlassClassificationPipeline
        X, y = X_y_multi

    automl.search(X, y)
    assert automl.results.keys() == {'pipeline_results', 'search_order'}
    assert automl.results['search_order'] == [0, 1]
    assert len(automl.results['pipeline_results']) == 2
    for pipeline_id, results in automl.results['pipeline_results'].items():
        assert results.keys() == {'id', 'pipeline_name', 'pipeline_class', 'pipeline_summary', 'parameters', 'score', 'high_variance_cv', 'training_time',
                                  'cv_data', 'percent_better_than_baseline'}
        assert results['id'] == pipeline_id
        assert isinstance(results['pipeline_name'], str)
        assert issubclass(results['pipeline_class'], expected_pipeline_class)
        assert isinstance(results['pipeline_summary'], str)
        assert isinstance(results['parameters'], dict)
        assert isinstance(results['score'], float)
        assert isinstance(results['high_variance_cv'], np.bool_)
        assert isinstance(results['cv_data'], list)
        for cv_result in results['cv_data']:
            assert cv_result.keys() == expected_cv_data_keys
            if automl_type == ProblemTypes.BINARY:
                assert isinstance(cv_result['binary_classification_threshold'], float)
            else:
                assert cv_result['binary_classification_threshold'] is None
            all_objective_scores = cv_result["all_objective_scores"]
            for score in all_objective_scores.values():
                assert score is not None
        assert automl.get_pipeline(pipeline_id).parameters == results['parameters']
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.full_rankings, pd.DataFrame)
    assert np.all(automl.rankings.dtypes == pd.Series(
        [np.dtype('int64'), np.dtype('O'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('O')],
        index=['id', 'pipeline_name', 'score', 'percent_better_than_baseline', 'high_variance_cv', 'parameters']))
    assert np.all(automl.full_rankings.dtypes == pd.Series(
        [np.dtype('int64'), np.dtype('O'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('O')],
        index=['id', 'pipeline_name', 'score', 'percent_better_than_baseline', 'high_variance_cv', 'parameters']))


@pytest.mark.parametrize("automl_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION])
@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_pipeline_limits(mock_fit_binary, mock_score_binary,
                         mock_fit_multi, mock_score_multi,
                         mock_fit_regression, mock_score_regression,
                         automl_type, caplog,
                         X_y_binary, X_y_multi, X_y_regression):
    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_binary
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
    elif automl_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression

    mock_score_binary.return_value = {'Log Loss Binary': 1.0}
    mock_score_multi.return_value = {'Log Loss Multiclass': 1.0}
    mock_score_regression.return_value = {'R2': 1.0}

    automl = AutoMLSearch(problem_type=automl_type, max_pipelines=1)
    automl.search(X, y)
    out = caplog.text
    assert "Searching up to 1 pipelines. " in out
    assert len(automl.results['pipeline_results']) == 1

    caplog.clear()
    automl = AutoMLSearch(problem_type=automl_type, max_time=1)
    automl.search(X, y)
    out = caplog.text
    assert "Will stop searching for new pipelines after 1 seconds" in out
    assert len(automl.results['pipeline_results']) >= 1

    caplog.clear()
    automl = AutoMLSearch(problem_type=automl_type, max_time=1, max_pipelines=5)
    automl.search(X, y)
    out = caplog.text
    assert "Searching up to 5 pipelines. " in out
    assert "Will stop searching for new pipelines after 1 seconds" in out
    assert len(automl.results['pipeline_results']) <= 5

    caplog.clear()
    automl = AutoMLSearch(problem_type=automl_type)
    automl.search(X, y)
    out = caplog.text
    assert "Using default limit of max_pipelines=5." in out
    assert len(automl.results['pipeline_results']) <= 5

    caplog.clear()
    automl = AutoMLSearch(problem_type=automl_type, max_time=1e-16)
    automl.search(X, y)
    out = caplog.text
    assert "Will stop searching for new pipelines after 0 seconds" in out
    # search will always run at least one pipeline
    assert len(automl.results['pipeline_results']) >= 1


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_pipeline_fit_raises(mock_fit, X_y_binary, caplog):
    msg = 'all your model are belong to us'
    mock_fit.side_effect = Exception(msg)
    X, y = X_y_binary

    automl = AutoMLSearch(problem_type='binary', max_pipelines=1)
    automl.search(X, y)
    out = caplog.text
    assert 'Exception during automl search' in out
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1

    cv_scores_all = pipeline_results[0].get('cv_data', {})
    for cv_scores in cv_scores_all:
        for name, score in cv_scores['all_objective_scores'].items():
            if name in ['# Training', '# Testing']:
                assert score > 0
            else:
                assert np.isnan(score)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
def test_pipeline_score_raises(mock_score, X_y_binary, caplog):
    msg = 'all your model are belong to us'
    mock_score.side_effect = Exception(msg)
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', max_pipelines=1)

    automl.search(X, y)
    out = caplog.text
    assert 'Exception during automl search' in out
    assert 'All scores will be replaced with nan.' in out
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1
    cv_scores_all = pipeline_results[0]["cv_data"][0]["all_objective_scores"]
    objective_scores = {o.name: cv_scores_all[o.name] for o in [automl.objective] + automl.additional_objectives}

    assert np.isnan(list(objective_scores.values())).all()


@patch('evalml.objectives.AUC.score')
def test_objective_score_raises(mock_score, X_y_binary, caplog):
    msg = 'all your model are belong to us'
    mock_score.side_effect = Exception(msg)
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', max_pipelines=1)

    automl.search(X, y)
    out = caplog.text

    assert msg in out
    pipeline_results = automl.results.get('pipeline_results')
    assert len(pipeline_results) == 1
    cv_scores_all = pipeline_results[0].get('cv_data')
    scores = cv_scores_all[0]['all_objective_scores']
    auc_score = scores.pop('AUC')
    assert np.isnan(auc_score)
    assert not np.isnan(list(scores.values())).any()


def test_rankings(X_y_binary, X_y_regression):
    X, y = X_y_binary
    model_families = ['random_forest']
    automl = AutoMLSearch(problem_type='binary', allowed_model_families=model_families, max_pipelines=3)
    automl.search(X, y)
    assert len(automl.full_rankings) == 3
    assert len(automl.rankings) == 2

    X, y = X_y_regression
    automl = AutoMLSearch(problem_type='regression', allowed_model_families=model_families, max_pipelines=3)
    automl.search(X, y)
    assert len(automl.full_rankings) == 3
    assert len(automl.rankings) == 2


@patch('evalml.objectives.BinaryClassificationObjective.optimize_threshold')
@patch('evalml.pipelines.BinaryClassificationPipeline.predict_proba')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_str_search(mock_fit, mock_score, mock_predict_proba, mock_optimize_threshold, X_y_binary):
    def _dummy_callback(param1, param2, param3):
        return None

    X, y = X_y_binary
    search_params = {
        'problem_type': 'binary',
        'objective': 'F1',
        'max_time': 100,
        'max_pipelines': 5,
        'patience': 2,
        'tolerance': 0.5,
        'allowed_model_families': ['random_forest', 'linear_model'],
        'data_split': StratifiedKFold(5),
        'tuner_class': RandomSearchTuner,
        'start_iteration_callback': _dummy_callback,
        'add_result_callback': None,
        'additional_objectives': ['Precision', 'AUC'],
        'n_jobs': 2,
        'optimize_thresholds': True
    }

    param_str_reps = {
        'Objective': search_params['objective'],
        'Max Time': search_params['max_time'],
        'Max Pipelines': search_params['max_pipelines'],
        'Allowed Pipelines': [],
        'Patience': search_params['patience'],
        'Tolerance': search_params['tolerance'],
        'Data Splitting': 'StratifiedKFold(n_splits=5, random_state=None, shuffle=False)',
        'Tuner': 'RandomSearchTuner',
        'Start Iteration Callback': '_dummy_callback',
        'Add Result Callback': None,
        'Additional Objectives': search_params['additional_objectives'],
        'Random State': 'RandomState(MT19937)',
        'n_jobs': search_params['n_jobs'],
        'Optimize Thresholds': search_params['optimize_thresholds']
    }

    automl = AutoMLSearch(**search_params)
    mock_score.return_value = {automl.objective.name: 1.0}
    mock_optimize_threshold.return_value = 0.62
    str_rep = str(automl)
    for param, value in param_str_reps.items():
        if isinstance(value, list):
            assert f"{param}" in str_rep
            for item in value:
                assert f"\t{str(item)}" in str_rep
        else:
            assert f"{param}: {str(value)}" in str_rep
    assert "Search Results" not in str_rep

    mock_score.return_value = {automl.objective.name: 1.0}
    automl.search(X, y)
    mock_fit.assert_called()
    mock_score.assert_called()
    mock_predict_proba.assert_called()
    mock_optimize_threshold.assert_called()

    str_rep = str(automl)
    assert "Search Results:" in str_rep
    assert automl.rankings.drop(['parameters'], axis='columns').to_string() in str_rep


def test_automl_data_check_results_is_none_before_search():
    automl = AutoMLSearch(problem_type='binary', max_pipelines=1)
    assert automl.data_check_results is None


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_empty_data_checks(mock_fit, mock_score):
    X = pd.DataFrame({"feature1": [1, 2, 3],
                      "feature2": [None, None, None]})
    y = pd.Series([1, 1, 1])

    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(problem_type="binary", max_pipelines=1)

    automl.search(X, y, data_checks=[])
    assert automl.data_check_results is None
    mock_fit.assert_called()
    mock_score.assert_called()

    automl.search(X, y, data_checks="disabled")
    assert automl.data_check_results is None

    automl.search(X, y, data_checks=None)
    assert automl.data_check_results is None


@patch('evalml.data_checks.DefaultDataChecks.validate')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_default_data_checks(mock_fit, mock_score, mock_validate, X_y_binary, caplog):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}
    mock_validate.return_value = [DataCheckWarning("default data check warning", "DefaultDataChecks")]
    automl = AutoMLSearch(problem_type='binary', max_pipelines=1)
    automl.search(X, y)
    out = caplog.text
    assert "default data check warning" in out
    assert automl.data_check_results == mock_validate.return_value
    mock_fit.assert_called()
    mock_score.assert_called()
    mock_validate.assert_called()


class MockDataCheckErrorAndWarning(DataCheck):
    def validate(self, X, y):
        return [DataCheckError("error one", self.name), DataCheckWarning("warning one", self.name)]


@pytest.mark.parametrize("data_checks",
                         [[MockDataCheckErrorAndWarning()],
                          DataChecks([MockDataCheckErrorAndWarning()])])
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_data_checks_raises_error(mock_fit, mock_score, data_checks, caplog):
    X = pd.DataFrame()
    y = pd.Series()

    automl = AutoMLSearch(problem_type="binary", max_pipelines=1)

    with pytest.raises(ValueError, match="Data checks raised"):
        automl.search(X, y, data_checks=data_checks)

    out = caplog.text
    assert "error one" in out
    assert "warning one" in out
    assert automl.data_check_results == MockDataCheckErrorAndWarning().validate(X, y)


def test_automl_bad_data_check_parameter_type():
    X = pd.DataFrame()
    y = pd.Series()

    automl = AutoMLSearch(problem_type="binary", max_pipelines=1)

    with pytest.raises(ValueError, match="Parameter data_checks must be a list. Received int."):
        automl.search(X, y, data_checks=1)
    with pytest.raises(ValueError, match="All elements of parameter data_checks must be an instance of DataCheck."):
        automl.search(X, y, data_checks=[1])
    with pytest.raises(ValueError, match="If data_checks is a string, it must be either 'auto' or 'disabled'. "
                                         "Received 'default'."):
        automl.search(X, y, data_checks="default")
    with pytest.raises(ValueError, match="All elements of parameter data_checks must be an instance of DataCheck."):
        automl.search(X, y, data_checks=[DataChecks([]), 1])


def test_automl_str_no_param_search():
    automl = AutoMLSearch(problem_type='binary')

    param_str_reps = {
        'Objective': 'Log Loss Binary',
        'Max Time': 'None',
        'Max Pipelines': 'None',
        'Allowed Pipelines': [],
        'Patience': 'None',
        'Tolerance': '0.0',
        'Data Splitting': 'StratifiedKFold(n_splits=3, random_state=0, shuffle=True)',
        'Tuner': 'SKOptTuner',
        'Additional Objectives': [
            'AUC',
            'Accuracy Binary',
            'Balanced Accuracy Binary',
            'F1',
            'MCC Binary',
            'Precision'],
        'Start Iteration Callback': 'None',
        'Add Result Callback': 'None',
        'Random State': 'RandomState(MT19937)',
        'n_jobs': '-1',
        'Verbose': 'True',
        'Optimize Thresholds': 'False'
    }

    str_rep = str(automl)
    for param, value in param_str_reps.items():
        assert f"{param}" in str_rep
        if isinstance(value, list):
            value = "\n".join(["\t{}".format(item) for item in value])
            assert value in str_rep
    assert "Search Results" not in str_rep


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_feature_selection(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    class MockFeatureSelectionPipeline(BinaryClassificationPipeline):
        component_graph = ['RF Classifier Select From Model', 'Logistic Regression Classifier']

        def fit(self, X, y):
            """Mock fit, noop"""

    allowed_pipelines = [MockFeatureSelectionPipeline]
    start_iteration_callback = MagicMock()
    automl = AutoMLSearch(problem_type='binary', max_pipelines=2, start_iteration_callback=start_iteration_callback, allowed_pipelines=allowed_pipelines)
    automl.search(X, y)

    assert start_iteration_callback.call_count == 2
    proposed_parameters = start_iteration_callback.call_args_list[1][0][1]
    assert proposed_parameters.keys() == {'RF Classifier Select From Model', 'Logistic Regression Classifier'}
    assert proposed_parameters['RF Classifier Select From Model']['number_features'] == X.shape[1]


@patch('evalml.tuners.random_search_tuner.RandomSearchTuner.is_search_space_exhausted')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_tuner_exception(mock_fit, mock_score, mock_is_search_space_exhausted, X_y_binary):
    mock_score.return_value = {'Log Loss Binary': 1.0}
    X, y = X_y_binary
    error_text = "Cannot create a unique set of unexplored parameters. Try expanding the search space."
    mock_is_search_space_exhausted.side_effect = NoParamsException(error_text)
    clf = AutoMLSearch(problem_type='regression', objective="R2", tuner_class=RandomSearchTuner, max_pipelines=10)
    with pytest.raises(NoParamsException, match=error_text):
        clf.search(X, y)


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_algorithm(mock_fit, mock_score, mock_algo_next_batch, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}
    mock_algo_next_batch.side_effect = StopIteration("that's all, folks")
    automl = AutoMLSearch(problem_type='binary', max_pipelines=5)
    automl.search(X, y)
    assert automl.data_check_results is None
    mock_fit.assert_called()
    mock_score.assert_called()
    assert mock_algo_next_batch.call_count == 1
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1
    assert pipeline_results[0].get('score') == 1.0


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.__init__')
def test_automl_allowed_pipelines_algorithm(mock_algo_init, dummy_binary_pipeline_class, X_y_binary):
    mock_algo_init.side_effect = Exception('mock algo init')
    X, y = X_y_binary

    allowed_pipelines = [dummy_binary_pipeline_class]
    automl = AutoMLSearch(problem_type='binary', allowed_pipelines=allowed_pipelines, max_pipelines=10)
    with pytest.raises(Exception, match='mock algo init'):
        automl.search(X, y)
    assert mock_algo_init.call_count == 1
    _, kwargs = mock_algo_init.call_args
    assert kwargs['max_pipelines'] == 10
    assert kwargs['allowed_pipelines'] == allowed_pipelines

    allowed_model_families = [ModelFamily.RANDOM_FOREST]
    automl = AutoMLSearch(problem_type='binary', allowed_model_families=allowed_model_families, max_pipelines=1)
    with pytest.raises(Exception, match='mock algo init'):
        automl.search(X, y)
    assert mock_algo_init.call_count == 2
    _, kwargs = mock_algo_init.call_args
    assert kwargs['max_pipelines'] == 1
    for actual, expected in zip(kwargs['allowed_pipelines'], [make_pipeline(X, y, estimator, ProblemTypes.BINARY) for estimator in get_estimators(ProblemTypes.BINARY, model_families=allowed_model_families)]):
        assert actual.parameters == expected.parameters


def test_automl_serialization(X_y_binary, tmpdir):
    X, y = X_y_binary
    path = os.path.join(str(tmpdir), 'automl.pkl')
    num_max_pipelines = 5
    automl = AutoMLSearch(problem_type='binary', max_pipelines=num_max_pipelines)
    automl.search(X, y)
    automl.save(path)
    loaded_automl = automl.load(path)
    for i in range(num_max_pipelines):
        assert automl.get_pipeline(i).__class__ == loaded_automl.get_pipeline(i).__class__
        assert automl.get_pipeline(i).parameters == loaded_automl.get_pipeline(i).parameters
        assert automl.results == loaded_automl.results
        pd.testing.assert_frame_equal(automl.rankings, loaded_automl.rankings)


@patch('cloudpickle.dump')
def test_automl_serialization_protocol(mock_cloudpickle_dump, tmpdir):
    path = os.path.join(str(tmpdir), 'automl.pkl')
    automl = AutoMLSearch(problem_type='binary', max_pipelines=5)

    automl.save(path)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert mock_cloudpickle_dump.call_args_list[0][1]['protocol'] == cloudpickle.DEFAULT_PROTOCOL

    mock_cloudpickle_dump.reset_mock()
    automl.save(path, pickle_protocol=42)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert mock_cloudpickle_dump.call_args_list[0][1]['protocol'] == 42


def test_invalid_data_splitter():
    data_splitter = pd.DataFrame()
    with pytest.raises(ValueError, match='Not a valid data splitter'):
        AutoMLSearch(problem_type='binary', data_split=data_splitter)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
def test_large_dataset_binary(mock_score):
    X = pd.DataFrame({'col_0': [i for i in range(101000)]})
    y = pd.Series([i % 2 for i in range(101000)])

    fraud_objective = FraudCost(amount_col='col_0')

    automl = AutoMLSearch(problem_type='binary',
                          objective=fraud_objective,
                          additional_objectives=['auc', 'f1', 'precision'],
                          max_time=1,
                          max_pipelines=1,
                          optimize_thresholds=True)
    mock_score.return_value = {automl.objective.name: 1.234}
    assert automl.data_split is None
    automl.search(X, y)
    assert isinstance(automl.data_split, TrainingValidationSplit)
    assert automl.data_split.get_n_splits() == 1

    for pipeline_id in automl.results['search_order']:
        assert len(automl.results['pipeline_results'][pipeline_id]['cv_data']) == 1
        assert automl.results['pipeline_results'][pipeline_id]['cv_data'][0]['score'] == 1.234


@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
def test_large_dataset_multiclass(mock_score):
    X = pd.DataFrame({'col_0': [i for i in range(101000)]})
    y = pd.Series([i % 4 for i in range(101000)])

    automl = AutoMLSearch(problem_type='multiclass', max_time=1, max_pipelines=1)
    mock_score.return_value = {automl.objective.name: 1.234}
    assert automl.data_split is None
    automl.search(X, y)
    assert isinstance(automl.data_split, TrainingValidationSplit)
    assert automl.data_split.get_n_splits() == 1

    for pipeline_id in automl.results['search_order']:
        assert len(automl.results['pipeline_results'][pipeline_id]['cv_data']) == 1
        assert automl.results['pipeline_results'][pipeline_id]['cv_data'][0]['score'] == 1.234


@patch('evalml.pipelines.RegressionPipeline.score')
def test_large_dataset_regression(mock_score):
    X = pd.DataFrame({'col_0': [i for i in range(101000)]})
    y = pd.Series([i for i in range(101000)])

    automl = AutoMLSearch(problem_type='regression', max_time=1, max_pipelines=1)
    mock_score.return_value = {automl.objective.name: 1.234}
    assert automl.data_split is None
    automl.search(X, y)
    assert isinstance(automl.data_split, TrainingValidationSplit)
    assert automl.data_split.get_n_splits() == 1

    for pipeline_id in automl.results['search_order']:
        assert len(automl.results['pipeline_results'][pipeline_id]['cv_data']) == 1
        assert automl.results['pipeline_results'][pipeline_id]['cv_data'][0]['score'] == 1.234


def test_allowed_pipelines_with_incorrect_problem_type(dummy_binary_pipeline_class):
    # checks that not setting allowed_pipelines does not error out
    AutoMLSearch(problem_type='binary')

    with pytest.raises(ValueError, match="is not compatible with problem_type"):
        AutoMLSearch(problem_type='regression', allowed_pipelines=[dummy_binary_pipeline_class])


def test_main_objective_problem_type_mismatch():
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(problem_type='binary', objective='R2')


def test_init_problem_type_error():
    with pytest.raises(ValueError, match=r"choose one of \(binary, multiclass, regression\) as problem_type"):
        AutoMLSearch()

    with pytest.raises(KeyError, match=r"does not exist"):
        AutoMLSearch(problem_type='multi')


def test_init_objective():
    defaults = {'multiclass': 'Log Loss Multiclass', 'binary': 'Log Loss Binary', 'regression': 'R2'}
    for problem_type in defaults:
        error_automl = AutoMLSearch(problem_type=problem_type)
        assert error_automl.objective.name == defaults[problem_type]


@patch('evalml.automl.automl_search.AutoMLSearch.search')
def test_checks_at_search_time(mock_search, dummy_regression_pipeline_class, X_y_multi):
    X, y = X_y_multi

    error_text = "in search, problem_type mismatches label type."
    mock_search.side_effect = ValueError(error_text)

    error_automl = AutoMLSearch(problem_type='regression', objective="R2")
    with pytest.raises(ValueError, match=error_text):
        error_automl.search(X, y)


def test_incompatible_additional_objectives():
    with pytest.raises(ValueError, match="is not compatible with a "):
        AutoMLSearch(problem_type='multiclass', additional_objectives=['Precision', 'AUC'])


def test_default_objective():
    correct_matches = {ProblemTypes.MULTICLASS: 'Log Loss Multiclass',
                       ProblemTypes.BINARY: 'Log Loss Binary',
                       ProblemTypes.REGRESSION: 'R2'}
    for problem_type in correct_matches:
        automl = AutoMLSearch(problem_type=problem_type)
        assert automl.objective.name == correct_matches[problem_type]

        automl = AutoMLSearch(problem_type=problem_type.name)
        assert automl.objective.name == correct_matches[problem_type]


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_add_to_rankings(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(problem_type='binary', max_pipelines=1, allowed_pipelines=[dummy_binary_pipeline_class])
    automl.search(X, y)

    mock_score.return_value = {'Log Loss Binary': 0.1234}

    test_pipeline = dummy_binary_pipeline_class(parameters={})
    automl.add_to_rankings(test_pipeline, X, y)

    assert len(automl.rankings) == 2
    assert 0.1234 in automl.rankings['score'].values


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_add_to_rankings_no_search(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', max_pipelines=1, allowed_pipelines=[dummy_binary_pipeline_class])

    mock_score.return_value = {'Log Loss Binary': 0.1234}
    test_pipeline = dummy_binary_pipeline_class(parameters={})
    with pytest.raises(RuntimeError, match="Please run automl"):
        automl.add_to_rankings(test_pipeline, X, y)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_add_to_rankings_duplicate(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 0.1234}

    automl = AutoMLSearch(problem_type='binary', max_pipelines=1, allowed_pipelines=[dummy_binary_pipeline_class])
    automl.search(X, y)

    test_pipeline = dummy_binary_pipeline_class(parameters={})
    automl.add_to_rankings(test_pipeline, X, y)

    test_pipeline_duplicate = dummy_binary_pipeline_class(parameters={})
    assert automl.add_to_rankings(test_pipeline_duplicate, X, y) is None


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_add_to_rankings_trained(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(problem_type='binary', max_pipelines=1, allowed_pipelines=[dummy_binary_pipeline_class])
    automl.search(X, y)

    mock_score.return_value = {'Log Loss Binary': 0.1234}
    test_pipeline = dummy_binary_pipeline_class(parameters={})
    automl.add_to_rankings(test_pipeline, X, y)

    class CoolBinaryClassificationPipeline(dummy_binary_pipeline_class):
        name = "Cool Binary Classification Pipeline"

    mock_fit.return_value = CoolBinaryClassificationPipeline(parameters={})
    test_pipeline_trained = CoolBinaryClassificationPipeline(parameters={}).fit(X, y)
    automl.add_to_rankings(test_pipeline_trained, X, y)

    assert list(automl.rankings['score'].values).count(0.1234) == 2


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_has_searched(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(problem_type='binary', max_pipelines=1)
    mock_score.return_value = {automl.objective.name: 1.0}
    assert not automl.has_searched

    automl.search(X, y)
    assert automl.has_searched


def test_no_search():
    automl = AutoMLSearch(problem_type='binary')
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.full_rankings, pd.DataFrame)

    df_columns = ["id", "pipeline_name", "score", "percent_better_than_baseline",
                  "high_variance_cv", "parameters"]
    assert (automl.rankings.columns == df_columns).all()
    assert (automl.full_rankings.columns == df_columns).all()

    assert automl._data_check_results is None

    with pytest.raises(PipelineNotFoundError):
        automl.best_pipeline

    with pytest.raises(PipelineNotFoundError):
        automl.get_pipeline(0)

    with pytest.raises(PipelineNotFoundError):
        automl.describe_pipeline(0)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_get_pipeline_invalid(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(problem_type='binary')
    with pytest.raises(PipelineNotFoundError, match="Pipeline not found in automl results"):
        automl.get_pipeline(1000)

    automl = AutoMLSearch(problem_type='binary', max_pipelines=1)
    automl.search(X, y)
    assert automl.get_pipeline(0).name == 'Mode Baseline Binary Classification Pipeline'
    automl._results['pipeline_results'][0].pop('pipeline_class')
    with pytest.raises(PipelineNotFoundError, match="Pipeline class or parameters not found in automl results"):
        automl.get_pipeline(0)

    automl = AutoMLSearch(problem_type='binary', max_pipelines=1)
    automl.search(X, y)
    assert automl.get_pipeline(0).name == 'Mode Baseline Binary Classification Pipeline'
    automl._results['pipeline_results'][0].pop('parameters')
    with pytest.raises(PipelineNotFoundError, match="Pipeline class or parameters not found in automl results"):
        automl.get_pipeline(0)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_describe_pipeline(mock_fit, mock_score, caplog, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(problem_type='binary', max_pipelines=1)
    automl.search(X, y)
    out = caplog.text

    assert "Searching up to 1 pipelines. " in out

    assert len(automl.results['pipeline_results']) == 1
    caplog.clear()
    automl.describe_pipeline(0)
    out = caplog.text
    assert "Mode Baseline Binary Classification Pipeline" in out
    assert "Problem Type: Binary Classification" in out
    assert "Model Family: Baseline" in out
    assert "* strategy : mode" in out
    assert "Total training time (including CV): " in out
    assert "Log Loss Binary # Training # Testing" in out
    assert "0                      1.000     66.000    34.000" in out
    assert "1                      1.000     67.000    33.000" in out
    assert "2                      1.000     67.000    33.000" in out
    assert "mean                   1.000          -         -" in out
    assert "std                    0.000          -         -" in out
    assert "coef of var            0.000          -         -" in out


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_results_getter(mock_fit, mock_score, caplog, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', max_pipelines=1)

    assert automl.results == {'pipeline_results': {}, 'search_order': []}

    mock_score.return_value = {'Log Loss Binary': 1.0}
    automl.search(X, y)

    assert automl.results['pipeline_results'][0]['score'] == 1.0

    with pytest.raises(AttributeError, match='set attribute'):
        automl.results = 2.0

    automl.results['pipeline_results'][0]['score'] = 2.0
    assert automl.results['pipeline_results'][0]['score'] == 1.0


@pytest.mark.parametrize("automl_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
@pytest.mark.parametrize("target_type", numeric_and_boolean_dtypes + categorical_dtypes)
def test_targets_data_types_classification(automl_type, target_type):
    if automl_type == ProblemTypes.BINARY:
        X, y = load_breast_cancer()
        if target_type == "bool":
            y = y.map({"malignant": False, "benign": True})
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = load_wine()
    if target_type == "category":
        y = pd.Categorical(y)
    elif "int" in target_type:
        unique_vals = y.unique()
        y = y.map({unique_vals[i]: int(i) for i in range(len(unique_vals))})
    elif "float" in target_type:
        unique_vals = y.unique()
        y = y.map({unique_vals[i]: float(i) for i in range(len(unique_vals))})

    unique_vals = y.unique()

    automl = AutoMLSearch(problem_type=automl_type, max_pipelines=3)
    automl.search(X, y)
    for pipeline_id, pipeline_result in automl.results['pipeline_results'].items():
        cv_data = pipeline_result['cv_data']
        for fold in cv_data:
            all_objective_scores = fold["all_objective_scores"]
            for score in all_objective_scores.values():
                assert score is not None

    assert len(automl.full_rankings) == 3
    assert not automl.full_rankings['score'].isnull().values.any()


class KeyboardInterruptOnKthPipeline:
    """Helps us time when the test will send a KeyboardInterrupt Exception to search."""

    def __init__(self, k):
        self.n_calls = 1
        self.k = k

    def __call__(self, pipeline_class, parameters, automl_obj):
        """Raises KeyboardInterrupt on the kth call.

        Arguments are ignored but included to meet the call back API.
        """
        if self.n_calls == self.k:
            self.n_calls += 1
            raise KeyboardInterrupt
        else:
            self.n_calls += 1


# These are used to mock return values to the builtin "input" function.
interrupt = ["y"]
interrupt_after_bad_message = ["No.", "Yes!", "y"]
dont_interrupt = ["n"]
dont_interrupt_after_bad_message = ["Yes", "yes.", "n"]


@pytest.mark.parametrize("when_to_interrupt,user_input,number_results",
                         [(1, interrupt, 0),
                          (1, interrupt_after_bad_message, 0),
                          (1, dont_interrupt, 5),
                          (1, dont_interrupt_after_bad_message, 5),
                          (2, interrupt, 1),
                          (2, interrupt_after_bad_message, 1),
                          (2, dont_interrupt, 5),
                          (2, dont_interrupt_after_bad_message, 5),
                          (3, interrupt, 2),
                          (3, interrupt_after_bad_message, 2),
                          (3, dont_interrupt, 5),
                          (3, dont_interrupt_after_bad_message, 5),
                          (5, interrupt, 4),
                          (5, interrupt_after_bad_message, 4),
                          (5, dont_interrupt, 5),
                          (5, dont_interrupt_after_bad_message, 5)])
@patch("builtins.input")
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"F1": 1.0})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_catch_keyboard_interrupt(mock_fit, mock_score, mock_input,
                                  when_to_interrupt, user_input, number_results,
                                  X_y_binary):

    mock_input.side_effect = user_input
    X, y = X_y_binary
    callback = KeyboardInterruptOnKthPipeline(k=when_to_interrupt)
    automl = AutoMLSearch(problem_type="binary", max_pipelines=5, start_iteration_callback=callback, objective="f1")
    automl.search(X, y)

    assert len(automl._results['pipeline_results']) == number_results


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch')
@patch('evalml.automl.AutoMLSearch._evaluate')
def test_pipelines_in_batch_return_nan(mock_evaluate, mock_next_batch, X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary
    mock_evaluate.side_effect = [{'cv_score_mean': 0}, {'cv_score_mean': np.nan}]
    mock_next_batch.side_effect = [[dummy_binary_pipeline_class(parameters={}), dummy_binary_pipeline_class(parameters={})]]
    automl = AutoMLSearch(problem_type='binary', allowed_pipelines=[dummy_binary_pipeline_class])
    automl.search(X, y)

    mock_evaluate.reset_mock()
    mock_next_batch.reset_mock()
    mock_evaluate.side_effect = [{'cv_score_mean': 0}, {'cv_score_mean': 0},  # first batch
                                 {'cv_score_mean': 0}, {'cv_score_mean': np.nan},  # second batch
                                 {'cv_score_mean': np.nan}, {'cv_score_mean': np.nan}]  # third batch, should raise error
    mock_next_batch.side_effect = [[dummy_binary_pipeline_class(parameters={}), dummy_binary_pipeline_class(parameters={})] for i in range(3)]
    automl = AutoMLSearch(problem_type='binary', allowed_pipelines=[dummy_binary_pipeline_class])
    with pytest.raises(AutoMLSearchException, match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective"):
        automl.search(X, y)


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch')
@patch('evalml.automl.AutoMLSearch._evaluate')
def test_pipelines_in_batch_return_none(mock_evaluate, mock_next_batch, X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary
    mock_evaluate.side_effect = [{'cv_score_mean': 0}, {'cv_score_mean': 0},  # first batch
                                 {'cv_score_mean': 0}, {'cv_score_mean': np.nan},  # second batch
                                 {'cv_score_mean': None}, {'cv_score_mean': None}]  # third batch, should raise error
    mock_next_batch.side_effect = [[dummy_binary_pipeline_class(parameters={}), dummy_binary_pipeline_class(parameters={})] for i in range(3)]
    automl = AutoMLSearch(problem_type='binary', allowed_pipelines=[dummy_binary_pipeline_class])
    with pytest.raises(AutoMLSearchException, match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective"):
        automl.search(X, y)


@patch('evalml.automl.automl_search.train_test_split')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_error_during_train_test_split(mock_fit, mock_score, mock_train_test_split, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}
    mock_train_test_split.side_effect = RuntimeError()
    automl = AutoMLSearch(problem_type='binary', objective='Accuracy Binary', max_pipelines=2, optimize_thresholds=True)
    automl.search(X, y)
    for pipeline in automl.results['pipeline_results'].values():
        assert np.isnan(pipeline['score'])


@pytest.mark.parametrize("objective_tuple,pipeline_scores,baseline_score",
                         product(_all_objectives_dict().items(),
                                 [(0.3, 0.4), (np.nan, 0.4), (0.3, np.nan), (np.nan, np.nan)],
                                 [0.1, np.nan]))
def test_percent_better_than_baseline_in_rankings(objective_tuple, pipeline_scores, baseline_score,
                                                  dummy_binary_pipeline_class, dummy_multiclass_pipeline_class,
                                                  dummy_regression_pipeline_class,
                                                  X_y_binary):

    # Ok to only use binary labels since score and fit methods are mocked
    X, y = X_y_binary

    name, objective = objective_tuple

    if objective in AutoMLSearch._objectives_not_allowed_in_automl and name != "cost benefit matrix":
        pytest.skip(f"Skipping because {name} is not allowed in automl as a string.")

    pipeline_class = {ProblemTypes.BINARY: dummy_binary_pipeline_class,
                      ProblemTypes.MULTICLASS: dummy_multiclass_pipeline_class,
                      ProblemTypes.REGRESSION: dummy_regression_pipeline_class}[objective.problem_type]
    baseline_pipeline_class = {ProblemTypes.BINARY: "evalml.pipelines.ModeBaselineBinaryPipeline",
                               ProblemTypes.MULTICLASS: "evalml.pipelines.ModeBaselineMulticlassPipeline",
                               ProblemTypes.REGRESSION: "evalml.pipelines.MeanBaselineRegressionPipeline",
                               }[objective.problem_type]

    class DummyPipeline(pipeline_class):
        problem_type = objective.problem_type

        def fit(self, *args, **kwargs):
            """Mocking fit"""

    class Pipeline1(DummyPipeline):
        name = "Pipeline1"

    class Pipeline2(DummyPipeline):
        name = "Pipeline2"

    mock_score_1 = MagicMock(return_value={objective.name: pipeline_scores[0]})
    mock_score_2 = MagicMock(return_value={objective.name: pipeline_scores[1]})
    Pipeline1.score = mock_score_1
    Pipeline2.score = mock_score_2

    if name == "cost benefit matrix":
        automl = AutoMLSearch(problem_type=objective.problem_type, max_pipelines=3,
                              allowed_pipelines=[Pipeline1, Pipeline2], objective=objective(0, 0, 0, 0))
    else:
        automl = AutoMLSearch(problem_type=objective.problem_type, max_pipelines=3,
                              allowed_pipelines=[Pipeline1, Pipeline2], objective=name)

    with patch(baseline_pipeline_class + ".score", return_value={objective.name: baseline_score}):
        automl.search(X, y, data_checks=None)
        scores = dict(zip(automl.rankings.pipeline_name, automl.rankings.percent_better_than_baseline))
        baseline_name = next(name for name in automl.rankings.pipeline_name if name not in {"Pipeline1", "Pipeline2"})
        answers = {"Pipeline1": round(objective.calculate_percent_difference(pipeline_scores[0], baseline_score), 2),
                   "Pipeline2": round(objective.calculate_percent_difference(pipeline_scores[1], baseline_score), 2),
                   baseline_name: round(objective.calculate_percent_difference(baseline_score, baseline_score), 2)}
        for name in answers:
            np.testing.assert_almost_equal(scores[name], answers[name], decimal=3)


@pytest.mark.parametrize("max_batches", [None, 1, 5, 8, 9, 10, 12])
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_max_batches_works(mock_pipeline_fit, mock_score, max_batches, X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(problem_type="binary", max_pipelines=None,
                          _max_batches=max_batches, objective="Log Loss Binary")
    automl.search(X, y, data_checks=None)

    if max_batches is None:
        n_results = 5
        max_batches = 1
        # _automl_algorithm will include all allowed_pipelines in the first batch even
        # if they are not searched over. That is why n_automl_pipelines does not equal
        # n_results when max_pipelines and max_batches are None
        n_automl_pipelines = 1 + len(automl.allowed_pipelines)
    else:
        # So that the test does not break when new estimator classes are added
        n_results = 1 + len(automl.allowed_pipelines) + (5 * (max_batches - 1))
        n_automl_pipelines = n_results

    assert automl._automl_algorithm.batch_number == max_batches
    # We add 1 to pipeline_number because _automl_algorithm does not know about the baseline
    assert automl._automl_algorithm.pipeline_number + 1 == n_automl_pipelines
    assert len(automl.results["pipeline_results"]) == n_results
    assert automl.rankings.shape[0] == min(1 + len(automl.allowed_pipelines), n_results)
    assert automl.full_rankings.shape[0] == n_results


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_max_batches_plays_nice_with_other_stopping_criteria(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary

    # Use the old default when all are None
    automl = AutoMLSearch(problem_type="binary", objective="Log Loss Binary")
    automl.search(X, y, data_checks=None)
    assert len(automl.results["pipeline_results"]) == 5

    # Use max_pipelines when both max_pipelines and max_batches are set
    automl = AutoMLSearch(problem_type="binary", objective="Log Loss Binary", _max_batches=10,
                          max_pipelines=6)
    automl.search(X, y, data_checks=None)
    assert len(automl.results["pipeline_results"]) == 6

    # Don't change max_pipelines when only max_pipelines is set
    automl = AutoMLSearch(problem_type="binary", max_pipelines=4)
    automl.search(X, y, data_checks=None)
    assert len(automl.results["pipeline_results"]) == 4


@pytest.mark.parametrize("max_batches", [0, -1, -10, -np.inf])
def test_max_batches_must_be_non_negative(max_batches):

    with pytest.raises(ValueError, match="Parameter max batches must be None or non-negative. Received {max_batches}."):
        AutoMLSearch(problem_type="binary", _max_batches=max_batches)


def test_can_print_out_automl_objective_names():
    AutoMLSearch.print_objective_names_allowed_in_automl()
