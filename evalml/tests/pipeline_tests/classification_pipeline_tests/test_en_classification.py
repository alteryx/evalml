from unittest.mock import patch

import numpy as np
import pytest

from evalml.pipelines import ENBinaryPipeline, ENMulticlassPipeline


@pytest.fixture
def dummy_en_multi_pipeline_class(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockENMultiClassificationPipeline(ENMulticlassPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator]

    return MockENMultiClassificationPipeline


@pytest.fixture
def dummy_en_binary_pipeline_class(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockENBinaryClassificationPipeline(ENBinaryPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator]

    return MockENBinaryClassificationPipeline


def test_en_init(X_y):
    X, y = X_y

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Elastic Net Classifier': {
            "alpha": 0.5,
            "l1_ratio": 0.5,
        }
    }
    clf = ENBinaryPipeline(parameters=parameters, random_state=2)
    expected_parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Elastic Net Classifier': {
            "alpha": 0.5,
            "l1_ratio": 0.5,
            "max_iter": 1000,
            "n_jobs": -1
        }
    }

    assert clf.parameters == expected_parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'Elastic Net Classifier w/ One Hot Encoder + Simple Imputer'


def test_summary():
    assert ENBinaryPipeline.summary == 'Elastic Net Classifier w/ One Hot Encoder + Simple Imputer'
    assert ENMulticlassPipeline.summary == 'Elastic Net Classifier w/ One Hot Encoder + Simple Imputer'


@patch('evalml.pipelines.PipelineBase._transform')
def test_en_binary_predict_pipeline_objective_mismatch(mock_transform, X_y, dummy_en_binary_pipeline_class):
    X, y = X_y
    binary_pipeline = dummy_en_binary_pipeline_class(parameters={})
    with pytest.raises(ValueError, match="You can only use a binary classification objective to make predictions for a binary classification pipeline."):
        binary_pipeline.predict(X, "precision_micro")
    mock_transform.assert_called()


@patch('evalml.pipelines.components.Estimator.predict')
def test_en_multi_classification_pipeline_predict(mock_predict, X_y, dummy_en_multi_pipeline_class):
    X, y = X_y
    multi_pipeline = dummy_en_multi_pipeline_class(parameters={})
    multi_pipeline.predict(X)
    mock_predict.assert_called()
    mock_predict.reset_mock()


@patch('evalml.objectives.BinaryClassificationObjective.decision_function')
@patch('evalml.pipelines.components.Estimator.predict_proba')
@patch('evalml.pipelines.components.Estimator.predict')
@patch('evalml.pipelines.PipelineBase._transform')
@patch('evalml.pipelines.PipelineBase.fit')
def test_en_binary_classification_pipeline_predict(mock_fit, mock_transform, mock_predict,
                                                   mock_predict_proba, mock_obj_decision, X_y,
                                                   dummy_en_multi_pipeline_class, dummy_en_binary_pipeline_class):
    X, y = X_y
    en_pipeline = dummy_en_binary_pipeline_class(parameters={})
    # test no objective passed and no custom threshold uses underlying estimator's predict method
    en_pipeline.predict(X)
    mock_predict.assert_called()
    mock_predict.reset_mock()

    # test objective passed but no custom threshold uses underlying estimator's predict method
    en_pipeline.predict(X, 'precision')
    mock_predict.assert_called()
    mock_predict.reset_mock()

    # test custom threshold set but no objective passed
    mock_predict_proba.return_value = np.array([[0.1, 0.2], [0.1, 0.2]])
    en_pipeline.threshold = 0.6
    en_pipeline.predict(X)
    mock_predict.assert_not_called()
    mock_predict_proba.assert_called()
    mock_obj_decision.assert_not_called()

    # test custom threshold set but no objective passed
    mock_predict.reset_mock()
    mock_predict_proba.return_value = np.array([[0.1, 0.2], [0.1, 0.2]])
    en_pipeline.threshold = 0.6
    en_pipeline.predict(X)
    mock_predict.assert_not_called()
    mock_predict_proba.assert_called()
    mock_obj_decision.assert_not_called()

    # test custom threshold set and objective passed
    mock_predict.reset_mock()
    mock_predict_proba.reset_mock()
    mock_predict_proba.return_value = np.array([[0.1, 0.2], [0.1, 0.2]])
    en_pipeline.threshold = 0.6
    en_pipeline.predict(X, 'precision')
    mock_predict.assert_not_called()
    mock_predict_proba.assert_called()
    mock_obj_decision.assert_called()
