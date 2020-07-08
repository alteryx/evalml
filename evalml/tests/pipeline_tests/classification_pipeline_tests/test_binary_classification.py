from unittest.mock import patch

import numpy as np
import pytest


@patch('evalml.pipelines.ClassificationPipeline._decode_targets')
@patch('evalml.pipelines.ClassificationPipeline._encode_targets')
@patch('evalml.objectives.BinaryClassificationObjective.decision_function')
@patch('evalml.pipelines.components.Estimator.predict_proba')
@patch('evalml.pipelines.components.Estimator.predict')
@patch('evalml.pipelines.PipelineBase._transform')
@patch('evalml.pipelines.PipelineBase.fit')
def test_binary_classification_pipeline_predict(mock_fit, mock_transform,
                                                mock_predict, mock_predict_proba,
                                                mock_obj_decision, mock_encode, mock_decode,
                                                X_y_binary, dummy_binary_pipeline_class):
    mock_decode.return_value = [0, 1]
    X, y = X_y_binary
    binary_pipeline = dummy_binary_pipeline_class(parameters={})
    # test no objective passed and no custom threshold uses underlying estimator's predict method
    binary_pipeline.predict(X)
    mock_decode.assert_called()
    mock_predict.assert_called()
    mock_predict.reset_mock()

    # test objective passed but no custom threshold uses underlying estimator's predict method
    binary_pipeline.predict(X, 'precision')
    mock_predict.assert_called()
    mock_predict.reset_mock()

    # test custom threshold set but no objective passed
    mock_predict_proba.return_value = np.array([[0.1, 0.2], [0.1, 0.2]])
    binary_pipeline.threshold = 0.6
    binary_pipeline.predict(X)
    mock_predict.assert_not_called()
    mock_predict_proba.assert_called()
    mock_obj_decision.assert_not_called()

    # test custom threshold set but no objective passed
    mock_predict.reset_mock()
    mock_predict_proba.return_value = np.array([[0.1, 0.2], [0.1, 0.2]])
    binary_pipeline.threshold = 0.6
    binary_pipeline.predict(X)
    mock_predict.assert_not_called()
    mock_predict_proba.assert_called()
    mock_obj_decision.assert_not_called()

    # test custom threshold set and objective passed
    mock_predict.reset_mock()
    mock_predict_proba.reset_mock()
    mock_predict_proba.return_value = np.array([[0.1, 0.2], [0.1, 0.2]])
    binary_pipeline.threshold = 0.6
    binary_pipeline.predict(X, 'precision')
    mock_predict.assert_not_called()
    mock_predict_proba.assert_called()
    mock_obj_decision.assert_called()


@patch('evalml.pipelines.PipelineBase._transform')
def test_binary_predict_pipeline_objective_mismatch(mock_transform, X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary
    binary_pipeline = dummy_binary_pipeline_class(parameters={})
    with pytest.raises(ValueError, match="You can only use a binary classification objective to make predictions for a binary classification pipeline."):
        binary_pipeline.predict(X, "precision_micro")
    mock_transform.assert_called()
