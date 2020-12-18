from unittest.mock import patch

import pandas as pd
import pytest


@patch('evalml.pipelines.ClassificationPipeline._decode_targets')
@patch('evalml.objectives.BinaryClassificationObjective.decision_function')
@patch('evalml.pipelines.components.Estimator.predict_proba')
@patch('evalml.pipelines.components.Estimator.predict')
def test_binary_classification_pipeline_predict(mock_predict, mock_predict_proba,
                                                mock_obj_decision, mock_decode,
                                                X_y_binary, dummy_binary_pipeline_class):
    mock_objs = [mock_decode, mock_predict]
    mock_decode.return_value = [0, 1]
    X, y = X_y_binary
    binary_pipeline = dummy_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    # test no objective passed and no custom threshold uses underlying estimator's predict method
    binary_pipeline.fit(X, y)
    binary_pipeline.predict(X)
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()

    # test objective passed but no custom threshold uses underlying estimator's predict method
    binary_pipeline.predict(X, 'precision')
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()

    mock_objs = [mock_decode, mock_predict_proba]
    # test custom threshold set but no objective passed
    mock_predict_proba.return_value = pd.DataFrame([[0.1, 0.2], [0.1, 0.2]])
    binary_pipeline.threshold = 0.6
    binary_pipeline._encoder.classes_ = [0, 1]
    binary_pipeline.predict(X)
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()
    mock_obj_decision.assert_not_called()
    mock_predict.assert_not_called()

    # test custom threshold set but no objective passed
    binary_pipeline.threshold = 0.6
    binary_pipeline.predict(X)
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()
    mock_obj_decision.assert_not_called()
    mock_predict.assert_not_called()

    # test custom threshold set and objective passed
    binary_pipeline.threshold = 0.6
    binary_pipeline.predict(X, 'precision')
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()
    mock_predict.assert_not_called()
    mock_obj_decision.assert_called()


@patch('evalml.pipelines.ComponentGraph._compute_features')
def test_binary_predict_pipeline_objective_mismatch(mock_transform, X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary
    binary_pipeline = dummy_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    binary_pipeline.fit(X, y)
    with pytest.raises(ValueError, match="You can only use a binary classification objective to make predictions for a binary classification pipeline."):
        binary_pipeline.predict(X, "precision micro")
    mock_transform.assert_called()
