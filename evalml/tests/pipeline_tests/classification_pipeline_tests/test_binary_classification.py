from unittest.mock import patch

import pandas as pd
import pytest
import woodwork as ww
from skopt.space import Categorical

from evalml.exceptions import PipelineScoreError
from evalml.objectives import FraudCost, get_objective
from evalml.pipelines import BinaryClassificationPipeline


def test_binary_init():
    clf = BinaryClassificationPipeline(component_graph=["Imputer", "One Hot Encoder", "Random Forest Classifier"])
    assert clf.parameters == {
        'Imputer': {
            'categorical_impute_strategy': 'most_frequent',
            'numeric_impute_strategy': 'mean',
            'categorical_fill_value': None,
            'numeric_fill_value': None
        },
        'One Hot Encoder': {
            'top_n': 10,
            'features_to_encode': None,
            'categories': None,
            'drop': 'if_binary',
            'handle_unknown': 'ignore',
            'handle_missing': 'error'
        },
        'Random Forest Classifier': {
            'n_estimators': 100,
            'max_depth': 6,
            'n_jobs': -1
        }
    }
    assert clf.custom_hyperparameters is None
    assert clf.name == "Random Forest Classifier w/ Imputer + One Hot Encoder"
    assert clf.random_seed == 0
    custom_hyperparameters = {"Imputer": {"numeric_impute_strategy": Categorical(["most_frequent", 'mean'])},
                              "Imputer_1": {"numeric_impute_strategy": Categorical(["median", 'mean'])},
                              "Random Forest Classifier": {"n_estimators": Categorical([50, 100])}}
    parameters = {
        "One Hot Encoder": {
            "top_n": 20
        }
    }
    clf = BinaryClassificationPipeline(component_graph=["Imputer", "One Hot Encoder", "Random Forest Classifier"],
                                       parameters=parameters,
                                       custom_hyperparameters=custom_hyperparameters,
                                       custom_name="Custom Pipeline",
                                       random_seed=42)

    assert clf.parameters == {
        'Imputer': {
            'categorical_impute_strategy': 'most_frequent',
            'numeric_impute_strategy': 'mean',
            'categorical_fill_value': None,
            'numeric_fill_value': None
        },
        'One Hot Encoder': {
            'top_n': 20,
            'features_to_encode': None,
            'categories': None,
            'drop': 'if_binary',
            'handle_unknown': 'ignore',
            'handle_missing': 'error'
        },
        'Random Forest Classifier': {
            'n_estimators': 100,
            'max_depth': 6,
            'n_jobs': -1
        }
    }
    assert clf.custom_hyperparameters == custom_hyperparameters
    assert clf.name == "Custom Pipeline"
    assert clf.random_seed == 42


@patch('evalml.pipelines.ClassificationPipeline._decode_targets', return_value=[0, 1])
@patch('evalml.objectives.BinaryClassificationObjective.decision_function', return_value=pd.Series([1, 0]))
@patch('evalml.pipelines.components.Estimator.predict_proba', return_value=ww.DataTable(pd.DataFrame([[0.1, 0.2], [0.1, 0.2]])))
@patch('evalml.pipelines.components.Estimator.predict', return_value=ww.DataColumn(pd.Series([1, 0])))
def test_binary_classification_pipeline_predict(mock_predict, mock_predict_proba,
                                                mock_obj_decision, mock_decode,
                                                X_y_binary, dummy_binary_pipeline_class):
    mock_objs = [mock_decode, mock_predict]
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


@patch('evalml.objectives.FraudCost.decision_function')
def test_binary_predict_pipeline_use_objective(mock_decision_function, X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    binary_pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    mock_decision_function.return_value = pd.Series([0] * 100)

    binary_pipeline.threshold = 0.7
    binary_pipeline.fit(X, y)
    fraud_cost = FraudCost(amount_col=0)
    binary_pipeline.score(X, y, ['precision', 'auc', fraud_cost])
    mock_decision_function.assert_called()


def test_binary_predict_pipeline_score_error(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    binary_pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    binary_pipeline.fit(X, y)
    with pytest.raises(PipelineScoreError, match='Invalid objective MCC Multiclass specified for problem type binary'):
        binary_pipeline.score(X, y, ['MCC Multiclass'])


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.predict_proba')
def test_pipeline_thresholding_errors(mock_binary_pred_proba, mock_binary_score, mock_binary_fit,
                                      make_data_type, logistic_regression_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    X = make_data_type('ww', X)
    y = make_data_type('ww', pd.Series([f"String value {i}" for i in y]))
    objective = get_objective("Log Loss Binary", return_instance=True)
    pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    pipeline.fit(X, y)
    pred_proba = pipeline.predict_proba(X, y).iloc[:, 1]
    with pytest.raises(ValueError, match="Problem type must be binary and objective must be optimizable"):
        pipeline.optimize_threshold(X, y, pred_proba, objective)
