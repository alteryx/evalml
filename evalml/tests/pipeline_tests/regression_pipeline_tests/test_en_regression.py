from unittest.mock import patch

import numpy as np

from evalml.pipelines import ENRegressionPipeline


def test_en_init(X_y_reg):
    X, y = X_y_reg

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'RF Regressor Select From Model': {
            "percent_features": 1.0,
            "number_features": len(X[0]),
            "n_estimators": 20,
            "max_depth": 5
        },
        'Elastic Net Regressor': {
            "alpha": 0.5,
            "l1_ratio": 0.5,
        }
    }
    clf = ENRegressionPipeline(parameters=parameters, random_state=2)
    expected_parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'RF Regressor Select From Model': {
            'percent_features': 1.0,
            'threshold': -np.inf
        },
        'Elastic Net Regressor': {
            "alpha": 0.5,
            "l1_ratio": 0.5,
        }
    }

    assert clf.parameters == expected_parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'Elastic Net Regressor w/ One Hot Encoder + Simple Imputer + RF Regressor Select From Model'


def test_summary():
    assert ENRegressionPipeline.summary == 'Elastic Net Regressor w/ One Hot Encoder + Simple Imputer + RF Regressor Select From Model'


@patch('evalml.pipelines.components.Estimator.predict')
@patch('evalml.pipelines.PipelineBase._transform')
@patch('evalml.pipelines.PipelineBase.fit')
def test_en_regression_pipeline_predict(mock_fit, mock_transform, mock_predict,
                                        X_y, dummy_en_regression_pipeline_class):
    X, y = X_y
    en_pipeline = dummy_en_regression_pipeline_class(parameters={})
    # test no objective passed and no custom threshold uses underlying estimator's predict method
    en_pipeline.predict(X)
    mock_predict.assert_called()
    mock_predict.reset_mock()

    # test objective passed but no custom threshold uses underlying estimator's predict method
    en_pipeline.predict(X, 'precision')
    mock_predict.assert_called()
    mock_predict.reset_mock()

    # test custom threshold set but no objective passed
    mock_predict.return_value = np.array([[0.1, 0.2], [0.1, 0.2]])
    en_pipeline.threshold = 0.6
    en_pipeline.predict(X)
    mock_predict.assert_called()

    # test custom threshold set but no objective passed
    mock_predict.reset_mock()
    mock_predict.return_value = np.array([[0.1, 0.2], [0.1, 0.2]])
    en_pipeline.threshold = 0.6
    en_pipeline.predict(X)
    mock_predict.assert_called()

    # test custom threshold set and objective passed
    mock_predict.reset_mock()
    mock_predict.reset_mock()
    mock_predict.return_value = np.array([[0.1, 0.2], [0.1, 0.2]])
    en_pipeline.threshold = 0.6
    en_pipeline.predict(X, 'precision')
    mock_predict.assert_called()
