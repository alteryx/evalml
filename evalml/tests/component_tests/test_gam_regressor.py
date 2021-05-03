from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines.components import GAMRegressor
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import make_h2o_ready

h2o = importorskip('h2o', reason='Skipping test because h2o not installed')
h2o.init()
gam = h2o.estimators.gam.H2OGeneralizedAdditiveEstimator


def test_gam_regressor_init():
    assert GAMRegressor.model_family == ModelFamily.GAM


def test_problem_types():
    assert set(GAMRegressor.supported_problem_types) == {ProblemTypes.REGRESSION}


def test_h2o_init():
    clf = GAMRegressor()
    assert clf.h2o_model_init.__name__ == 'H2OGeneralizedAdditiveEstimator'
    assert clf.h2o.__name__ == 'h2o'


def test_h2o_parameters():
    clf = GAMRegressor()
    assert clf.parameters == {'family': 'AUTO',
                              'solver': 'AUTO',
                              'stopping_metric': 'deviance',
                              'keep_cross_validation_models': False,
                              'seed': 0}


def test_fit_predict_returns(X_y_binary):
    X, y = X_y_binary

    clf = GAMRegressor()
    fit_return = clf.fit(X, y)
    predictions = clf.predict(X)

    assert isinstance(fit_return, h2o.estimators.gam.H2OGeneralizedAdditiveEstimator)
    assert isinstance(predictions, pd.Series)


def test_no_y(X_y_binary):
    X, y = X_y_binary

    clf = GAMRegressor()
    with pytest.raises(ValueError, match="GAM Regressor requires y as input."):
        clf.fit(X)


def test_fit_predict_regression(X_y_regression):
    X, y = X_y_regression

    X, y, training_frame = make_h2o_ready(X, y, [ProblemTypes.REGRESSION])
    X_cols = [str(col_) for col_ in list(X.columns)]

    clf_sk = gam(family='Gaussian', link='Identity', gam_columns=X_cols, lambda_search=True, seed=0)
    clf_sk.train(x=list(X.columns), y=y.name, training_frame=training_frame)
    y_pred_h2o = clf_sk.predict(h2o.H2OFrame(X))
    y_pred_h2o = y_pred_h2o['predict'].as_data_frame().iloc[:, 0].values

    clf = GAMRegressor()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    pd.testing.assert_series_equal(pd.Series(y_pred_h2o, name='predict'), y_pred)


def test_family_link_solver_param_updates(X_y_regression):
    clf = GAMRegressor()

    X, y = X_y_regression
    X, y, training_frame = make_h2o_ready(X, y, supported_problem_types=GAMRegressor.supported_problem_types)

    clf.fit(X, y)
    assert clf._update_params(X, y) == {'gam_columns': [str(col_) for col_ in list(X.columns)],
                                        "lambda_search": True,
                                        "family": "Gaussian",
                                        "link": "Identity"}


def test_feature_importance(X_y_regression):
    X, y = X_y_regression

    X, y, training_frame = make_h2o_ready(X, y, [ProblemTypes.REGRESSION])
    X_cols = [str(col_) for col_ in list(X.columns)]

    clf_h2o = gam(family='Gaussian', link='Identity', gam_columns=X_cols, lambda_search=True, seed=0)
    clf_h2o.train(x=list(X.columns), y=y.name, training_frame=training_frame)
    feat_imp_h2o = clf_h2o.varimp()

    clf = GAMRegressor()
    clf.fit(X, y)
    feat_imp_ = clf.feature_importance

    np.testing.assert_array_equal(feat_imp_h2o, feat_imp_)


@patch('evalml.pipelines.components.estimators.regressors.gam_regressor.GAMRegressor.predict')
@patch('evalml.pipelines.components.estimators.regressors.gam_regressor.GAMRegressor.fit')
def test_fit_no_categories(mock_fit, mock_predict, X_y_regression):
    X, y = X_y_regression

    X2 = pd.DataFrame(X)
    X2.columns = np.arange(len(X2.columns))
    clf = GAMRegressor()
    clf.fit(X, y)
    arg_X = mock_fit.call_args[0][0]
    np.testing.assert_array_equal(arg_X, X2)

    clf.predict(X[:10])
    arg_X = mock_predict.call_args[0][0]
    np.testing.assert_array_equal(arg_X, X2[:10])
