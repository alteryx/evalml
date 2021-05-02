from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines.components import GAMClassifier
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import make_h2o_ready

h2o = importorskip('h2o', reason='Skipping test because h2o not installed')
h2o.init()
gam = h2o.estimators.gam.H2OGeneralizedAdditiveEstimator


def test_gam_classifier_init():
    assert GAMClassifier.model_family == ModelFamily.GAM


def test_problem_types():
    assert set(GAMClassifier.supported_problem_types) == {ProblemTypes.MULTICLASS, ProblemTypes.BINARY}


def test_fit_predict_binary(X_y_binary):
    X, y = X_y_binary

    X, y, training_frame = make_h2o_ready(X, y, [ProblemTypes.BINARY])
    X_cols = [str(col_) for col_ in list(X.columns)]

    clf_h2o = gam(family='binomial', link='Logit', gam_columns=X_cols, lambda_search=True, seed=0)
    clf_h2o.train(x=list(X.columns), y=y.name, training_frame=training_frame)
    y_pred_h2o = clf_h2o.predict(h2o.H2OFrame(X))
    y_pred_h2o = y_pred_h2o['predict'].as_data_frame().iloc[:, 0].values

    clf = GAMClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    pd.testing.assert_series_equal(pd.Series(y_pred_h2o, name='predict'), y_pred)


@pytest.mark.parametrize("dataset_type", ["multi", "multi_plus"])
def test_fit_predict_multi(dataset_type, X_y_multi, X_y_multi_more_classes):
    if dataset_type == 'multi':
        X, y = X_y_multi
    else:
        X, y = X_y_multi_more_classes

    X, y, training_frame = make_h2o_ready(X, y, [ProblemTypes.MULTICLASS])
    X_cols = [str(col_) for col_ in list(X.columns)]

    clf_h2o = gam(family='multinomial', link='Family_Default', gam_columns=X_cols, lambda_search=True, seed=0)
    clf_h2o.train(x=list(X.columns), y=y.name, training_frame=training_frame)
    y_pred_h2o = clf_h2o.predict(h2o.H2OFrame(X))
    y_pred_h2o = y_pred_h2o['predict'].as_data_frame().iloc[:, 0].values

    clf = GAMClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    pd.testing.assert_series_equal(pd.Series(y_pred_h2o, name='predict'), y_pred)


@pytest.mark.parametrize("dataset_type", ["binary", "multi", "multi_plus"])
def test_family_link_solver_param_updates(dataset_type, X_y_binary, X_y_multi, X_y_multi_more_classes):
    clf = GAMClassifier()
    if dataset_type == 'binary':
        X, y = X_y_binary
    elif dataset_type == 'multi':
        X, y = X_y_multi
    else:
        X, y = X_y_multi_more_classes

    clf.fit(X, y)
    if dataset_type == 'binary':
        assert clf.parameters['family'] == "binomial"
        assert clf.parameters['link'] == "Logit"
        assert clf.parameters['lambda_search']
    else:
        assert clf.parameters['family'] == "multinomial"
        assert clf.parameters['link'] == "Family_Default"
        assert clf.parameters['lambda_search']


def test_feature_importance(X_y_binary):
    X, y = X_y_binary

    X, y, training_frame = make_h2o_ready(X, y, [ProblemTypes.BINARY])
    X_cols = [str(col_) for col_ in list(X.columns)]

    clf_h2o = gam(family='binomial', link='Logit', gam_columns=X_cols, lambda_search=True, seed=0)
    clf_h2o.train(x=list(X.columns), y=y.name, training_frame=training_frame)
    feat_imp_h2o = clf_h2o.varimp()

    clf = GAMClassifier()
    clf.fit(X, y)
    feat_imp_ = clf.feature_importance

    np.testing.assert_array_equal(feat_imp_h2o, feat_imp_)


@patch('evalml.pipelines.components.estimators.classifiers.gam_classifier.GAMClassifier.predict')
@patch('evalml.pipelines.components.estimators.classifiers.gam_classifier.GAMClassifier.fit')
def test_fit_predict_no_categories(mock_fit, mock_predict, X_y_binary):
    X, y = X_y_binary

    X2 = pd.DataFrame(X)
    X2.columns = np.arange(len(X2.columns))
    clf = GAMClassifier()
    clf.fit(X, y)
    arg_X = mock_fit.call_args[0][0]
    np.testing.assert_array_equal(arg_X, X2)

    clf.predict(X[:10])
    arg_X = mock_predict.call_args[0][0]
    np.testing.assert_array_equal(arg_X, X[:10])
