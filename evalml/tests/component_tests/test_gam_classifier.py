from unittest.mock import patch

import numpy as np
import pandas as pd
from pytest import importorskip
from sklearn import datasets

from evalml.model_family import ModelFamily
from evalml.pipelines.components import GAMClassifier
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import SEED_BOUNDS, make_h2o_ready

h2o = importorskip('h2o', reason='Skipping test because h2o not installed')
h2o.connect()
gam = h2o.estimators.gam.H2OGeneralizedAdditiveEstimator


def test_gam_classifier_init():
    assert GAMClassifier.model_family == ModelFamily.LINEAR_MODEL


def test_problem_types():
    assert set(GAMClassifier.supported_problem_types) == {ProblemTypes.MULTICLASS, ProblemTypes.BINARY,
                                                          ProblemTypes.TIME_SERIES_MULTICLASS,
                                                          ProblemTypes.TIME_SERIES_BINARY}


def test_gam_classifier_random_state_bounds_seed(X_y_binary):
    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = GAMClassifier(random_state=SEED_BOUNDS.min_bound)
    clf.fit(X, y)
    clf = GAMClassifier(random_state=SEED_BOUNDS.max_bound)
    clf.fit(X, y)


def test_fit_predict_binary(X_y_binary):
    X, y = X_y_binary

    X, y, training_frame = make_h2o_ready(X, y, [ProblemTypes.BINARY])
    X_cols = list(X.columns)[:9]
    X_cols = [str(col_) for col_ in X_cols]

    gam = h2o.estimators.gam.H2OGeneralizedAdditiveEstimator
    clf_sk = gam(family='binomial', link='Logit', gam_columns=X_cols, lambda_search=True, seed=42)
    clf_sk.train(x=list(X.columns), y=y.name, training_frame=training_frame)
    y_pred_sk = clf_sk.predict(h2o.H2OFrame(X))
    y_pred_sk = y_pred_sk['predict']
    y_pred_sk = y_pred_sk.as_data_frame().iloc[:, 0].values

    clf = GAMClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)


def test_fit_predict_multi(X_y_multi):
    X, y = X_y_multi

    X, y, training_frame = make_h2o_ready(X, y, [ProblemTypes.MULTICLASS])
    X_cols = list(X.columns)[:9]
    X_cols = [str(col_) for col_ in X_cols]

    gam = h2o.estimators.gam.H2OGeneralizedAdditiveEstimator
    clf_sk = gam(family='multinomial', link='Family_Default', gam_columns=X_cols, lambda_search=True, seed=42)
    clf_sk.train(x=list(X.columns), y=y.name, training_frame=training_frame)
    y_pred_sk = clf_sk.predict(h2o.H2OFrame(X))
    y_pred_sk = y_pred_sk['predict']
    y_pred_sk = y_pred_sk.as_data_frame().iloc[:, 0].values

    clf = GAMClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)


def test_family_link_solver_param_updates(X_y_binary, X_y_multi):
    X, y = X_y_binary
    clf = GAMClassifier()
    clf.fit(X, y)
    assert clf.parameters['family'] == "binomial"
    assert clf.parameters['link'] == "Logit"
    assert clf.parameters['lambda_search']

    X, y = X_y_multi
    clf = GAMClassifier()
    clf.fit(X, y)
    assert clf.parameters['family'] == "multinomial"
    assert clf.parameters['link'] == "Family_Default"
    assert clf.parameters['lambda_search']

    X, y = datasets.make_classification(n_samples=100, n_features=20, n_classes=4, n_informative=4, n_redundant=2, random_state=0)
    clf = GAMClassifier()
    clf.fit(X, y)
    assert clf.parameters['family'] == "ordinal"
    assert clf.parameters['solver'] == "GRADIENT_DESCENT_LH"
    assert clf.parameters['link'] == "Family_Default"
    assert not clf.parameters['lambda_search']


def test_feature_importance(X_y_binary):
    X, y = X_y_binary
    clf = GAMClassifier()
    clf.fit(X, y)
    feat_imp_ = clf.feature_importance
    print(feat_imp_)


@patch('evalml.pipelines.components.estimators.classifiers.gam_classifier.GAMClassifier.predict')
@patch('evalml.pipelines.components.estimators.classifiers.gam_classifier.GAMClassifier.fit')
def test_fit_no_categories(mock_fit, mock_predict, X_y_binary):
    X, y = X_y_binary
    X2 = pd.DataFrame(X)
    X2.columns = np.arange(len(X2.columns))
    clf = GAMClassifier()
    clf.fit(X, y)
    arg_X = mock_fit.call_args[0][0]
    np.testing.assert_array_equal(arg_X, X2)

    clf.predict(X)
    arg_X = mock_predict.call_args[0][0]
    np.testing.assert_array_equal(arg_X, X2)
