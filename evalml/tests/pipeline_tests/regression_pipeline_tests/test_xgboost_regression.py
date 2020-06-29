import category_encoders as ce
import numpy as np
from pytest import importorskip
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import R2
from evalml.pipelines import XGBoostRegressionPipeline
from evalml.pipelines.components import XGBoostRegressor
from evalml.utils import get_random_seed, get_random_state, import_or_raise

importorskip('xgboost', reason='Skipping test because xgboost not installed')


def test_xg_init(X_y_regression):
    X, y = X_y_regression

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'median',
            'fill_value': None
        },
        'One Hot Encoder': {
            'top_n': 10,
            'categories': None,
            'drop': None,
            'handle_unknown': 'ignore',
            'handle_missing': 'error'},
        'XGBoost Regressor': {
            'eta': 0.2,
            'max_depth': 5,
            'min_child_weight': 3,
            'n_estimators': 20
        }
    }

    clf = XGBoostRegressionPipeline(parameters=parameters, random_state=1)

    assert clf.parameters == parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(1).get_state()[0])
    assert clf.summary == 'XGBoost Regressor w/ One Hot Encoder + Simple Imputer'


def test_summary():
    assert XGBoostRegressionPipeline.summary == 'XGBoost Regressor w/ One Hot Encoder + Simple Imputer'


def test_xgboost_regression(X_y_regression):
    X, y = X_y_regression

    random_seed = 42
    xgb_random_seed = get_random_seed(get_random_state(random_seed), min_bound=XGBoostRegressor.SEED_MIN, max_bound=XGBoostRegressor.SEED_MAX)
    xgb = import_or_raise("xgboost")
    imputer = SimpleImputer(strategy='mean')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    estimator = xgb.XGBRegressor(random_state=xgb_random_seed,
                                 eta=0.1,
                                 max_depth=3,
                                 min_child_weight=1,
                                 n_estimators=10)
    sk_pipeline = Pipeline([("encoder", enc),
                            ("imputer", imputer),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'XGBoost Regressor': {
            "n_estimators": 10,
            "eta": 0.1,
            "min_child_weight": 1,
            "max_depth": 3
        }
    }

    clf = XGBoostRegressionPipeline(parameters=parameters, random_state=get_random_state(random_seed))
    clf.fit(X, y)
    y_pred = clf.predict(X)
    objective = R2()
    clf_scores = clf.score(X, y, [objective])
    np.testing.assert_almost_equal(y_pred, sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_scores[objective.name], decimal=5)

    # testing objective parameter passed in does not change results
    clf.fit(X, y)
    y_pred_with_objective = clf.predict(X)
    np.testing.assert_almost_equal(y_pred.to_numpy(), y_pred_with_objective.to_numpy(), decimal=5)


def test_xgr_input_feature_names(X_y_categorical_regression):
    X, y = X_y_categorical_regression
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'median'
        },
        'XGBoost Regressor': {
            "n_estimators": 20,
            "eta": 0.2,
            "min_child_weight": 3,
            "max_depth": 5,
        }
    }
    clf = XGBoostRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    assert not clf.feature_importance.isnull().all().all()
