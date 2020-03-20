import category_encoders as ce
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import Precision, PrecisionMicro
from evalml.pipelines import XGBoostBinaryPipeline, XGBoostMulticlassPipeline
from evalml.utils import import_or_raise


def test_xg_init(X_y):
    X, y = X_y

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'median'
        },
        'RF Classifier Select From Model': {
            "percent_features": 1.0,
            "number_features": len(X[0]),
            "n_estimators": 20,
            "max_depth": 5
        },
        'XGBoost Classifier': {
            "n_estimators": 20,
            "eta": 0.2,
            "min_child_weight": 3,
            "max_depth": 5,
        }
    }

    clf = XGBoostBinaryPipeline(parameters=parameters)

    expected_parameters = {
        'Simple Imputer': {
            'impute_strategy': 'median'
        },
        'RF Classifier Select From Model': {
            'percent_features': 1.0,
            'threshold': -np.inf,
        },
        'XGBoost Classifier': {
            'eta': 0.2,
            'max_depth': 5,
            'min_child_weight': 3,
            'n_estimators': 20
        }
    }

    assert clf.parameters == expected_parameters


def test_lor_objective_tuning(X_y):
    X, y = X_y

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'median'
        },
        'RF Classifier Select From Model': {
            "percent_features": 1.0,
            "number_features": len(X[0]),
            "n_estimators": 20,
            "max_depth": 5
        },
        'XGBoost Classifier': {
            "n_estimators": 20,
            "eta": 0.2,
            "min_child_weight": 3,
            "max_depth": 5,
        }
    }

    clf = XGBoostBinaryPipeline(parameters=parameters)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    objective = PrecisionMicro()
    with pytest.raises(ValueError, match="You can only use a binary classification objective to make predictions for a binary classification pipeline."):
        y_pred_with_objective = clf.predict(X, objective)

    # testing objective parameter passed in does not change results
    objective = Precision()
    y_pred_with_objective = clf.predict(X, objective)
    np.testing.assert_almost_equal(y_pred, y_pred_with_objective, decimal=5)

    # testing objective parameter passed and set threshold does change results
    with pytest.raises(AssertionError):
        clf.threshold = 0.01
        y_pred_with_objective = clf.predict(X, objective)
        np.testing.assert_almost_equal(y_pred, y_pred_with_objective, decimal=5)


def test_xg_multi(X_y_multi):
    X, y = X_y_multi

    xgb = import_or_raise("xgboost")
    imputer = SimpleImputer(strategy='mean')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    estimator = xgb.XGBClassifier(random_state=0,
                                  eta=0.1,
                                  max_depth=3,
                                  min_child_weight=1,
                                  n_estimators=10)
    rf_estimator = SKRandomForestClassifier(random_state=0, n_estimators=10, max_depth=3)
    feature_selection = SelectFromModel(estimator=rf_estimator,
                                        max_features=max(1, int(1 * len(X[0]))),
                                        threshold=-np.inf)
    sk_pipeline = Pipeline([("encoder", enc),
                            ("imputer", imputer),
                            ("feature_selection", feature_selection),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = PrecisionMicro()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'RF Classifier Select From Model': {
            "percent_features": 1.0,
            "number_features": len(X[0]),
            "n_estimators": 20,
            "max_depth": 3,
        },
        'XGBoost Classifier': {
            "n_estimators": 10,
            "eta": 0.1,
            "min_child_weight": 1,
            "max_depth": 3
        }
    }

    clf = XGBoostMulticlassPipeline(parameters=parameters)
    clf.fit(X, y)
    clf_scores = clf.score(X, y, [objective])
    y_pred = clf.predict(X)

    assert((y_pred == sk_pipeline.predict(X)).all())
    assert (sk_score == clf_scores[objective.name])
    assert len(np.unique(y_pred)) == 3
    assert len(clf.feature_importances) == len(X[0])
    assert not clf.feature_importances.isnull().all().all()

    # testing objective parameter passed in does not change results
    clf.fit(X, y)
    y_pred_with_objective = clf.predict(X)
    assert((y_pred == y_pred_with_objective).all())


def test_xg_input_feature_names(X_y):
    X, y = X_y
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'median'
        },
        'RF Classifier Select From Model': {
            "percent_features": 1.0,
            "number_features": X.shape[1],
            "n_estimators": 20,
            "max_depth": 5
        },
        'XGBoost Classifier': {
            "n_estimators": 20,
            "eta": 0.2,
            "min_child_weight": 3,
            "max_depth": 5,
        }
    }

    clf = XGBoostBinaryPipeline(parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name
