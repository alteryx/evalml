import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import R2
from evalml.pipelines import RFRegressionPipeline


def test_rf_init(X_y_reg):
    X, y = X_y_reg

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'RF Classifier Select From Model': {
            "percent_features": 1.0,
            "number_features": len(X[0]),
            "n_estimators": 20
        },
        'Random Forest Regressor': {
            "n_estimators": 20,
            "max_depth": 5,
        }
    }
    clf = RFRegressionPipeline(parameters=parameters)
    assert clf.parameters == parameters


def test_rf_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression

    imputer = SimpleImputer(strategy='most_frequent')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    estimator = RandomForestRegressor(random_state=0,
                                      n_estimators=10,
                                      max_depth=3,
                                      n_jobs=-1)
    feature_selection = SelectFromModel(estimator=estimator,
                                        max_features=max(1, int(1 * X.shape[1])),
                                        threshold=-np.inf)
    sk_pipeline = Pipeline([("encoder", enc),
                            ("imputer", imputer),
                            ("feature_selection", feature_selection),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = R2()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'RF Regressor Select From Model': {
            "percent_features": 1.0,
            "number_features": X.shape[1],
            "n_estimators": 10,
            "max_depth": 3,
        },
        'Random Forest Regressor': {
            "n_estimators": 10,
            "max_depth": 3,
        }
    }
    clf = RFRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    clf_score = clf.score(X, y, [objective])
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_score[objective.name], decimal=5)


def test_rfr_input_feature_names(X_y_reg):
    X, y = X_y_reg
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    objective = R2()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'RF Classifier Select From Model': {
            "percent_features": 1.0,
            "number_features": X.shape[1],
            "n_estimators": 20
        },
        'Random Forest Regressor': {
            "n_estimators": 20,
            "max_depth": 5,
        }
    }
    clf = RFRegressionPipeline(parameters=parameters)
    clf.fit(X, y, objective)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name
