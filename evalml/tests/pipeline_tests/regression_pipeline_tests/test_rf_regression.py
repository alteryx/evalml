import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import R2
from evalml.pipelines import RFRegressionPipeline


def test_rf_init(X_y_reg):
    X, y = X_y_reg

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Random Forest Regressor': {
            "n_estimators": 20,
            "max_depth": 5,
        }
    }
    clf = RFRegressionPipeline(parameters=parameters, random_state=2)
    expected_parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {
            'top_n': 10,
            'categories': None,
            'drop': None,
            'handle_unknown': 'ignore',
            'handle_missing': 'error'},
        'Random Forest Regressor': {
            'max_depth': 5,
            'n_estimators': 20,
            'n_jobs': -1
        }
    }

    assert clf.parameters == expected_parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'Random Forest Regressor w/ One Hot Encoder + Simple Imputer'


def test_summary():
    assert RFRegressionPipeline.summary == 'Random Forest Regressor w/ One Hot Encoder + Simple Imputer'


def test_rf_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression

    imputer = SimpleImputer(strategy='most_frequent')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    estimator = RandomForestRegressor(random_state=0,
                                      n_estimators=10,
                                      max_depth=3,
                                      n_jobs=-1)
    sk_pipeline = Pipeline([("encoder", enc),
                            ("imputer", imputer),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = R2()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'Random Forest Regressor': {
            "n_estimators": 10,
            "max_depth": 3,
        }
    }
    clf = RFRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    clf_scores = clf.score(X, y, [objective])
    y_pred = clf.predict(X)
    np.testing.assert_almost_equal(y_pred.to_numpy(), sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_scores[objective.name], decimal=5)

    # testing objective parameter passed in does not change results
    y_pred_with_objective = clf.predict(X, objective)
    np.testing.assert_almost_equal(y_pred.to_numpy(), y_pred_with_objective.to_numpy(), decimal=5)


def test_rfr_input_feature_names(X_y_reg):
    X, y = X_y_reg
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Random Forest Regressor': {
            "n_estimators": 20,
            "max_depth": 5,
        }
    }
    clf = RFRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importance) == len(X.columns)
    assert not clf.feature_importance.isnull().all().all()
    for col_name in clf.feature_importance["feature"]:
        assert "col_" in col_name
