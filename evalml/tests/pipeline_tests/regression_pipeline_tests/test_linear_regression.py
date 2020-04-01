import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evalml.objectives import R2
from evalml.pipelines import LinearRegressionPipeline


def test_lr_init(X_y_categorical_regression):
    X, y = X_y_categorical_regression

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Linear Regressor': {
            'fit_intercept': True,
            'normalize': True,
        },
    }
    clf = LinearRegressionPipeline(parameters=parameters, random_state=2)
    assert clf.parameters == parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])


def test_linear_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    imputer = SimpleImputer(strategy='most_frequent')
    scaler = StandardScaler()
    estimator = LinearRegression(normalize=False, fit_intercept=True, n_jobs=-1)
    sk_pipeline = Pipeline([("encoder", enc),
                            ("imputer", imputer),
                            ("scaler", scaler),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = R2()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'Linear Regressor': {
            'fit_intercept': True,
            'normalize': False,
        }
    }
    clf = LinearRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    clf_scores = clf.score(X, y, [objective])
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_scores[objective.name], decimal=5)
    assert not clf.feature_importances.isnull().all().all()

    # testing objective parameter passed in does not change results
    clf.fit(X, y, objective)
    y_pred_with_objective = clf.predict(X, objective)
    assert((y_pred == y_pred_with_objective).all())


def test_lr_input_feature_names(X_y):
    X, y = X_y
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    objective = R2()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Linear Regressor': {
            'fit_intercept': True,
            'normalize': True,
        }
    }
    clf = LinearRegressionPipeline(parameters=parameters)
    clf.fit(X, y, objective)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name
