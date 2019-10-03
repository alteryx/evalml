import category_encoders as ce
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SKPipeline

from evalml.objectives import PrecisionMicro
from evalml.pipelines import LogisticRegressionPipeline


def test_lr_multi(X_y_multi):
    X, y = X_y_multi
    imputer = SimpleImputer(strategy='mean')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    estimator = LogisticRegression(random_state=0,
                                   penalty='l2',
                                   C=1.0,
                                   multi_class='auto',
                                   solver="lbfgs",
                                   n_jobs=-1)
    sk_pipeline = SKPipeline(
        [("imputer", imputer),
         ("encoder", enc),
         ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = PrecisionMicro()
    clf = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]))
    clf.fit(X, y)
    clf_score = clf.score(X, y)
    y_pred = clf.predict(X)
    assert (sk_score == clf_score[0])
    assert len(np.unique(y_pred)) == 3
    # assert len(clf.feature_importances) == len(X[0])
