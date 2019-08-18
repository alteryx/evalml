from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from skopt.space import Integer

from .pipeline_base import PipelineBase


class RFPipeline(PipelineBase):
    name = "Random Forest w/ Imputation"
    model_type = "random_forest"

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 1000),
        "strategy": ["mean", "median", "most_frequent"],
    }

    def __init__(self, objective, n_estimators, max_depth, strategy, n_jobs=1, random_state=0):
        imputer = SimpleImputer(strategy=strategy)

        feature_selection = SelectFromModel(LassoCV(cv=3))

        forest = RandomForestClassifier(random_state=random_state,
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        n_jobs=n_jobs)

        self.pipeline = Pipeline(
            [("imputer", imputer),
             ("feature_selection", feature_selection),
             ("forest", forest)]
        )

        super().__init__(objective=objective, random_state=random_state)
