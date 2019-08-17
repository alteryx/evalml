from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from skopt.space import Integer, Real
from xgboost import XGBClassifier

from .pipeline_base import PipelineBase


class XGBoostPipeline(PipelineBase):
    name = "XGBoost w/ Imputation"
    model_type = "random_forest"

    hyperparameters = {
        "eta": Real(0, 1),
        "min_child_weight": Real(1, 1000),
        "max_depth": Integer(1, 20),
        "strategy": ["mean", "median", "most_frequent"],
    }

    def __init__(self, eta, min_child_weight, max_depth, strategy, n_jobs=1, random_state=0):
        imputer = SimpleImputer(strategy=strategy)

        feature_selection = SelectFromModel(LassoCV(cv=3))

        forest = XGBClassifier(
            random_state=random_state,
            eta=eta,
            max_depth=max_depth,
            min_child_weight=min_child_weight
            )

        self.pipeline = Pipeline(
            [("imputer", imputer),
             ("feature_selection", feature_selection),
             ("forest", forest)]
        )

    @classmethod
    def get_hyperparameters(cls):
        return
