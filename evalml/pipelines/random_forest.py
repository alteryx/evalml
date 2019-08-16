from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .pipeline_base import PipelineBase


class RFPipeline(PipelineBase):
    model = "xgboost"

    def __init__(self, n_estimators, strategy, random_state=None):
        self.pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy=strategy)),
             ("forest", RandomForestClassifier(random_state=random_state,
                                               n_estimators=n_estimators))]
        )

    def get_hyperparameters():
        return {
            "n_estimators": [0, 100],
            "strategy": ["mean", "median", "most_frequent"]
        }
