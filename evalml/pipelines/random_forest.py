from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .pipeline_base import PipelineBase

from skopt.space import Real, Integer



class RFPipeline(PipelineBase):
    model = "xgboost"

    def __init__(self, n_estimators, max_depth, min_samples_split, strategy, n_jobs=1, random_state=None):
        self.pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy=strategy)),
             ("forest", RandomForestClassifier(random_state=random_state,
                                               n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               min_samples_split=min_samples_split,
                                               n_jobs=n_jobs))]
        )

    def get_hyperparameters():
        return {
            "n_estimators": Integer(10, 1000),
            "max_depth": Integer(1, 1000),
            "min_samples_split": Integer(2, 100),
            "strategy": ["mean", "median", "most_frequent"],
        }
