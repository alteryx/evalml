"""Component that selects top features based on importance weights using a Random Forest classifier."""
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.feature_selection import SelectFromModel as SkSelect
from skopt.space import Real

from evalml.pipelines.components.transformers.feature_selection.feature_selector import (
    FeatureSelector,
)


class RFClassifierSelectFromModel(FeatureSelector):
    """Selects top features based on importance weights using a Random Forest classifier.

    Args:
        number_features (int): The maximum number of features to select.
            If both percent_features and number_features are specified, take the greater number of features. Defaults to 0.5.
            Defaults to None.
        n_estimators (float): The number of trees in the forest. Defaults to 100.
        max_depth (int): Maximum tree depth for base learners. Defaults to 6.
        percent_features (float): Percentage of features to use.
            If both percent_features and number_features are specified, take the greater number of features. Defaults to 0.5.
        threshold (string or float): The threshold value to use for feature selection.
            Features whose importance is greater or equal are kept while the others are discarded.
            If "median", then the threshold value is the median of the feature importances.
            A scaling factor (e.g., "1.25*mean") may also be used. Defaults to -np.inf.
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "RF Classifier Select From Model"
    hyperparameter_ranges = {
        "percent_features": Real(0.01, 1),
        "threshold": ["mean", "median"],
    }
    """{
        "percent_features": Real(0.01, 1),
        "threshold": ["mean", "median"],
    }"""

    def __init__(
        self,
        number_features=None,
        n_estimators=10,
        max_depth=None,
        percent_features=0.5,
        threshold="median",
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "number_features": number_features,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "percent_features": percent_features,
            "threshold": threshold,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        estimator = SKRandomForestClassifier(
            random_state=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
        )
        max_features = (
            max(1, int(percent_features * number_features)) if number_features else None
        )
        feature_selection = SkSelect(
            estimator=estimator,
            max_features=max_features,
            threshold=threshold,
            **kwargs,
        )
        super().__init__(
            parameters=parameters,
            component_obj=feature_selection,
            random_seed=random_seed,
        )
