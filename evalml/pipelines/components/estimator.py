from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.linear_model import LogisticRegression as LogisticRegression
from skopt.space import Integer, Real
from xgboost import XGBClassifier

from .component_base import ComponentBase
from .component_types import ComponentTypes


class Estimator(ComponentBase):
    """A component that fits and predicts given data"""

    def __init__(self, name, component_type, hyperparameters={}, parameters={}, needs_fitting=False, component_obj=None, random_state=0):
        super().__init__(name=name, component_type=component_type, hyperparameters=hyperparameters, parameters=parameters, needs_fitting=needs_fitting,
                         component_obj=component_obj, random_state=random_state)

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (DataFrame) : features

        Returns:
            Series : estimated labels
        """
        return self._component_obj.predict(X)

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (DataFrame) : features

        Returns:
            DataFrame : probability estimates
        """
        return self._component_obj.predict_proba(X)


class LogisticRegressionClassifier(Estimator):
    """
    Logistic Regression Classifier
    """

    def __init__(self, penalty="l2", C=1.0, n_jobs=-1, random_state=0):
        self.name = "Logistic Regression Classifier"
        self.component_type = ComponentTypes.CLASSIFIER
        self.penalty = penalty
        self.C = C
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.hyperparameters = {
            "penalty": ["l2"],
            "C": Real(.01, 10),
        }
        self._component_obj = LogisticRegression(penalty=self.penalty,
                                                 C=self.C,
                                                 random_state=self.random_state,
                                                 multi_class="auto",
                                                 solver="lbfgs",
                                                 n_jobs=self.n_jobs)

        self.parameters = {"penalty": self.penalty, "C": self.C}
        super().__init__(name=self.name, component_type=self.component_type, hyperparameters=self.hyperparameters, parameters=self.parameters, component_obj=self._component_obj)


class RandomForestClassifier(Estimator):
    """Random Forest Classifier"""

    def __init__(self, n_estimators, max_depth=None, n_jobs=-1, random_state=0):
        self.name = "Random Forest Classifier"
        self.component_type = ComponentTypes.CLASSIFIER
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.hyperparameters = {
            "n_estimators": Integer(10, 1000),
            "max_depth": Integer(1, 32),
        }
        self._component_obj = SKRandomForestClassifier(random_state=self.random_state,
                                                       n_estimators=self.n_estimators,
                                                       max_depth=self.max_depth,
                                                       n_jobs=self.n_jobs)
        self.parameters = {"n_estimators": self.n_estimators, "max_depth": self.max_depth}
        super().__init__(name=self.name, component_type=self.component_type, hyperparameters=self.hyperparameters, parameters=self.parameters, component_obj=self._component_obj)


class XGBoostClassifier(Estimator):
    """XGBoost Classifier"""

    def __init__(self, eta, max_depth, min_child_weight, random_state=0, **kwargs):
        self.name = "XGBoost Classifier"
        self.component_type = ComponentTypes.CLASSIFIER
        self.eta = eta
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.hyperparameters = {
            "eta": Real(0, 1),
            "max_depth": Integer(1, 20),
            "min_child_weight": Real(1, 10),
        }
        self.parameters = {"eta": self.eta, "max_depth": self.max_depth, "min_child_weight": self.min_child_weight}
        self._component_obj = XGBClassifier(random_state=self.random_state,
                                            eta=self.eta,
                                            max_depth=self.max_depth,
                                            min_child_weight=self.min_child_weight)
        super().__init__(name=self.name, component_type=self.component_type, hyperparameters=self.hyperparameters, parameters=self.parameters, component_obj=self._component_obj)


class RandomForestRegressor(Estimator):
    """Random Forest Regressor"""

    def __init__(self, n_estimators, max_depth=None, n_jobs=-1, random_state=0):
        self.name = "Random Forest Regressor"
        self.component_type = ComponentTypes.REGRESSOR
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.hyperparameters = {
            "n_estimators": Integer(10, 1000),
            "max_depth": Integer(1, 32),
        }
        self._component_obj = SKRandomForestRegressor(random_state=self.random_state,
                                                      n_estimators=self.n_estimators,
                                                      max_depth=self.max_depth,
                                                      n_jobs=self.n_jobs)
        self.parameters = {"n_estimators": self.n_estimators, "max_depth": self.max_depth}
        super().__init__(name=self.name, component_type=self.component_type, hyperparameters=self.hyperparameters, parameters=self.parameters, component_obj=self._component_obj)


class LinearRegressor(Estimator):
    """Linear Regressor"""

    def __init__(self, n_jobs=-1):
        self.name = "Linear Regressor"
        self.component_type = ComponentTypes.REGRESSOR
        self.hyperparameters = {}
        self._component_obj = SKLinearRegression()
        self.parameters = {}
        super().__init__(name=self.name, component_type=self.component_type, hyperparameters=self.hyperparameters, parameters=self.parameters, component_obj=self._component_obj)
