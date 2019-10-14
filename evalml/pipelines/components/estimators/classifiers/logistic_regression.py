from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimator import Estimator
from sklearn.linear_model import LogisticRegression as LogisticRegression
from skopt.space import Integer, Real

class LogisticRegressionClassifier(Estimator):
    """
    Logistic Regression Classifier
    """

    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
    }

    def __init__(self, penalty="l2", C=1.0, n_jobs=-1, random_state=0):
        self.name = "Logistic Regression Classifier"
        self.component_type = ComponentTypes.CLASSIFIER
        self.penalty = penalty
        self.C = C
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._component_obj = LogisticRegression(penalty=self.penalty,
                                                 C=self.C,
                                                 random_state=self.random_state,
                                                 multi_class="auto",
                                                 solver="lbfgs",
                                                 n_jobs=self.n_jobs)

        self.parameters = {"penalty": self.penalty, "C": self.C}
        super().__init__(name=self.name, component_type=self.component_type, parameters=self.parameters, component_obj=self._component_obj)
