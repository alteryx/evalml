from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ZeroRClassifier(Estimator):
    """Classifier that predicts using the mode."""
    name = "ZeroR Classifier"
    hyperparameter_ranges = {}
    model_family = ModelFamily.NONE
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self):
        parameters = {}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)


    def fit(self, X, y=None):
        """Fits component to data

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training labels of length [n_samples]

        Returns:
            self
        """
        pass


    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame) : features

        Returns:
            pd.Series : estimated labels
        """
        pass


    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (pd.DataFrame) : features

        Returns:
            pd.DataFrame : probability estimates
        """
        pass


    @property
    def feature_importances(self):
        """Returns feature importances.

        Returns:
            list(float) : importance associated with each feature


            based on %?
        """
        pass
