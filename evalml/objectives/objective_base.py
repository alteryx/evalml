from abc import ABC, abstractmethod


class ObjectiveBase(ABC):
    name = None
    greater_is_better = True
    score_needs_proba = False

    def __init__(self, verbose=False):
        self.verbose = verbose

    @abstractmethod
    def objective_function(self, y_predicted, y_true, X=None):
        raise NotImplementedError

    def score(self, y_predicted, y_true, X=None):
        """Calculate score from applying fitted objective to predicted values

        If a higher score is better than a lower score, set greater_is_better attribute to True

        Arguments:
            y_predicted (list): the predictions from the model. If needs_proba is True,
                it is the probability estimates

            y_true (list): the ground truth for the predictions.

            X (pd.DataFrame): any extra columns that are needed from training
                data to fit. Only provided if uses_extra_columns is True.

        Returns:
            score

        """
        return self.objective_function(y_predicted, y_true, X)
