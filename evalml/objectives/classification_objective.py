from abc import abstractmethod

from .objective_base import ObjectiveBase


class ClassificationObjective(ObjectiveBase):
    """Base class for all classification objectives."""

    @abstractmethod
    def decision_function(self, ypred_proba, threshold=None, X=None):
        """
        Decision function used to determine class labels.
        Arguments:
            ypred_proba (pd.Series): Predicted probablities
            threshold (float): Threshold to determine class label
            X (pd.DataFrame): Additional information to use to make a decision

            Returns:
                pd.Series: Series of predicted class labels
        """
        raise NotImplementedError("decision_function for this classification objective is not yet defined!")
