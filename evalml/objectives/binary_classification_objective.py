from .objective_base import ObjectiveBase


class BinaryClassificationObjective(ObjectiveBase):
    can_optimize_bin_class_threshold = False
    optimal = None

    # def decision_function(self, ypred_proba, classification_threshold=0.0, X=None):
    #     """Apply the learned objective function to the output of a model.
    #     note to self (delete later): old predict()
    #     Arguments:
    #         ypred_proba: the prediction to transform to final prediction
    #     Returns:
    #         predictions
    #     """
    #     return ypred_proba > classification_threshold

    # def optimize_threshold(self, ypred_proba, y_true, X=None):
    #     """Learn and optimize the threshold objective function based on the predictions from a model.
    #     Arguments:
    #         ypred_proba (list): the predictions from the model. If needs_proba is True,
    #             it is the probability estimates
    #         y_true (list): the ground truth for the predictions.
    #         X (pd.DataFrame): any extra columns that are needed from training
    #             data to fit. Only provided if uses_extra_columns is True.
    #     Returns:
    #         optimal threshold
    #     """
    #     def cost(threshold):
    #         predictions = self.decision_function(ypred_proba, threshold, X=X)
    #         cost = self.objective_function(ypred_proba, y_true, X=X)
    #         return -cost if self.greater_is_better else cost
    #     optimal = minimize_scalar(cost, method='Golden', options={"maxiter": 100})
    #     self.optimal = optimal.x
    #     return self.optimal
