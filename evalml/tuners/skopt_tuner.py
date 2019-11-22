import pandas as pd
from skopt import Optimizer


class SKOptTuner:
    """Bayesian Optimizer"""

    def __init__(self, space, random_state=0):
        """ Init SkOptTuner
        Arguments:
            space (dict): search space for hyperparameters
            random_state (int): random state

        Returns:
            SKoptTuner: self
        """
        self.opt = Optimizer(space, "ET", acq_optimizer="sampling", random_state=random_state)

    def add(self, parameters, score):
        """ Add score to sample

        Arguments:
            parameters (dict): hyper-parameters
            score (float): associated score

        Returns:
            None
        """
        # skip adding nan scores for
        if not pd.isnull(score):
            return self.opt.tell(list(parameters), score)

    def propose(self):
        """ Returns hyper-parameters based off search space and samples

        Arguments:
            None:

        Returns:
            dict: proposed hyper-parameters
        """
        return self.opt.ask()
