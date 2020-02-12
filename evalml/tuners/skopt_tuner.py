import pandas as pd
from skopt import Optimizer

from .tuner import Tuner


class SKOptTuner(Tuner):
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
            parameters (dict): hyperparameters
            score (float): associated score

        Returns:
            None
        """
        # skip adding nan scores
        if pd.isnull(score): return
        self.opt.tell(list(parameters), score)

    def propose(self):
        """ Returns hyperparameters based off search space and samples

        Returns:
            dict: proposed hyperparameters
        """
        return self.opt.ask()
