import pandas as pd
from skopt import Optimizer

from .tuner import Tuner


class SKOptTuner(Tuner):
    """Bayesian Optimizer"""

    def __init__(self, pipeline_hyperparameter_ranges, random_state=0):
        """ Init SkOptTuner

        Arguments:
            pipeline_hyperparameter_ranges (dict): a set of hyperparameter ranges corresponding to a pipeline's parameters
            random_state (int, np.random.RandomState): The random state
        """
        super().__init__(pipeline_hyperparameter_ranges, random_state=random_state)
        self.opt = Optimizer(self._search_space_ranges, "ET", acq_optimizer="sampling", random_state=random_state)

    def add(self, pipeline_parameters, score):
        """ Add score to sample

        Arguments:
            pipeline_parameters (dict): a dict of the parameters used to evaluate a pipeline
            score (float): the score obtained by evaluating the pipeline with the provided parameters

        Returns:
            None
        """
        # skip adding nan scores
        if pd.isnull(score):
            return
        flat_parameter_values = self._convert_to_flat_parameters(pipeline_parameters)
        self.opt.tell(flat_parameter_values, score)

    def propose(self):
        """Returns a suggested set of parameters to train and score a pipeline with, based off the search space dimensions and prior samples.

        Returns:
            dict: proposed pipeline parameters
        """
        flat_parameters = self.opt.ask()
        return self._convert_to_pipeline_parameters(flat_parameters)
