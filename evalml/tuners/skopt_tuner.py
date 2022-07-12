"""Bayesian Optimizer."""
import logging
import warnings

import pandas as pd
from skopt import Optimizer

from evalml.tuners.tuner import Tuner
from evalml.tuners.tuner_exceptions import ParameterError

logger = logging.getLogger(__name__)


class SKOptTuner(Tuner):
    """Bayesian Optimizer.

    Args:
        pipeline_hyperparameter_ranges (dict): A set of hyperparameter ranges corresponding to a pipeline's parameters.
        random_seed (int): The seed for the random number generator. Defaults to 0.

    Examples:
        >>> tuner = SKOptTuner({'My Component': {'param a': [0.0, 10.0], 'param b': ['a', 'b', 'c']}})
        >>> proposal = tuner.propose()
        ...
        >>> assert proposal.keys() == {'My Component'}
        >>> assert proposal['My Component'] == {'param a': 5.928446182250184, 'param b': 'c'}

        Determines points using a Bayesian Optimizer approach.

        >>> for each in range(7):
        ...     print(tuner.propose())
        {'My Component': {'param a': 8.57945617622757, 'param b': 'c'}}
        {'My Component': {'param a': 6.235636967859724, 'param b': 'b'}}
        {'My Component': {'param a': 2.9753460654447235, 'param b': 'a'}}
        {'My Component': {'param a': 2.7265629458011325, 'param b': 'b'}}
        {'My Component': {'param a': 8.121687287754932, 'param b': 'b'}}
        {'My Component': {'param a': 3.927847961008298, 'param b': 'c'}}
        {'My Component': {'param a': 3.3739616041726843, 'param b': 'b'}}
    """

    def __init__(self, pipeline_hyperparameter_ranges, random_seed=0):
        super().__init__(pipeline_hyperparameter_ranges, random_seed=random_seed)
        self.opt = Optimizer(
            self._search_space_ranges,
            "ET",
            acq_optimizer="sampling",
            random_state=random_seed,
        )

    def add(self, pipeline_parameters, score):
        """Add score to sample.

        Args:
            pipeline_parameters (dict): A dict of the parameters used to evaluate a pipeline
            score (float): The score obtained by evaluating the pipeline with the provided parameters

        Returns:
            None

        Raises:
            Exception: If skopt tuner errors.
            ParameterError: If skopt receives invalid parameters.
        """
        # skip adding nan scores
        if pd.isnull(score):
            return
        flat_parameter_values = self._convert_to_flat_parameters(pipeline_parameters)
        try:
            self.opt.tell(flat_parameter_values, score)
        except Exception as e:
            logger.debug(
                "SKOpt tuner received error during add. Score: {}\nParameters: {}\nFlat parameter values: {}\nError: {}".format(
                    pipeline_parameters,
                    score,
                    flat_parameter_values,
                    e,
                ),
            )
            if str(e) == "'<=' not supported between instances of 'int' and 'NoneType'":
                msg = "Invalid parameters specified to SKOptTuner.add: parameters {} error {}".format(
                    pipeline_parameters,
                    str(e),
                )
                logger.error(msg)
                raise ParameterError(msg)
            raise (e)

    def propose(self):
        """Returns a suggested set of parameters to train and score a pipeline with, based off the search space dimensions and prior samples.

        Returns:
            dict: Proposed pipeline parameters.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not len(self._search_space_ranges):
                return self._convert_to_pipeline_parameters({})
            flat_parameters = self.opt.ask()
            return self._convert_to_pipeline_parameters(flat_parameters)
