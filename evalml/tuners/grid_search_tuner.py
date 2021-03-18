import itertools

from skopt.space import Integer, Real

from .tuner import Tuner
from .tuner_exceptions import NoParamsException


class GridSearchTuner(Tuner):
    """Grid Search Optimizer.

    Example:
        >>> tuner = GridSearchTuner({'My Component': {'param a': [0.0, 10.0], 'param b': ['a', 'b', 'c']}}, n_points=5)
        >>> proposal = tuner.propose()
        >>> assert proposal.keys() == {'My Component'}
        >>> assert proposal['My Component'] == {'param a': 0.0, 'param b': 'a'}
    """

    def __init__(self, pipeline_hyperparameter_ranges, n_points=10, random_seed=0):
        """ Generate all of the possible points to search for in the grid

        Arguments:
            pipeline_hyperparameter_ranges (dict): a set of hyperparameter ranges corresponding to a pipeline's parameters
            n_points (int): The number of points to sample from along each dimension
                defined in the ``space`` argument
            random_seed (int): Seed for random number generator. Unused in this class, defaults to 0.
        """
        super().__init__(pipeline_hyperparameter_ranges, random_seed=random_seed)
        raw_dimensions = list()
        for dimension in self._search_space_ranges:
            # Categorical dimension
            if isinstance(dimension, list):
                range_values = dimension
            elif isinstance(dimension, (Real, Integer, tuple)):
                if isinstance(dimension, (tuple)) and isinstance(dimension[0], (int, float)) and isinstance(dimension[1], (int, float)):
                    if dimension[1] > dimension[0]:
                        low = dimension[0]
                        high = dimension[1]
                    else:
                        error_text = "Upper bound must be greater than lower bound. Parameter lower bound is {0} and upper bound is {1}"
                        error_text = error_text.format(dimension[0], dimension[1])
                        raise ValueError(error_text)
                else:
                    low = dimension.low
                    high = dimension.high
                delta = (high - low) / (n_points - 1)
                if isinstance(dimension, Integer):
                    range_values = [int((x * delta) + low) for x in range(n_points)]
                else:
                    range_values = [(x * delta) + low for x in range(n_points)]
            raw_dimensions.append(range_values)
        self._grid_points = itertools.product(*raw_dimensions)
        self.curr_params = None

    def add(self, pipeline_parameters, score):
        """Not applicable to grid search tuner as generated parameters are
        not dependent on scores of previous parameters.

        Arguments:
            pipeline_parameters (dict): a dict of the parameters used to evaluate a pipeline
            score (float): the score obtained by evaluating the pipeline with the provided parameters
        """
        pass

    def propose(self):
        """Returns parameters from _grid_points iterations

        If all possible combinations of parameters have been scored, then ``NoParamsException`` is raised.

        Returns:
            dict: proposed pipeline parameters
        """
        if not self.curr_params:
            self.is_search_space_exhausted()
        params = self.curr_params
        self.curr_params = None
        return self._convert_to_pipeline_parameters(params)

    def is_search_space_exhausted(self):
        """Checks if it is possible to generate a set of valid parameters. Stores generated parameters in
        ``self.curr_params`` to be returned by ``propose()``.

        Raises:
            NoParamsException: If a search space is exhausted, then this exception is thrown.

        Returns:
            bool: If no more valid parameters exists in the search space, return false.
        """
        try:
            self.curr_params = next(self._grid_points)
            return False
        except StopIteration:
            raise NoParamsException("Grid search has exhausted all possible parameters.")
            return True
