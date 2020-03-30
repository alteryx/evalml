import itertools

from skopt.space import Integer, Real

from .tuner import Tuner
from .tuner_exceptions import NoParamsException


class GridSearchTuner(Tuner):
    """Grid Search Optimizer

    Example:
        >>> tuner = GridSearchTuner([(1,10), ['A', 'B']], n_points=5)
        >>> print(tuner.propose())
        (1.0, 'A')
        >>> print(tuner.propose())
        (1.0, 'B')
        >>> print(tuner.propose())
        (3.25, 'A')
    """

    def __init__(self, space, n_points=10, random_state=None):
        """ Generate all of the possible points to search for in the grid

        Arguments:
            space: A list of all dimensions available to tune
            n_points: The number of points to sample from along each dimension
                defined in the ``space`` argument
            random_state: Unused in this class
        """
        raw_dimensions = list()
        for dimension in space:
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
            else:
                raise TypeError("Invalid dimension type in tuner")
            raw_dimensions.append(range_values)
        self._grid_points = itertools.product(*raw_dimensions)
        self.curr_params = None

    def add(self, parameters, score):
        """Not applicable to grid search tuner as generated parameters are
        not dependent on scores of previous parameters.

        Arguments:
            parameters: Hyperparameters used
            score: Associated score
        """
        pass

    def propose(self):
        """ Returns hyperparameters from _grid_points iterations

        If all possible combinations of parameters have been scored, then ``NoParamsException`` is raised.

        Returns:
            dict: proposed hyperparameters
        """
        if not self.curr_params:
            self.is_search_space_exhausted()
        params = self.curr_params
        self.curr_params = None
        return params

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
