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
            n_points: The number of points to uniformly sample from \
                Real dimensions.
            space: A list of all dimensions available to tune
            random_state: Unused in this class
        """
        raw_dimensions = list()
        for dimension in space:
            # Categorical dimension
            print(type(dimension))
            if isinstance(dimension, list):
                range_values = dimension
            elif isinstance(dimension, Real) or isinstance(dimension, Integer) or isinstance(dimension, tuple):
                if isinstance(dimension, tuple) and isinstance(dimension[0], (int, float)) and isinstance(dimension[1], (int, float)):
                    if dimension[1] > dimension[0]:
                        low = dimension[0]
                        high = dimension[1]
                    else:
                        raise TypeError("Invalid dimension type in tuner")
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

        Returns:
            dict: proposed hyperparameters
        """
        try:
            return next(self._grid_points)
        except StopIteration:
            raise NoParamsException("Grid search has exhausted all possible parameters.")
