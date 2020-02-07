import itertools

from skopt.space import Integer, Real

from .tuner_exceptions import NoParamsException


class GridSearchTuner:
    """Grid Search Optimizer"""

    def __init__(self, space, n_points=10, random_state=None):
        """ Generate all of the possible points to search for in the grid

        Arguments:
            n_points: The number of points to uniformly sample from \
                Real dimensions.
            space: A list of all dimensions available to tune
            random_state: Unused in this class

        Example:
            >>> from skopt.space import Real
            >>> GridSearchTuner([Real(1,10)], n_points=5)
        """
        raw_dimensions = list()
        for dimension in space:
            # Categorical dimension
            if isinstance(dimension, list):
                range_values = dimension
            elif isinstance(dimension, Real) or isinstance(dimension, Integer):
                low = dimension.low
                high = dimension.high
                delta = (high - low) / (n_points - 1)
                if isinstance(dimension, Integer):
                    range_values = [int((x * delta) + low) for x in range(n_points)]
                else:
                    range_values = [(x * delta) + low for x in range(n_points)]
            else:
                return Exception("Invalid dimension type in tuner")
            raw_dimensions.append(range_values)
        self._grid_points = itertools.product(*raw_dimensions)

    def add(self, parameters, score):
        pass

    def propose(self):
        try:
            return next(self._grid_points)
        except StopIteration:
            raise NoParamsException("Grid search has exhausted all possible parameters.")
