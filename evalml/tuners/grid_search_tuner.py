import itertools

from skopt.space import Integer, Real

from .tuner_exceptions import NoParamsException


class GridSearchTuner:
    """Grid Search Optimizer"""

    def __init__(self, space, points=10, random_state=None):
        """ Generate all of the possible points to search for in the grid

        Arguments:
            points: The number of points to uniformly sample from \
                Real dimensions.
            random_state: Not used in grid search, kept for compatibility
        """
        raw_dimensions = list()
        for dimension in space:
            # Categorical dimension
            if isinstance(dimension, list):
                range_values = dimension
            elif isinstance(dimension, Real) or isinstance(dimension, Integer):
                low = dimension.low
                high = dimension.high
                delta = (high - low) / (points - 1)

                if isinstance(dimension, Integer):
                    range_values = [int((x * delta) + low) for x in range(points)]
                else:
                    range_values = [(x * delta) + low for x in range(points)]
            else:
                return Exception("Invalid dimension type in tuner")
            raw_dimensions.append(range_values)
        self.grid_points = list(itertools.product(*raw_dimensions))

    def add(self, parameters, score):
        # Since this is a grid search, we don't need to store the results.
        pass

    def propose(self):
        if len(self.grid_points) > 0:
            return self.grid_points.pop()
        else:
            raise NoParamsException("Grid search has exhausted all possible parameters.")
