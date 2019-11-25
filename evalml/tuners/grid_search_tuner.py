import itertools

from skopt import Space
from skopt.space import Real, Integer, Categorical


class GridSearchTuner:
    """Grid Search Optimizer"""

    def __init__(self, space, points=10):
        """ Generate all of the possible points to search for in the grid

        Arguments:
            points: The number of points to uniformly sample from Real \
                    dimensions.
        """
        self.space = Space(space)
        raw_dimensions = list()
        for dimension in self.space:
            if isinstance(dimension, Categorical):
                range_values = dimension.categories
            elif isinstance(dimension, Integer):
                range_values = list(range(dimension.low, dimension.high))
            elif isinstance(dimension, Real):
                low = dimension.low
                high = dimension.high
                points = 100
                delta = (low + high) / points
                range_values = [low + (delta * i) for i in range(points)]
            else:
                return Exception("Invalid dimension type in tuner")
            raw_dimensions.append(range_values)
        self.grid_points = list(itertools.product(*raw_dimensions))

    def add(self, parameters, score):
        # Since this is a grid search, we don't need to store the results.
        return 0

    def propose(self):
        if len(self.grid_points) > 0:
            return self.grid_points.pop()
        else:
            return Exception("There are no more parameters left to search")
