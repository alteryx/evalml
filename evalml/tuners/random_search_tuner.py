from numpy.random import RandomState
from skopt import Space

from .tuner import Tuner
from .tuner_exceptions import NoParamsException


class RandomSearchTuner(Tuner):
    """Random Search Optimizer

    Example:
        >>> tuner = RandomSearchTuner([(1,10)], n_points=5)
    """

    def __init__(self, space, random_state=None, check_duplicates=True):
        """ Sets up check for duplication if needed

        Arguments:
            space: A list of all dimensions available to tune
            random_state: Unused in this class
            check_duplicates: A boolean that determines if hyperparameters should be unique
        """
        self.space = Space(space)
        self.random_state = RandomState(random_state)
        self.check_duplicates = check_duplicates
        if self.check_duplicates is True:
            self.used_parameters = set()

    def add(self, parameters, score):
        pass

    def propose(self, max_attempts=10):
        """Generate a unique set of parameters.

        Arguments:
            max_attempts (Object): The maximum number of tries to get a unique
                set of random parameters. Only used if tuner is initalized with
                self.check_duplicates=True

        Returns:
            A list of unique parameters
        """
        curr_parameters = self.space.rvs(random_state=self.random_state)[0]
        if self.check_duplicates is True:
            param_tuple = tuple(curr_parameters)
            attempts = 0
            while param_tuple in self.used_parameters and attempts < max_attempts:
                attempts += 1
                curr_parameters = self.space.rvs(random_state=self.random_state)[0]
                param_tuple = tuple(curr_parameters)
            if attempts >= max_attempts:
                raise NoParamsException("Cannot create a unique set of unexplored parameters. Try expanding the search space.")
            else:
                self.used_parameters.add(param_tuple)
        return curr_parameters
