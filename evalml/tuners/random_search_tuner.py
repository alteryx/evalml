from numpy.random import RandomState
from skopt import Space


class RandomSearchTuner:
    """Random Search Optimizer"""

    def __init__(self, space, random_state=None):
        self.space = Space(space)
        self.random_state = RandomState(random_state)
        self.used_parameters = set()

    def add(self, parameters, score):
        # Since this is a random search, we don't need to store the results.
        return 0

    def propose(self, max_attempts=10):
        """Generate a unique set of parameters.

        Arguments:
            max_attempts (Object): The maximum number of tries to get a unique
                set of random parameters

        Returns:
            A list of unique parameters
        """
        curr_parameters = self.space.rvs(random_state=self.random_state)[0]
        param_tuple = tuple(curr_parameters)
        attempts = 0
        while param_tuple in self.used_parameters and attempts < max_attempts:
            attempts += 1
            curr_parameters = self.space.rvs(random_state=self.random_state)[0]
        if attempts >= max_attempts:
            Exception("Cannot create a unique set of unexplored parameters. \
                       Try expanding the search space.")
        else:
            self.used_parameters.add(param_tuple)
            return curr_parameters
