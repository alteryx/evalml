from numpy.random import RandomState
from skopt import Space

from .tuner import Tuner
from .tuner_exceptions import NoParamsException


class RandomSearchTuner(Tuner):
    """Random Search Optimizer

    Example:
        >>> tuner = RandomSearchTuner([(1,10), ['A', 'B']], random_state=0)
        >>> print(tuner.propose())
        (6, 'B')
        >>> print(tuner.propose())
        (4, 'B')
        >>> print(tuner.propose())
        (5, 'A')
    """

    def __init__(self, space, random_state=None, with_replacement=False, replacement_max_attempts=10):
        """ Sets up check for duplication if needed.

        Arguments:
            space: A list of all dimensions available to tune
            random_state: Unused in this class
            with_replacement: If false, only unique hyperparameters will be shown
            replacement_max_attempts: The maximum number of tries to get a unique
                set of random parameters. Only used if tuner is initalized with
                with_replacement=True
        """
        self._space = Space(space)
        self._random_state = RandomState(random_state)
        self._with_replacement = with_replacement
        self._replacement_max_attempts = replacement_max_attempts
        self._used_parameters = set()
        self._used_parameters.add(())
        self.curr_params = None

    def add(self, parameters, score):
        """Not applicable to random search tuner as generated parameters are
        not dependent on scores of previous parameters.

        Arguments:
            parameters: Hyperparameters used
            score: Associated score
        """
        pass

    def _get_sample(self):
        return tuple(self._space.rvs(random_state=self._random_state)[0])

    def propose(self):
        """Generate a unique set of parameters.

        If tuner was initialized with ``with_replacement=True`` and the tuner is unable to
        generate a unique set of parameters after ``replacement_max_attempts`` tries, then ``NoParamsException`` is raised.

        Returns:
            A list of unique parameters
        """
        if self._with_replacement:
            return self._get_sample()
        elif not self.curr_params:
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
        if self._with_replacement:
            return False
        else:
            curr_params = ()
            attempts = 0
            while curr_params in self._used_parameters:
                if attempts >= self._replacement_max_attempts:
                    raise NoParamsException("Cannot create a unique set of unexplored parameters. Try expanding the search space.")
                    return True
                attempts += 1
                curr_params = self._get_sample()
            self._used_parameters.add(curr_params)
            self.curr_params = curr_params
            return False
