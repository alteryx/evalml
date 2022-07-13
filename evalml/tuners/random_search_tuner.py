"""Random Search Optimizer."""
from skopt import Space

from evalml.tuners import NoParamsException, Tuner
from evalml.utils import get_random_state


class RandomSearchTuner(Tuner):
    """Random Search Optimizer.

    Args:
        pipeline_hyperparameter_ranges (dict): a set of hyperparameter ranges corresponding to a pipeline's parameters
        with_replacement (bool): If false, only unique hyperparameters will be shown
        replacement_max_attempts (int): The maximum number of tries to get a unique
            set of random parameters. Only used if tuner is initalized with
            with_replacement=True
        random_seed (int): Seed for random number generator. Defaults to 0.

    Example:
        >>> tuner = RandomSearchTuner({'My Component': {'param a': [0.0, 10.0], 'param b': ['a', 'b', 'c']}}, random_seed=42)
        >>> proposal = tuner.propose()
        ...
        >>> assert proposal.keys() == {'My Component'}
        >>> assert proposal['My Component'] == {'param a': 3.7454011884736254, 'param b': 'c'}

        Determines points using a random search approach.

        >>> for each in range(7):
        ...     print(tuner.propose())
        {'My Component': {'param a': 7.3199394181140525, 'param b': 'b'}}
        {'My Component': {'param a': 1.5601864044243654, 'param b': 'a'}}
        {'My Component': {'param a': 0.5808361216819947, 'param b': 'c'}}
        {'My Component': {'param a': 6.011150117432089, 'param b': 'c'}}
        {'My Component': {'param a': 0.2058449429580245, 'param b': 'c'}}
        {'My Component': {'param a': 8.32442640800422, 'param b': 'a'}}
        {'My Component': {'param a': 1.8182496720710064, 'param b': 'a'}}
    """

    def __init__(
        self,
        pipeline_hyperparameter_ranges,
        with_replacement=False,
        replacement_max_attempts=10,
        random_seed=0,
    ):
        super().__init__(pipeline_hyperparameter_ranges, random_seed=random_seed)
        self._space = Space(self._search_space_ranges)
        self._random_state = get_random_state(random_seed)
        self._with_replacement = with_replacement
        self._replacement_max_attempts = replacement_max_attempts
        self._used_parameters = set()
        self._used_parameters.add(())
        self.curr_params = None

    def add(self, pipeline_parameters, score):
        """Not applicable to random search tuner as generated parameters are not dependent on scores of previous parameters.

        Args:
            pipeline_parameters (dict): A dict of the parameters used to evaluate a pipeline
            score (float): The score obtained by evaluating the pipeline with the provided parameters
        """
        pass

    def _get_sample(self):
        return tuple(self._space.rvs(random_state=self._random_state)[0])

    def propose(self):
        """Generate a unique set of parameters.

        If tuner was initialized with ``with_replacement=True`` and the tuner is unable to
        generate a unique set of parameters after ``replacement_max_attempts`` tries, then ``NoParamsException`` is raised.

        Returns:
            dict: Proposed pipeline parameters
        """
        if not len(self._search_space_ranges):
            return self._convert_to_pipeline_parameters({})
        if self._with_replacement:
            return self._convert_to_pipeline_parameters(self._get_sample())
        elif not self.curr_params:
            self.is_search_space_exhausted()
        params = self.curr_params
        self.curr_params = None
        return self._convert_to_pipeline_parameters(params)

    def is_search_space_exhausted(self):
        """Checks if it is possible to generate a set of valid parameters. Stores generated parameters in ``self.curr_params`` to be returned by ``propose()``.

        Returns:
            bool: If no more valid parameters exists in the search space, return False.

        Raises:
            NoParamsException: If a search space is exhausted, then this exception is thrown.
        """
        if self._with_replacement:
            return False
        else:
            curr_params = ()
            attempts = 0
            while curr_params in self._used_parameters:
                if attempts >= self._replacement_max_attempts:
                    raise NoParamsException(
                        "Cannot create a unique set of unexplored parameters. Try expanding the search space.",
                    )
                    return True
                attempts += 1
                curr_params = self._get_sample()
            self._used_parameters.add(curr_params)
            self.curr_params = curr_params
            return False
