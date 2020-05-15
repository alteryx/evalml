from skopt import Space

from evalml.tuners import NoParamsException, Tuner
from evalml.utils import get_random_state


class RandomSearchTuner(Tuner):
    """Random Search Optimizer

    Example:
        >>> from evalml.pipelines.classification import LogisticRegressionBinaryPipeline
        >>> tuner = RandomSearchTuner(LogisticRegressionBinaryPipeline, random_state=42)
        >>> proposal = tuner.propose()
        >>> assert sorted(proposal.keys()) == ['Logistic Regression Classifier', 'One Hot Encoder', 'Simple Imputer', 'Standard Scaler']
        >>> assert proposal['One Hot Encoder'] == {}
        >>> assert proposal['Simple Imputer'] == {'impute_strategy': 'median'}
        >>> assert proposal['Standard Scaler'] == {}
        >>> assert proposal['Logistic Regression Classifier']['penalty'] == 'l2'
        >>> assert abs(proposal['Logistic Regression Classifier']['C'] - 7.322619478695938) < 1e-5
    """

    def __init__(self, pipeline_class, random_state=0, with_replacement=False, replacement_max_attempts=10):
        """ Sets up check for duplication if needed.

        Arguments:
            pipeline_class (PipelineBase subclass): the pipeline class to tune
            random_state: Unused in this class
            with_replacement: If false, only unique hyperparameters will be shown
            replacement_max_attempts: The maximum number of tries to get a unique
                set of random parameters. Only used if tuner is initalized with
                with_replacement=True
        """
        super().__init__(pipeline_class, random_state=random_state)
        self._space = Space(self._search_space_ranges)
        self._random_state = get_random_state(random_state)
        self._with_replacement = with_replacement
        self._replacement_max_attempts = replacement_max_attempts
        self._used_parameters = set()
        self._used_parameters.add(())
        self.curr_params = None

    def add(self, pipeline_parameters, score):
        """Not applicable to random search tuner as generated parameters are
        not dependent on scores of previous parameters.

        Arguments:
            pipeline_parameters (dict): a dict of the parameters used to evaluate a pipeline
            score (float): the score obtained by evaluating the pipeline with the provided parameters
        """
        pass

    def _get_sample(self):
        return tuple(self._space.rvs(random_state=self._random_state)[0])

    def propose(self):
        """Generate a unique set of parameters.

        If tuner was initialized with ``with_replacement=True`` and the tuner is unable to
        generate a unique set of parameters after ``replacement_max_attempts`` tries, then ``NoParamsException`` is raised.

        Returns:
            dict: proposed pipeline parameters
        """
        if self._with_replacement:
            return self._convert_to_pipeline_parameters(self._get_sample())
        elif not self.curr_params:
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
