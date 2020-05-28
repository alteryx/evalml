from abc import ABC, abstractmethod

from evalml.model_family import handle_model_family
from evalml.pipelines import get_pipelines
from evalml.tuners import SKOptTuner
from evalml.utils import get_random_state


class AutoMLAlgorithmException(Exception):
    """Exception raised when an error is encountered during the computation of the automl algorithm"""
    pass


class AutoMLAlgorithm(ABC):
    """Base class for the automl algorithms which power evalml."""

    def __init__(self,
                 objective,
                 max_pipelines=None,
                 allowed_model_families=None,
                 allowed_pipelines=None,
                 tuner_class=None,
                 random_state=0):
        """This class represents an automated machine learning (AutoML) algorithm. It encapsulates the decision-making logic behind an automl search, by both deciding which pipelines to evaluate next and by deciding what set of parameters to configure the pipeline with.

        To use this interface, you must define a next_batch method which returns the next group of pipelines to evaluate on the training data. That method may access state and results recorded from the previous batches, although that information is not tracked in a general way in this base class. You must also define a can_continue method which tells the caller whether the automl algorithm has more pipelines to recommend for evaluation. Finally, overriding add_result is a convenient way to record pipeline evaluation info if necessary.

        Arguments:
            objective (ObjectiveBase): An objective which defines the problem type and whether larger or smaller scores are more optimal
            max_pipelines (int): The maximum number of pipelines to be evaluated.
            allowed_model_families (list(str, ModelFamily)): The model families enabled in the search. The default value of None indicates all model families are allowed.
            allowed_pipelines (list(class)): A list of PipelineBase subclasses indicating the pipelines allowed in the search. The default of None indicates all pipelines for this problem type are allowed.
            tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
        """
        self.random_state = get_random_state(random_state)
        self.objective = objective
        self.max_pipelines = max_pipelines
        self.allowed_pipelines = allowed_pipelines or get_pipelines(problem_type=self.objective.problem_type, model_families=allowed_model_families)
        self.allowed_model_families = [handle_model_family(f) for f in
                                       (allowed_model_families or list(set([p.model_family for p in self.allowed_pipelines])))]
        self._tuner_class = tuner_class or SKOptTuner
        self._tuners = {}
        for p in self.allowed_pipelines:
            self._tuners[p.name] = self._tuner_class(p.hyperparameters, random_state=self.random_state)
        self._pipeline_number = 0
        self._batch_number = 0

    @property
    def pipeline_number(self):
        return self._pipeline_number

    @property
    def batch_number(self):
        return self._batch_number

    @abstractmethod
    def can_continue(self):
        """Are there more pipelines to evaluate?"""

    @abstractmethod
    def next_batch(self):
        """Get the next batch of pipelines to evaluate"""

    def add_result(self, score, pipeline):
        """Register results from evaluating a pipeline"""
        score_to_minimize = -score if self.objective.greater_is_better else score
        self._tuners[pipeline.name].add(pipeline.parameters, score_to_minimize)
