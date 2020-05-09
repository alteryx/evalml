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
