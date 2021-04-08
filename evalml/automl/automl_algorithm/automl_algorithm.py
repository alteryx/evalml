from abc import ABC, abstractmethod

from evalml.exceptions import PipelineNotFoundError
from evalml.tuners import SKOptTuner


class AutoMLAlgorithmException(Exception):
    """Exception raised when an error is encountered during the computation of the automl algorithm"""
    pass


class AutoMLAlgorithm(ABC):
    """Base class for the automl algorithms which power evalml."""

    def __init__(self,
                 allowed_pipelines=None,
                 max_iterations=None,
                 tuner_class=None,
                 random_seed=0):
        """This class represents an automated machine learning (AutoML) algorithm. It encapsulates the decision-making logic behind an automl search, by both deciding which pipelines to evaluate next and by deciding what set of parameters to configure the pipeline with.

        To use this interface, you must define a next_batch method which returns the next group of pipelines to evaluate on the training data. That method may access state and results recorded from the previous batches, although that information is not tracked in a general way in this base class. Overriding add_result is a convenient way to record pipeline evaluation info if necessary.

        Arguments:
            allowed_pipelines (list(class)): A list of PipelineBase subclasses indicating the pipelines allowed in the search. The default of None indicates all pipelines for this problem type are allowed.
            max_iterations (int): The maximum number of iterations to be evaluated.
            tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        self.random_seed = random_seed
        self.allowed_pipelines = allowed_pipelines or []
        self.max_iterations = max_iterations
        self._tuner_class = tuner_class or SKOptTuner
        self._tuners = {}
        for pipeline in self.allowed_pipelines:
            self._tuners[pipeline.name] = self._tuner_class(pipeline.hyperparameters, random_seed=self.random_seed)
        self._pipeline_number = 0
        self._batch_number = 0

    @abstractmethod
    def next_batch(self):
        """Get the next batch of pipelines to evaluate

        Returns:
            list(PipelineBase): a list of instances of PipelineBase subclasses, ready to be trained and evaluated.
        """

    def add_result(self, score_to_minimize, pipeline, trained_pipeline_results):
        """Register results from evaluating a pipeline

        Arguments:
            score_to_minimize (float): The score obtained by this pipeline on the primary objective, converted so that lower values indicate better pipelines.
            pipeline (PipelineBase): The trained pipeline object which was used to compute the score.
            trained_pipeline_results (dict): Results from training a pipeline.
        """
        if pipeline.name not in self._tuners:
            raise PipelineNotFoundError(f"No such pipeline allowed in this AutoML search: {pipeline.name}")
        self._tuners[pipeline.name].add(pipeline.parameters, score_to_minimize)

    @property
    def pipeline_number(self):
        """Returns the number of pipelines which have been recommended so far."""
        return self._pipeline_number

    @property
    def batch_number(self):
        """Returns the number of batches which have been recommended so far."""
        return self._batch_number
