import inspect
from abc import ABC, abstractmethod

from evalml.pipelines import PipelineBase


class Tuner(ABC):
    """Defines API for Tuners

    Tuners implement different strategies for sampling from a search space. They're used in EvalML to search the space of pipeline hyperparameters.
    """

    def __init__(self, pipeline_class, random_state=0):
        """Init Tuner

        Arguments:
            pipeline_class (PipelineBase subclass): the pipeline class to tune
            random_state (int, np.random.RandomState): The random state

        Returns:
            Tuner: self
        """
        if not inspect.isclass(pipeline_class) and issubclass(pipeline_class, PipelineBase):
            raise Exception()
        self._pipeline_class = pipeline_class
        self._component_names = set()
        self._parameter_names_map = dict()
        self._search_space_names = []
        self._search_space_ranges = []
        hyperparameter_ranges = pipeline_class.hyperparameters
        for component_name, component_ranges in hyperparameter_ranges.items():
            self._component_names.add(component_name)
            for parameter_name, parameter_range in component_ranges.items():
                flat_parameter_name = '{}: {}'.format(component_name, parameter_name)
                self._parameter_names_map[flat_parameter_name] = (component_name, parameter_name)
                self._search_space_names.append(flat_parameter_name)
                self._search_space_ranges.append(parameter_range)

    def _convert_to_flat_parameters(self, pipeline_parameters):
        """Convert from pipeline parameters to a flat list of values"""
        flat_parameter_values = []
        for flat_parameter_name in self._search_space_names:
            component_name, parameter_name = self._parameter_names_map[flat_parameter_name]
            flat_parameter_values.append(pipeline_parameters[component_name][parameter_name])
        return flat_parameter_values

    def _convert_to_pipeline_parameters(self, flat_parameters):
        """Convert from a flat list of values to a dict of pipeline parameters"""
        pipeline_parameters = {component_name: dict() for component_name in self._component_names}
        for i, parameter_value in enumerate(flat_parameters):
            flat_parameter_name = self._search_space_names[i]
            component_name, parameter_name = self._parameter_names_map[flat_parameter_name]
            pipeline_parameters[component_name][parameter_name] = parameter_value
        return pipeline_parameters

    @abstractmethod
    def add(self, pipeline_parameters, score):
        """ Register a set of hyperparameters with the score obtained from training a pipeline with those hyperparameters.

        Arguments:
            pipeline_parameters (dict): a dict of the parameters used to evaluate a pipeline
            score (float): the score obtained by evaluating the pipeline with the provided parameters

        Returns:
            None
        """

    @abstractmethod
    def propose(self):
        """Returns a suggested set of parameters to train and score a pipeline with, based off the search space dimensions and prior samples.

        Returns:
            dict: proposed pipeline parameters
        """

    def is_search_space_exhausted(self):
        """ Optional. If possible search space for tuner is finite, this method indicates
        whether or not all possible parameters have been scored.

        Returns:
            bool: Returns true if all possible parameters in a search space has been scored.
        """
        return False
