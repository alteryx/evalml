"""Base Tuner class."""
from abc import ABC, abstractmethod

from skopt.space import Categorical, Integer, Real


class Tuner(ABC):
    """Base Tuner class.

    Tuners implement different strategies for sampling from a search space. They're used in EvalML to search the space of pipeline hyperparameters.

    Args:
        pipeline_hyperparameter_ranges (dict): a set of hyperparameter ranges corresponding to a pipeline's parameters.
        random_seed (int): The random state. Defaults to 0.
    """

    def __init__(self, pipeline_hyperparameter_ranges, random_seed=0):
        self._pipeline_hyperparameter_ranges = pipeline_hyperparameter_ranges
        self._parameter_names_map = dict()
        self._search_space_names = []
        self._search_space_ranges = []
        self.random_seed = random_seed
        if not isinstance(pipeline_hyperparameter_ranges, dict):
            raise ValueError(
                "pipeline_hyperparameter_ranges must be a dict but is of type {}".format(
                    type(pipeline_hyperparameter_ranges)
                )
            )
        self._component_names = list(pipeline_hyperparameter_ranges.keys())
        for component_name, component_ranges in pipeline_hyperparameter_ranges.items():
            if not isinstance(component_ranges, dict):
                raise ValueError(
                    "pipeline_hyperparameter_ranges has invalid entry for {}: {}".format(
                        component_name, component_ranges
                    )
                )
            for parameter_name, parameter_range in component_ranges.items():
                if parameter_range is None:
                    raise ValueError(
                        "pipeline_hyperparameter_ranges has invalid dimensions for "
                        + "{} parameter {}: None.".format(
                            component_name, parameter_name
                        )
                    )
                if not isinstance(
                    parameter_range, (Real, Integer, Categorical, list, tuple)
                ):
                    continue
                flat_parameter_name = "{}: {}".format(component_name, parameter_name)
                self._parameter_names_map[flat_parameter_name] = (
                    component_name,
                    parameter_name,
                )
                self._search_space_names.append(flat_parameter_name)
                self._search_space_ranges.append(parameter_range)

    def _convert_to_flat_parameters(self, pipeline_parameters):
        """Convert from pipeline parameters to a flat list of values."""
        flat_parameter_values = []
        for flat_parameter_name in self._search_space_names:
            component_name, parameter_name = self._parameter_names_map[
                flat_parameter_name
            ]
            if (
                component_name not in pipeline_parameters
                or parameter_name not in pipeline_parameters[component_name]
            ):
                raise TypeError(
                    'Pipeline parameters missing required field "{}" for component "{}"'.format(
                        parameter_name, component_name
                    )
                )
            flat_parameter_values.append(
                pipeline_parameters[component_name][parameter_name]
            )
        return flat_parameter_values

    def _convert_to_pipeline_parameters(self, flat_parameters):
        """Convert from a flat list of values to a dict of pipeline parameters."""
        pipeline_parameters = {
            component_name: dict() for component_name in self._component_names
        }
        for flat_parameter_name, parameter_value in zip(
            self._search_space_names, flat_parameters
        ):
            component_name, parameter_name = self._parameter_names_map[
                flat_parameter_name
            ]
            pipeline_parameters[component_name][parameter_name] = parameter_value
        return pipeline_parameters

    @abstractmethod
    def add(self, pipeline_parameters, score):
        """Register a set of hyperparameters with the score obtained from training a pipeline with those hyperparameters.

        Args:
            pipeline_parameters (dict): a dict of the parameters used to evaluate a pipeline
            score (float): the score obtained by evaluating the pipeline with the provided parameters

        Returns:
            None
        """

    @abstractmethod
    def propose(self):
        """Returns a suggested set of parameters to train and score a pipeline with, based off the search space dimensions and prior samples.

        Returns:
            dict: Proposed pipeline parameters
        """

    def is_search_space_exhausted(self):
        """Optional. If possible search space for tuner is finite, this method indicates whether or not all possible parameters have been scored.

        Returns:
            bool: Returns true if all possible parameters in a search space has been scored.
        """
        return False
