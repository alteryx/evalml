import copy

from evalml.model_family import handle_model_family
from evalml.pipelines import (
    CatBoostClassificationPipeline,
    CatBoostRegressionPipeline,
    LinearRegressionPipeline,
    LogisticRegressionPipeline,
    PipelineBase,
    RFClassificationPipeline,
    RFRegressionPipeline,
    XGBoostPipeline
)
from evalml.problem_types import handle_problem_types
from evalml.utils import classproperty, import_or_raise

_DEFAULT_PIPELINES = [RFClassificationPipeline,
                      XGBoostPipeline,
                      LogisticRegressionPipeline,
                      LinearRegressionPipeline,
                      RFRegressionPipeline,
                      CatBoostClassificationPipeline,
                      CatBoostRegressionPipeline]


class Registry:
    """Registry containing default pipelines and additional pipelines for automl"""

    other_pipelines = []
    """list(PipelineBase): additional pipelines registered in Registry"""

    @classproperty
    def default_pipelines(cls):
        """Returns a list of default pipelines provided by evalml

        Returns:
            list[PipelineBase]: a list of pipeline classes
        """
        pipelines = copy.copy(_DEFAULT_PIPELINES)
        try:
            import_or_raise("xgboost", error_msg="XGBoost not installed.")
        except ImportError:
            pipelines.remove(XGBoostPipeline)
        try:
            import_or_raise("catboost", error_msg="Catboost not installed.")
        except ImportError:
            pipelines.remove(CatBoostClassificationPipeline)
            pipelines.remove(CatBoostRegressionPipeline)
        return pipelines

    @classmethod
    def all_pipelines(cls):
        """Returns a list of all pipelines currently registered

        Returns:
            list[PipelineBase]: a list of pipeline classes
        """
        return cls.default_pipelines + cls.other_pipelines

    @classmethod
    def register(cls, pipeline_class):
        """Registers a given pipeline class. Note: pipeline must inherit from `PipelineBase`

        Arguments:
            pipeline_class (PipelineBase): pipeline class which inherits from `PipelineBase`
        """
        if issubclass(pipeline_class, PipelineBase):
            cls.other_pipelines.append(pipeline_class)
        else:
            raise TypeError("Provided pipeline {} is not a subclass of `PipelineBase`".format(pipeline_class))

    @classmethod
    def register_from_components(cls, component_graph, supported_problem_types, name):
        """Registers a given pipeline with a component graph, supported problem types and pipeline class name

        Arguments:
            component_graph (list(str/ComponentBase)): list of components that make up this pipeline
            supported_problem_types (list(str/ProblemTypes)): list of problem types this pipeline supports
            name (str): name of this pipeline class
        """
        base_class = PipelineBase
        class_dict = {
            'component_graph': component_graph,
            'supported_problem_types': supported_problem_types
        }
        temp_pipeline = type(name, (base_class,), class_dict)
        cls.register(temp_pipeline)

    @classmethod
    def reset_pipelines(cls):
        """Reset non-default additional pipelines"""
        cls.other_pipelines = []

    @classmethod
    def find_pipeline(cls, name):
        """Finds and returns pipeline given pipeline class name

        Arguments:
            name (str): pipeline class name
        """
        for pipeline in cls.all_pipelines():
            if pipeline.name == name:
                return pipeline
        return None

    @classmethod
    def get_pipelines(cls, problem_type, model_families=None):
        """Returns the pipelines allowed for a particular problem type.

        Can also optionally filter by a list of model types.

        Arguments:

        Returns:
            list[PipelineBase]: a list of pipeline classes
        """
        if model_families is not None and not isinstance(model_families, list):
            raise TypeError("model_families parameter is not a list.")

        problem_pipelines = []

        if model_families:
            model_families = [handle_model_family(model_family) for model_family in model_families]

        problem_type = handle_problem_types(problem_type)
        for p in cls.all_pipelines():
            problem_types = [handle_problem_types(pt) for pt in p.supported_problem_types]
            if problem_type in problem_types:
                problem_pipelines.append(p)

        if model_families is None:
            return problem_pipelines

        all_model_families = cls.list_model_families(problem_type)
        for model_family in model_families:
            if model_family not in all_model_families:
                raise RuntimeError("Unrecognized model type for problem type %s: %s" % (problem_type, model_family))

        pipelines = []

        for p in problem_pipelines:
            if p.model_family in model_families:
                pipelines.append(p)

        return pipelines

    @classmethod
    def list_model_families(cls, problem_type):
        """List model type for a particular problem type

        Args:
            problem_types (ProblemTypes or str): binary, multiclass, or regression

        Returns:
            list[ModelFamily]: a list of model families
        """

        problem_pipelines = []
        problem_type = handle_problem_types(problem_type)
        for p in cls.all_pipelines():
            problem_types = [handle_problem_types(pt) for pt in p.supported_problem_types]
            if problem_type in problem_types:
                problem_pipelines.append(p)

        return list(set([p.model_family for p in problem_pipelines]))
