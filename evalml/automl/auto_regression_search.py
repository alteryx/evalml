from sklearn.model_selection import KFold

from .auto_base import AutoBase
from .pipeline_template import PipelineTemplate

# from evalml.pipelines import LinearRegressionPipeline, RFRegressionPipeline
from evalml.pipelines.components import (
    LinearRegressor,
    OneHotEncoder,
    RandomForestRegressor,
    RFRegressorSelectFromModel,
    SimpleImputer,
    StandardScaler
)
from evalml.problem_types import ProblemTypes


class AutoRegressionSearch(AutoBase):
    """Automatic pipeline search for regression problems"""

    def __init__(self,
                 objective=None,
                 max_pipelines=None,
                 max_time=None,
                 patience=None,
                 tolerance=None,
                 model_types=None,
                 cv=None,
                 tuner=None,
                 detect_label_leakage=True,
                 start_iteration_callback=None,
                 add_result_callback=None,
                 additional_objectives=None,
                 random_state=0,
                 verbose=True):
        """Automated regressors pipeline search

        Arguments:
            objective (Object): the objective to optimize

            max_pipelines (int): Maximum number of pipelines to search. If max_pipelines and
                max_time is not set, then max_pipelines will default to max_pipelines of 5.

            max_time (int, str): Maximum time to search for pipelines.
                This will not start a new pipeline search after the duration
                has elapsed. If it is an integer, then the time will be in seconds.
                For strings, time can be specified as seconds, minutes, or hours.

            model_types (list): The model types to search. By default searches over all
                model_types. Run evalml.list_model_types("regression") to see options.

            patience (int): Number of iterations without improvement to stop search early. Must be positive.
                If None, early stopping is disabled. Defaults to None.

            tolerance (float): Minimum percentage difference to qualify as score improvement for early stopping.
                Only applicable if patience is not None. Defaults to None.

            cv: cross validation method to use. By default StratifiedKFold

            tuner: the tuner class to use. Defaults to scikit-optimize tuner

            detect_label_leakage (bool): If True, check input features for label leakage and
                warn if found. Defaults to true.

            start_iteration_callback (callable): function called before each pipeline training iteration.
                Passed two parameters: pipeline_class, parameters.

            add_result_callback (callable): function called after each pipeline training iteration.
                Passed two parameters: results, trained_pipeline.

            additional_objectives (list): Custom set of objectives to score on.
                Will override default objectives for problem type if not empty.

            random_state (int): the random_state

            verbose (boolean): If True, turn verbosity on. Defaults to True

        """
        if objective is None:
            objective = "R2"

        problem_type = ProblemTypes.REGRESSION

        if cv is None:
            cv = KFold(n_splits=3, random_state=random_state)

        templates = self._generate_pipeline_templates(model_types)

        super().__init__(tuner=tuner,
                         objective=objective,
                         cv=cv,
                         max_pipelines=max_pipelines,
                         max_time=max_time,
                         patience=patience,
                         tolerance=tolerance,
                         model_types=model_types,
                         problem_type=problem_type,
                         detect_label_leakage=detect_label_leakage,
                         start_iteration_callback=start_iteration_callback,
                         add_result_callback=add_result_callback,
                         additional_objectives=additional_objectives,
                         random_state=random_state,
                         verbose=verbose,
                         templates=templates
                         )

    def _generate_pipeline_templates(self, model_types):
        rfr = [OneHotEncoder, SimpleImputer, RFRegressorSelectFromModel, RandomForestRegressor]
        lrp = [OneHotEncoder, SimpleImputer, StandardScaler, LinearRegressor]
        pipelines = [rfr, lrp]
        all_templates = []
        for pipeline in pipelines:
            template = PipelineTemplate(pipeline)
            all_templates.append(template)

        templates = all_templates
        if model_types:
            templates = [template for templates in all_templates if templates.model_type in model_types]

        return templates
