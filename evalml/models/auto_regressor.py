from sklearn.model_selection import KFold

from .auto_base import AutoBase

from evalml.problem_types import ProblemTypes


class AutoRegressor(AutoBase):
    """Automatic pipeline search for regression problems"""

    def __init__(self,
                 objective=None,
                 max_pipelines=None,
                 max_time=None,
                 model_types=None,
                 cv=None,
                 tuner=None,
                 detect_label_leakage=True,
                 start_iteration_callback=None,
                 add_result_callback=None,
                 additional_objectives=None,
                 id_cols_threshold=1.0,
                 null_threshold=0.95,
                 check_outliers=False,
                 random_state=0,
                 verbose=True):
        """Automated regressors pipeline search

        Arguments:
            objective (Object): the objective to optimize

            max_pipelines (int): maximum number of pipelines to search

            max_time (int): maximum time in seconds to search for pipelines.
                won't start new pipeline search after this duration has elapsed

            model_types (list): The model types to search. By default searches over all
                model_types. Run evalml.list_model_types("regression") to see options.

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

            null_threshold(float): Float in range [0,1] that represents what percentage of a feature needs to be
                null values for the feature to be considered "highly-null". Default is 0.95.

            check_outliers(bool): If True, checks if there are any outliers in data. Default is False.

            id_cols_threshold(float): Float in range [0,1] that represents the probability threshold for
            a feature to be considered an ID column. Default is 1.0.

            random_state (int): the random_state

            verbose (boolean): If True, turn verbosity on. Defaults to True

        """
        if objective is None:
            objective = "R2"

        problem_type = ProblemTypes.REGRESSION

        if cv is None:
            cv = KFold(n_splits=3, random_state=random_state)

        super().__init__(
            tuner=tuner,
            objective=objective,
            cv=cv,
            max_pipelines=max_pipelines,
            max_time=max_time,
            model_types=model_types,
            problem_type=problem_type,
            detect_label_leakage=detect_label_leakage,
            start_iteration_callback=start_iteration_callback,
            add_result_callback=add_result_callback,
            id_cols_threshold=id_cols_threshold,
            additional_objectives=additional_objectives,
            null_threshold=null_threshold,
            check_outliers=check_outliers,
            random_state=random_state,
            verbose=verbose
        )
