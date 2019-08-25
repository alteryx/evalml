from sklearn.model_selection import KFold

from .auto_base import AutoBase

from evalml.objectives import standard_metrics


class AutoRegressor(AutoBase):
    """Automatic pipeline search for regression problems"""

    def __init__(self,
                 objective=None,
                 max_pipelines=5,
                 max_time=None,
                 model_types=None,
                 cv=None,
                 tuner=None,
                 detect_label_leakage=True,
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

            random_state (int): the random_state

            verbose (boolean): If True, turn verbosity on. Defaults to True

        """
        if objective is None:
            objective = "R2"

        default_objectives = [
            standard_metrics.R2(),
        ]

        if cv is None:
            cv = KFold(n_splits=3, random_state=random_state)

        problem_type = "regression"

        super().__init__(
            tuner=tuner,
            objective=objective,
            cv=cv,
            max_pipelines=max_pipelines,
            max_time=max_time,
            model_types=model_types,
            problem_type=problem_type,
            default_objectives=default_objectives,
            detect_label_leakage=detect_label_leakage,
            random_state=random_state,
            verbose=verbose,
        )
