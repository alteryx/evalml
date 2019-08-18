from .auto_base import AutoBase


class AutoRegressor(AutoBase):
    def __init__(self,
                 objective=None,
                 max_pipelines=5,
                 max_time=60 * 5,
                 model_types=None,
                 cv=None,
                 random_state=0,
                 tuner=None,
                 verbose=True):
        """Automated regressor pipeline search

        Arguments:
            objective (Object): the objective to optimize
            max_pipelines (int): maximum number of models to search
            max_time (int): maximum time in seconds to search for models
            model_types (list): The model types to search. By default searches over
                model_types
            cv (): cross validation method to use. By default StratifiedKFold
            tuner (): the tuner class to use. Defaults to scikit-optimize tuner
            random_state ():
            verbose (boolean): If True, turn verbosity on. Defaults to True

        """
        raise NotImplementedError("")

    def fit(self, X, y, feature_types=None):
        """Find best classifier

        Arguments:
            X (pd.DataFrame): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

            feature_types (list, optional): list of feature types. either numeric of categorical.
                categorical features will automatically be encoded

        Returns:

            self
        """
        raise NotImplementedError("")

    def _select_pipeline(self):
        raise NotImplementedError("")

    def _propose_parameters(self, pipeline_class):
        raise NotImplementedError("")

    def _add_result(self, trained_pipeline, parameters, scores, training_time):
        raise NotImplementedError("")

    def _save_pipeline(self, pipeline_name, parameters, trained_pipeline):
        raise NotImplementedError("")

    def _get_pipeline(self, pipeline_name, parameters):
        raise NotImplementedError("")

    @property
    def rankings(self):
        """Returns the rankings of the models searched"""
        raise NotImplementedError("")

    @property
    def best_pipeline(self):
        """Returns the best model found"""
        raise NotImplementedError("")
