from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    SEED_BOUNDS,
    get_logger,
    get_random_seed,
    import_or_raise
)
from evalml.utils.gen_utils import make_h2o_ready

logger = get_logger(__file__)


class GAMClassifier(Estimator):
    """GAM Classifier"""
    name = "GAM Classifier"
    hyperparameter_ranges = {
        "solver": ["IRLSM", "L_BFGS"],
        "alpha": Real(0.000001, 1),
        "lambda": Real(0.000001, 1)
    }
    model_family = ModelFamily.LINEAR_MODEL
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]

    SEED_MIN = 0
    SEED_MAX = SEED_BOUNDS.max_bound

    def __init__(self, family='AUTO', solver="AUTO", stopping_metric="logloss", keep_cross_validation_models=False, random_state=0, **kwargs):
        random_seed = get_random_seed(random_state, self.SEED_MIN, self.SEED_MAX)

        self._parameters = {"family": family,
                            "solver": solver,
                            "stopping_metric": stopping_metric,
                            "keep_cross_validation_models": keep_cross_validation_models,
                            "seed": random_seed}
        self._parameters.update(kwargs)

        h2o_error_msg = "H2O is not installed. please install using `pip install h2o`."
        self.h2o = import_or_raise("h2o", error_msg=h2o_error_msg)
        self.h2o.init()

        self.h2o_model_init = self.h2o.estimators.gam.H2OGeneralizedAdditiveEstimator

        super().__init__(parameters=self._parameters,
                         component_obj=None,
                         random_state=random_seed)

    def _update_params(self, X, y):
        X_cols = [str(col_) for col_ in list(X.columns)]
        new_params = {'gam_columns': X_cols,
                      "lambda_search": True}
        if y.nunique() == 3:
            new_params.update({"family": "multinomial",
                               "link": "Family_Default"})
        elif y.nunique() > 3:
            new_params.update({"family": "ordinal",
                               "solver": "GRADIENT_DESCENT_LH",
                               "link": "Family_Default",
                               "lambda_search": False})
        else:
            new_params.update({"family": "binomial",
                               "link": "Logit"})
        return new_params

    def _retry_fit(self, X, y, training_frame, error=None):
        error = str(error)
        array_exception = "ArrayIndexOutOfBoundsException"
        if error.find(array_exception) != -1:
            index_start = error.find(array_exception) + 38
            index_end = error.find(" out")
            new_col = int(error[index_start:index_end]) if index_end != 1 else None
            logger.info(f"Encountered ArrayIndexOutOfBoundsException, limiting number of gam_columns to {new_col}")
        else:
            raise error
        self._parameters['gam_columns'] = self._parameters['gam_columns'][:new_col]
        self.h2o_model = self.h2o_model_init(**self._parameters)
        self.h2o_model.train(x=list(X.columns), y=y.name, training_frame=training_frame)

    def fit(self, X, y=None, retrying=False):
        if not retrying:
            X, y, training_frame = make_h2o_ready(X, y, supported_problem_types=GAMClassifier.supported_problem_types)
            new_params = self._update_params(X, y)
            self._parameters.update(new_params)
        self.h2o_model = self.h2o_model_init(**self._parameters)
        try:
            self.h2o_model.train(x=list(X.columns), y=y.name, training_frame=training_frame)
        except OSError as e:
            self._retry_fit(X, y, training_frame, e)
        return self.h2o_model

    def predict(self, X):
        X = make_h2o_ready(X, supported_problem_types=GAMClassifier.supported_problem_types)
        X = self.h2o.H2OFrame(X)
        predictions = self.h2o_model.predict(X)
        predictions = predictions.as_data_frame(use_pandas=True).iloc[:, 0]
        return predictions

    @property
    def feature_importance(self):
        return self.h2o_model.varimp()
