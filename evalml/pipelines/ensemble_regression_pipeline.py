"""Pipeline subclass for all binary classification pipelines."""
from multiprocessing.sharedctypes import Value

from matplotlib.cbook import Stack
from evalml.problem_types.utils import is_binary, is_multiclass
from .binary_classification_pipeline_mixin import (
    BinaryClassificationPipelineMixin,
)

from evalml.objectives import get_objective
from evalml.pipelines.regression_pipeline import RegressionPipeline
from evalml.utils import infer_feature_types
from evalml.pipelines.components import LabelEncoder, StackedEnsembleRegressor
from evalml.automl.utils import make_data_splitter
from evalml.problem_types import ProblemTypes
import woodwork as ww
import numpy as np
import pandas as pd

class EnsembleRegressionPipeline(RegressionPipeline):
    """Pipeline subclass for all binary classification pipelines.

    Args:
        component_graph (ComponentGraph, list, dict): ComponentGraph instance, list of components in order, or dictionary of components.
            Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Regressor] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Regressor"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary or None implies using all default values for component parameters. Defaults to None.
        custom_name (str): Custom name for the pipeline. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Example:
        >>> pipeline = BinaryClassificationPipeline(component_graph=["Simple Imputer", "Logistic Regression Regressor"],
        ...                                         parameters={"Logistic Regression Regressor": {"penalty": "elasticnet",
        ...                                                                                        "solver": "liblinear"}},
        ...                                         custom_name="My Binary Pipeline")
        ...
        >>> assert pipeline.custom_name == "My Binary Pipeline"
        >>> assert pipeline.component_graph.component_dict.keys() == {'Simple Imputer', 'Logistic Regression Regressor'}

        The pipeline parameters will be chosen from the default parameters for every component, unless specific parameters
        were passed in as they were above.

        >>> assert pipeline.parameters == {
        ...     'Simple Imputer': {'impute_strategy': 'most_frequent', 'fill_value': None},
        ...     'Logistic Regression Regressor': {'penalty': 'elasticnet',
        ...                                        'C': 1.0,
        ...                                        'n_jobs': -1,
        ...                                        'multi_class': 'auto',
        ...                                        'solver': 'liblinear'}}
    """
    name = "V3 Stacked Ensemble Regressor"
    def __init__(
        self,
        input_pipelines,
        component_graph=None,
        parameters=None,
        custom_name=None,
        random_seed=0,
    ):
        self.input_pipelines = input_pipelines

        if component_graph is None:
            component_graph = {
                "Stacked Ensembler": ["Stacked Ensemble Regressor", "X", "y"]
            }
        super().__init__(
            component_graph,
            custom_name=custom_name,
            parameters=parameters,
            random_seed=random_seed,
        )
        self._is_stacked_ensemble = True

    def predict(self, X, objective=None):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features]
            objective (Object or string): The objective to use to make predictions.

        Returns:
            pd.Series: Estimated labels
        """
        X = infer_feature_types(X)
        metalearner_X = self.transform(X)
        predictions = self.component_graph.predict(metalearner_X)
        predictions = self.inverse_transform(predictions)
        predictions.name = self.input_target_name
        return infer_feature_types(predictions)

    @property
    def _all_input_pipelines_fitted(self):
        for pipeline in self.input_pipelines:
            if not pipeline._is_fitted:
                return False
        return True

    def _fit_input_pipelines(self, X, y, force_retrain=False):
        fitted_pipelines = []
        for pipeline in self.input_pipelines:
            if pipeline._is_fitted and not force_retrain:
                fitted_pipelines.append(pipeline)
            else:
                if force_retrain:
                    new_pl = pipeline.clone()
                else:
                    new_pl = pipeline
                fitted_pipelines.append(new_pl.fit(X, y))
        self.input_pipelines = fitted_pipelines
        
    def fit(self, X, y, data_splitter=None, force_retrain=False):
        """Build a classification model. For string and categorical targets, classes are sorted by sorted(set(y)) and then are mapped to values between 0 and n_classes-1.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, np.ndarray): The target training labels of length [n_samples]

        Returns:
            self

        Raises:
            ValueError: If the number of unique classes in y are not appropriate for the type of pipeline.
        """

        X = infer_feature_types(X)
        y = infer_feature_types(y)

        if "numeric" not in y.ww.semantic_tags:
            raise ValueError(f"Regression pipeline can only handle numeric target data")

        if not self._all_input_pipelines_fitted or force_retrain is True:
            self._fit_input_pipelines(X, y, force_retrain=True)

        if data_splitter is None:
            data_splitter = make_data_splitter(X, y, problem_type=ProblemTypes.REGRESSION)

        splits = data_splitter.split(X, y)

        metalearner_X = []
        metalearner_y = []

        pred_pls = []
        for pipeline in self.input_pipelines:
            pred_pls.append(pipeline.clone())

        # Split off pipelines for CV
        for i, (train, valid) in enumerate(splits):
            fold_X = {}
            X_train, X_valid = X.ww.iloc[train], X.ww.iloc[valid]
            y_train, y_valid = y.ww.iloc[train], y.ww.iloc[valid]

            for pipeline in pred_pls:
                pipeline.fit(X_train, y_train)
                pl_preds = pipeline.predict(X_valid)
                fold_X[pipeline.name] = pl_preds
            
            metalearner_X.append(pd.DataFrame(fold_X))
            metalearner_y.append(y_valid)

        metalearner_X = pd.concat(metalearner_X)
        metalearner_y = pd.concat(metalearner_y)

        self.component_graph.fit(metalearner_X, metalearner_y)        
        return self

    def transform(self, X, y=None):
        if not self._all_input_pipelines_fitted:
            raise ValueError("Input pipelines needs to be fitted before transform")
        input_pipeline_preds = {}
        for pipeline in self.input_pipelines:
            pl_preds = pipeline.predict(X)
            input_pipeline_preds[pipeline.name] = pl_preds
        
        return pd.DataFrame(input_pipeline_preds)

    def clone(self):
        """Constructs a new pipeline with the same components, parameters, and random seed.

        Returns:
            A new instance of this pipeline with identical components, parameters, and random seed.
        """
        clone = self.__class__(
            input_pipelines=self.input_pipelines,
            component_graph=self.component_graph,
            parameters=self.parameters,
            custom_name=self.custom_name,
            random_seed=self.random_seed,
        )
        if is_binary(self.problem_type):
            clone.threshold = self.threshold
        return clone

    def new(self, parameters, random_seed=0):
        """Constructs a new instance of the pipeline with the same component graph but with a different set of parameters. Not to be confused with python's __new__ method.

        Args:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary or None implies using all default values for component parameters. Defaults to None.
            random_seed (int): Seed for the random number generator. Defaults to 0.

        Returns:
            A new instance of this pipeline with identical components.
        """
        return self.__class__(
            self.input_pipelines,
            self.component_graph,
            parameters=parameters,
            custom_name=self.custom_name,
            random_seed=random_seed,
        )