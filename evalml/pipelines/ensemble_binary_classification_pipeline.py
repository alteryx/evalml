"""Pipeline subclass for all binary classification pipelines."""
from multiprocessing.sharedctypes import Value

from matplotlib.cbook import Stack
from evalml.problem_types.utils import is_binary, is_multiclass
from .binary_classification_pipeline_mixin import (
    BinaryClassificationPipelineMixin,
)

from evalml.objectives import get_objective
from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline
from evalml.utils import infer_feature_types
from evalml.pipelines.components import LabelEncoder, StackedEnsembleClassifier
from evalml.automl.utils import make_data_splitter
from evalml.problem_types import ProblemTypes
import woodwork as ww
import numpy as np
import pandas as pd

class EnsembleBinaryClassificationPipeline(BinaryClassificationPipeline):
    """Pipeline subclass for all binary classification pipelines.

    Args:
        component_graph (ComponentGraph, list, dict): ComponentGraph instance, list of components in order, or dictionary of components.
            Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary or None implies using all default values for component parameters. Defaults to None.
        custom_name (str): Custom name for the pipeline. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Example:
        >>> pipeline = BinaryClassificationPipeline(component_graph=["Simple Imputer", "Logistic Regression Classifier"],
        ...                                         parameters={"Logistic Regression Classifier": {"penalty": "elasticnet",
        ...                                                                                        "solver": "liblinear"}},
        ...                                         custom_name="My Binary Pipeline")
        ...
        >>> assert pipeline.custom_name == "My Binary Pipeline"
        >>> assert pipeline.component_graph.component_dict.keys() == {'Simple Imputer', 'Logistic Regression Classifier'}

        The pipeline parameters will be chosen from the default parameters for every component, unless specific parameters
        were passed in as they were above.

        >>> assert pipeline.parameters == {
        ...     'Simple Imputer': {'impute_strategy': 'most_frequent', 'fill_value': None},
        ...     'Logistic Regression Classifier': {'penalty': 'elasticnet',
        ...                                        'C': 1.0,
        ...                                        'n_jobs': -1,
        ...                                        'multi_class': 'auto',
        ...                                        'solver': 'liblinear'}}
    """
    name = "V3 Stacked Ensemble Classifier"
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
                "Label Encoder": ["Label Encoder", "X", "y"],
                "Stacked Ensembler": ["Stacked Ensemble Classifier", "X", "Label Encoder.y"]
            }
        super().__init__(
            component_graph,
            custom_name=custom_name,
            parameters=parameters,
            random_seed=random_seed,
        )
        self._is_stacked_ensemble = True

    def _predict(self, X, objective=None):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features]
            objective (Object or string): The objective to use to make predictions.

        Returns:
            pd.Series: Estimated labels
        """
        if objective is not None:
            objective = get_objective(objective, return_instance=True)
            if not objective.is_defined_for_problem_type(self.problem_type):
                raise ValueError(
                    "You can only use a binary classification objective to make predictions for a binary classification pipeline."
                )

        metalearner_X = self.transform(X)
        if self.threshold is None:
            return self.component_graph.predict(metalearner_X)
        ypred_proba = self.predict_proba(metalearner_X)
        predictions = self._predict_with_objective(X, ypred_proba, objective)        
        return infer_feature_types(predictions)

    def predict_proba(self, X, X_train=None, y_train=None):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features]
            objective (Object or string): The objective to use to make predictions.

        Returns:
            pd.Series: Estimated labels
        """

        metalearner_X = self.transform(X)
        return super().predict_proba(metalearner_X)

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

        if is_binary(self.problem_type) and y.nunique() != 2:
            raise ValueError("Binary pipelines require y to have 2 unique classes!")
        elif is_multiclass(self.problem_type) and y.nunique() in [1, 2]:
            raise ValueError(
                "Multiclass pipelines require y to have 3 or more unique classes!"
            )

        if not self._all_input_pipelines_fitted or force_retrain is True:
            self._fit_input_pipelines(X, y, force_retrain=True)

        if data_splitter is None:
            data_splitter = make_data_splitter(X, y, problem_type=ProblemTypes.BINARY)

        splits = data_splitter.split(X, y)

        metalearner_X = []
        metalearner_y = []

        pred_pls = []
        for pipeline in self.input_pipelines:
            pred_pls.append(pipeline.clone())

        # Split off pipelines for CV 
        for i, (train, valid) in enumerate(splits):
            fold_X = []
            X_train, X_valid = X.ww.iloc[train], X.ww.iloc[valid]
            y_train, y_valid = y.ww.iloc[train], y.ww.iloc[valid]

            for pipeline in pred_pls:
                pipeline.fit(X_train, y_train)
                pl_preds = pipeline.predict_proba(X_valid)
                if isinstance(pl_preds, pd.DataFrame):
                    new_columns = {}
                    for i, column in enumerate(pl_preds.columns):
                        new_columns[column] = i
                    pl_preds.ww.rename(new_columns, inplace=True)
                    if len(pl_preds.columns) == 2:
                        # If it is a binary problem, drop the first column since both columns are colinear
                        pl_preds = pl_preds.ww.drop(pl_preds.columns[0])
                    pl_preds = pl_preds.ww.rename(
                        {
                            col: f"Col {str(col)} {pipeline.name}.x"
                            for col in pl_preds.columns
                        }
                    )
                fold_X.append(pl_preds)
            
            metalearner_X.append(ww.concat_columns(fold_X))
            metalearner_y.append(y_valid)

        metalearner_X = pd.concat(metalearner_X)
        metalearner_y = pd.concat(metalearner_y)

        self.component_graph.fit(metalearner_X, metalearner_y)        

        self._classes_ = list(ww.init_series(np.unique(metalearner_y)))
        return self

    def transform(self, X, y=None):
        if not self._all_input_pipelines_fitted:
            raise ValueError("Input pipelines needs to be fitted before transform")
        input_pipeline_preds = []
        for pipeline in self.input_pipelines:
            pl_preds = pipeline.predict_proba(X)
            if isinstance(pl_preds, pd.DataFrame):
                new_columns = {}
                for i, column in enumerate(pl_preds.columns):
                    new_columns[column] = i
                pl_preds.ww.rename(new_columns, inplace=True)
                if len(pl_preds.columns) == 2:
                    # If it is a binary problem, drop the first column since both columns are colinear
                    pl_preds = pl_preds.ww.drop(pl_preds.columns[0])
                pl_preds = pl_preds.ww.rename(
                    {
                        col: f"Col {str(col)} {pipeline.name}.x"
                        for col in pl_preds.columns
                    }
                )
            input_pipeline_preds.append(pl_preds)
        
        return ww.concat_columns(input_pipeline_preds)

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