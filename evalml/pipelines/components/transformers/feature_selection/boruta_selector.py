"""Base class for Boruta Selectors"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from skopt.space import Integer

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components.transformers.feature_selection.feature_selector import (
    FeatureSelector,
)
from evalml.utils import infer_feature_types


class BorutaSelector(FeatureSelector):
    """Selects relevant features using Boruta

    Args:
        estimator (Estimator): The maximum number of features to select.
            If both percent_features and number_features are specified, take the greater number of features. Defaults to 0.5.
            Defaults to None.
        n_estimators (int or string): The number of estimators.
        perc (int): percentile used as our threshold for copmarison between shadow and real features
        alpha (float): Level at which the corrected p-values will get rejected in both
            correction steps.
        two_step (boolean): if False, will use Bonferroni correction only
        max_iter (int): maximum number of iterations to perfrom
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
        early_stopping (boolean): whether to use early stopping
        n_iter_no_change (int): Maximum number of itertaions to do without confriming a tenative feature
    """

    hyperparameter_ranges = {
        "perc": Integer(0, 100),
    }
    """{
        "percent_features": Real(0.01, 1),
        "threshold": ["mean", "median"],
    }"""

    def __init__(
        self,
        number_features=None,
        n_estimators=10,
        max_depth=None,
        perc=50,
        alpha=0.05,
        two_step=False,
        max_iter=100,
        n_jobs=-1,
        random_seed=0,
        use_weak=False,
        **kwargs,
    ):
        parameters = {
            "perc": perc,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_iter": max_iter,
            "alpha": alpha,
            "two_step": two_step,
            "n_jobs": n_jobs,
            "use_weak": use_weak,
        }
        parameters.update(kwargs)
        self.use_weak = use_weak
        estimator = self._get_estimator(
            random_seed=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
        )

        feature_selection = BorutaPy(
            estimator=estimator,
            n_estimators=n_estimators,
            perc=perc,
            alpha=alpha,
            two_step=two_step,
            max_iter=max_iter,
            random_state=random_seed,
            **kwargs,
        )
        super().__init__(
            parameters=parameters,
            component_obj=feature_selection,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features]
            y (pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self

        Raises:
            MethodPropertyNotFoundError: If component does not have a fit method or a component_obj that implements fit.
        """
        X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        try:
            self._component_obj.fit(X.values, y.values)
            return self
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Component requires a fit method or a component_obj that implements fit",
            )

    def transform(self, X, y=None):
        """Transforms input data by selecting features. If the component_obj does not have a transform method, will raise an MethodPropertyNotFoundError exception.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Target data. Ignored.

        Returns:
            pd.DataFrame: Transformed X

        Raises:
            MethodPropertyNotFoundError: If feature selector does not have a transform method or a component_obj that implements transform
        """
        X_ww = infer_feature_types(X)
        self.input_feature_names = list(X_ww.columns.values)

        X_t = self._component_obj.transform(X.values, weak=self.use_weak)

        selected_col_names = self.get_names()
        features = pd.DataFrame(X_t, columns=selected_col_names, index=X_ww.index)
        features.ww.init(schema=X_ww.ww.schema.get_subset_schema(selected_col_names))
        return features

    def get_names(self):
        """Get names of selected features.

        Returns:
            list[str]: List of the names of features selected.
        """
        selected_masks = self._component_obj.support_
        if self.use_weak:
            selected_masks = np.logical_or(self._component_obj.support_, self._component_obj.support_weak_)
        return [
            feature_name
            for (selected, feature_name) in zip(
                selected_masks,
                self.input_feature_names,
            )
            if selected
        ]

    @abstractmethod
    def _get_estimator(self, random_seed, n_estimators, max_depth, n_jobs):
        '''Return estimator with supplied parameters'''


class RFClassifierBorutaSelector(BorutaSelector):
    name = "Boruta Selector with RF Classifier"

    def _get_estimator(self, random_seed, n_estimators, max_depth, n_jobs):
        return SKRandomForestClassifier(
                random_state=random_seed,
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=n_jobs,
            )


class RFRegressorBorutaSelector(BorutaSelector):
    name = "Boruta Selector with RF Regressor"

    def _get_estimator(self, random_seed, n_estimators, max_depth, n_jobs):
        return SKRandomForestRegressor(
                random_state=random_seed,
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=n_jobs,
            )
