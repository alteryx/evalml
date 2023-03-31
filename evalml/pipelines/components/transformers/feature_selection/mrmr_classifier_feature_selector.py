"""Component that selects top features based on importance weights."""
import pandas as pd
from mrmr import mrmr_classif
from skopt.space import Real

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class MRMRClassifierFeatureSelector(Transformer):
    """Selects top features based on importance weights.

    Args:
        parameters (dict): Dictionary of parameters for the component. Defaults to None.
        component_obj (obj): Third-party objects useful in component implementation. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "MRMR Classifier Feature Selector"
    hyperparameter_ranges = {
        "percent_features": Real(0.01, 1),
    }
    """{
        "percent_features": Real(0.01, 1),
        "threshold": ["mean", "median"],
    }"""

    def __init__(
        self,
        number_features=None,
        percent_features=0.5,
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "number_features": number_features,
            "percent_features": percent_features,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def get_names(self):
        """Get names of selected features.

        Returns:
            list[str]: List of the names of features selected.
        """
        return self.selected_col_names

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
        features = pd.DataFrame(X_ww, columns=self.selected_col_names, index=X_ww.index)
        features.ww.init(
            schema=X_ww.ww.schema.get_subset_schema(self.selected_col_names),
        )
        return features

    def fit(self, X, y):
        """Fits component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features]
            y (pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self

        Raises:
            MethodPropertyNotFoundError: If component does not have a fit method or a component_obj that implements fit.
        """
        X_ww = infer_feature_types(X)
        self.input_feature_names = list(X_ww.columns.values)

        percent_num_features_from_X = int(
            len(self.input_feature_names) * self.parameters["percent_features"],
        )
        number_features = (
            max(self.parameters["number_features"], percent_num_features_from_X)
            if self.parameters["number_features"]
            else percent_num_features_from_X
        )
        self.selected_col_names = mrmr_classif(
            X=X,
            y=y,
            K=number_features,
            show_progress=False,
        )

        return self

    def fit_transform(self, X, y):
        """Fit and transform data using the feature selector.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.
        """
        return self.fit(X, y).transform(X, y)

    def _handle_partial_dependence_fast_mode(
        self,
        pipeline_parameters,
        X=None,
        target=None,
    ):
        """Updates pipeline parameters to not drop any features based off of feature importance.

            This is needed, because fast mode refits cloned pipelines on single columns,
            so categorical columns that have one-hot encoding applied may lose some
            of their encoded columns with the default parameters. Therefore, we update the
            parameters here to not drop any columns, and use the original
            pipeline to determine if that feature gets dropped or not.

        Args:
            pipeline_parameters (dict): Pipeline parameters that will be used to create the pipelines
                used in partial dependence fast mode.
            X (pd.DataFrame, optional): Holdout data being used for partial dependence calculations.
            target (str, optional): The target whose values we are trying to predict.

        Return:
            pipeline_parameters (dict): Pipeline parameters updated to allow the FeatureSelector component
                to not drop any features.
        """
        # Raise the percent of features we want to keep to not lose any
        pipeline_parameters[self.name]["percent_features"] = 1.0
        # Lower the threshold for feature importance above which we keep features
        pipeline_parameters[self.name]["threshold"] = 0.0

        return pipeline_parameters
