"""Component that reduces the number of features by using Linear Discriminant Analysis."""
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types, is_all_numeric


class LinearDiscriminantAnalysis(Transformer):
    """Reduces the number of features by using Linear Discriminant Analysis.

    Args:
        n_components (int): The number of features to maintain after computation. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Linear Discriminant Analysis Transformer"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, n_components=None, random_seed=0, **kwargs):
        if n_components and n_components < 1:
            raise ValueError(
                "Invalid number of compponents for Linear Discriminant Analysis",
            )
        parameters = {"n_components": n_components}
        parameters.update(kwargs)
        lda = SkLDA(n_components=n_components, **kwargs)
        super().__init__(
            parameters=parameters,
            component_obj=lda,
            random_seed=random_seed,
        )

    def fit(self, X, y):
        """Fits the LDA component.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If input data is not all numeric.
        """
        X = infer_feature_types(X)
        if not is_all_numeric(X):
            raise ValueError("LDA input must be all numeric")
        y = infer_feature_types(y)
        n_features = X.shape[1]
        n_classes = y.nunique()
        n_components = self.parameters["n_components"]
        if n_components is not None and n_components > min(n_classes, n_features):
            raise ValueError(f"n_components value {n_components} is too large")

        self._component_obj.fit(X, y)
        return self

    def transform(self, X, y=None):
        """Transform data using the fitted LDA component.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.

        Raises:
            ValueError: If input data is not all numeric.
        """
        X_ww = infer_feature_types(X)
        if not is_all_numeric(X_ww):
            raise ValueError("LDA input must be all numeric")
        X_t = self._component_obj.transform(X)
        X_t = pd.DataFrame(
            X_t,
            index=X_ww.index,
            columns=[f"component_{i}" for i in range(X_t.shape[1])],
        )
        X_t.ww.init()
        return X_t

    def fit_transform(self, X, y=None):
        """Fit and transform data using the LDA component.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.

        Raises:
            ValueError: If input data is not all numeric.
        """
        X_ww = infer_feature_types(X)
        if not is_all_numeric(X_ww):
            raise ValueError("LDA input must be all numeric")
        y = infer_feature_types(y)

        X_t = self._component_obj.fit_transform(X, y)
        X_t = pd.DataFrame(
            X_t,
            index=X_ww.index,
            columns=[f"component_{i}" for i in range(X_t.shape[1])],
        )
        X_t.ww.init()
        return X_t
