from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_understanding.permutation_importance import (
    calculate_permutation_importance,
    calculate_permutation_importance_one_column,
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    RegressionPipeline,
    Transformer,
)
from evalml.pipelines.components import (
    PCA,
    DateTimeFeaturizer,
    DFSTransformer,
    OneHotEncoder,
    TextFeaturizer,
)
from evalml.utils import infer_feature_types


class DoubleColumns(Transformer):
    """Custom transformer for testing permutation importance implementation.

    We don't have any transformers that create features that you can repeatedly "stack" on the previous output.
    That being said, I want to test that our implementation can handle that case in the event we add a transformer like
    that in the future.
    """

    name = "DoubleColumns"
    hyperparameter_ranges = {}

    def __init__(self, drop_old_columns=True, random_seed=0):
        self._provenance = {}
        self.drop_old_columns = drop_old_columns
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        self._provenance = {col: [f"{col}_doubled"] for col in X.columns}
        new_X = X.assign(**{f"{col}_doubled": 2 * X.loc[:, col] for col in X.columns})
        if self.drop_old_columns:
            new_X = new_X.drop(columns=X.columns)
        return infer_feature_types(new_X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def _get_feature_provenance(self):
        return self._provenance


class LinearPipelineWithDropCols(BinaryClassificationPipeline):
    component_graph = [
        "Drop Columns Transformer",
        OneHotEncoder,
        DateTimeFeaturizer,
        "Random Forest Classifier",
    ]


class LinearPipelineWithImputer(BinaryClassificationPipeline):
    component_graph = [
        "Imputer",
        OneHotEncoder,
        DateTimeFeaturizer,
        "Random Forest Classifier",
    ]


class LinearPipelineSameFeatureUsedByTwoComponents(BinaryClassificationPipeline):
    component_graph = [
        "Select Columns Transformer",
        "Imputer",
        DateTimeFeaturizer,
        OneHotEncoder,
        "Random Forest Classifier",
    ]


class LinearPipelineTwoEncoders(BinaryClassificationPipeline):
    component_graph = [
        "Select Columns Transformer",
        "Imputer",
        DateTimeFeaturizer,
        OneHotEncoder,
        OneHotEncoder,
        "Random Forest Classifier",
    ]


class LinearPipelineWithTextFeatures(BinaryClassificationPipeline):
    component_graph = [
        "Select Columns Transformer",
        "Imputer",
        TextFeaturizer,
        OneHotEncoder,
        "Random Forest Classifier",
    ]


class LinearPipelineWithTextFeaturizerNoTextFeatures(LinearPipelineWithTextFeatures):
    """Testing a pipeline with TextFeaturizer but no text features."""


class LinearPipelineWithDoubling(BinaryClassificationPipeline):
    component_graph = [
        "Select Columns Transformer",
        DoubleColumns,
        DoubleColumns,
        DoubleColumns,
        "Random Forest Classifier",
    ]


class LinearPipelineWithTargetEncoderAndOHE(BinaryClassificationPipeline):
    component_graph = [
        "Select Columns Transformer",
        "Imputer",
        DateTimeFeaturizer,
        OneHotEncoder,
        "Target Encoder",
        "Random Forest Classifier",
    ]


class LinearPipelineCreateFeatureThenDropIt(BinaryClassificationPipeline):
    component_graph = [
        "Select Columns Transformer",
        DoubleColumns,
        "Drop Columns Transformer",
        "Random Forest Classifier",
    ]


class DagTwoEncoders(BinaryClassificationPipeline):
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "SelectNumeric": ["Select Columns Transformer", "Imputer.x", "y"],
        "SelectCategorical1": ["Select Columns Transformer", "Imputer.x", "y"],
        "SelectCategorical2": ["Select Columns Transformer", "Imputer.x", "y"],
        "OHE_1": ["One Hot Encoder", "SelectCategorical1.x", "y"],
        "OHE_2": ["One Hot Encoder", "SelectCategorical2.x", "y"],
        "DT": ["DateTime Featurization Component", "SelectNumeric.x", "y"],
        "Estimator": ["Random Forest Classifier", "DT.x", "OHE_1.x", "OHE_2.x", "y"],
    }


class DagReuseFeatures(BinaryClassificationPipeline):
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "SelectDate": ["Select Columns Transformer", "Imputer.x", "y"],
        "SelectCategorical1": ["Select Columns Transformer", "Imputer.x", "y"],
        "SelectCategorical2": ["Select Columns Transformer", "Imputer.x", "y"],
        "OHE_1": ["One Hot Encoder", "SelectCategorical1.x", "y"],
        "OHE_2": ["One Hot Encoder", "SelectCategorical2.x", "y"],
        "DT": ["DateTime Featurization Component", "SelectDate.x", "y"],
        "OHE_3": ["One Hot Encoder", "DT.x", "y"],
        "Estimator": ["Random Forest Classifier", "OHE_3.x", "OHE_1.x", "OHE_2.x", "y"],
    }


class PipelineWithTargetTransformer(RegressionPipeline):
    component_graph = {
        "Log": ["Log Transformer", "X", "y"],
        "SelectNumeric": ["Select Columns Transformer", "X", "y"],
        "Estimator": ["Random Forest Regressor", "SelectNumeric.x", "Log.y"],
    }


test_cases = [
    (
        LinearPipelineWithDropCols,
        {
            "Drop Columns Transformer": {
                "columns": [
                    "country",
                    "customer_present",
                    "provider",
                    "region",
                    "expiration_date",
                    "lat",
                    "card_id",
                ]
            }
        },
    ),
    (
        LinearPipelineWithImputer,
        {
            "Select Columns Transformer": {
                "columns": ["provider", "lng", "datetime", "card_id", "country"]
            }
        },
    ),
    (
        LinearPipelineSameFeatureUsedByTwoComponents,
        {
            "Select Columns Transformer": {
                "columns": ["expiration_date", "datetime", "amount"]
            },
            "DateTime Featurization Component": {"encode_as_categories": True},
        },
    ),
    (
        LinearPipelineTwoEncoders,
        {
            "Select Columns Transformer": {
                "columns": [
                    "currency",
                    "expiration_date",
                    "region",
                    "country",
                    "amount",
                ]
            },
            "One Hot Encoder": {
                "features_to_encode": [
                    "currency",
                    "expiration_date",
                ]
            },
            "One Hot Encoder_2": {"features_to_encode": ["region", "country"]},
        },
    ),
    (
        LinearPipelineWithTextFeatures,
        {"Select Columns Transformer": {"columns": ["provider", "amount", "currency"]}},
    ),
    (
        LinearPipelineWithTextFeaturizerNoTextFeatures,
        {"Select Columns Transformer": {"columns": ["amount", "currency"]}},
    ),
    (
        LinearPipelineWithDoubling,
        {"Select Columns Transformer": {"columns": ["amount"]}},
    ),
    (
        LinearPipelineWithDoubling,
        {
            "Select Columns Transformer": {"columns": ["amount"]},
            "DoubleColumns": {"drop_old_columns": False},
        },
    ),
    (
        DagTwoEncoders,
        {
            "SelectNumeric": {
                "columns": [
                    "card_id",
                    "store_id",
                    "datetime",
                ]
            },
            "SelectCategorical1": {"columns": ["currency", "provider"]},
            "SelectCategorical2": {"columns": ["region", "country"]},
            "OHE_1": {"features_to_encode": ["currency", "provider"]},
            "OHE_2": {"features_to_encode": ["region", "country"]},
        },
    ),
    (
        DagReuseFeatures,
        {
            "SelectDate": {
                "columns": [
                    "datetime",
                ]
            },
            "SelectCategorical1": {"columns": ["currency", "provider"]},
            "SelectCategorical2": {"columns": ["region"]},
            "OHE_1": {"features_to_encode": ["currency", "provider"]},
            "OHE_2": {"features_to_encode": ["region"]},
            "DT": {"encode_as_categories": True},
        },
    ),
    (
        LinearPipelineWithTargetEncoderAndOHE,
        {
            "Select Columns Transformer": {
                "columns": ["currency", "provider", "region", "country"]
            },
            "One Hot Encoder": {"features_to_encode": ["currency", "provider"]},
            "Target Encoder": {"cols": ["region", "country"]},
        },
    ),
    (
        LinearPipelineCreateFeatureThenDropIt,
        {
            "Select Columns Transformer": {"columns": ["amount"]},
            "DoubleColumns": {"drop_old_columns": False},
            "Drop Columns Transformer": {"columns": ["amount_doubled"]},
        },
    ),
    (
        PipelineWithTargetTransformer,
        {"SelectNumeric": {"columns": ["card_id", "store_id", "lat", "lng"]}},
    ),
]


@pytest.mark.parametrize("pipeline_class, parameters", test_cases)
@patch(
    "evalml.pipelines.PipelineBase._supports_fast_permutation_importance",
    new_callable=PropertyMock,
)
def test_fast_permutation_importance_matches_slow_output(
    mock_supports_fast_importance,
    pipeline_class,
    parameters,
    has_minimal_dependencies,
    fraud_100,
):
    if (
        has_minimal_dependencies
        and pipeline_class == LinearPipelineWithTargetEncoderAndOHE
    ):
        pytest.skip(
            "Skipping test_fast_permutation_importance_matches_sklearn_output for target encoder cause "
            "dependency not installed."
        )
    X, y = fraud_100

    objective = "Log Loss Binary"
    if pipeline_class == LinearPipelineWithTextFeatures:
        X.ww.set_types(logical_types={"provider": "NaturalLanguage"})
    elif pipeline_class == PipelineWithTargetTransformer:
        y = X.ww.pop("amount")
        objective = "R2"

    mock_supports_fast_importance.return_value = True
    parameters["Random Forest Classifier"] = {"n_jobs": 1}
    pipeline = pipeline_class(pipeline_class.component_graph, parameters=parameters)
    pipeline.fit(X, y)
    fast_scores = calculate_permutation_importance(
        pipeline, X, y, objective=objective, random_seed=0
    )
    mock_supports_fast_importance.return_value = False
    slow_scores = calculate_permutation_importance(
        pipeline, X, y, objective=objective, random_seed=0
    )

    # The row order is not guaranteed to be equal in the case of ties
    # so converting to a dict after rounding to use dict equality over
    # assert_frame_equal
    fast_scores = dict(zip(fast_scores.feature, fast_scores.importance.round(5)))
    slow_scores = dict(zip(slow_scores.feature, slow_scores.importance.round(5)))
    assert slow_scores == fast_scores

    precomputed_features = pipeline.compute_estimator_features(X, y)
    # Run one column of each logical type
    for col in [
        "card_id",
        "datetime",
        "currency",
        "customer_present",
        "region",
        "amount",
    ]:
        if col == "amount" and pipeline_class == PipelineWithTargetTransformer:
            # amount is the target for this pipeline
            continue
        mock_supports_fast_importance.return_value = True
        permutation_importance_one_col_fast = (
            calculate_permutation_importance_one_column(
                pipeline,
                X,
                y,
                col,
                objective=objective,
                fast=True,
                precomputed_features=precomputed_features,
            )
        )

        mock_supports_fast_importance.return_value = False
        permutation_importance_one_col_slow = (
            calculate_permutation_importance_one_column(
                pipeline, X, y, col, objective=objective, fast=False
            )
        )
        np.testing.assert_almost_equal(
            permutation_importance_one_col_fast, permutation_importance_one_col_slow
        )


class PipelineWithDimReduction(BinaryClassificationPipeline):
    component_graph = [PCA, "Logistic Regression Classifier"]

    def __init__(self, parameters, random_seed=0):
        super().__init__(
            self.component_graph,
            parameters=parameters,
            random_seed=random_seed,
        )


class EnsembleDag(BinaryClassificationPipeline):
    component_graph = {
        "Imputer_1": ["Imputer", "X", "y"],
        "Imputer_2": ["Imputer", "X", "y"],
        "OHE_1": ["One Hot Encoder", "Imputer_1.x", "y"],
        "OHE_2": ["One Hot Encoder", "Imputer_2.x", "y"],
        "DT_1": ["DateTime Featurization Component", "OHE_1.x", "y"],
        "DT_2": ["DateTime Featurization Component", "OHE_2.x", "y"],
        "Estimator_1": ["Random Forest Classifier", "DT_1.x", "y"],
        "Estimator_2": ["Extra Trees Classifier", "DT_2.x", "y"],
        "Ensembler": [
            "Logistic Regression Classifier",
            "Estimator_1.x",
            "Estimator_2.x",
            "y",
        ],
    }

    def __init__(self, parameters, random_seed=0):
        super().__init__(
            self.component_graph,
            parameters=parameters,
            random_seed=random_seed,
        )


class PipelineWithDFS(BinaryClassificationPipeline):
    component_graph = [DFSTransformer, "Logistic Regression Classifier"]

    def __init__(self, parameters, random_seed=0):
        super().__init__(
            self.component_graph,
            parameters=parameters,
            random_seed=random_seed,
        )


class PipelineWithCustomComponent(BinaryClassificationPipeline):
    component_graph = [DoubleColumns, "Logistic Regression Classifier"]

    def __init__(self, parameters, random_seed=0):
        super().__init__(
            self.component_graph,
            parameters=parameters,
            random_seed=random_seed,
        )


class SklearnStackedEnsemblePipeline(BinaryClassificationPipeline):
    component_graph = ["Sklearn Stacked Ensemble Classifier"]

    def __init__(self, parameters, random_seed=0):
        super().__init__(
            self.component_graph,
            parameters=parameters,
            random_seed=random_seed,
        )


pipelines_that_do_not_support_fast_permutation_importance = [
    PipelineWithDimReduction,
    PipelineWithDFS,
    PipelineWithCustomComponent,
    EnsembleDag,
    SklearnStackedEnsemblePipeline,
]


@pytest.mark.parametrize(
    "pipeline_class", pipelines_that_do_not_support_fast_permutation_importance
)
def test_supports_fast_permutation_importance(pipeline_class):
    params = {
        "Sklearn Stacked Ensemble Classifier": {
            "input_pipelines": [PipelineWithDFS({})]
        }
    }
    assert not pipeline_class(params)._supports_fast_permutation_importance


def test_get_permutation_importance_invalid_objective(
    X_y_regression, linear_regression_pipeline_class
):
    X, y = X_y_regression
    pipeline = linear_regression_pipeline_class(parameters={}, random_seed=42)
    with pytest.raises(
        ValueError,
        match=f"Given objective 'MCC Multiclass' cannot be used with '{pipeline.name}'",
    ):
        calculate_permutation_importance(pipeline, X, y, "mcc multiclass")


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_get_permutation_importance_binary(
    X_y_binary,
    data_type,
    logistic_regression_binary_pipeline_class,
    binary_test_objectives,
    make_data_type,
):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}, random_seed=42
    )
    pipeline.fit(X, y)
    for objective in binary_test_objectives:
        permutation_importance = calculate_permutation_importance(
            pipeline, X, y, objective
        )
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()

        permutation_importance_sorted = permutation_importance.sort_values(
            "feature", ascending=True
        ).reset_index(drop=True)
        X = pd.DataFrame(X)
        for col in X.columns[:3]:
            permutation_importance_one_col = (
                calculate_permutation_importance_one_column(
                    pipeline, X, y, col, objective, fast=False
                )
            )
            np.testing.assert_almost_equal(
                permutation_importance_sorted["importance"][col],
                permutation_importance_one_col,
            )


def test_get_permutation_importance_multiclass(
    X_y_multi, logistic_regression_multiclass_pipeline_class, multiclass_test_objectives
):
    X, y = X_y_multi
    X = pd.DataFrame(X)
    pipeline = logistic_regression_multiclass_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}, random_seed=42
    )
    pipeline.fit(X, y)
    for objective in multiclass_test_objectives:
        permutation_importance = calculate_permutation_importance(
            pipeline, X, y, objective
        )
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()

        permutation_importance_sorted = permutation_importance.sort_values(
            "feature", ascending=True
        ).reset_index(drop=True)
        for col in X.columns[:3]:
            permutation_importance_one_col = (
                calculate_permutation_importance_one_column(
                    pipeline, X, y, col, objective, fast=False
                )
            )
            np.testing.assert_almost_equal(
                permutation_importance_sorted["importance"][col],
                permutation_importance_one_col,
            )


def test_get_permutation_importance_regression(
    linear_regression_pipeline_class, regression_test_objectives
):
    X = pd.DataFrame([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    y = pd.Series([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    pipeline = linear_regression_pipeline_class(
        parameters={"Linear Regressor": {"n_jobs": 1}}, random_seed=42
    )
    pipeline.fit(X, y)

    for objective in regression_test_objectives:
        permutation_importance = calculate_permutation_importance(
            pipeline, X, y, objective
        )
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()

        permutation_importance_sorted = permutation_importance.sort_values(
            "feature", ascending=True
        ).reset_index(drop=True)
        for col in X.columns:
            permutation_importance_one_col = (
                calculate_permutation_importance_one_column(
                    pipeline, X, y, col, objective, fast=False
                )
            )
            np.testing.assert_almost_equal(
                permutation_importance_sorted["importance"][col],
                permutation_importance_one_col,
            )


def test_get_permutation_importance_correlated_features(
    logistic_regression_binary_pipeline_class,
):
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["correlated"] = y * 2
    X["not correlated"] = [-1, -1, -1, 0]
    y = y.astype(bool)
    pipeline = logistic_regression_binary_pipeline_class(parameters={}, random_seed=42)
    pipeline.fit(X, y)
    importance = calculate_permutation_importance(
        pipeline, X, y, objective="Log Loss Binary", random_seed=0
    )
    assert list(importance.columns) == ["feature", "importance"]
    assert not importance.isnull().all().all()
    correlated_importance_val = importance["importance"][
        importance.index[importance["feature"] == "correlated"][0]
    ]
    not_correlated_importance_val = importance["importance"][
        importance.index[importance["feature"] == "not correlated"][0]
    ]
    assert correlated_importance_val > not_correlated_importance_val


def test_permutation_importance_oversampler(fraud_100):
    pytest.importorskip(
        "imblearn.over_sampling",
        reason="Skipping test because imbalanced-learn not installed",
    )
    X, y = fraud_100
    pipeline = BinaryClassificationPipeline(
        component_graph={
            "Imputer": ["Imputer", "X", "y"],
            "One Hot Encoder": ["One Hot Encoder", "Imputer.x", "y"],
            "DateTime Featurization Component": [
                "DateTime Featurization Component",
                "One Hot Encoder.x",
                "y",
            ],
            "Oversampler": [
                "Oversampler",
                "DateTime Featurization Component.x",
                "y",
            ],
            "Decision Tree Classifier": [
                "Decision Tree Classifier",
                "Oversampler.x",
                "Oversampler.y",
            ],
        }
    )
    pipeline.fit(X=X, y=y)
    pipeline.predict(X)
    importance = calculate_permutation_importance(
        pipeline, X, y, objective="Log Loss Binary"
    )
    assert not importance.isnull().all().all()


def test_get_permutation_importance_one_column_fast_no_precomputed_features(
    X_y_binary, logistic_regression_binary_pipeline_class
):
    X, y = X_y_binary
    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}, random_seed=42
    )
    with pytest.raises(
        ValueError,
        match="Fast method of calculating permutation importance requires precomputed_features",
    ):
        calculate_permutation_importance_one_column(
            pipeline, X, y, 0, "log loss binary", fast=True
        )


@pytest.mark.parametrize(
    "pipeline_class", pipelines_that_do_not_support_fast_permutation_importance
)
def test_get_permutation_importance_one_column_pipeline_does_not_support_fast(
    X_y_binary, pipeline_class
):
    X, y = X_y_binary
    params = {
        "Sklearn Stacked Ensemble Classifier": {
            "input_pipelines": [PipelineWithDFS({})]
        }
    }
    assert not pipeline_class(params)._supports_fast_permutation_importance
    with pytest.raises(
        ValueError,
        match="Pipeline does not support fast permutation importance calculation",
    ):
        calculate_permutation_importance_one_column(
            pipeline_class(params), X, y, 0, "log loss binary", fast=True
        )


def test_permutation_importance_unknown(X_y_binary):
    # test to see if we can get permutation importance fine with a dataset that has unknown features
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X.ww.init(logical_types={0: "unknown"})
    pl = BinaryClassificationPipeline(["Random Forest Classifier"])
    pl.fit(X, y)
    s = calculate_permutation_importance(pl, X, y, objective="Log Loss Binary")
    assert not s.isnull().any().any()


def test_permutation_importance_url_email(df_with_url_and_email):
    X = df_with_url_and_email.ww.select(["numeric", "url", "EmailAddress"])
    y = pd.Series([0, 1, 1, 0, 1])

    pl = BinaryClassificationPipeline(
        [
            "URL Featurizer",
            "Email Featurizer",
            "One Hot Encoder",
            "Random Forest Classifier",
        ]
    )
    pl.fit(X, y)
    data = calculate_permutation_importance(pl, X, y, objective="Log Loss Binary")
    assert not data.isnull().any().any()
    assert "url" in data["feature"].tolist()
    assert "email" in data["feature"].tolist()
