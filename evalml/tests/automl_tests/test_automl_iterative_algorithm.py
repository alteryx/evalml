import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from skopt.space import Categorical, Integer

from evalml import AutoMLSearch
from evalml.automl.automl_algorithm import IterativeAlgorithm
from evalml.automl.automl_algorithm.iterative_algorithm import _ESTIMATOR_FAMILY_ORDER
from evalml.automl.callbacks import raise_error_callback
from evalml.exceptions import ParameterNotUsedWarning
from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline, ComponentGraph, Estimator
from evalml.pipelines.components.utils import allowed_model_families, get_estimators
from evalml.problem_types import ProblemTypes
from evalml.tuners import SKOptTuner


def test_automl_feature_selection_with_allowed_component_graphs(
    AutoMLTestEnv,
    X_y_binary,
):
    X, y = X_y_binary

    start_iteration_callback = MagicMock()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=2,
        start_iteration_callback=start_iteration_callback,
        allowed_component_graphs={
            "Name": [
                "RF Classifier Select From Model",
                "Logistic Regression Classifier",
            ],
        },
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 1.0, "F1": 0.5}):
        automl.search()

    assert start_iteration_callback.call_count == 2
    proposed_parameters = start_iteration_callback.call_args_list[1][0][0].parameters
    assert proposed_parameters.keys() == {
        "RF Classifier Select From Model",
        "Logistic Regression Classifier",
    }
    assert (
        proposed_parameters["RF Classifier Select From Model"]["number_features"]
        == X.shape[1]
    )


def test_automl_allowed_component_graphs_iterative_algorithm(
    dummy_classifier_estimator_class,
    X_y_binary,
):
    X, y = X_y_binary
    allowed_component_graphs = {
        "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
    }
    aml = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
        max_iterations=10,
        automl_algorithm="iterative",
    )

    assert aml.automl_algorithm.allowed_component_graphs == allowed_component_graphs


def test_component_graph_with_incorrect_problem_type(
    dummy_classifier_estimator_class,
    X_y_binary,
):
    X, y = X_y_binary
    with pytest.raises(ValueError, match="not valid for this component graph"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="regression",
            allowed_component_graphs={
                "Mock Binary Classification Pipeline": [
                    dummy_classifier_estimator_class,
                ],
            },
            automl_algorithm="iterative",
        )


@pytest.mark.parametrize("return_dict", [True, False])
def test_describe_pipeline_with_ensembling(
    return_dict,
    X_y_binary,
    AutoMLTestEnv,
    caplog,
):
    X, y = X_y_binary
    two_stacking_batches = 1 + 2 * (len(get_estimators(ProblemTypes.BINARY)) + 1)
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=two_stacking_batches,
        objective="Log Loss Binary",
        ensembling=True,
        optimize_thresholds=False,
        error_callback=raise_error_callback,
        automl_algorithm="iterative",
    )

    score_side_effect = [
        {"Log Loss Binary": score}
        for score in np.arange(
            0,
            -1 * automl.max_iterations * automl.data_splitter.get_n_splits(),
            -0.1,
        )
    ]  # Decreases with each call

    test_env = AutoMLTestEnv("binary")
    with test_env.test_context(mock_score_side_effect=score_side_effect):
        automl.search()
    pipeline_names = automl.rankings["pipeline_name"]
    assert pipeline_names.str.contains("Ensemble").any()

    ensemble_ids = [
        _get_first_stacked_classifier_no() - 1,
        len(automl.results["pipeline_results"]) - 1,
    ]
    for i, ensemble_id in enumerate(ensemble_ids):
        caplog.clear()
        automl_dict = automl.describe_pipeline(ensemble_id, return_dict=return_dict)
        out = caplog.text
        assert "Stacked Ensemble Classification Pipeline" in out
        assert "* final_estimator : Elastic Net Classifier" in out
        assert "Problem Type: binary" in out
        assert "Model Family: Ensemble" in out
        assert "Total training time (including CV): " in out
        assert "Log Loss Binary # Training # Validation" in out
        assert "Input for ensembler are pipelines with IDs:" in out

        if return_dict:
            assert automl_dict["id"] == ensemble_id
            assert (
                automl_dict["pipeline_name"]
                == "Stacked Ensemble Classification Pipeline"
            )
            assert "Stacked Ensemble Classifier" in automl_dict["pipeline_summary"]
            assert isinstance(automl_dict["mean_cv_score"], float)
            assert not automl_dict["high_variance_cv"]
            assert isinstance(automl_dict["training_time"], float)
            assert isinstance(
                automl_dict["percent_better_than_baseline_all_objectives"],
                dict,
            )
            assert isinstance(automl_dict["percent_better_than_baseline"], float)
            assert isinstance(automl_dict["validation_score"], float)
            assert len(automl_dict["input_pipeline_ids"]) == len(
                allowed_model_families("binary"),
            )
            assert all(
                input_id < ensemble_id for input_id in automl_dict["input_pipeline_ids"]
            )
            if i > 0:
                assert all(
                    input_id > ensemble_ids[i - 1]
                    for input_id in automl_dict["input_pipeline_ids"]
                )
        else:
            assert automl_dict is None


def _get_first_stacked_classifier_no(model_families=None):
    """Gets the number of iterations necessary before the stacked ensemble will be used."""
    num_classifiers = len(
        get_estimators(ProblemTypes.BINARY, model_families=model_families),
    )
    # Baseline + first batch + each pipeline iteration (5 is current default pipelines_per_batch) + 1
    return 1 + num_classifiers + num_classifiers * 5 + 1


@pytest.mark.parametrize(
    "max_iterations",
    [
        None,
        1,
        8,
        10,
        _get_first_stacked_classifier_no(),
        _get_first_stacked_classifier_no() + 2,
    ],
)
@pytest.mark.parametrize("use_ensembling", [True, False])
def test_max_iteration_works_with_stacked_ensemble(
    max_iterations,
    use_ensembling,
    AutoMLTestEnv,
    X_y_binary,
    caplog,
):
    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=max_iterations,
        objective="Log Loss Binary",
        optimize_thresholds=False,
        ensembling=use_ensembling,
        verbose=True,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.8}):
        automl.search()
    # every nth batch a stacked ensemble will be trained
    if max_iterations is None:
        max_iterations = 5  # Default value for max_iterations

    pipeline_names = automl.rankings["pipeline_name"]
    if max_iterations < _get_first_stacked_classifier_no():
        assert not pipeline_names.str.contains("Ensemble").any()
    elif use_ensembling:
        assert pipeline_names.str.contains("Ensemble").any()
        assert (
            f"Ensembling will run at the {_get_first_stacked_classifier_no()} iteration"
            in caplog.text
        )

    else:
        assert not pipeline_names.str.contains("Ensemble").any()


def test_automl_one_allowed_component_graph_ensembling_disabled(
    AutoMLTestEnv,
    X_y_binary,
    caplog,
):
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.RANDOM_FOREST]) + 1
    # Checks that when len(allowed_component_graphs) == 1, ensembling is not run, even if set to True
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=max_iterations,
        optimize_thresholds=False,
        allowed_model_families=[ModelFamily.RANDOM_FOREST],
        ensembling=True,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    assert (
        "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run."
        in caplog.text
    )

    pipeline_names = automl.rankings["pipeline_name"]
    assert not pipeline_names.str.contains("Ensemble").any()

    caplog.clear()
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.LINEAR_MODEL]) + 1
    allowed_component_graph = {
        "Logistic Regression Binary Pipeline": [
            "Imputer",
            "One Hot Encoder",
            "Standard Scaler",
            "Logistic Regression Classifier",
        ],
    }
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=max_iterations,
        allowed_component_graphs=allowed_component_graph,
        optimize_thresholds=False,
        ensembling=True,
        automl_algorithm="iterative",
    )
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    pipeline_names = automl.rankings["pipeline_name"]
    assert not pipeline_names.str.contains("Ensemble").any()
    assert (
        "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run."
        in caplog.text
    )
    # Check that ensembling runs when len(allowed_model_families) == 1 but len(allowed_component_graphs) > 1
    caplog.clear()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=max_iterations,
        optimize_thresholds=False,
        allowed_model_families=[ModelFamily.LINEAR_MODEL],
        ensembling=True,
    )
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    pipeline_names = automl.rankings["pipeline_name"]
    assert pipeline_names.str.contains("Ensemble").any()
    assert (
        "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run."
        not in caplog.text
    )


def test_automl_max_iterations_less_than_ensembling_disabled(
    AutoMLTestEnv,
    X_y_binary,
    caplog,
):
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.LINEAR_MODEL])
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=max_iterations - 1,
        optimize_thresholds=False,
        allowed_model_families=[ModelFamily.LINEAR_MODEL],
        ensembling=True,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    assert (
        f"Ensembling is set to True, but max_iterations is too small, so ensembling will not run. Set max_iterations >= {max_iterations} to run ensembling."
        in caplog.text
    )

    pipeline_names = automl.rankings["pipeline_name"]
    assert not pipeline_names.str.contains("Ensemble").any()


def test_automl_max_batches_less_than_ensembling_disabled(
    AutoMLTestEnv,
    X_y_binary,
    caplog,
):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=2,
        optimize_thresholds=False,
        allowed_model_families=[ModelFamily.LINEAR_MODEL],
        ensembling=True,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    first_ensemble_batch = (
        1 + len(automl.allowed_pipelines) + 1
    )  # First batch + each pipeline batch
    assert (
        f"Ensembling is set to True, but max_batches is too small, so ensembling will not run. Set max_batches >= {first_ensemble_batch} to run ensembling."
        in caplog.text
    )

    pipeline_names = automl.rankings["pipeline_name"]
    assert not pipeline_names.str.contains("Ensemble").any()


@pytest.mark.parametrize("max_batches", [None, 1, 5, 8, 9, 10, 12, 20])
@pytest.mark.parametrize("use_ensembling", [True, False])
@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.REGRESSION])
def test_max_batches_num_pipelines(
    max_batches,
    use_ensembling,
    problem_type,
    AutoMLTestEnv,
    X_y_binary,
    X_y_regression,
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary

    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        max_iterations=None,
        max_batches=max_batches,
        optimize_thresholds=False,
        ensembling=use_ensembling,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv(problem_type)
    with env.test_context(score_return_value={automl.objective.name: 0.3}):
        automl.search()
    # every nth batch a stacked ensemble will be trained
    ensemble_nth_batch = len(automl.allowed_pipelines) + 1

    if max_batches is None:
        n_results = len(automl.allowed_pipelines) + 1
        # automl_algorithm will include all allowed_pipelines in the first batch even
        # if they are not searched over. That is why n_automl_pipelines does not equal
        # n_results when max_iterations and max_batches are None
        n_automl_pipelines = 1 + len(automl.allowed_pipelines)
        num_ensemble_batches = 0
    else:
        # automl algorithm does not know about the additional stacked ensemble pipelines
        num_ensemble_batches = (
            (max_batches - 1) // ensemble_nth_batch if use_ensembling else 0
        )
        # So that the test does not break when new estimator classes are added
        n_results = (
            1
            + len(automl.allowed_pipelines)
            + (5 * (max_batches - 1 - num_ensemble_batches))
            + num_ensemble_batches
        )
        n_automl_pipelines = n_results
    if max_batches is None:
        max_batches = automl.automl_algorithm.default_max_batches
        assert automl.automl_algorithm.batch_number == max_batches
        assert automl.automl_algorithm.pipeline_number + 1 == n_automl_pipelines
    else:
        assert automl.automl_algorithm.batch_number == max_batches
        assert automl.automl_algorithm.pipeline_number + 1 == n_automl_pipelines
    assert len(automl.results["pipeline_results"]) == n_results
    if num_ensemble_batches == 0:
        assert automl.rankings.shape[0] == min(
            1 + len(automl.allowed_pipelines),
            n_results,
        )  # add one for baseline
    else:
        assert automl.rankings.shape[0] == min(
            2 + len(automl.allowed_pipelines),
            n_results,
        )  # add two for baseline and stacked ensemble
    assert automl.full_rankings.shape[0] == n_results


@patch("evalml.tuners.skopt_tuner.SKOptTuner.add")
def test_pipeline_hyperparameters_make_pipeline_other_errors(
    mock_add,
    AutoMLTestEnv,
    X_y_multi,
):
    X, y = X_y_multi
    search_parameters = {
        "Imputer": {"numeric_impute_strategy": Categorical(["most_frequent", "mean"])},
    }
    estimators = get_estimators("multiclass", [ModelFamily.EXTRA_TREES])

    component_graphs = {}
    for ind, estimator in enumerate(estimators):
        component_graphs[f"CG_{ind}"] = [estimator]

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        allowed_component_graphs=component_graphs,
        search_parameters=search_parameters,
        n_jobs=1,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("multiclass")

    mock_add.side_effect = ValueError("Alternate error that can be thrown")
    with pytest.raises(ValueError) as error:
        with env.test_context(score_return_value={"Log Loss Multiclass": 1.0}):
            automl.search()
    assert "Alternate error that can be thrown" in str(error.value)
    assert "Default parameters for components" not in str(error.value)


@pytest.mark.parametrize("component_graphs", [True, False])
@pytest.mark.parametrize("automl_parameters", [True, False])
@pytest.mark.parametrize("custom_hyperparameters", [True, False])
def test_pipeline_custom_hyperparameters_make_pipeline(
    custom_hyperparameters,
    automl_parameters,
    component_graphs,
    X_y_multi,
    AutoMLTestEnv,
):
    X, y = X_y_multi
    X.ww.columns = [f"Column_{i}" for i in range(20)]

    component_graph_ = None
    search_parameters_ = {}

    if component_graphs:
        component_graph_ = {
            "Name_0": [
                "Drop Columns Transformer",
                "Imputer",
                "Random Forest Classifier",
            ],
        }

    if automl_parameters:
        search_parameters_ = {
            "Drop Columns Transformer": {
                "columns": ["Column_0", "Column_1", "Column_2"],
            },
        }
    if custom_hyperparameters:
        search_parameters_.update(
            {
                "Imputer": {"numeric_impute_strategy": Categorical(["mean"])},
                "Random Forest Classifier": {
                    "max_depth": Integer(4, 7),
                    "n_estimators": Integer(190, 210),
                },
            },
        )

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        allowed_component_graphs=component_graph_,
        search_parameters=search_parameters_,
        max_batches=4,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("multiclass")
    with env.test_context(score_return_value={"Log Loss Multiclass": 1.0}):
        automl.search()

    for i, row in automl.full_rankings.iterrows():
        if (
            "Random Forest Classifier" in row["pipeline_name"]
            or "Name_0" in row["pipeline_name"]
        ):
            if component_graph_ and automl_parameters:
                assert row["parameters"]["Drop Columns Transformer"]["columns"] == [
                    "Column_0",
                    "Column_1",
                    "Column_2",
                ]
            if custom_hyperparameters:
                assert (
                    row["parameters"]["Imputer"]["numeric_impute_strategy"]
                    in search_parameters_["Imputer"]["numeric_impute_strategy"]
                )
                assert (
                    4 <= row["parameters"]["Random Forest Classifier"]["max_depth"] <= 7
                )
                assert (
                    190
                    <= row["parameters"]["Random Forest Classifier"]["n_estimators"]
                    <= 210
                )
            else:
                assert row["parameters"]["Imputer"]["numeric_impute_strategy"] in [
                    "mean",
                    "median",
                    "most_frequent",
                    "knn",
                ]
                assert (
                    1
                    <= row["parameters"]["Random Forest Classifier"]["max_depth"]
                    <= 10
                )
                assert (
                    10
                    <= row["parameters"]["Random Forest Classifier"]["n_estimators"]
                    <= 1000
                )


def test_passes_njobs_to_pipelines(
    dummy_classifier_estimator_class,
    X_y_binary,
    AutoMLTestEnv,
):
    X, y = X_y_binary

    class MockEstimatorWithNJobs(Estimator):
        name = "Mock Classifier with njobs"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        hyperparameter_ranges = {}

        def __init__(self, n_jobs=-1, random_seed=0):
            super().__init__(
                parameters={"n_jobs": n_jobs},
                component_obj=None,
                random_seed=random_seed,
            )

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        n_jobs=3,
        max_batches=2,
        allowed_component_graphs={
            "Pipeline 1": [MockEstimatorWithNJobs],
            "Pipeline 2": [MockEstimatorWithNJobs],
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
        },
        optimize_thresholds=False,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.2}):
        automl.search()
    for parameters in automl.full_rankings.parameters:
        if "Mock Classifier with njobs" in parameters:
            assert parameters["Mock Classifier with njobs"]["n_jobs"] == 3
        else:
            assert all(
                "n_jobs" not in component_params
                for component_params in parameters.values()
            )


def test_automl_ensembling_false(AutoMLTestEnv, X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_time="60 seconds",
        max_batches=20,
        optimize_thresholds=False,
        ensembling=False,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.32}):
        automl.search()
    assert not automl.rankings["pipeline_name"].str.contains("Ensemble").any()


@pytest.mark.parametrize(
    "problem_type,pipeline_name,ensemble_name",
    [
        (
            "binary",
            "Stacked Ensemble Classification Pipeline",
            "Stacked Ensemble Classifier",
        ),
        (
            "multiclass",
            "Stacked Ensemble Classification Pipeline",
            "Stacked Ensemble Classifier",
        ),
        (
            "regression",
            "Stacked Ensemble Regression Pipeline",
            "Stacked Ensemble Regressor",
        ),
    ],
)
@pytest.mark.parametrize("df_text", [True, False])
@patch("evalml.automl.automl_search.IterativeAlgorithm")
def test_search_with_text_and_ensembling(
    mock_iter,
    df_text,
    problem_type,
    pipeline_name,
    ensemble_name,
):
    X_with_text = pd.DataFrame(
        {
            "col_1": [
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm gonna be the main event, like no king was before! I'm brushing up on looking down, I'm working on my ROAR!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "I dreamed a dream in days gone by, when hope was high and life worth living",
                "Red, the blood of angry men - black, the dark of ages past",
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "Red, the blood of angry men - black, the dark of ages past",
                "It was red and yellow and green and brown and scarlet and black and ochre and peach and ruby and olive and violet and fawn...",
            ],
        },
    )
    X_no_text = pd.DataFrame({"col_1": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]})

    if df_text:
        X = X_with_text
        X.ww.init(logical_types={"col_1": "NaturalLanguage"})
    else:
        X = X_no_text
    if problem_type == "binary":
        y = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    elif problem_type == "multiclass":
        y = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    else:
        y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    _ = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        allowed_model_families=["random_forest", "decision_tree"],
        optimize_thresholds=False,
        max_batches=4,
        ensembling=True,
        automl_algorithm="iterative",
    )

    call_args = mock_iter.call_args[1]["text_in_ensembling"]
    if df_text:
        assert call_args is True
    else:
        assert call_args is False


def test_pipelines_per_batch(AutoMLTestEnv, X_y_binary):
    def total_pipelines(automl, num_batches, batch_size):
        total = 1 + len(automl.allowed_pipelines)
        total += (num_batches - 1) * batch_size
        return total

    X, y = X_y_binary

    # Checking for default of _pipelines_per_batch
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=2,
        optimize_thresholds=False,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.30}):
        automl.search()
    assert automl._pipelines_per_batch == 5
    assert automl.automl_algorithm.pipelines_per_batch == 5
    assert total_pipelines(automl, 2, 5) == len(automl.full_rankings)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=1,
        optimize_thresholds=False,
        _pipelines_per_batch=2,
        automl_algorithm="iterative",
    )
    with env.test_context(score_return_value={"Log Loss Binary": 0.30}):
        automl.search()
    assert automl._pipelines_per_batch == 2
    assert automl.automl_algorithm.pipelines_per_batch == 2
    assert total_pipelines(automl, 1, 2) == len(automl.full_rankings)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=2,
        optimize_thresholds=False,
        _pipelines_per_batch=10,
        automl_algorithm="iterative",
    )
    with env.test_context(score_return_value={"Log Loss Binary": 0.30}):
        automl.search()
    assert automl._pipelines_per_batch == 10
    assert automl.automl_algorithm.pipelines_per_batch == 10
    assert total_pipelines(automl, 2, 10) == len(automl.full_rankings)


def test_automl_respects_random_seed(X_y_binary, dummy_classifier_estimator_class):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        allowed_component_graphs={"Name_0": [dummy_classifier_estimator_class]},
        optimize_thresholds=False,
        random_seed=42,
        max_iterations=10,
        automl_algorithm="iterative",
    )
    pipelines = [
        BinaryClassificationPipeline(
            component_graph=[dummy_classifier_estimator_class],
            random_seed=42,
        ),
    ]
    automl.automl_algorithm = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        tuner_class=SKOptTuner,
        random_seed=42,
        n_jobs=1,
        number_features=X.shape[1],
        pipelines_per_batch=5,
        ensembling=False,
        text_in_ensembling=False,
        search_parameters={},
    )
    automl.automl_algorithm.allowed_pipelines = pipelines
    assert automl.allowed_pipelines[0].random_seed == 42


def test_automl_respects_pipeline_parameters_with_duplicate_components(
    AutoMLTestEnv,
    X_y_binary,
):
    X, y = X_y_binary
    # Pass the input of the first imputer to the second imputer
    component_graph_dict = {
        "Imputer": ["Imputer", "X", "y"],
        "Imputer_1": ["Imputer", "Imputer.x", "y"],
        "Random Forest Classifier": ["Random Forest Classifier", "Imputer_1.x", "y"],
    }

    automl = AutoMLSearch(
        X,
        y,
        problem_type="binary",
        allowed_component_graphs={"Pipeline from dict": component_graph_dict},
        search_parameters={
            "Imputer": {"numeric_impute_strategy": "most_frequent"},
            "Imputer_1": {"numeric_impute_strategy": "median"},
        },
        optimize_thresholds=False,
        max_batches=1,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 0.63}):
        automl.search()
    for row in automl.full_rankings.iloc[1:].parameters:
        assert row["Imputer"]["numeric_impute_strategy"] == "most_frequent"
        assert row["Imputer_1"]["numeric_impute_strategy"] == "median"

    component_graph_dict = {
        "One Hot Encoder": ["One Hot Encoder", "X", "y"],
        "One Hot Encoder_1": ["One Hot Encoder", "One Hot Encoder.x", "y"],
        "Random Forest Classifier": [
            "Random Forest Classifier",
            "One Hot Encoder_1.x",
            "y",
        ],
    }

    automl = AutoMLSearch(
        X,
        y,
        problem_type="binary",
        allowed_component_graphs={
            "Pipeline from dict": component_graph_dict,
        },
        search_parameters={
            "One Hot Encoder": {"top_n": 15},
            "One Hot Encoder_1": {"top_n": 25},
        },
        optimize_thresholds=False,
        max_batches=1,
        automl_algorithm="iterative",
    )
    with env.test_context(score_return_value={automl.objective.name: 0.63}):
        automl.search()
    for row in automl.full_rankings.iloc[1:].parameters:
        assert row["One Hot Encoder"]["top_n"] == 15
        assert row["One Hot Encoder_1"]["top_n"] == 25


def test_automl_respects_pipeline_custom_hyperparameters_with_duplicate_components(
    AutoMLTestEnv,
    X_y_binary,
):
    X, y = X_y_binary
    search_parameters = {
        "Imputer": {"numeric_impute_strategy": Categorical(["most_frequent", "mean"])},
        "Imputer_1": {"numeric_impute_strategy": Categorical(["median", "mean"])},
        "Random Forest Classifier": {"n_estimators": Categorical([50, 100])},
    }
    component_graph = {
        "Name_dict": {
            "Imputer": ["Imputer", "X", "y"],
            "Imputer_1": ["Imputer", "Imputer.x", "y"],
            "Random Forest Classifier": [
                "Random Forest Classifier",
                "Imputer_1.x",
                "y",
            ],
        },
    }

    automl = AutoMLSearch(
        X,
        y,
        problem_type="binary",
        allowed_component_graphs=component_graph,
        search_parameters=search_parameters,
        optimize_thresholds=False,
        max_batches=5,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 0.12}):
        automl.search()
    for i, row in automl.full_rankings.iterrows():
        if "Mode Baseline Binary" in row["pipeline_name"]:
            continue
        if row["pipeline_name"] == "Name_dict":
            assert row["parameters"]["Imputer"]["numeric_impute_strategy"] in {
                "most_frequent",
                "mean",
            }
            assert row["parameters"]["Imputer_1"]["numeric_impute_strategy"] in {
                "median",
                "mean",
            }
            assert row["parameters"]["Random Forest Classifier"]["n_estimators"] in {
                50,
                100,
            }


@pytest.mark.parametrize(
    "allowed_component_graphs",
    [None, {"graph": ["Imputer", "Logistic Regression Classifier"]}],
)
@pytest.mark.parametrize(
    "search_parameters,set_values",
    [
        ({"Logistic Regression Classifier": {"penalty": "l1"}}, {}),
        (
            {
                "Imputer": {"numeric_impute_strategy": "mean"},
                "Logistic Regression Classifier": {"penalty": "l1"},
            },
            {},
        ),
        (
            {
                "Undersampler": {"sampling_ratio": 0.05},
                "Logistic Regression Classifier": {"penalty": "l1"},
            },
            {"Undersampler"},
        ),
        (
            {
                "Undersampler": {"sampling_ratio": 0.05},
                "Oversampler": {"sampling_ratio": 0.10},
            },
            {"Undersampler", "Oversampler"},
        ),
    ],
)
def test_pipeline_parameter_warnings_component_graphs(
    search_parameters,
    set_values,
    allowed_component_graphs,
    AutoMLTestEnv,
    X_y_binary,
):
    X, y = X_y_binary
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(
            "always",
            category=ParameterNotUsedWarning,
        )
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            max_batches=2,
            n_jobs=1,
            allowed_component_graphs=allowed_component_graphs,
            search_parameters=search_parameters,
            automl_algorithm="iterative",
        )
        env = AutoMLTestEnv("binary")
        with env.test_context(score_return_value={automl.objective.name: 1.0}):
            automl.search()
    assert len(w) == (1 if len(set_values) else 0)
    if len(w):
        assert w[0].message.components == set_values


def test_graph_automl(X_y_multi):
    X, y = X_y_multi
    X = pd.DataFrame(X, columns=[f"Column_{i}" for i in range(20)])

    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "OneHot_RandomForest": ["One Hot Encoder", "Imputer.x", "y"],
        "OneHot_ElasticNet": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest": ["Random Forest Classifier", "OneHot_RandomForest.x", "y"],
        "Elastic Net": ["Elastic Net Classifier", "OneHot_ElasticNet.x", "y"],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }
    automl_parameters_ = {
        "OneHot_ElasticNet": {"top_n": 5},
        "Random Forest": {"n_estimators": 201},
        "Elastic Net": {"C": 0.42, "l1_ratio": 0.2},
    }
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        allowed_component_graphs={"Name_0": ComponentGraph(component_graph)},
        search_parameters=automl_parameters_,
        max_batches=2,
        automl_algorithm="iterative",
    )

    dag_dict = automl.allowed_pipelines[0].graph_dict()
    for node_, params_ in automl_parameters_.items():
        for key_, val_ in params_.items():
            assert (
                dag_dict["Nodes"][node_]["Parameters"][key_]
                == automl_parameters_[node_][key_]
            )


def test_automl_respects_pipeline_order(X_y_binary, AutoMLTestEnv):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X,
        y,
        problem_type="binary",
        engine="sequential",
        max_iterations=5,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 0.2}):
        automl.search()
    searched_model_families = [
        automl.get_pipeline(search_index).estimator.model_family
        for search_index in automl.results["search_order"][1:]
    ]
    # Check that sorting via the model family order results in the same list.
    assert searched_model_families == sorted(
        searched_model_families,
        key=lambda model_family: _ESTIMATOR_FAMILY_ORDER.index(model_family),
    )


def test_get_ensembler_input_pipelines(X_y_binary, AutoMLTestEnv):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=_get_first_stacked_classifier_no(),
        objective="Log Loss Binary",
        ensembling=True,
        automl_algorithm="iterative",
    )

    score_side_effect = [
        {"Log Loss Binary": score}
        for score in np.arange(
            0,
            -1 * automl.max_iterations * automl.data_splitter.get_n_splits(),
            -0.1,
        )
    ]  # Decreases with each call

    test_env = AutoMLTestEnv("binary")
    with test_env.test_context(mock_score_side_effect=score_side_effect):
        automl.search()

    best_pipeline_ids = [
        pipeline["id"]
        for pipeline in list(automl.automl_algorithm._best_pipeline_info.values())
    ]
    best_pipeline_ids.sort()

    input_pipeline_ids = automl.get_ensembler_input_pipelines(
        _get_first_stacked_classifier_no() - 1,
    )
    input_pipeline_ids.sort()

    assert best_pipeline_ids == input_pipeline_ids

    two_stacking_batches = 1 + 2 * (len(get_estimators(ProblemTypes.BINARY)) + 1)
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=two_stacking_batches,
        objective="Log Loss Binary",
        ensembling=True,
        automl_algorithm="iterative",
    )

    test_env = AutoMLTestEnv("binary")
    with test_env.test_context(mock_score_side_effect=score_side_effect):
        automl.search()

    ensemble_ids = [
        _get_first_stacked_classifier_no() - 1,
        len(automl.results["pipeline_results"]) - 1,
    ]

    final_best_pipeline_ids = [
        pipeline["id"]
        for pipeline in list(automl.automl_algorithm._best_pipeline_info.values())
    ]
    final_best_pipeline_ids.sort()

    input_pipeline_0_ids = automl.get_ensembler_input_pipelines(ensemble_ids[0])
    input_pipeline_0_ids.sort()

    input_pipeline_1_ids = automl.get_ensembler_input_pipelines(ensemble_ids[1])
    input_pipeline_1_ids.sort()

    assert final_best_pipeline_ids != input_pipeline_0_ids
    assert final_best_pipeline_ids == input_pipeline_1_ids

    error_text = "Pipeline ID 12 is not a valid ensemble pipeline"
    with pytest.raises(ValueError, match=error_text):
        automl.get_ensembler_input_pipelines(12)

    error_text = "Pipeline ID 500 is not a valid ensemble pipeline"
    with pytest.raises(ValueError, match=error_text):
        automl.get_ensembler_input_pipelines(500)
