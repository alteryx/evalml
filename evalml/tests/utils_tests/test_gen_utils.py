import inspect
import os
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import ValidationErrorCode
from evalml.model_understanding.visualizations import visualize_decision_tree
from evalml.pipelines.components import ComponentBase
from evalml.utils.gen_utils import (
    SEED_BOUNDS,
    _rename_column_names_to_numeric,
    are_datasets_separated_by_gap_time_index,
    are_ts_parameters_valid_for_split,
    classproperty,
    contains_all_ts_parameters,
    convert_to_seconds,
    deprecate_arg,
    get_importable_subclasses,
    get_random_seed,
    get_time_index,
    import_or_raise,
    is_categorical_actually_boolean,
    jupyter_check,
    pad_with_nans,
    save_plot,
    validate_holdout_datasets,
)


@pytest.fixture(scope="module")
def in_container_arm64():
    """Helper fixture to run chromium as a single process for kaleido.

    The env var is set in the Dockerfile.arm for the purposes of local
    testing in a container on a mac M1, otherwise it's a noop.
    """
    if os.getenv("DOCKER_ARM", None):
        import plotly.io as pio

        pio.kaleido.scope.chromium_args += ("--single-process",)


@patch("importlib.import_module")
def test_import_or_raise_errors(dummy_importlib):
    def _mock_import_function(library_str):
        if library_str == "_evalml":
            raise ImportError("Mock ImportError executed!")
        if library_str == "attr_error_lib":
            raise Exception("Mock Exception executed!")

    dummy_importlib.side_effect = _mock_import_function

    with pytest.raises(ImportError, match="Missing optional dependency '_evalml'"):
        import_or_raise("_evalml")
    with pytest.raises(
        ImportError,
        match="Missing optional dependency '_evalml'. Please use pip to install _evalml. Additional error message",
    ):
        import_or_raise("_evalml", "Additional error message")
    with pytest.raises(
        Exception,
        match="An exception occurred while trying to import `attr_error_lib`: Mock Exception executed!",
    ):
        import_or_raise("attr_error_lib")


def test_import_or_raise_imports():
    math = import_or_raise("math", "error message")
    assert math.ceil(0.1) == 1


def test_convert_to_seconds():
    assert convert_to_seconds("10 s") == 10
    assert convert_to_seconds("10 sec") == 10
    assert convert_to_seconds("10 second") == 10
    assert convert_to_seconds("10 seconds") == 10

    assert convert_to_seconds("10 m") == 600
    assert convert_to_seconds("10 min") == 600
    assert convert_to_seconds("10 minute") == 600
    assert convert_to_seconds("10 minutes") == 600

    assert convert_to_seconds("10 h") == 36000
    assert convert_to_seconds("10 hr") == 36000
    assert convert_to_seconds("10 hour") == 36000
    assert convert_to_seconds("10 hours") == 36000

    with pytest.raises(AssertionError, match="Invalid unit."):
        convert_to_seconds("10 years")


def test_get_random_seed_rng():
    def make_mock_random_state(return_value):
        class MockRandomState(np.random.RandomState):
            def __init__(self):
                self.min_bound = None
                self.max_bound = None
                super().__init__()

            def randint(self, min_bound, max_bound):
                self.min_bound = min_bound
                self.max_bound = max_bound
                return return_value

        return MockRandomState()

    rng = make_mock_random_state(42)
    assert get_random_seed(rng) == 42
    assert rng.min_bound == SEED_BOUNDS.min_bound
    assert rng.max_bound == SEED_BOUNDS.max_bound


def test_get_random_seed_int():
    # ensure the invariant "min_bound < max_bound" is enforced
    with pytest.raises(ValueError):
        get_random_seed(0, min_bound=0, max_bound=0)
    with pytest.raises(ValueError):
        get_random_seed(0, min_bound=0, max_bound=-1)

    # test default boundaries to show the provided value should modulate within the default range
    assert get_random_seed(SEED_BOUNDS.max_bound - 2) == SEED_BOUNDS.max_bound - 2
    assert get_random_seed(SEED_BOUNDS.max_bound - 1) == SEED_BOUNDS.max_bound - 1
    assert get_random_seed(SEED_BOUNDS.max_bound) == SEED_BOUNDS.min_bound
    assert get_random_seed(SEED_BOUNDS.max_bound + 1) == SEED_BOUNDS.min_bound + 1
    assert get_random_seed(SEED_BOUNDS.max_bound + 2) == SEED_BOUNDS.min_bound + 2
    assert get_random_seed(SEED_BOUNDS.min_bound - 2) == SEED_BOUNDS.max_bound - 2
    assert get_random_seed(SEED_BOUNDS.min_bound - 1) == SEED_BOUNDS.max_bound - 1
    assert get_random_seed(SEED_BOUNDS.min_bound) == SEED_BOUNDS.min_bound
    assert get_random_seed(SEED_BOUNDS.min_bound + 1) == SEED_BOUNDS.min_bound + 1
    assert get_random_seed(SEED_BOUNDS.min_bound + 2) == SEED_BOUNDS.min_bound + 2

    # vectorize get_random_seed via a wrapper for easy evaluation
    default_min_bound = (
        inspect.signature(get_random_seed).parameters["min_bound"].default
    )
    default_max_bound = (
        inspect.signature(get_random_seed).parameters["max_bound"].default
    )
    assert default_min_bound == SEED_BOUNDS.min_bound
    assert default_max_bound == SEED_BOUNDS.max_bound

    def get_random_seed_vec(
        min_bound=None,
        max_bound=None,
    ):  # passing None for either means no value is provided to get_random_seed
        def get_random_seed_wrapper(random_seed):
            return get_random_seed(
                random_seed,
                min_bound=min_bound if min_bound is not None else default_min_bound,
                max_bound=max_bound if max_bound is not None else default_max_bound,
            )

        return np.vectorize(get_random_seed_wrapper)

    # ensure that regardless of the setting of min_bound and max_bound, the output of get_random_seed always stays
    # between the min_bound (inclusive) and max_bound (exclusive), and wraps neatly around that range using modular arithmetic.
    vals = np.arange(-100, 100)

    def make_expected_values(vals, min_bound, max_bound):
        return np.array(
            [
                i
                if (min_bound <= i and i < max_bound)
                else ((i - min_bound) % (max_bound - min_bound)) + min_bound
                for i in vals
            ],
        )

    np.testing.assert_equal(
        get_random_seed_vec(min_bound=None, max_bound=None)(vals),
        make_expected_values(
            vals,
            min_bound=SEED_BOUNDS.min_bound,
            max_bound=SEED_BOUNDS.max_bound,
        ),
    )
    np.testing.assert_equal(
        get_random_seed_vec(min_bound=None, max_bound=10)(vals),
        make_expected_values(vals, min_bound=SEED_BOUNDS.min_bound, max_bound=10),
    )
    np.testing.assert_equal(
        get_random_seed_vec(min_bound=-10, max_bound=None)(vals),
        make_expected_values(vals, min_bound=-10, max_bound=SEED_BOUNDS.max_bound),
    )
    np.testing.assert_equal(
        get_random_seed_vec(min_bound=0, max_bound=5)(vals),
        make_expected_values(vals, min_bound=0, max_bound=5),
    )
    np.testing.assert_equal(
        get_random_seed_vec(min_bound=-5, max_bound=0)(vals),
        make_expected_values(vals, min_bound=-5, max_bound=0),
    )
    np.testing.assert_equal(
        get_random_seed_vec(min_bound=-5, max_bound=5)(vals),
        make_expected_values(vals, min_bound=-5, max_bound=5),
    )
    np.testing.assert_equal(
        get_random_seed_vec(min_bound=5, max_bound=10)(vals),
        make_expected_values(vals, min_bound=5, max_bound=10),
    )
    np.testing.assert_equal(
        get_random_seed_vec(min_bound=-10, max_bound=-5)(vals),
        make_expected_values(vals, min_bound=-10, max_bound=-5),
    )


def test_class_property():
    class MockClass:
        name = "MockClass"

        @classproperty
        def caps_name(cls):
            return cls.name.upper()

    assert MockClass.caps_name == "MOCKCLASS"


def test_get_importable_subclasses_wont_get_custom_classes():
    class ChildClass(ComponentBase):
        pass

    assert ChildClass not in get_importable_subclasses(ComponentBase)


@patch("importlib.import_module")
def test_import_or_warn_errors(dummy_importlib):
    def _mock_import_function(library_str):
        if library_str == "_evalml":
            raise ImportError("Mock ImportError executed!")
        if library_str == "attr_error_lib":
            raise Exception("Mock Exception executed!")

    dummy_importlib.side_effect = _mock_import_function

    with pytest.warns(UserWarning, match="Missing optional dependency '_evalml'"):
        import_or_raise("_evalml", warning=True)
    with pytest.warns(
        UserWarning,
        match="Missing optional dependency '_evalml'. Please use pip to install _evalml. Additional error message",
    ):
        import_or_raise("_evalml", "Additional error message", warning=True)
    with pytest.warns(
        UserWarning,
        match="An exception occurred while trying to import `attr_error_lib`: Mock Exception executed!",
    ):
        import_or_raise("attr_error_lib", warning=True)


@patch("evalml.utils.gen_utils.import_or_raise")
def test_jupyter_check_errors(mock_import_or_raise):
    mock_import_or_raise.side_effect = ImportError
    assert not jupyter_check()

    mock_import_or_raise.side_effect = Exception
    assert not jupyter_check()


@patch("evalml.utils.gen_utils.import_or_raise")
def test_jupyter_check(mock_import_or_raise):
    mock_import_or_raise.return_value = MagicMock()
    mock_import_or_raise().core.getipython.get_ipython.return_value = True
    assert jupyter_check()
    mock_import_or_raise().core.getipython.get_ipython.return_value = False
    assert not jupyter_check()
    mock_import_or_raise().core.getipython.get_ipython.return_value = None
    assert not jupyter_check()


def _check_equality(data, expected, check_index_type=True):
    if isinstance(data, pd.Series):
        pd.testing.assert_series_equal(data, expected, check_index_type)
    else:
        pd.testing.assert_frame_equal(data, expected, check_index_type)


@pytest.mark.parametrize(
    "data,num_to_pad,expected",
    [
        (pd.Series([1, 2, 3]), 1, pd.Series([np.nan, 1, 2, 3], dtype="float64")),
        (pd.Series([1, 2, 3]), 0, pd.Series([1, 2, 3])),
        (
            pd.Series([1, 2, 3, 4], index=pd.date_range("2020-10-01", "2020-10-04")),
            2,
            pd.Series([np.nan, np.nan, 1, 2, 3, 4], dtype="float64"),
        ),
        (
            pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
            0,
            pd.DataFrame(
                {
                    "a": pd.Series([1.0, 2.0, 3.0], dtype="float64"),
                    "b": pd.Series([4.0, 5.0, 6.0], dtype="float64"),
                },
            ),
        ),
        (
            pd.DataFrame({"a": [4, 5, 6], "b": ["a", "b", "c"]}),
            1,
            pd.DataFrame(
                {
                    "a": pd.Series([np.nan, 4, 5, 6], dtype="float64"),
                    "b": [np.nan, "a", "b", "c"],
                },
            ),
        ),
        (
            pd.DataFrame({"a": [1, 0, 1]}),
            2,
            pd.DataFrame({"a": pd.Series([np.nan, np.nan, 1, 0, 1], dtype="float64")}),
        ),
    ],
)
def test_pad_with_nans(data, num_to_pad, expected):
    padded = pad_with_nans(data, num_to_pad)
    _check_equality(padded, expected)


def test_pad_with_nans_with_series_name():
    name = "data to pad"
    data = pd.Series([1, 2, 3], name=name)
    padded = pad_with_nans(data, 1)
    _check_equality(padded, pd.Series([np.nan, 1, 2, 3], name=name, dtype="float64"))


def test_rename_column_names_to_numeric():
    X = pd.DataFrame(np.array([[1, 2], [3, 4]]))
    X.ww.init()
    pd.testing.assert_frame_equal(_rename_column_names_to_numeric(X), pd.DataFrame(X))

    X = pd.DataFrame({"<>": [1, 2], ">>": [2, 4]})
    X.ww.init()
    pd.testing.assert_frame_equal(
        _rename_column_names_to_numeric(X),
        pd.DataFrame({0: [1, 2], 1: [2, 4]}),
    )

    X.ww.init(logical_types={"<>": "categorical", ">>": "categorical"})
    X_renamed = _rename_column_names_to_numeric(X)
    X_expected = pd.DataFrame(
        {
            0: pd.Series([1, 2], dtype="category"),
            1: pd.Series([2, 4], dtype="category"),
        },
    )
    pd.testing.assert_frame_equal(X_renamed, X_expected)


@pytest.mark.parametrize(
    "file_name,format,interactive",
    [
        ("test_plot", "png", False),
        ("test_plot.png", "png", False),
        ("test_plot.", "png", False),
        ("test_plot.png", "jpeg", False),
    ],
)
def test_save_plotly_static_default_format(
    in_container_arm64,
    file_name,
    format,
    interactive,
    fitted_decision_tree_classification_pipeline,
    tmpdir,
):
    pipeline = fitted_decision_tree_classification_pipeline
    feat_fig_ = pipeline.graph_feature_importance()

    filepath = os.path.join(str(tmpdir), f"{file_name}")
    no_output_ = save_plot(
        fig=feat_fig_,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=False,
    )
    output_ = save_plot(
        fig=feat_fig_,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=True,
    )

    assert not no_output_
    assert os.path.exists(output_)
    assert isinstance(output_, str)
    assert os.path.basename(output_) == "test_plot.png"


@pytest.mark.parametrize("file_name,format,interactive", [("test_plot", "jpeg", False)])
def test_save_plotly_static_different_format(
    file_name,
    format,
    interactive,
    fitted_decision_tree_classification_pipeline,
    tmpdir,
):
    feat_fig_ = fitted_decision_tree_classification_pipeline.graph_feature_importance()

    filepath = os.path.join(str(tmpdir), f"{file_name}")
    no_output_ = save_plot(
        fig=feat_fig_,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=False,
    )
    output_ = save_plot(
        fig=feat_fig_,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=True,
    )

    assert not no_output_
    assert os.path.exists(output_)
    assert isinstance(output_, str)
    assert os.path.basename(output_) == "test_plot.jpeg"


@pytest.mark.parametrize("file_name,format,interactive", [(None, "jpeg", False)])
def test_save_plotly_static_no_filepath(
    file_name,
    format,
    interactive,
    fitted_decision_tree_classification_pipeline,
    tmpdir,
):
    feat_fig_ = fitted_decision_tree_classification_pipeline.graph_feature_importance()

    filepath = os.path.join(str(tmpdir), f"{file_name}") if file_name else None
    output_ = save_plot(
        fig=feat_fig_,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=True,
    )

    assert os.path.exists(output_)
    assert isinstance(output_, str)
    assert os.path.basename(output_) == "test_plot.jpeg"
    os.remove("test_plot.jpeg")


@pytest.mark.parametrize(
    "file_name,format,interactive",
    [
        ("test_plot", "html", True),
        ("test_plot.png", "html", True),
        ("test_plot.", "html", True),
        ("test_plot.png", "jpeg", True),
        ("test_plot", None, True),
        ("test_plot.html", None, True),
    ],
)
def test_save_plotly_interactive(
    file_name,
    format,
    interactive,
    fitted_decision_tree_classification_pipeline,
    tmpdir,
):
    feat_fig_ = fitted_decision_tree_classification_pipeline.graph_feature_importance()

    filepath = os.path.join(str(tmpdir), f"{file_name}") if file_name else None
    no_output_ = save_plot(
        fig=feat_fig_,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=False,
    )
    output_ = save_plot(
        fig=feat_fig_,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=True,
    )

    assert not no_output_
    assert os.path.exists(output_)
    assert isinstance(output_, str)
    assert os.path.basename(output_) == "test_plot.html"


@pytest.mark.parametrize(
    "file_name,format,interactive",
    [
        ("test_plot", "png", False),
        ("test_plot.png", "png", False),
        ("test_plot.", "png", False),
    ],
)
def test_save_graphviz_default_format(
    file_name,
    format,
    interactive,
    fitted_tree_estimators,
    tmpdir,
):
    est_class, _ = fitted_tree_estimators
    src = visualize_decision_tree(estimator=est_class, filled=True, max_depth=3)

    filepath = os.path.join(str(tmpdir), f"{file_name}") if file_name else None
    no_output_ = save_plot(
        fig=src,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=False,
    )
    output_ = save_plot(
        fig=src,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=True,
    )

    assert not no_output_
    assert os.path.exists(output_)
    assert isinstance(output_, str)
    assert os.path.basename(output_) == "test_plot.png"


@pytest.mark.parametrize("file_name,format,interactive", [("test_plot", "jpeg", False)])
def test_save_graphviz_different_format(
    file_name,
    format,
    interactive,
    fitted_tree_estimators,
    tmpdir,
):
    est_class, _ = fitted_tree_estimators
    src = visualize_decision_tree(estimator=est_class, filled=True, max_depth=3)

    filepath = os.path.join(str(tmpdir), f"{file_name}")
    no_output_ = save_plot(
        fig=src,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=False,
    )
    output_ = save_plot(
        fig=src,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=True,
    )

    assert not no_output_
    assert os.path.exists(output_)
    assert isinstance(output_, str)
    assert os.path.basename(output_) == "test_plot.png"


@pytest.mark.parametrize(
    "file_name,format,interactive",
    [("Output/in_folder_plot", "jpeg", True)],
)
def test_save_graphviz_invalid_filepath(
    file_name,
    format,
    interactive,
    fitted_tree_estimators,
    tmpdir,
):
    est_class, _ = fitted_tree_estimators
    src = visualize_decision_tree(estimator=est_class, filled=True, max_depth=3)

    filepath = f"{file_name}.{format}"

    with pytest.raises(ValueError, match="Specified filepath is not writeable"):
        save_plot(
            fig=src,
            filepath=filepath,
            format=format,
            interactive=interactive,
            return_filepath=False,
        )


@pytest.mark.parametrize(
    "file_name,format,interactive",
    [("example_plot", None, False), ("example_plot", "png", False)],
)
def test_save_graphviz_different_filename_output(
    file_name,
    format,
    interactive,
    fitted_tree_estimators,
    tmpdir,
):
    est_class, _ = fitted_tree_estimators
    src = visualize_decision_tree(estimator=est_class, filled=True, max_depth=3)

    filepath = os.path.join(str(tmpdir), f"{file_name}")
    no_output_ = save_plot(
        fig=src,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=False,
    )
    output_ = save_plot(
        fig=src,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=True,
    )

    assert not no_output_
    assert os.path.exists(output_)
    assert isinstance(output_, str)
    assert os.path.basename(output_) == "example_plot.png"


@pytest.mark.parametrize(
    "file_name,format,interactive",
    [
        ("test_plot", "png", False),
        ("test_plot.png", "png", False),
        ("test_plot.", "png", False),
        ("test_plot.png", "jpeg", False),
    ],
)
def test_save_matplotlib_default_format(
    file_name,
    format,
    interactive,
    fitted_tree_estimators,
    tmpdir,
):
    from matplotlib import pyplot as plt

    def setup_plt():
        fig_ = plt.figure(figsize=(4.5, 4.5))
        plt.plot(range(5))
        return fig_

    fig = setup_plt()
    filepath = os.path.join(str(tmpdir), f"{file_name}")
    no_output_ = save_plot(
        fig=fig,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=False,
    )
    output_ = save_plot(
        fig=fig,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=True,
    )

    assert not no_output_
    assert os.path.exists(output_)
    assert isinstance(output_, str)
    assert os.path.basename(output_) == "test_plot.png"


@pytest.mark.parametrize(
    "file_name,format,interactive",
    [
        ("test_plot", "png", False),
        ("test_plot.png", "png", False),
        ("test_plot.", "png", False),
        ("test_plot.png", "jpeg", False),
    ],
)
def test_save_seaborn_default_format(
    file_name,
    format,
    interactive,
    fitted_tree_estimators,
    tmpdir,
):
    import seaborn as sns

    def setup_plt():
        data_ = [0, 1, 2, 3, 4]
        fig = sns.scatterplot(data=data_)
        return fig

    fig = setup_plt()
    filepath = os.path.join(str(tmpdir), f"{file_name}")
    no_output_ = save_plot(
        fig=fig,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=False,
    )
    output_ = save_plot(
        fig=fig,
        filepath=filepath,
        format=format,
        interactive=interactive,
        return_filepath=True,
    )

    assert not no_output_
    assert os.path.exists(output_)
    assert isinstance(output_, str)
    assert os.path.basename(output_) == "test_plot.png"


def test_deprecate_arg():
    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter("always")
        assert deprecate_arg("foo", "bar", None, 5) == 5
        assert not warn

    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter("always")
        assert deprecate_arg("foo", "bar", 4, 7) == 4
        assert len(warn) == 1
        assert str(warn[0].message).startswith(
            "Argument 'foo' has been deprecated in favor of 'bar'",
        )


def test_contains_all_ts_parameters():
    is_valid, msg = contains_all_ts_parameters(
        {"time_index": "date", "max_delay": 1, "forecast_horizon": 3, "gap": 7},
    )
    assert is_valid and not msg

    is_valid, msg = contains_all_ts_parameters(
        {"time_index": None, "max_delay": 1, "forecast_horizon": 3, "gap": 7},
    )
    assert not is_valid and msg

    is_valid, msg = contains_all_ts_parameters({"time_index": "date"})

    assert not is_valid and msg


def test_are_ts_parameters_valid():
    result = are_ts_parameters_valid_for_split(
        gap=1,
        max_delay=8,
        forecast_horizon=3,
        n_obs=20,
        n_splits=3,
    )
    assert not result.is_valid and result.msg

    result = are_ts_parameters_valid_for_split(
        gap=1,
        max_delay=6,
        forecast_horizon=3,
        n_obs=20,
        n_splits=3,
    )
    assert result.is_valid and not result.msg

    result = are_ts_parameters_valid_for_split(
        gap=1,
        max_delay=8,
        forecast_horizon=3,
        n_obs=200,
        n_splits=3,
    )
    assert result.is_valid and not result.msg


@pytest.mark.parametrize(
    "gap,reset_index,freq",
    [
        (0, False, "1D"),
        (0, True, "3D"),
        (1, False, "1D"),
        (1, True, "1D"),
        (5, False, "1D"),
        (5, True, "1D"),
        (5, False, None),
    ],
)
def test_noninferrable_data(gap, reset_index, freq):
    date_range_ = pd.date_range("1/1/21", freq=freq, periods=100)
    training_date_range = date_range_[:80]
    if freq is None:
        training_date_range = pd.DatetimeIndex(["12/12/1984"]).append(date_range_[1:])
    testing_date_range = date_range_[80 + gap : 85 + gap]

    X_train = pd.DataFrame(training_date_range, columns=["date"])
    X = pd.DataFrame(testing_date_range, columns=["date"])

    if not reset_index:
        X.index = [i for i in range(80, 85)]

    problem_config = {
        "max_delay": 0,
        "forecast_horizon": 1,
        "time_index": "date",
        "gap": gap,
    }

    assert are_datasets_separated_by_gap_time_index(X_train, X, problem_config)


@pytest.mark.parametrize("gap", [0, 1, 5])
@pytest.mark.parametrize("forecast_horizon", [1, 5, 10])
@pytest.mark.parametrize("length_or_freq", ["length", "freq"])
def test_time_series_pipeline_validates_holdout_data(
    length_or_freq,
    forecast_horizon,
    gap,
    ts_data,
):
    X, _, y = ts_data()
    problem_config = {
        "time_index": "date",
        "gap": gap,
        "max_delay": 2,
        "forecast_horizon": forecast_horizon,
    }
    TRAIN_LENGTH = 15
    X_train = X.iloc[:TRAIN_LENGTH]

    if length_or_freq == "length":
        X = X.iloc[TRAIN_LENGTH + gap : TRAIN_LENGTH + gap + forecast_horizon + 2]
    elif length_or_freq == "freq":
        dates = pd.date_range("2020-10-16", periods=16)
        X = X.iloc[TRAIN_LENGTH + gap : TRAIN_LENGTH + gap + forecast_horizon]
        X["date"] = dates[gap + 1 : gap + 1 + len(X)]

    length_error = (
        f"Holdout data X must have {forecast_horizon} rows (value of forecast horizon) "
        f"Data received - Length X: {len(X)}"
    )
    gap_error = (
        f"The first value indicated by the column date needs to start {gap + 1} "
        f"units ahead of the training data. "
        f"X value start: {X['date'].iloc[0]}, X_train value end {X_train['date'].iloc[-1]}."
    )

    result = validate_holdout_datasets(X, X_train, problem_config)

    assert not result.is_valid
    if length_or_freq == "length":
        assert result.error_messages[0] == length_error
        assert result.error_codes[0] == ValidationErrorCode.INVALID_HOLDOUT_LENGTH
    else:
        assert result.error_messages[0] == gap_error
        assert (
            result.error_codes[0] == ValidationErrorCode.INVALID_HOLDOUT_GAP_SEPARATION
        )


def test_year_start_separated_by_gap():
    X = pd.DataFrame(
        {
            "time_index": pd.Series(
                pd.date_range("1960-01-01", freq="AS-JAN", periods=35),
            ),
        },
    )
    train = X.iloc[:30]
    test = X.iloc[32:36]
    assert are_datasets_separated_by_gap_time_index(
        train,
        test,
        {"time_index": "time_index", "gap": 2},
    )


def test_is_categorical_actually_boolean():
    X = pd.DataFrame(
        {
            "categorical": ["a", "b", "c"],
            "boolean_categorical": [True, False, None],
            "boolean": [True, False, True],
        },
    )

    assert not is_categorical_actually_boolean(X, "categorical")
    assert is_categorical_actually_boolean(X, "boolean_categorical")
    assert not is_categorical_actually_boolean(X, "boolean")


@pytest.mark.parametrize("X_num_time_columns", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "X_has_time_index",
    ["X_has_time_index", "X_doesnt_have_time_index"],
)
@pytest.mark.parametrize(
    "y_has_time_index",
    ["y_has_time_index", "y_doesnt_have_time_index"],
)
@pytest.mark.parametrize(
    "time_index_specified",
    [
        "time_index_is_specified",
        "time_index_not_specified",
        "time_index_is_specified_but_wrong",
    ],
)
def test_get_time_index(
    ts_data,
    X_num_time_columns,
    X_has_time_index,
    y_has_time_index,
    time_index_specified,
):
    X, _, y = ts_data()
    time_index_col_name = "date"
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)

    # Modify time series data to match testing conditions
    if X_has_time_index == "X_doesnt_have_time_index":
        X = X.ww.reset_index(drop=True)
    if y_has_time_index == "y_doesnt_have_time_index":
        y = y.reset_index(drop=True)
    if X_num_time_columns == 0:
        X = X.ww.drop(columns=[time_index_col_name])
    elif X_num_time_columns > 1:
        for addn_col in range(X_num_time_columns - 1):
            X.ww[time_index_col_name + str(addn_col + 1)] = X.ww[time_index_col_name]
    time_index = {
        "time_index_is_specified": "date",
        "time_index_not_specified": None,
        "time_index_is_specified_but_wrong": "d4t3s",
    }[time_index_specified]

    err_msg = None
    # The time series data has no time data
    if (
        X_num_time_columns == 0
        and X_has_time_index == "X_doesnt_have_time_index"
        and y_has_time_index == "y_doesnt_have_time_index"
    ):
        err_msg = "There are no Datetime features in the feature data and neither the feature nor the target data have a DateTime index."

    # The time series data has too much time data
    elif (
        X_num_time_columns > 1
        and time_index_specified == "time_index_not_specified"
        and y_has_time_index == "y_doesnt_have_time_index"
        and X_has_time_index != "X_has_time_index"
    ):
        err_msg = "Too many Datetime features provided in data but no time_index column specified during __init__."

    # If the wrong time_index column is specified with multiple datetime columns
    elif (
        time_index_specified == "time_index_is_specified_but_wrong"
        and X_num_time_columns > 1
        and X_has_time_index != "X_has_time_index"
        and y_has_time_index != "y_has_time_index"
    ):
        err_msg = "Too many Datetime features provided in data and provided time_index column d4t3s not present in data."

    if err_msg is not None:
        with pytest.raises(
            ValueError,
            match=err_msg,
        ):
            get_time_index(X, y, time_index)
    else:
        idx = get_time_index(X, y, time_index)
        assert isinstance(idx, pd.DatetimeIndex)


def test_get_time_index_maintains_freq():
    idx = pd.DatetimeIndex(
        [
            "1992-01-01T00:00:00Z",
            "1992-02-01T00:00:00Z",
            "1992-03-01T00:00:00Z",
            "1992-04-01T00:00:00Z",
            "1992-05-01T00:00:00Z",
            "1992-06-01T00:00:00Z",
            "1992-07-01T00:00:00Z",
            "1992-08-01T00:00:00Z",
            "1992-09-01T00:00:00Z",
            "1992-10-01T00:00:00Z",
            "1992-11-01T00:00:00Z",
            "1992-12-01T00:00:00Z",
            "1993-01-01T00:00:00Z",
            "1993-02-01T00:00:00Z",
        ],
    )
    X = pd.DataFrame(index=idx)
    y = pd.Series(range(len(idx)), idx)
    X.ww.init()
    y.ww.init()
    time_idx = get_time_index(X, y, None)
    assert X.index.equals(time_idx)
    assert y.index.equals(time_idx)
    assert time_idx.freq is not None

    X = pd.DataFrame({"date": idx})
    y = pd.Series(range(len(idx)))
    X.ww.init()
    y.ww.init()
    time_idx = get_time_index(X, y, None)
    assert time_idx.freq is not None
