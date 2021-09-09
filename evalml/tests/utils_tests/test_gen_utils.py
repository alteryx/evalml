import inspect
import os
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_understanding.graphs import visualize_decision_tree
from evalml.pipelines.components import ComponentBase
from evalml.utils.gen_utils import (
    SEED_BOUNDS,
    _rename_column_names_to_numeric,
    classproperty,
    convert_to_seconds,
    deprecate_arg,
    get_importable_subclasses,
    get_random_seed,
    import_or_raise,
    jupyter_check,
    pad_with_nans,
    save_plot,
)


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
        min_bound=None, max_bound=None
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
            ]
        )

    np.testing.assert_equal(
        get_random_seed_vec(min_bound=None, max_bound=None)(vals),
        make_expected_values(
            vals, min_bound=SEED_BOUNDS.min_bound, max_bound=SEED_BOUNDS.max_bound
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
                }
            ),
        ),
        (
            pd.DataFrame({"a": [4, 5, 6], "b": ["a", "b", "c"]}),
            1,
            pd.DataFrame(
                {
                    "a": pd.Series([np.nan, 4, 5, 6], dtype="float64"),
                    "b": [np.nan, "a", "b", "c"],
                }
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
    X = np.array([[1, 2], [3, 4]])
    pd.testing.assert_frame_equal(_rename_column_names_to_numeric(X), pd.DataFrame(X))

    X = pd.DataFrame({"<>": [1, 2], ">>": [2, 4]})
    pd.testing.assert_frame_equal(
        _rename_column_names_to_numeric(X), pd.DataFrame({0: [1, 2], 1: [2, 4]})
    )

    X.ww.init(logical_types={"<>": "categorical", ">>": "categorical"})
    X_renamed = _rename_column_names_to_numeric(X)
    X_expected = pd.DataFrame(
        {0: pd.Series([1, 2], dtype="category"), 1: pd.Series([2, 4], dtype="category")}
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
    file_name,
    format,
    interactive,
    decision_tree_classification_pipeline_class,
    tmpdir,
    has_minimal_dependencies,
):
    if not has_minimal_dependencies:
        pipeline = decision_tree_classification_pipeline_class
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
    decision_tree_classification_pipeline_class,
    tmpdir,
    has_minimal_dependencies,
):
    if not has_minimal_dependencies:
        pipeline = decision_tree_classification_pipeline_class
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
        assert os.path.basename(output_) == "test_plot.jpeg"


@pytest.mark.parametrize("file_name,format,interactive", [(None, "jpeg", False)])
def test_save_plotly_static_no_filepath(
    file_name,
    format,
    interactive,
    decision_tree_classification_pipeline_class,
    tmpdir,
    has_minimal_dependencies,
):
    if not has_minimal_dependencies:
        pipeline = decision_tree_classification_pipeline_class
        feat_fig_ = pipeline.graph_feature_importance()

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
    decision_tree_classification_pipeline_class,
    tmpdir,
    has_minimal_dependencies,
):
    if not has_minimal_dependencies:
        pipeline = decision_tree_classification_pipeline_class
        feat_fig_ = pipeline.graph_feature_importance()

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
    has_minimal_dependencies,
):
    if not has_minimal_dependencies:
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
    has_minimal_dependencies,
):
    if not has_minimal_dependencies:
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
    "file_name,format,interactive", [("Output/in_folder_plot", "jpeg", True)]
)
def test_save_graphviz_invalid_filepath(
    file_name,
    format,
    interactive,
    fitted_tree_estimators,
    tmpdir,
    has_minimal_dependencies,
):
    if not has_minimal_dependencies:
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
    has_minimal_dependencies,
):
    if not has_minimal_dependencies:
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
    file_name, format, interactive, fitted_tree_estimators, tmpdir
):
    plt = pytest.importorskip("matplotlib.pyplot")

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
    has_minimal_dependencies,
):
    sns = pytest.importorskip("seaborn")

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
            "Argument 'foo' has been deprecated in favor of 'bar'"
        )
