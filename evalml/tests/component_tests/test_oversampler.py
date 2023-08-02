import copy

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from imblearn import over_sampling as im
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.exceptions import ComponentNotYetFittedError
from evalml.pipelines.components import Oversampler
from evalml.utils.woodwork_utils import infer_feature_types


def test_init():
    parameters = {
        "sampling_ratio": 0.5,
        "k_neighbors_default": 2,
        "n_jobs": -1,
        "sampling_ratio_dict": None,
    }
    oversampler = Oversampler(**parameters)
    assert oversampler.parameters == parameters


def test_oversampler_raises_error_if_y_is_None():
    oversampler = Oversampler(sampling_ratio=1)
    X = pd.DataFrame({"a": [i for i in range(5)], "b": [1 for i in range(5)]})
    X = infer_feature_types(X, feature_types={"a": "Categorical"})
    with pytest.raises(ValueError, match="y cannot be None"):
        oversampler.fit(X, None)
    with pytest.raises(ValueError, match="y cannot be None"):
        oversampler.fit_transform(X, None)
    oversampler.fit(X, pd.Series([0] * 3 + [1] * 2))
    with pytest.raises(ValueError, match="y cannot be None"):
        oversampler.transform(X, None)


@pytest.mark.parametrize("categorical_columns", ["none", "all", "some"])
@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_sampler_selection(
    problem_type,
    categorical_columns,
    mock_imbalanced_data_X_y,
):
    from imblearn import over_sampling as im

    X, y = mock_imbalanced_data_X_y(problem_type, categorical_columns, "small")

    oversampler = Oversampler(sampling_ratio=1)
    assert oversampler.sampler is None
    oversampler.fit(X, y)

    if categorical_columns == "none":
        assert oversampler.sampler == im.SMOTE
    elif categorical_columns == "some":
        assert oversampler.sampler == im.SMOTENC
    else:
        assert oversampler.sampler == im.SMOTEN


@pytest.mark.parametrize("oversampler_type", ["numeric", "categorical"])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_no_oversample(data_type, oversampler_type, make_data_type, X_y_binary):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    oversampler = Oversampler(sampling_ratio=1)

    if oversampler_type == "categorical":
        X2 = infer_feature_types(X, feature_types={1: "Categorical"})
        if data_type == "ww":
            X2.ww.set_types({0: "Categorical"})
        new_X, new_y = oversampler.fit_transform(X2, y)
    else:
        new_X, new_y = oversampler.fit_transform(X, y)

    np.testing.assert_equal(X, new_X.values)
    np.testing.assert_equal(y, new_y.values)


@pytest.mark.parametrize("oversampler_type", ["numeric", "categorical"])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_oversample_imbalanced_binary(data_type, oversampler_type, make_data_type):
    X = np.array(
        [
            [i for i in range(1000)],
            [i % 7 for i in range(1000)],
            [0.3 * (i % 3) for i in range(1000)],
        ],
    ).T
    y = np.array([0] * 150 + [1] * 850)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    oversampler = Oversampler(sampling_ratio=1)
    if oversampler_type == "categorical":
        X = infer_feature_types(X, feature_types={1: "Categorical"})
        if data_type == "ww":
            X.ww.set_types({0: "Categorical"})
    fit_transformed_X, fit_transformed_y = oversampler.fit_transform(X, y)

    new_length = 1700
    assert len(fit_transformed_X) == new_length
    assert len(fit_transformed_y) == new_length
    value_counts = fit_transformed_y.value_counts()
    assert value_counts.values[0] == value_counts.values[1]
    pd.testing.assert_series_equal(
        value_counts,
        pd.Series([850, 850]),
        check_dtype=False,
        check_names=False,
    )

    oversampler = Oversampler(sampling_ratio=1)
    oversampler.fit(X, y)
    transformed_X, transformed_y = oversampler.transform(X, y)
    assert_frame_equal(transformed_X, fit_transformed_X)
    assert_series_equal(transformed_y, fit_transformed_y)


@pytest.mark.parametrize("sampling_ratio", [0.2, 0.5])
@pytest.mark.parametrize("oversampler_type", ["numeric", "categorical"])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_oversample_imbalanced_multiclass(
    data_type,
    oversampler_type,
    sampling_ratio,
    make_data_type,
):
    X = np.array(
        [
            [i for i in range(1000)],
            [i % 7 for i in range(1000)],
            [0.3 * (i % 3) for i in range(1000)],
        ],
    ).T
    y = np.array([0] * 800 + [1] * 100 + [2] * 100)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    def initalize_oversampler(X):
        oversampler = Oversampler(sampling_ratio=sampling_ratio)
        if oversampler_type == "categorical":
            X = infer_feature_types(X, feature_types={0: "Categorical"})
            if data_type == "ww":
                X.ww.set_types({0: "Categorical"})
            oversampler = Oversampler(sampling_ratio=sampling_ratio)
        return X, oversampler

    X, oversampler = initalize_oversampler(X)
    fit_transformed_X, fit_transformed_y = oversampler.fit_transform(X, y)

    num_samples = [800, 800 * sampling_ratio, 800 * sampling_ratio]
    # check the lengths and sampled values are as we expect
    assert len(fit_transformed_X) == sum(num_samples)
    assert len(fit_transformed_y) == sum(num_samples)
    value_counts = fit_transformed_y.value_counts()
    assert value_counts.values[1] == value_counts.values[2]
    np.testing.assert_equal(value_counts.values, np.array(num_samples))

    X, oversampler = initalize_oversampler(X)
    oversampler.fit(X, y)
    transformed_X, transformed_y = oversampler.transform(X, y)
    assert_frame_equal(transformed_X, fit_transformed_X)
    assert_series_equal(transformed_y, fit_transformed_y)


@pytest.mark.parametrize("sampler", ["SMOTE", "SMOTEN", "SMOTENC"])
def test_oversample_seed_same_outputs(sampler, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    y = pd.Series([0] * 90 + [1] * 10)

    samplers = []
    for seed in [0, 0, 1]:
        oversampler = Oversampler(sampling_ratio=1, random_seed=seed)
        if "N" in sampler:
            X = infer_feature_types(
                X,
                feature_types={0: "Categorical", 1: "Categorical"},
            )
            if sampler == "SMOTEN" and X.shape[1] > 2:
                X = X.drop([i for i in range(2, 20)], axis=1)
        samplers.append(oversampler)

    # iterate through different indices in samplers
    # in group 1, first two oversamplers in samplers should be equal
    # in group 2, calling same oversamplers twice should be equal
    # in group 3, last two oversamplers in samplers should be different
    for s1, s2 in [[0, 1], [1, 1], [1, 2]]:
        X1, y1 = samplers[s1].fit_transform(X, y)
        X2, y2 = samplers[s2].fit_transform(X, y)
        if s2 == 2 and sampler != "SMOTEN":
            # group 3, SMOTEN performance doesn't change with different random states
            with pytest.raises(AssertionError):
                pd.testing.assert_frame_equal(X1, X2)
        else:
            pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)


@pytest.mark.parametrize(
    "component_sampler",
    [
        "SMOTE",
        "SMOTENC",
        "SMOTEN",
    ],
)
@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_samplers_perform_equally(
    problem_type,
    component_sampler,
    X_y_binary,
    X_y_multi,
):
    from imblearn import over_sampling as im

    imblearn_sampler = getattr(im, component_sampler)
    if problem_type == "binary":
        X, _ = X_y_binary
        y = np.array([0] * 90 + [1] * 10)
        imb_learn_sampling_ratio = 0.5
        expected_y = np.array([0] * 90 + [1] * 45)
    else:
        X, _ = X_y_multi
        y = np.array([0] * 70 + [1] * 20 + [2] * 10)
        imb_learn_sampling_ratio = {0: 70, 1: 35, 2: 35}
        expected_y = np.array([0] * 70 + [1] * 35 + [2] * 35)
    sampling_ratio = 0.5
    sampling_dic = {"sampling_ratio": sampling_ratio}
    random_seed = 1
    if component_sampler != "SMOTE":
        X.ww.set_types(
            {
                0: "Categorical",
                1: "Categorical",
                2: "Categorical",
                3: "Categorical",
            },
        )
        if component_sampler == "SMOTEN":
            X = pd.DataFrame(X).drop([i for i in range(4, X.shape[1])], axis=1)
    component = Oversampler(**sampling_dic, random_seed=random_seed)
    if component_sampler == "SMOTENC":
        imb_sampler = imblearn_sampler(
            sampling_strategy=imb_learn_sampling_ratio,
            categorical_features=[0, 1, 2, 3],
            random_state=random_seed,
        )
    else:
        imb_sampler = imblearn_sampler(
            sampling_strategy=imb_learn_sampling_ratio,
            random_state=random_seed,
        )

    X_com, y_com = component.fit_transform(X, y)
    X_im, y_im = imb_sampler.fit_resample(X, y)

    np.testing.assert_equal(X_com.values, X_im)
    np.testing.assert_equal(y_com.values, y_im)
    np.testing.assert_equal(sorted(y_im), expected_y)


@pytest.mark.parametrize("use_nullable", [True, False])
def test_smoten_categorical_boolean(X_y_binary, im, use_nullable):
    X, y = X_y_binary
    bool_ltype = "BooleanNullable" if use_nullable else "Boolean"
    # Set the 0 column of float values to be Categorical type
    X.ww.set_types({0: "Categorical"})
    # Add a Boolean column into X
    X.ww[len(X.columns)] = ww.init_series(
        pd.Series([True, False] * 50),
        logical_type=bool_ltype,
    )
    X = X.ww.select(["Boolean", "BooleanNullable", "Categorical"])

    sn = Oversampler()
    _ = sn.fit_transform(X, y)
    assert sn.sampler == im.SMOTEN


@pytest.mark.parametrize("use_nullable_1", [True, False])
@pytest.mark.parametrize("use_nullable_2", [True, False])
def test_smotenc_boolean_numeric(X_y_binary, im, use_nullable_1, use_nullable_2):
    bool_ltype_1 = "BooleanNullable" if use_nullable_1 else "Boolean"
    bool_ltype_2 = "BooleanNullable" if use_nullable_2 else "Boolean"
    X, y = X_y_binary
    # Add two boolean columns into X
    X.ww[len(X.columns)] = ww.init_series(
        pd.Series([True, False] * 50),
        logical_type=bool_ltype_1,
    )
    X.ww[len(X.columns)] = ww.init_series(
        pd.Series([True, False] * 50),
        logical_type=bool_ltype_2,
    )
    snc = Oversampler()
    _, _ = snc.fit_transform(X, y)
    assert snc.sampler == im.SMOTENC


def test_smotenc_categorical_features(X_y_binary):
    X, y = X_y_binary
    X.ww.set_types({0: "Categorical", 1: "Categorical"})
    snc = Oversampler()
    _ = snc.fit_transform(X, y)
    assert snc.categorical_features == [0, 1]

    X[0] = pd.Series(["A", "B", "C", "A", "C"] * 20, dtype="category")
    X.ww.init()
    snc = Oversampler()
    _ = snc.fit_transform(X, y)
    assert snc.categorical_features == [0, 1]


def test_smotenc_category_features(X_y_binary):
    X, y = X_y_binary
    X.columns = X.columns.astype(str)
    X = pd.DataFrame(X).rename({0: "postal", 1: "country"}, axis="columns")
    X["2"] = [i % 2 for i in range(X.shape[0])]

    # Replace postal and country features with plausible postal/country codes
    X["postal"] = np.arange(10001, 10001 + X.shape[0])
    X["country"] = np.arange(20001, 20001 + X.shape[0])

    # Add different types of categorical data - boolean values, string categories, and postal codes
    X_ww = infer_feature_types(
        X,
        feature_types={
            "postal": "PostalCode",
            "country": "CountryCode",
            "2": "Boolean",
        },
    )
    X_ww.ww[str(len(X.columns))] = ww.init_series(
        pd.Series([True, False] * 50),
        logical_type="BooleanNullable",
    )
    snc = Oversampler()
    _ = snc.fit_transform(X_ww, y)
    assert snc.categorical_features == [20, 21, 2, len(X.columns)]


def test_smotenc_output_shape(X_y_binary):
    X, y = X_y_binary
    y_imbalanced = pd.Series([0] * 90 + [1] * 10)
    X_ww = infer_feature_types(X, feature_types={0: "Categorical", 1: "Categorical"})
    snc = Oversampler()
    with pytest.raises(ComponentNotYetFittedError, match="You must fit Oversampler"):
        snc.transform(X_ww, y)
    # test sampling and no sampling
    for y_value in [y, y_imbalanced]:
        snc.fit(X_ww, y_value)
        X_out, y_out = snc.transform(X_ww, y_value)
        assert X_out.shape[1] == X_ww.shape[1]

        X_out, y_out = snc.fit_transform(X_ww, y)
        assert X_out.shape[1] == X_ww.shape[1]
        assert y_out.shape[0] == X_out.shape[0]


@pytest.mark.parametrize(
    "sampling_ratio_dict,expected_dict_values",
    [
        ({0: 0.5, 1: 1}, {0: 425, 1: 850}),
        ({0: 0.1, 1: 1}, {0: 150, 1: 850}),
        ({0: 1, 1: 1}, {0: 850, 1: 850}),
        ({0: 0.5, 1: 0.1}, {0: 425, 1: 850}),
    ],
)
def test_oversampler_sampling_dict(sampling_ratio_dict, expected_dict_values):
    X = np.array(
        [
            [i for i in range(1000)],
            [i % 7 for i in range(1000)],
            [0.3 * (i % 3) for i in range(1000)],
        ],
    ).T
    X_ww = infer_feature_types(X, feature_types={0: "Categorical", 1: "Categorical"})
    y = np.array([0] * 150 + [1] * 850)
    oversampler = Oversampler(sampling_ratio_dict=sampling_ratio_dict, random_seed=12)
    new_X, new_y = oversampler.fit_transform(X_ww, y)

    assert len(new_X) == sum(expected_dict_values.values())
    assert new_y.value_counts().to_dict() == expected_dict_values
    assert oversampler.random_seed == 12


def test_oversampler_dictionary_overrides_ratio():
    X = np.array(
        [
            [i for i in range(1000)],
            [i % 7 for i in range(1000)],
            [0.3 * (i % 3) for i in range(1000)],
        ],
    ).T
    X_ww = infer_feature_types(X, feature_types={0: "Categorical", 1: "Categorical"})
    y = np.array([0] * 150 + [1] * 850)
    dictionary = {0: 0.5, 1: 1}
    expected_result = {0: 425, 1: 850}
    overs = Oversampler(sampling_ratio=0.1, sampling_ratio_dict=dictionary)
    new_X, new_y = overs.fit_transform(X_ww, y)

    assert len(new_X) == sum(expected_result.values())
    assert new_y.value_counts().to_dict() == expected_result


def test_oversampler_sampling_dict_strings():
    X = np.array(
        [
            [i for i in range(1000)],
            [i % 7 for i in range(1000)],
            [0.3 * (i % 3) for i in range(1000)],
        ],
    ).T
    X_ww = infer_feature_types(X, feature_types={0: "Categorical", 1: "Categorical"})
    y = np.array(["minority"] * 150 + ["majority"] * 850)
    dictionary = {"minority": 0.5, "majority": 1}
    expected_result = {"minority": 425, "majority": 850}
    overs = Oversampler(sampling_ratio_dict=dictionary)
    new_X, new_y = overs.fit_transform(X_ww, y)

    assert len(new_X) == sum(expected_result.values())
    assert new_y.value_counts().to_dict() == expected_result


@pytest.mark.parametrize(
    "minority,expected,fails",
    [(1, 0, True), (2, 1, False), (5, 4, False), (10, 5, False)],
)
def test_oversampler_sampling_k_neighbors(minority, expected, fails):
    X = np.array(
        [
            [i for i in range(1000)],
            [i % 7 for i in range(1000)],
            [0.3 * (i % 3) for i in range(1000)],
        ],
    ).T
    X_ww = infer_feature_types(X, feature_types={0: "Categorical", 1: "Categorical"})
    y = np.array(["minority"] * minority + ["majority"] * (1000 - minority))
    overs = Oversampler(k_neighbors_default=5)
    if fails:
        with pytest.raises(
            ValueError,
            match="Minority class needs more than 1 sample to use SMOTE",
        ):
            overs.fit_transform(X_ww, y)
        return
    overs.fit_transform(X_ww, y)
    assert overs._component_obj.k_neighbors == expected
    assert overs.parameters["k_neighbors"] == expected


def test_oversampler_copy(X_y_binary):
    X, y = X_y_binary

    oversampler = Oversampler()
    oversampler_copy = copy.deepcopy(oversampler)
    assert oversampler == oversampler_copy

    oversampler.fit(X, y)
    oversampler_fit_copy = copy.deepcopy(oversampler)
    assert oversampler == oversampler_fit_copy


@pytest.mark.parametrize(
    "nullable_y_ltype",
    ["IntegerNullable", "AgeNullable", "BooleanNullable"],
)
def test_oversampler_handle_nullable_types(
    nullable_type_test_data,
    nullable_type_target,
    nullable_y_ltype,
):
    X = nullable_type_test_data(has_nans=False)
    # Oversampler can only handle numeric and boolean columns
    X = X.ww.select(include=["numeric", "Boolean", "BooleanNullable", "category"])
    original_schema = X.ww.schema
    y = nullable_type_target(ltype=nullable_y_ltype, has_nans=False)

    oversampler = Oversampler(sampling_ratio=0.5)
    oversampler.fit(X.ww.copy(), y)
    X_t, y_t = oversampler.transform(X.ww.copy(), y)

    # Confirm oversampling happened by checking the length increased
    assert len(X_t) > len(X)
    assert len(y_t) > len(y)

    # Confirm the original types are maintained
    assert original_schema == X_t.ww.schema


@pytest.mark.parametrize(
    "nullable_y_ltype",
    ["IntegerNullable", "AgeNullable", "BooleanNullable"],
)
@pytest.mark.parametrize(
    "handle_incompatibility",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(raises=TypeError),
        ),
    ],
)
def test_oversampler_nullable_type_incompatibility(
    nullable_type_test_data,
    handle_incompatibility,
    nullable_y_ltype,
):
    """Testing that the nullable type incompatibility that caused us to add handling for Oversampler
    is still present in imblearn's SMOTE oversamplers. If this test is causing the test suite to fail
    because the code below no longer raises the expected ValueError, we should confirm that the nullable
    types now work for our use case and remove the nullable type handling logic from Oversampler.
    """
    X = nullable_type_test_data(has_nans=False)
    X = X.ww[
        ["categorical col", "int col nullable", "bool col nullable", "age col nullable"]
    ]
    y = ww.init_series(pd.Series([1, 0, 0, 1, 1] * 4), logical_type=nullable_y_ltype)

    if handle_incompatibility:
        evalml_oversampler = Oversampler(sampling_ratio=0.5)
        X, y = evalml_oversampler._handle_nullable_types(X, y)

    im_oversampler = im.SMOTENC(categorical_features=[0])
    im_oversampler.fit_resample(X, y)


@pytest.mark.parametrize(
    "handle_incompatibility",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(strict=True, raises=ValueError),
        ),
    ],
)
def test_oversampler_category_dtype_incompatibility(
    nullable_type_test_data,
    handle_incompatibility,
):
    """Testing that the sklearn error the SMOTENC oversampler produces with categorical columns
    is still present. If this test is causing the test suite to fail because the code
    below no longer raises the expected ValueError, we should confirm that the issue with categorical
    columns is truly gone and remove the logic to convert categorical columns to the object dtype.
    """
    X = nullable_type_test_data(has_nans=False)
    X = X.ww[["categorical col", "bool col"]]
    y = ww.init_series(pd.Series([1, 0, 0, 1, 1] * 4))

    if handle_incompatibility:
        X["categorical col"] = X["categorical col"].astype("object")

    im_oversampler = im.SMOTEN()
    im_oversampler.fit_resample(X, y)
