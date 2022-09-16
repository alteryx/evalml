from unittest.mock import patch

import featuretools as ft
import pandas as pd
import pytest
import woodwork as ww
from featuretools.feature_base import IdentityFeature
from pandas.testing import assert_frame_equal
from woodwork.logical_types import Boolean, Categorical, Datetime, Double, Integer

from evalml.demos import load_diabetes
from evalml.pipelines.components import DFSTransformer


def test_index_errors(X_y_binary):
    with pytest.raises(TypeError, match="Index provided must be string"):
        DFSTransformer(index=0)

    with pytest.raises(TypeError, match="Index provided must be string"):
        DFSTransformer(index=None)


def test_numeric_columns(X_y_multi):
    X, y = X_y_multi
    X_pd = pd.DataFrame(X)

    feature = DFSTransformer()
    feature.fit(X_pd, y)
    feature.transform(X_pd)


@patch("evalml.pipelines.components.transformers.preprocessing.featuretools.dfs")
@patch(
    "evalml.pipelines.components.transformers.preprocessing.featuretools.calculate_feature_matrix",
)
def test_featuretools_index(mock_calculate_feature_matrix, mock_dfs, X_y_multi):
    X, y = X_y_multi
    X_pd = pd.DataFrame(X)
    X_new_index = X_pd.copy()
    index = [i for i in range(len(X))]
    new_index = [i * 2 for i in index]
    X_new_index["index"] = new_index
    mock_calculate_feature_matrix.return_value = pd.DataFrame({})

    # check if _make_entity_set keeps the intended index
    feature = DFSTransformer()
    feature.fit(X_new_index)
    feature.transform(X_new_index)
    arg_es = mock_dfs.call_args[1]["entityset"].dataframes[0].index
    arg_tr = mock_calculate_feature_matrix.call_args[1]["entityset"].dataframes[0].index
    assert arg_es.to_list() == new_index
    assert arg_tr.to_list() == new_index

    # check if _make_entity_set fills in the proper index values
    feature.fit(X_pd)
    feature.transform(X_pd)
    arg_es = mock_dfs.call_args[1]["entityset"].dataframes[0].index
    arg_tr = mock_calculate_feature_matrix.call_args[1]["entityset"].dataframes[0].index
    assert arg_es.to_list() == index
    assert arg_tr.to_list() == index


def test_transform(X_y_binary, X_y_multi, X_y_regression):
    datasets = locals()
    for dataset in datasets.values():
        X, y = dataset
        X = pd.DataFrame(X)  # Drop ww information since setting column types fails
        X.columns = X.columns.astype(str)
        es = ft.EntitySet()
        es = es.add_dataframe(
            dataframe_name="X",
            dataframe=X,
            index="index",
            make_index=True,
        )
        feature_matrix, features = ft.dfs(entityset=es, target_dataframe_name="X")

        feature = DFSTransformer()
        feature.fit(X)
        X_t = feature.transform(X)

        assert_frame_equal(feature_matrix, X_t)
        assert features == feature.features

        feature.fit(X, y)
        feature.transform(X)

        X.ww.init(logical_types={col: "double" for col in X.columns})
        feature.fit(X)
        feature.transform(X)


def test_transform_subset(X_y_binary, X_y_multi, X_y_regression):
    datasets = locals()
    for dataset in datasets.values():
        X, y = dataset
        X_pd = pd.DataFrame(X)
        X_pd.columns = X_pd.columns.astype(str)
        X_fit = X_pd.iloc[: len(X) // 3]
        X_transform = X_pd.iloc[len(X) // 3 :]

        es = ft.EntitySet()
        es = es.add_dataframe(
            dataframe_name="X",
            dataframe=X_transform,
            index="index",
            make_index=True,
        )
        feature_matrix, features = ft.dfs(entityset=es, target_dataframe_name="X")

        feature = DFSTransformer()
        feature.fit(X_fit)
        X_t = feature.transform(X_transform)

        assert_frame_equal(feature_matrix, X_t)


@pytest.mark.parametrize(
    "X_df",
    [
        pd.DataFrame(
            pd.to_datetime(["20190902", "20200519", "20190607"], format="%Y%m%d"),
        ),
        pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64")),
        pd.DataFrame(pd.Series([1.0, 2.0, 3.0], dtype="float")),
        pd.DataFrame(pd.Series(["a", "b", "a"], dtype="category")),
    ],
)
def test_ft_woodwork_custom_overrides_returned_by_components(X_df):
    y = pd.Series([1, 2, 1])
    override_types = [Integer, Double, Categorical, Datetime, Boolean]
    for logical_type in override_types:
        try:
            X = X_df.copy()
            X.ww.init(logical_types={0: logical_type})
        except (ww.exceptions.TypeConversionError, ValueError):
            continue
        if X.loc[:, 0].isna().all():
            # Casting the fourth dataframe to datetime will produce all NaNs
            continue

        dft = DFSTransformer()
        dft.fit(X, y)
        transformed = dft.transform(X, y)
        assert isinstance(transformed, pd.DataFrame)
        if logical_type == Datetime:
            assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
                "DAY(0)": Categorical,
                "MONTH(0)": Categorical,
                "WEEKDAY(0)": Categorical,
                "YEAR(0)": Categorical,
            }
        else:
            assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
                "0": logical_type,
            }


@patch("evalml.pipelines.components.transformers.preprocessing.featuretools.dfs")
def test_dfs_sets_max_depth_1(mock_dfs, X_y_multi):
    X, y = X_y_multi
    X_pd = pd.DataFrame(X)

    feature = DFSTransformer()
    feature.fit(X_pd, y)
    _, kwargs = mock_dfs.call_args
    assert kwargs["max_depth"] == 1


@patch("evalml.pipelines.components.transformers.preprocessing.featuretools.dfs")
def test_dfs_with_serialized_features(mock_dfs, X_y_binary):
    X, y = X_y_binary
    X_pd = pd.DataFrame(X)
    X_pd.columns = X_pd.columns.astype(str)

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X_pd,
        index="index",
        make_index=True,
    )
    feature_matrix, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
    )

    dfs = DFSTransformer(features=features)
    dfs.fit(X_pd)  # no-op
    assert not mock_dfs.called

    X_t = dfs.transform(X_pd)
    assert_frame_equal(feature_matrix, X_t)
    assert features == dfs.features


@patch("evalml.pipelines.components.transformers.preprocessing.featuretools.dfs")
@patch(
    "evalml.pipelines.components.transformers.preprocessing.featuretools.calculate_feature_matrix",
)
def test_dfs_skip_transform(mock_calculate_feature_matrix, mock_dfs, X_y_binary):
    X, y = X_y_binary
    X_pd = pd.DataFrame(X)
    X_pd.columns = X_pd.columns.astype(str)
    X_fit = X_pd.iloc[: len(X) // 3]
    X_transform = X_pd.iloc[len(X) // 3 :]

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X_transform,
        index="index",
        make_index=True,
    )
    feature_matrix, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
    )
    features = list(filter(lambda f: not isinstance(f, IdentityFeature), features))
    dfs = DFSTransformer(features=features)
    dfs.fit(X_fit)  # no-op
    X_t = dfs.transform(
        feature_matrix,
    )  # no-op as well, feature_matrix contains features already
    assert not mock_dfs.called
    assert not mock_calculate_feature_matrix.called

    assert_frame_equal(feature_matrix, X_t)
    assert features == dfs.features


@patch("evalml.pipelines.components.transformers.preprocessing.featuretools.dfs")
def test_dfs_does_not_skip_transform_with_non_identity_feature(mock_dfs, X_y_binary):
    X, y = X_y_binary
    X_pd = pd.DataFrame(X)
    X_pd.columns = X_pd.columns.astype(str)
    X_fit = X_pd.iloc[: len(X) // 3]
    X_transform = X_pd.iloc[len(X) // 3 :]

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X_transform,
        index="index",
        make_index=True,
    )
    feature_matrix, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
    )

    non_identity_features = list(
        filter(lambda feature: not isinstance(feature, IdentityFeature), features),
    )
    dfs = DFSTransformer(features=non_identity_features)
    dfs.fit(X_fit)  # no-op
    X_t = dfs.transform(X_pd)  # calculate_feature matrix is called
    assert not mock_dfs.called

    # assert that all non-identity features are calculated
    for col in X_t.columns:
        assert "ABSOLUTE" in col


@patch("evalml.pipelines.components.transformers.preprocessing.featuretools.dfs")
def test_dfs_missing_feature_column(mock_dfs, X_y_binary):
    X, y = X_y_binary
    X_pd = pd.DataFrame(X)
    X_pd.columns = X_pd.columns.astype(str)
    X_fit = X_pd.iloc[: len(X) // 3]
    X_transform = X_pd.iloc[len(X) // 3 :]

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X_transform,
        index="index",
        make_index=True,
    )
    feature_matrix, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
    )

    dfs = DFSTransformer(features=features)
    dfs.fit(X_fit)  # no-op
    X_pd = X_pd.drop("1", axis=1)
    X_t = dfs.transform(X_pd)  # calculate_feature matrix is called
    assert not mock_dfs.called

    assert "1" not in list(X_t.columns)
    assert "ABSOLUTE(1)" not in list(X_t.columns)

    for col in X_pd.columns:
        assert col in list(X_t.columns)
        assert f"ABSOLUTE({col})" in list(X_t.columns)


def test_transform_identity_and_non_identity():
    X, y = load_diabetes()
    del X.ww

    X_fit = X.iloc[: X.shape[0] // 2]

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X_fit,
        index="index",
        make_index=True,
    )
    feature_matrix, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
    )

    dfs = DFSTransformer(features=features)
    dfs.fit(X_fit)
    X_t = dfs.transform(feature_matrix)

    pd.testing.assert_frame_equal(X_t, feature_matrix)


def test_dfs_multi_input_primitive(X_y_binary):
    X, y = X_y_binary
    X_pd = pd.DataFrame(X)
    X_pd.columns = X_pd.columns.astype(str)
    X_fit = X_pd.iloc[: len(X) // 3]
    X_transform = X_pd.iloc[len(X) // 3 :]

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X_transform,
        index="index",
        make_index=True,
    )
    feature_matrix, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["divide_numeric"],
    )  # divide_numeric is a primitive that generates features with 2 input columns

    dfs = DFSTransformer(features=features)
    dfs.fit(X_fit)

    X_t = dfs.transform(X_transform)  # transform case
    assert_frame_equal(feature_matrix, X_t)
    assert features == dfs.features

    X_t = dfs.transform(feature_matrix)  # skip transform case
    assert_frame_equal(feature_matrix, X_t)
    assert features == dfs.features

    X_transform = X_transform.drop("1", axis=1)  # missing input case
    X_t = dfs.transform(X_transform)

    excluded_cols = ["1"]
    for i in range(20):
        excluded_cols.append(f"1 / {i}")
    for col in excluded_cols:
        assert col not in X_t.columns
