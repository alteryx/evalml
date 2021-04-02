import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import (
    SMOTENCSampler,
    SMOTENSampler,
    SMOTESampler
)

pytest.importorskip('imblearn.over_sampling', reason='Skipping test because imbalanced-learn not installed')


@pytest.mark.parametrize("sampler", [SMOTESampler, SMOTENCSampler, SMOTENSampler])
def test_init(sampler):
    parameters = {
        "sampling_strategy": 1.0,
        "k_neighbors": 2
    }
    if 'SMOTENC' in sampler.name:
        parameters['categorical_features'] = [0]
    oversampler = sampler(**parameters)
    assert oversampler.parameters == parameters


@pytest.mark.parametrize("sampler", [SMOTESampler(), SMOTENCSampler(categorical_features=[0]), SMOTENSampler()])
def test_none_y(sampler):
    X = pd.DataFrame([[i] for i in range(5)])
    oversampler = sampler
    with pytest.raises(ValueError, match="y cannot be none"):
        oversampler.fit(X, None)
    with pytest.raises(ValueError, match="y cannot be none"):
        oversampler.fit_transform(X, None)
    oversampler.fit(X, pd.Series([0] * 4 + [1]))
    oversampler.transform(X, None)


@pytest.mark.parametrize("sampler", [SMOTESampler(), SMOTENCSampler(categorical_features=[0]), SMOTENSampler()])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_no_oversample(data_type, sampler, make_data_type, X_y_binary):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    oversampler = sampler
    new_X, new_y = oversampler.fit_transform(X, y)

    if data_type == "ww":
        X = X.to_dataframe().values
        y = y.to_series().values
    elif data_type == "pd":
        X = X.values
        y = y.values

    np.testing.assert_equal(X, new_X.to_dataframe().values)
    np.testing.assert_equal(y, new_y.to_series().values)


@pytest.mark.parametrize("sampler", [SMOTESampler(), SMOTENCSampler(categorical_features=[1]), SMOTENSampler()])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_oversample_imbalanced_binary(data_type, sampler, make_data_type):
    X = np.array([[i for i in range(1000)],
                  [i % 7 for i in range(1000)],
                  [0.3 * (i % 3) for i in range(1000)]]).T
    y = np.array([0] * 150 + [1] * 850)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    oversampler = sampler
    new_X, new_y = oversampler.fit_transform(X, y)

    new_length = 1700
    assert len(new_X) == new_length
    assert len(new_y) == new_length
    value_counts = new_y.to_series().value_counts()
    assert value_counts.values[0] == value_counts.values[1]
    pd.testing.assert_series_equal(value_counts, pd.Series([850, 850]), check_dtype=False)

    transform_X, transform_y = oversampler.transform(X, y)

    if data_type == "ww":
        X = X.to_dataframe().values
        y = y.to_series().values
    elif data_type == "pd":
        X = X.values
        y = y.values

    np.testing.assert_equal(X, transform_X.to_dataframe().values)
    np.testing.assert_equal(y, transform_y.to_series().values)


@pytest.mark.parametrize("sampling_strategy", ['auto', {0: 800, 1: 300, 2: 300}])
@pytest.mark.parametrize("sampler", [SMOTESampler, SMOTENCSampler, SMOTENSampler])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_oversample_imbalanced_multiclass(data_type, sampler, sampling_strategy, make_data_type):
    X = np.array([[i for i in range(1000)],
                  [i % 7 for i in range(1000)],
                  [0.3 * (i % 3) for i in range(1000)]]).T
    y = np.array([0] * 800 + [1] * 100 + [2] * 100)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    oversampler = sampler(sampling_strategy=sampling_strategy)
    if 'NC' in sampler.name:
        oversampler = sampler(categorical_features=[1], sampling_strategy=sampling_strategy)

    new_X, new_y = oversampler.fit_transform(X, y)

    if sampling_strategy == 'auto':
        new_length = 2400
        num_samples = [800, 800, 800]
    else:
        new_length = sum(sampling_strategy.values())
        num_samples = [800, 300, 300]
    # check the lengths and sampled values are as we expect
    assert len(new_X) == new_length
    assert len(new_y) == new_length
    value_counts = new_y.to_series().value_counts()
    assert value_counts.values[1] == value_counts.values[2]
    np.testing.assert_equal(value_counts.values, np.array(num_samples))

    transform_X, transform_y = oversampler.transform(X, y)

    if data_type == "ww":
        X = X.to_dataframe().values
        y = y.to_series().values
    elif data_type == "pd":
        X = X.values
        y = y.values

    np.testing.assert_equal(X, transform_X.to_dataframe().values)
    np.testing.assert_equal(y, transform_y.to_series().values)
