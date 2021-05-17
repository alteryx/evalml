import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import ComponentNotYetFittedError
from evalml.pipelines.components import (
    SMOTENCSampler,
    SMOTENSampler,
    SMOTESampler
)
from evalml.utils.woodwork_utils import infer_feature_types

im = pytest.importorskip('imblearn.over_sampling', reason='Skipping test because imbalanced-learn not installed')


@pytest.mark.parametrize("sampler", [SMOTESampler, SMOTENCSampler, SMOTENSampler])
def test_init(sampler):
    parameters = {
        "sampling_ratio": 0.5,
        "k_neighbors": 2,
        "n_jobs": -1
    }
    oversampler = sampler(**parameters)
    assert oversampler.parameters == parameters


@pytest.mark.parametrize("sampler", [SMOTESampler(sampling_ratio=1),
                                     SMOTENCSampler(sampling_ratio=1),
                                     SMOTENSampler(sampling_ratio=1)])
def test_none_y(sampler):
    X = pd.DataFrame({"a": [i for i in range(5)],
                      "b": [1 for i in range(5)]})
    X = infer_feature_types(X, feature_types={"a": "Categorical"})
    oversampler = sampler
    with pytest.raises(ValueError, match="y cannot be none"):
        oversampler.fit(X, None)
    with pytest.raises(ValueError, match="y cannot be none"):
        oversampler.fit_transform(X, None)
    oversampler.fit(X, pd.Series([0] * 4 + [1]))
    oversampler.transform(X, None)


@pytest.mark.parametrize("sampler", [SMOTESampler(sampling_ratio=1),
                                     SMOTENCSampler(sampling_ratio=1),
                                     SMOTENSampler(sampling_ratio=1)])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_no_oversample(data_type, sampler, make_data_type, X_y_binary):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    oversampler = sampler
    if oversampler.name == "SMOTENC Oversampler":
        X2 = infer_feature_types(X, feature_types={1: "Categorical"})
        if data_type == "ww":
            X2 = X2.set_types({0: "Categorical"})
        new_X, new_y = oversampler.fit_transform(X2, y)
    else:
        new_X, new_y = oversampler.fit_transform(X, y)

    if data_type == "ww":
        X = X.to_dataframe().values
        y = y.to_series().values
    elif data_type == "pd":
        X = X.values
        y = y.values

    np.testing.assert_equal(X, new_X.to_dataframe().values)
    np.testing.assert_equal(y, new_y.to_series().values)


@pytest.mark.parametrize("sampler", [SMOTESampler(sampling_ratio=1),
                                     SMOTENCSampler(sampling_ratio=1),
                                     SMOTENSampler(sampling_ratio=1)])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_oversample_imbalanced_binary(data_type, sampler, make_data_type):
    X = np.array([[i for i in range(1000)],
                  [i % 7 for i in range(1000)],
                  [0.3 * (i % 3) for i in range(1000)]]).T
    y = np.array([0] * 150 + [1] * 850)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    oversampler = sampler
    if oversampler.name == "SMOTENC Oversampler":
        X2 = infer_feature_types(X, feature_types={1: "Categorical"})
        if data_type == "ww":
            X2 = X2.set_types({0: "Categorical"})
        new_X, new_y = oversampler.fit_transform(X2, y)
    else:
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
    np.testing.assert_equal(None, transform_y)


@pytest.mark.parametrize("sampling_ratio", [0.2, 0.5])
@pytest.mark.parametrize("sampler", [SMOTESampler, SMOTENCSampler, SMOTENSampler])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_oversample_imbalanced_multiclass(data_type, sampler, sampling_ratio, make_data_type):
    X = np.array([[i for i in range(1000)],
                  [i % 7 for i in range(1000)],
                  [0.3 * (i % 3) for i in range(1000)]]).T
    y = np.array([0] * 800 + [1] * 100 + [2] * 100)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    X2 = X
    oversampler = sampler(sampling_ratio=sampling_ratio)
    if sampler.name == 'SMOTENC Oversampler':
        X2 = infer_feature_types(X, feature_types={0: "Categorical"})
        if data_type == "ww":
            X2 = X2.set_types({0: "Categorical"})
        oversampler = sampler(sampling_ratio=sampling_ratio)

    new_X, new_y = oversampler.fit_transform(X2, y)

    num_samples = [800, 800 * sampling_ratio, 800 * sampling_ratio]
    # check the lengths and sampled values are as we expect
    assert len(new_X) == sum(num_samples)
    assert len(new_y) == sum(num_samples)
    value_counts = new_y.to_series().value_counts()
    assert value_counts.values[1] == value_counts.values[2]
    np.testing.assert_equal(value_counts.values, np.array(num_samples))

    transform_X, transform_y = oversampler.transform(X2, y)

    if data_type == "ww":
        X = X.to_dataframe().values
        y = y.to_series().values
    elif data_type == "pd":
        X = X.values
        y = y.values

    np.testing.assert_equal(X, transform_X.to_dataframe().values)
    np.testing.assert_equal(None, transform_y)


@pytest.mark.parametrize("sampler", [SMOTESampler, SMOTENCSampler, SMOTENSampler])
def test_oversample_seed_same_outputs(sampler, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    y = pd.Series([0] * 90 + [1] * 10)

    samplers = []
    for seed in [0, 0, 1]:
        oversampler = sampler(sampling_ratio=1, random_seed=seed)
        if 'NC' in sampler.name:
            X = infer_feature_types(X, feature_types={1: "Categorical"})
            oversampler = sampler(sampling_ratio=1, random_seed=seed)
        samplers.append(oversampler)

    # iterate through different indices in samplers
    # in group 1, first two oversamplers in samplers should be equal
    # in group 2, calling same oversamplers twice should be equal
    # in group 3, last two oversamplers in samplers should be different
    for s1, s2 in [[0, 1], [1, 1], [1, 2]]:
        X1, y1 = samplers[s1].fit_transform(X, y)
        X2, y2 = samplers[s2].fit_transform(X, y)
        if s2 == 2 and sampler != SMOTENSampler:
            # group 3, SMOTENSampler performance doesn't change with different random states
            with pytest.raises(AssertionError):
                pd.testing.assert_frame_equal(X1.to_dataframe(), X2.to_dataframe())
        else:
            pd.testing.assert_frame_equal(X1.to_dataframe(), X2.to_dataframe())
        pd.testing.assert_series_equal(y1.to_series(), y2.to_series())


@pytest.mark.parametrize("component_sampler,imblearn_sampler",
                         [(SMOTESampler, im.SMOTE),
                          (SMOTENCSampler, im.SMOTENC),
                          (SMOTENSampler, im.SMOTEN)])
@pytest.mark.parametrize("problem_type", ['binary', 'multiclass'])
def test_samplers_perform_equally(problem_type, component_sampler, imblearn_sampler, X_y_binary, X_y_multi):
    if problem_type == 'binary':
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
    sampling_dic = {'sampling_ratio': sampling_ratio}
    X2 = X
    random_seed = 1
    if component_sampler != SMOTENCSampler:
        component = component_sampler(**sampling_dic, random_seed=random_seed)
        imb_sampler = imblearn_sampler(sampling_strategy=imb_learn_sampling_ratio, random_state=random_seed)
    else:
        X2 = infer_feature_types(X, feature_types={1: "Categorical", 2: "Categorical", 3: "Categorical", 4: "Categorical"})
        component = component_sampler(**sampling_dic, random_seed=random_seed)
        imb_sampler = imblearn_sampler(sampling_strategy=imb_learn_sampling_ratio, categorical_features=[1, 2, 3, 4], random_state=random_seed)

    X_com, y_com = component.fit_transform(X2, y)
    X_im, y_im = imb_sampler.fit_resample(X, y)

    np.testing.assert_equal(X_com.to_dataframe().values, X_im)
    np.testing.assert_equal(y_com.to_series().values, y_im)
    np.testing.assert_equal(sorted(y_im), expected_y)


def test_smotenc_categorical_features(X_y_binary):
    X, y = X_y_binary
    X_ww = infer_feature_types(X, feature_types={0: 'Categorical', 1: 'Categorical'})
    snc = SMOTENCSampler()
    X_out, y_out = snc.fit_transform(X_ww, y)
    assert snc.categorical_features == [0, 1]


def test_smotenc_output_shape(X_y_binary):
    X, y = X_y_binary
    y_imbalanced = pd.Series([0] * 90 + [1] * 10)
    X_ww = infer_feature_types(X, feature_types={0: 'Categorical', 1: 'Categorical'})
    snc = SMOTENCSampler()
    with pytest.raises(ComponentNotYetFittedError, match=f'You must fit SMOTENCSampler'):
        snc.transform(X_ww, y)
    # test sampling and no sampling
    for y_value in [y, y_imbalanced]:
        snc.fit(X_ww, y_value)
        X_out, y_out = snc.transform(X_ww, y_value)
        assert X_out.shape[1] == X_ww.shape[1]

        X_out, y_out = snc.fit_transform(X_ww, y)
        assert X_out.shape[1] == X_ww.shape[1]
