#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest

from evalml.tuners.skopt_tuner import SKOptTuner
from evalml.tuners.tuner import Tuner


def assert_params_almost_equal(a, b, decimal=7):
    """Given two sets of numeric/str parameter lists, assert numerics are approx equal and strs are equal"""
    def separate_numeric_and_str(values):
        def is_numeric(val):
            return isinstance(val, (int, float))

        def extract(vals, invert):
            return [el for el in vals if (invert ^ is_numeric(el))]

        return extract(values, False), extract(values, True)
    a_num, a_str = separate_numeric_and_str(a)
    b_num, b_str = separate_numeric_and_str(a)
    assert a_str == b_str
    np.testing.assert_almost_equal(a_num, b_num, decimal=decimal,
                                   err_msg="Numeric parameter values are not approximately equal")


random_state = 0


def test_tuner_base(test_space):
    with pytest.raises(TypeError):
        Tuner(test_space)


def test_skopt_tuner_basic(test_space, test_space_unicode):
    tuner = SKOptTuner(test_space, random_state=random_state)
    assert isinstance(tuner, Tuner)
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5, 8.442657485810175, 'option_c'])
    tuner.add(proposed_params, 0.5)

    tuner = SKOptTuner(test_space_unicode, random_state=random_state)
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5, 8.442657485810175, 'option_c 💩'])
    tuner.add(proposed_params, 0.5)


def test_skopt_tuner_space_types():
    tuner = SKOptTuner([(0, 10)], random_state=random_state)
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5.928446182250184])
    tuner.add(proposed_params, 0.5)

    tuner = SKOptTuner([(0, 10.0)], random_state=random_state)
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5.928446182250184])
    tuner.add(proposed_params, 0.5)


def test_skopt_tuner_invalid_space():
    with pytest.raises(TypeError):
        SKOptTuner(False)
    with pytest.raises(ValueError):
        SKOptTuner([(0)])
    with pytest.raises(ValueError):
        SKOptTuner(((0, 1)))
    with pytest.raises(ValueError):
        SKOptTuner([(0, 0)])


def test_skopt_tuner_invalid_parameters_score(test_space):
    tuner = SKOptTuner(test_space)
    with pytest.raises(TypeError):
        tuner.add(0, 0.5)
    with pytest.raises(ValueError):
        tuner.add([0, 1], 0.5)
    with pytest.raises(ValueError):
        tuner.add([0, 1, 2], 0.5)
    with pytest.raises(ValueError):
        tuner.add([0, 1, '2'], 0.5)
    with pytest.raises(ValueError):
        tuner.add([-1, 1, 'option_a'], 0.5)
    with pytest.raises(ValueError):
        tuner.add([0, -1, 'option_a'], 0.5)
    with pytest.raises(ValueError):
        tuner.add([0, 1, 'option_a', 3], 0.5)
    with pytest.raises(ValueError):
        tuner.add([np.nan, 1, 'option_a'], 0.5)
    with pytest.raises(ValueError):
        tuner.add([np.inf, 1, 'option_a'], 0.5)
    with pytest.raises(TypeError):
        tuner.add([None, 1, 'option_a'], 0.5)
    tuner.add((0, 1, 'option_a'), 0.5)
    tuner.add([0, 1, 'option_a'], np.nan)
    tuner.add([0, 1, 'option_a'], np.inf)
    tuner.add([0, 1, 'option_a'], None)
    tuner.propose()
    print(random_state)
