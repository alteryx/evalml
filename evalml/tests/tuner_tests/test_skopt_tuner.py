#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unittest.mock import patch

import numpy as np
import pytest
from skopt.space import Integer, Real

from evalml.tuners import ParameterError, Tuner
from evalml.tuners.skopt_tuner import SKOptTuner

random_seed = 0


def test_tuner_init():
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class Tuner with abstract methods add, propose",
    ):
        Tuner({})


def test_skopt_tuner_init():
    with pytest.raises(
        ValueError,
        match="pipeline_hyperparameter_ranges must be a dict but is of type <class 'set'>",
    ):
        SKOptTuner({"My Component"})
    with pytest.raises(
        ValueError,
        match="pipeline_hyperparameter_ranges has invalid entry for My Component: True",
    ):
        SKOptTuner({"My Component": True})
    with pytest.raises(
        ValueError,
        match="pipeline_hyperparameter_ranges has invalid entry for My Component",
    ):
        SKOptTuner({"My Component": 0})
    with pytest.raises(
        ValueError,
        match="pipeline_hyperparameter_ranges has invalid entry for My Component",
    ):
        SKOptTuner({"My Component": None})
    with pytest.raises(
        ValueError,
        match="pipeline_hyperparameter_ranges has invalid dimensions for My Component parameter param a: None",
    ):
        SKOptTuner({"My Component": {"param a": None}})
    SKOptTuner({})
    SKOptTuner({"My Component": {}})


def test_skopt_tuner_is_search_space_exhausted():
    tuner = SKOptTuner({})
    assert not tuner.is_search_space_exhausted()


def test_skopt_tuner_basic():
    pipeline_hyperparameter_ranges = {
        "Mock Classifier": {
            "parameter a": Integer(0, 10),
            "parameter b": Real(0, 10),
            "parameter c": (0, 10),
            "parameter d": (0, 10.0),
            "parameter e": ["option a", "option b", "option c"],
            "parameter f": ["option a 💩", "option b 💩", "option c 💩"],
            "parameter g": ["option a", "option b", 100, np.inf],
        },
    }

    tuner = SKOptTuner(pipeline_hyperparameter_ranges, random_seed=random_seed)
    assert isinstance(tuner, Tuner)
    first_params = tuner.get_starting_parameters({})
    assert first_params == {}
    first_params = tuner.get_starting_parameters(pipeline_hyperparameter_ranges)
    assert first_params == {
        "Mock Classifier": {
            "parameter a": 5,
            "parameter b": 5.488135039273248,
            "parameter c": 0,
            "parameter d": 0,
            "parameter e": "option a",
            "parameter f": "option a 💩",
            "parameter g": "option a",
        },
    }
    proposed_params = tuner.propose()
    assert proposed_params == {
        "Mock Classifier": {
            "parameter a": 5,
            "parameter b": 8.442657485810175,
            "parameter c": 3,
            "parameter d": 8.472517387841256,
            "parameter e": "option b",
            "parameter f": "option b 💩",
            "parameter g": "option b",
        },
    }
    tuner.add(proposed_params, 0.5)


def test_skopt_tuner_invalid_ranges():
    with pytest.raises(
        ValueError,
        match="Invalid dimension \\[\\]. Read the documentation for supported types.",
    ):
        SKOptTuner(
            {
                "Mock Classifier": {
                    "param a": Integer(0, 10),
                    "param b": Real(0, 10),
                    "param c": [],
                },
            },
            random_seed=random_seed,
        )
    with pytest.raises(
        ValueError,
        match="pipeline_hyperparameter_ranges has invalid dimensions for Mock Classifier parameter param c",
    ):
        SKOptTuner(
            {
                "Mock Classifier": {
                    "param a": Integer(0, 10),
                    "param b": Real(0, 10),
                    "param c": None,
                },
            },
            random_seed=random_seed,
        )


def test_skopt_tuner_single_value():
    expected_params = {"Mock Classifier": {}}
    params = {"Mock Classifier": {"param c": 10}}
    tuner = SKOptTuner(params, random_seed=random_seed)
    starting_params = tuner.get_starting_parameters(params)
    assert starting_params == expected_params
    proposed_params = tuner.propose()
    assert proposed_params == expected_params


def test_skopt_tuner_invalid_parameters_score():
    pipeline_hyperparameter_ranges = {
        "Mock Classifier": {
            "param a": Integer(0, 10),
            "param b": Real(0, 10),
            "param c": ["option a", "option b", "option c"],
        },
    }
    tuner = SKOptTuner(pipeline_hyperparameter_ranges, random_seed=random_seed)
    with pytest.raises(
        TypeError,
        match='Pipeline parameters missing required field "param a" for component "Mock Classifier"',
    ):
        tuner.add({}, 0.5)
    with pytest.raises(
        TypeError,
        match='Pipeline parameters missing required field "param a" for component "Mock Classifier"',
    ):
        tuner.add({"Mock Classifier": {}}, 0.5)
    with pytest.raises(
        TypeError,
        match='Pipeline parameters missing required field "param b" for component "Mock Classifier"',
    ):
        tuner.add({"Mock Classifier": {"param a": 0}}, 0.5)
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add(
            {"Mock Classifier": {"param a": 0, "param b": 0.0, "param c": 0}},
            0.5,
        )
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add(
            {"Mock Classifier": {"param a": -1, "param b": 0.0, "param c": "option a"}},
            0.5,
        )
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add(
            {"Mock Classifier": {"param a": 0, "param b": 11.0, "param c": "option a"}},
            0.5,
        )
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add(
            {"Mock Classifier": {"param a": 0, "param b": 0.0, "param c": "option d"}},
            0.5,
        )
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add(
            {
                "Mock Classifier": {
                    "param a": np.nan,
                    "param b": 0.0,
                    "param c": "option a",
                },
            },
            0.5,
        )
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add(
            {
                "Mock Classifier": {
                    "param a": np.inf,
                    "param b": 0.0,
                    "param c": "option a",
                },
            },
            0.5,
        )
    with pytest.raises(
        ParameterError,
        match="Invalid parameters specified to SKOptTuner.add",
    ):
        tuner.add(
            {
                "Mock Classifier": {
                    "param a": None,
                    "param b": 0.0,
                    "param c": "option a",
                },
            },
            0.5,
        )
    with patch("evalml.tuners.skopt_tuner.Optimizer.tell") as mock_optimizer_tell:
        msg = "Mysterious internal error"
        mock_optimizer_tell.side_effect = Exception(msg)
        with pytest.raises(Exception, match=msg):
            tuner.add(
                {
                    "Mock Classifier": {
                        "param a": 0,
                        "param b": 0.0,
                        "param c": "option a",
                    },
                },
                0.5,
            )
    tuner.add(
        {"Mock Classifier": {"param a": 0, "param b": 1.0, "param c": "option a"}},
        0.5,
    )
    tuner.add(
        {"Mock Classifier": {"param a": 0, "param b": 1.0, "param c": "option a"}},
        np.nan,
    )
    tuner.add(
        {"Mock Classifier": {"param a": 0, "param b": 1.0, "param c": "option a"}},
        np.inf,
    )
    tuner.add(
        {"Mock Classifier": {"param a": 0, "param b": 1.0, "param c": "option a"}},
        None,
    )
    tuner.propose()


def test_skopt_tuner_propose():
    pipeline_hyperparameter_ranges = {
        "Mock Classifier": {
            "param a": Integer(0, 10),
            "param b": Real(0, 10),
            "param c": ["option a", "option b", "option c"],
        },
    }
    tuner = SKOptTuner(pipeline_hyperparameter_ranges, random_seed=random_seed)
    first_params = tuner.get_starting_parameters(pipeline_hyperparameter_ranges)
    assert first_params == {
        "Mock Classifier": {
            "param a": 5,
            "param b": 5.488135039273248,
            "param c": "option a",
        },
    }
    tuner.add(
        {"Mock Classifier": {"param a": 0, "param b": 1.0, "param c": "option a"}},
        0.5,
    )
    parameters = tuner.propose()
    assert parameters == {
        "Mock Classifier": {
            "param a": 5,
            "param b": 8.442657485810175,
            "param c": "option c",
        },
    }
