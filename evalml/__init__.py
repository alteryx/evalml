"""
Copyright (c) 2019 Feature Labs, Inc.

The usage of this software is governed by the Feature Labs End User License Agreement available at https://www.featurelabs.com/eula/. If you do not agree to the terms set out in this agreement, do not use the software, and immediately contact Feature Labs or your supplier.
"""
# flake8:noqa

import warnings

# hack to prevent warnings from skopt
# must import sklearn first
import sklearn

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    import skopt

import evalml.demos
import evalml.model_family
import evalml.objectives
import evalml.pipelines
import evalml.preprocessing
import evalml.problem_types
import evalml.utils
import evalml.data_checks

from evalml.pipelines import list_model_families
from evalml.automl import AutoClassificationSearch, AutoRegressionSearch
from evalml.utils import print_info

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


__version__ = '0.9.0'
