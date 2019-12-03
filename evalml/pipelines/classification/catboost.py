from skopt.space import Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    CatBoostClassifier,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler
)
from evalml.problem_types import ProblemTypes

