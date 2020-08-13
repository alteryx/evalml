=============
API Reference
=============

.. currentmodule:: evalml.demos

Demo Datasets
=============

.. autosummary::
    :toctree: generated
    :nosignatures:

    load_fraud
    load_wine
    load_breast_cancer
    load_diabetes


.. currentmodule:: evalml.preprocessing

Preprocessing
=============

Utilities to preprocess data before using evalml.

.. autosummary::
    :toctree: generated
    :nosignatures:

    drop_nan_target_rows
    label_distribution
    load_data
    number_of_features
    split_data


.. currentmodule:: evalml.automl

AutoML
======

AutoML Search Classes
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: class_with_properties.rst
    :nosignatures:

    AutoMLSearch


.. currentmodule:: evalml.automl.automl_algorithm

AutoML Algorithm Classes
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: class_with_properties.rst
    :nosignatures:

    AutoMLAlgorithm
    IterativeAlgorithm


.. currentmodule:: evalml.pipelines

Pipelines
=========

Pipeline Base Classes
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    PipelineBase
    ClassificationPipeline
    BinaryClassificationPipeline
    MulticlassClassificationPipeline
    RegressionPipeline

Classification Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: pipeline_class.rst
    :nosignatures:

    BaselineBinaryPipeline
    BaselineMulticlassPipeline
    ModeBaselineBinaryPipeline
    ModeBaselineMulticlassPipeline

Regression Pipelines
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: pipeline_class.rst
    :nosignatures:

    BaselineRegressionPipeline
    MeanBaselineRegressionPipeline


Pipeline Graph Utils
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    precision_recall_curve
    graph_precision_recall_curve
    roc_curve
    graph_roc_curve
    graph_confusion_matrix
    calculate_permutation_importance
    graph_permutation_importance


.. currentmodule:: evalml.pipelines.utils

Pipeline Utils
~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    make_pipeline


.. currentmodule:: evalml.pipelines.components

Components
==========

Component Base Classes
~~~~~~~~~~~~~~~~~~~~~~
Components represent a step in a pipeline.

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ComponentBase
    Transformer
    Estimator

.. currentmodule:: evalml.pipelines.components.utils

Component Utils
~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    allowed_model_families
    get_estimators


.. currentmodule:: evalml.pipelines.components

Transformers
~~~~~~~~~~~~
Transformers are components that take in data as input and output transformed data.

.. autosummary::
    :toctree: generated
    :template: transformer_class.rst
    :nosignatures:

    DropColumns
    SelectColumns
    OneHotEncoder
    PerColumnImputer
    Imputer
    SimpleImputer
    StandardScaler
    RFRegressorSelectFromModel
    RFClassifierSelectFromModel
    DropNullColumns
    DateTimeFeaturizer
    TextFeaturizer

Estimators
~~~~~~~~~~

Classifiers
-----------

Classifiers are components that output a predicted class label.

.. autosummary::
    :toctree: generated
    :template: estimator_class.rst
    :nosignatures:

    CatBoostClassifier
    ElasticNetClassifier
    ExtraTreesClassifier
    RandomForestClassifier
    LogisticRegressionClassifier
    XGBoostClassifier
    BaselineClassifier

Regressors
-----------

Regressors are components that output a predicted target value.

.. autosummary::
    :toctree: generated
    :template: estimator_class.rst
    :nosignatures:

    CatBoostRegressor
    ElasticNetRegressor
    LinearRegressor
    ExtraTreesRegressor
    RandomForestRegressor
    XGBoostRegressor
    BaselineRegressor


.. currentmodule:: evalml.pipelines.prediction_explanations

Prediction Explanations
========================

.. autosummary::
    :toctree: generated
    :nosignatures:

    explain_prediction
    explain_predictions
    explain_predictions_best_worst


.. currentmodule:: evalml.objectives

Objective Functions
====================

Objective Base Classes
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ObjectiveBase
    BinaryClassificationObjective
    MulticlassClassificationObjective
    RegressionObjective



Domain-Specific Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    FraudCost
    LeadScoring
    CostBenefitMatrix


Classification Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    AccuracyBinary
    AccuracyMulticlass
    AUC
    AUCMacro
    AUCMicro
    AUCWeighted
    BalancedAccuracyBinary
    BalancedAccuracyMulticlass
    F1
    F1Micro
    F1Macro
    F1Weighted
    LogLossBinary
    LogLossMulticlass
    MCCBinary
    MCCMulticlass
    Precision
    PrecisionMicro
    PrecisionMacro
    PrecisionWeighted
    Recall
    RecallMicro
    RecallMacro
    RecallWeighted



Regression Objectives
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    R2
    MAE
    MSE
    MeanSquaredLogError
    MedianAE
    MaxError
    ExpVariance
    RootMeanSquaredError
    RootMeanSquaredLogError


.. currentmodule:: evalml.problem_types

Problem Types
=============

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ProblemTypes

.. autosummary::
    :toctree: generated
    :nosignatures:

    handle_problem_types


.. currentmodule:: evalml.model_family

Model Family
============

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ModelFamily

.. autosummary::
    :toctree: generated
    :nosignatures:

    handle_model_family


.. currentmodule:: evalml.tuners

Tuners
======

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    Tuner
    SKOptTuner
    GridSearchTuner
    RandomSearchTuner


.. currentmodule:: evalml.data_checks

Data Checks
===========

Data Check Classes
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: data_check_class.rst
    :nosignatures:

    DataCheck
    InvalidTargetDataCheck
    HighlyNullDataCheck
    IDColumnsDataCheck
    LabelLeakageDataCheck
    OutliersDataCheck
    NoVarianceDataCheck


.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    DataChecks
    DefaultDataChecks


Data Check Messages
~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :template: data_check_message.rst
    :nosignatures:

    DataCheckMessage
    DataCheckError
    DataCheckWarning


Data Check Message Types
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    DataCheckMessageType


.. currentmodule:: evalml.utils

Utils
=====

General Utils
~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    import_or_raise
    convert_to_seconds
    get_random_state
    get_random_seed


Graph Utils
~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    confusion_matrix
    normalize_confusion_matrix
