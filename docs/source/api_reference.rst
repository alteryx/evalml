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

.. autosummary::
    :toctree: generated
    :nosignatures:

    load_data
    split_data


.. currentmodule:: evalml.automl

AutoML
======

.. autosummary::
    :toctree: generated
    :template: class_with_properties.rst
    :nosignatures:

    AutoClassificationSearch
    AutoRegressionSearch


Plotting
~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: accessor_method.rst
    :nosignatures:

    AutoClassificationSearch.plot.get_roc_data
    AutoClassificationSearch.plot.generate_roc_plot
    AutoClassificationSearch.plot.get_confusion_matrix_data
    AutoClassificationSearch.plot.generate_confusion_matrix


.. currentmodule:: evalml.model_family

Model Family
============

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ModelFamily


.. currentmodule:: evalml.pipelines

.. autosummary::
    :toctree: generated
    :nosignatures:

    list_model_families


Components
==========

Transformers
~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    OneHotEncoder
    RFRegressorSelectFromModel
    RFClassifierSelectFromModel
    SimpleImputer
    StandardScaler

Estimators
~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    LogisticRegressionClassifier
    RandomForestClassifier
    XGBoostClassifier
    LinearRegressor
    RandomForestRegressor


Pipelines
=========

.. autosummary::
    :toctree: generated
    :nosignatures:

    get_pipelines

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    PipelineBase
    BinaryClassificationPipeline
    MulticlassClassificationPipeline
    CatBoostBinaryClassificationPipeline
    CatBoostMulticlassClassificationPipeline
    LogisticRegressionBinaryPipeline
    LogisticRegressionMulticlassPipeline
    RFBinaryClassificationPipeline
    RFMulticlassClassificationPipeline
    XGBoostBinaryPipeline
    XGBoostMulticlassPipeline


Plotting
~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: accessor_callable.rst

   PipelineBase.graph
   PipelineBase.feature_importance_graph


.. currentmodule:: evalml.objectives

Objective Functions
====================

Domain Specific
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    FraudCost
    LeadScoring


Classification
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    F1
    F1Micro
    F1Macro
    F1Weighted
    Precision
    PrecisionMicro
    PrecisionMacro
    PrecisionWeighted
    Recall
    RecallMicro
    RecallMacro
    RecallWeighted
    AUC
    AUCMicro
    AUCMacro
    AUCWeighted
    LogLoss
    MCC
    ROC
    ConfusionMatrix


Regression
~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    R2
    MAE
    MSE
    MSLE
    MedianAE
    MaxError
    ExpVariance


Plot Metrics
~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    ROC
    ConfusionMatrix


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


.. currentmodule:: evalml.guardrails

Guardrails
=============

.. autosummary::
    :toctree: generated
    :nosignatures:

    detect_highly_null
    detect_label_leakage
    detect_outliers
    detect_id_columns


.. currentmodule:: evalml.utils

Guardrails
=============

.. autosummary::
    :toctree: generated
    :nosignatures:

    convert_to_seconds
    normalize_confusion_matrix