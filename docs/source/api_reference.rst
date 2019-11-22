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


.. currentmodule:: evalml

Models
======

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    AutoClassifier
    AutoRegressor

Model Types
===========

.. autosummary::
    :toctree: generated
    :nosignatures:

    list_model_types


.. currentmodule:: evalml.pipelines

Pipelines
=========

.. autosummary::
    :toctree: generated
    :nosignatures:

    get_pipelines
    save_pipeline
    load_pipeline

.. autosummary::
    :toctree: generated
    :template: class.rst
    :nosignatures:

    RFClassificationPipeline
    XGBoostPipeline
    LogisticRegressionPipeline
    RFRegressionPipeline
    LinearRegressionPipeline


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

    SKOptTuner


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
