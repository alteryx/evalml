=============
API Reference
=============


Demo Datasets
=============

.. autoapisummary::
    :nosignatures:

    evalml.demos.load_breast_cancer
    evalml.demos.load_diabetes
    evalml.demos.load_fraud
    evalml.demos.load_wine
    evalml.demos.load_churn


Preprocessing
=============

Utilities to preprocess data before using evalml.

.. autoapisummary::
    :nosignatures:

    evalml.preprocessing.load_data
    evalml.preprocessing.target_distribution
    evalml.preprocessing.number_of_features
    evalml.preprocessing.split_data


Exceptions
=============

.. autoapisummary::

    evalml.exceptions.MethodPropertyNotFoundError
    evalml.exceptions.PipelineNotFoundError
    evalml.exceptions.ObjectiveNotFoundError
    evalml.exceptions.MissingComponentError
    evalml.exceptions.ComponentNotYetFittedError
    evalml.exceptions.PipelineNotYetFittedError
    evalml.exceptions.AutoMLSearchException
    evalml.exceptions.PipelineScoreError
    evalml.exceptions.DataCheckInitError
    evalml.exceptions.NullsInColumnWarning


AutoML
======

AutoML Search Interface
~~~~~~~~~~~~~~~~~~~~~~~
.. autoapisummary::

    evalml.automl.AutoMLSearch


AutoML Utils
~~~~~~~~~~~~
.. autoapisummary::
    :nosignatures:

    evalml.automl.search
    evalml.automl.get_default_primary_search_objective
    evalml.automl.make_data_splitter


AutoML Algorithm Classes
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoapisummary::

    evalml.automl.automl_algorithm.AutoMLAlgorithm
    evalml.automl.automl_algorithm.IterativeAlgorithm


AutoML Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoapisummary::
    :nosignatures:

    evalml.automl.callbacks.silent_error_callback
    evalml.automl.callbacks.log_error_callback
    evalml.automl.callbacks.raise_error_callback


AutoML Engines
~~~~~~~~~~~~~~
.. autoapisummary::

    evalml.automl.engine.sequential_engine.SequentialEngine
    evalml.automl.engine.cf_engine.CFEngine
    evalml.automl.engine.dask_engine.DaskEngine

Pipelines
=========

Pipeline Base Classes
~~~~~~~~~~~~~~~~~~~~~
.. autoapisummary::

    evalml.pipelines.PipelineBase
    evalml.pipelines.classification_pipeline.ClassificationPipeline
    evalml.pipelines.binary_classification_pipeline.BinaryClassificationPipeline
    evalml.pipelines.MulticlassClassificationPipeline
    evalml.pipelines.RegressionPipeline
    evalml.pipelines.TimeSeriesClassificationPipeline
    evalml.pipelines.TimeSeriesBinaryClassificationPipeline
    evalml.pipelines.TimeSeriesMulticlassClassificationPipeline
    evalml.pipelines.TimeSeriesRegressionPipeline


Pipeline Utils
~~~~~~~~~~~~~~
.. autoapisummary::
    :nosignatures:

    evalml.pipelines.utils.make_pipeline
    evalml.pipelines.utils.generate_pipeline_code
    evalml.pipelines.utils.rows_of_interest



Component Graphs
================

.. autoapisummary::

    evalml.pipelines.ComponentGraph


Components
==========

Component Base Classes
~~~~~~~~~~~~~~~~~~~~~~
Components represent a step in a pipeline.

.. autoapisummary::

    evalml.pipelines.components.ComponentBase
    evalml.pipelines.Transformer
    evalml.pipelines.Estimator


Component Utils
~~~~~~~~~~~~~~~
.. autoapisummary::
    :nosignatures:

    evalml.pipelines.components.utils.allowed_model_families
    evalml.pipelines.components.utils.get_estimators
    evalml.pipelines.components.utils.generate_component_code


Transformers
~~~~~~~~~~~~
Transformers are components that take in data as input and output transformed data.

.. autoapisummary::

    evalml.pipelines.components.DropColumns
    evalml.pipelines.components.SelectColumns
    evalml.pipelines.components.SelectByType
    evalml.pipelines.components.OneHotEncoder
    evalml.pipelines.components.TargetEncoder
    evalml.pipelines.components.PerColumnImputer
    evalml.pipelines.components.Imputer
    evalml.pipelines.components.SimpleImputer
    evalml.pipelines.components.TimeSeriesImputer
    evalml.pipelines.components.StandardScaler
    evalml.pipelines.components.RFRegressorSelectFromModel
    evalml.pipelines.components.RFClassifierSelectFromModel
    evalml.pipelines.components.DropNullColumns
    evalml.pipelines.components.DateTimeFeaturizer
    evalml.pipelines.components.NaturalLanguageFeaturizer
    evalml.pipelines.components.TimeSeriesFeaturizer
    evalml.pipelines.components.TimeSeriesRegularizer
    evalml.pipelines.components.DFSTransformer
    evalml.pipelines.components.PolynomialDetrender
    evalml.pipelines.components.Undersampler
    evalml.pipelines.components.Oversampler


Estimators
~~~~~~~~~~

Classifiers
-----------

Classifiers are components that output a predicted class label.

.. autoapisummary::

    evalml.pipelines.components.CatBoostClassifier
    evalml.pipelines.components.ElasticNetClassifier
    evalml.pipelines.components.ExtraTreesClassifier
    evalml.pipelines.components.RandomForestClassifier
    evalml.pipelines.components.LightGBMClassifier
    evalml.pipelines.components.LogisticRegressionClassifier
    evalml.pipelines.components.XGBoostClassifier
    evalml.pipelines.components.BaselineClassifier
    evalml.pipelines.components.StackedEnsembleClassifier
    evalml.pipelines.components.DecisionTreeClassifier
    evalml.pipelines.components.KNeighborsClassifier
    evalml.pipelines.components.SVMClassifier
    evalml.pipelines.components.VowpalWabbitBinaryClassifier
    evalml.pipelines.components.VowpalWabbitMulticlassClassifier


Regressors
-----------

Regressors are components that output a predicted target value.

.. autoapisummary::

    evalml.pipelines.components.ARIMARegressor
    evalml.pipelines.components.CatBoostRegressor
    evalml.pipelines.components.ElasticNetRegressor
    evalml.pipelines.components.ExponentialSmoothingRegressor
    evalml.pipelines.components.LinearRegressor
    evalml.pipelines.components.ExtraTreesRegressor
    evalml.pipelines.components.RandomForestRegressor
    evalml.pipelines.components.XGBoostRegressor
    evalml.pipelines.components.BaselineRegressor
    evalml.pipelines.components.TimeSeriesBaselineEstimator
    evalml.pipelines.components.StackedEnsembleRegressor
    evalml.pipelines.components.DecisionTreeRegressor
    evalml.pipelines.components.LightGBMRegressor
    evalml.pipelines.components.SVMRegressor
    evalml.pipelines.components.VowpalWabbitRegressor


Model Understanding
===================

Utility Methods
~~~~~~~~~~~~~~~
.. autoapisummary::
    :nosignatures:

    evalml.model_understanding.confusion_matrix
    evalml.model_understanding.normalize_confusion_matrix
    evalml.model_understanding.precision_recall_curve
    evalml.model_understanding.roc_curve
    evalml.model_understanding.calculate_permutation_importance
    evalml.model_understanding.calculate_permutation_importance_one_column
    evalml.model_understanding.binary_objective_vs_threshold
    evalml.model_understanding.get_prediction_vs_actual_over_time_data
    evalml.model_understanding.partial_dependence
    evalml.model_understanding.get_prediction_vs_actual_data
    evalml.model_understanding.get_linear_coefficients
    evalml.model_understanding.t_sne
    evalml.model_understanding.find_confusion_matrix_per_thresholds


Graph Utility Methods
~~~~~~~~~~~~~~~~~~~~~~~
.. autoapisummary::
    :nosignatures:

    evalml.model_understanding.graph_precision_recall_curve
    evalml.model_understanding.graph_roc_curve
    evalml.model_understanding.graph_confusion_matrix
    evalml.model_understanding.graph_permutation_importance
    evalml.model_understanding.graph_binary_objective_vs_threshold
    evalml.model_understanding.graph_prediction_vs_actual
    evalml.model_understanding.graph_prediction_vs_actual_over_time
    evalml.model_understanding.graph_partial_dependence
    evalml.model_understanding.graph_t_sne


Prediction Explanations
~~~~~~~~~~~~~~~~~~~~~~~

.. autoapisummary::
    :nosignatures:

    evalml.model_understanding.explain_predictions
    evalml.model_understanding.explain_predictions_best_worst


Objectives
====================

Objective Base Classes
~~~~~~~~~~~~~~~~~~~~~~

.. autoapisummary::

    evalml.objectives.ObjectiveBase
    evalml.objectives.BinaryClassificationObjective
    evalml.objectives.MulticlassClassificationObjective
    evalml.objectives.RegressionObjective


Domain-Specific Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoapisummary::

    evalml.objectives.FraudCost
    evalml.objectives.LeadScoring
    evalml.objectives.CostBenefitMatrix


Classification Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoapisummary::

    evalml.objectives.AccuracyBinary
    evalml.objectives.AccuracyMulticlass
    evalml.objectives.AUC
    evalml.objectives.AUCMacro
    evalml.objectives.AUCMicro
    evalml.objectives.AUCWeighted
    evalml.objectives.Gini
    evalml.objectives.BalancedAccuracyBinary
    evalml.objectives.BalancedAccuracyMulticlass
    evalml.objectives.F1
    evalml.objectives.F1Micro
    evalml.objectives.F1Macro
    evalml.objectives.F1Weighted
    evalml.objectives.LogLossBinary
    evalml.objectives.LogLossMulticlass
    evalml.objectives.MCCBinary
    evalml.objectives.MCCMulticlass
    evalml.objectives.Precision
    evalml.objectives.PrecisionMicro
    evalml.objectives.PrecisionMacro
    evalml.objectives.PrecisionWeighted
    evalml.objectives.Recall
    evalml.objectives.RecallMicro
    evalml.objectives.RecallMacro
    evalml.objectives.RecallWeighted


Regression Objectives
~~~~~~~~~~~~~~~~~~~~~

.. autoapisummary::

    evalml.objectives.R2
    evalml.objectives.MAE
    evalml.objectives.MAPE
    evalml.objectives.MSE
    evalml.objectives.MeanSquaredLogError
    evalml.objectives.MedianAE
    evalml.objectives.MaxError
    evalml.objectives.ExpVariance
    evalml.objectives.RootMeanSquaredError
    evalml.objectives.RootMeanSquaredLogError


Objective Utils
~~~~~~~~~~~~~~~~~~~~~

.. autoapisummary::
    :nosignatures:

    evalml.objectives.get_all_objective_names
    evalml.objectives.get_core_objectives
    evalml.objectives.get_core_objective_names
    evalml.objectives.get_non_core_objectives
    evalml.objectives.get_objective


Problem Types
=============

.. autoapisummary::
    :nosignatures:

    evalml.problem_types.handle_problem_types
    evalml.problem_types.detect_problem_type
    evalml.problem_types.ProblemTypes


Model Family
============

.. autoapisummary::
    :nosignatures:

    evalml.model_family.handle_model_family
    evalml.model_family.ModelFamily


Tuners
======

.. autoapisummary::

    evalml.tuners.Tuner
    evalml.tuners.SKOptTuner
    evalml.tuners.GridSearchTuner
    evalml.tuners.RandomSearchTuner


Data Checks
===========

Data Check Classes
~~~~~~~~~~~~~~~~~~
.. autoapisummary::

    evalml.data_checks.DataCheck
    evalml.data_checks.InvalidTargetDataCheck
    evalml.data_checks.NullDataCheck
    evalml.data_checks.IDColumnsDataCheck
    evalml.data_checks.TargetLeakageDataCheck
    evalml.data_checks.OutliersDataCheck
    evalml.data_checks.NoVarianceDataCheck
    evalml.data_checks.ClassImbalanceDataCheck
    evalml.data_checks.MulticollinearityDataCheck
    evalml.data_checks.DateTimeFormatDataCheck
    evalml.data_checks.TimeSeriesParametersDataCheck
    evalml.data_checks.TimeSeriesSplittingDataCheck

    evalml.data_checks.DataChecks
    evalml.data_checks.DefaultDataChecks


Data Check Messages
~~~~~~~~~~~~~~~~~~~
.. autoapisummary::

    evalml.data_checks.DataCheckMessage
    evalml.data_checks.DataCheckError
    evalml.data_checks.DataCheckWarning


Data Check Message Types
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoapisummary::

    evalml.data_checks.DataCheckMessageType

Data Check Message Codes
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoapisummary::

    evalml.data_checks.DataCheckMessageCode


Utils
=====

General Utils
~~~~~~~~~~~~~

.. autoapisummary::
    :nosignatures:

    evalml.utils.import_or_raise
    evalml.utils.convert_to_seconds
    evalml.utils.get_random_state
    evalml.utils.get_random_seed
    evalml.utils.pad_with_nans
    evalml.utils.drop_rows_with_nans
    evalml.utils.infer_feature_types
    evalml.utils.save_plot
    evalml.utils.is_all_numeric
    evalml.utils.get_importable_subclasses


.. toctree::
    :hidden:

    autoapi/evalml/index
