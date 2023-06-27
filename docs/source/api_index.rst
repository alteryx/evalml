=============
API Reference
=============


Demo Datasets
=============

.. autoapisummary::
    :nosignatures:

    evalml.demos.load_breast_cancer
    evalml.demos.load_churn
    evalml.demos.load_diabetes
    evalml.demos.load_fraud
    evalml.demos.load_weather
    evalml.demos.load_wine


Preprocessing
=============

Preprocessing Utils
~~~~~~~~~~~~~~~~~~~
Utilities to preprocess data before using evalml.

.. autoapisummary::
    :nosignatures:

    evalml.preprocessing.load_data
    evalml.preprocessing.number_of_features
    evalml.preprocessing.split_data
    evalml.preprocessing.target_distribution

Data Splitters
~~~~~~~~~~~~~~

.. autoapisummary::
    :nosignatures:

    evalml.preprocessing.data_splitters.NoSplit
    evalml.preprocessing.data_splitters.KFold
    evalml.preprocessing.data_splitters.StratifiedKFold
    evalml.preprocessing.data_splitters.TrainingValidationSplit
    evalml.preprocessing.data_splitters.TimeSeriesSplit

Exceptions
=============

.. autoapisummary::

    evalml.exceptions.AutoMLSearchException
    evalml.exceptions.ComponentNotYetFittedError
    evalml.exceptions.DataCheckInitError
    evalml.exceptions.MethodPropertyNotFoundError
    evalml.exceptions.MissingComponentError
    evalml.exceptions.NoPositiveLabelException
    evalml.exceptions.ObjectiveCreationError
    evalml.exceptions.ObjectiveNotFoundError
    evalml.exceptions.PartialDependenceError
    evalml.exceptions.PipelineError
    evalml.exceptions.PipelineNotFoundError
    evalml.exceptions.PipelineNotYetFittedError
    evalml.exceptions.PipelineScoreError

Warnings
~~~~~~~~

.. autoapisummary::

    evalml.exceptions.NullsInColumnWarning
    evalml.exceptions.ParameterNotUsedWarning

Error Codes
~~~~~~~~~~~

.. autoapisummary::

    evalml.exceptions.PartialDependenceErrorCode
    evalml.exceptions.PipelineErrorCodeEnum
    evalml.exceptions.ValidationErrorCode

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

    evalml.automl.get_default_primary_search_objective
    evalml.automl.get_threshold_tuning_info
    evalml.automl.make_data_splitter
    evalml.automl.resplit_training_data
    evalml.automl.search
    evalml.automl.search_iterative
    evalml.automl.tune_binary_threshold


AutoML Algorithm Classes
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoapisummary::

    evalml.automl.automl_algorithm.AutoMLAlgorithm
    evalml.automl.automl_algorithm.DefaultAlgorithm
    evalml.automl.automl_algorithm.IterativeAlgorithm


AutoML Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoapisummary::
    :nosignatures:

    evalml.automl.callbacks.log_error_callback
    evalml.automl.callbacks.raise_error_callback
    evalml.automl.callbacks.silent_error_callback


AutoML Engines
~~~~~~~~~~~~~~
.. autoapisummary::

    evalml.automl.engine.CFEngine
    evalml.automl.engine.DaskEngine
    evalml.automl.engine.EngineBase
    evalml.automl.engine.SequentialEngine

Pipelines
=========

Pipeline Base Classes
~~~~~~~~~~~~~~~~~~~~~
.. autoapisummary::

    evalml.pipelines.BinaryClassificationPipeline
    evalml.pipelines.ClassificationPipeline
    evalml.pipelines.MulticlassClassificationPipeline
    evalml.pipelines.PipelineBase
    evalml.pipelines.RegressionPipeline
    evalml.pipelines.TimeSeriesBinaryClassificationPipeline
    evalml.pipelines.TimeSeriesClassificationPipeline
    evalml.pipelines.TimeSeriesMulticlassClassificationPipeline
    evalml.pipelines.TimeSeriesRegressionPipeline


Pipeline Utils
~~~~~~~~~~~~~~
.. autoapisummary::
    :nosignatures:

    evalml.pipelines.utils.get_actions_from_option_defaults
    evalml.pipelines.utils.generate_pipeline_code
    evalml.pipelines.utils.generate_pipeline_example
    evalml.pipelines.utils.make_pipeline
    evalml.pipelines.utils.make_pipeline_from_actions
    evalml.pipelines.utils.make_pipeline_from_data_check_output
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
    evalml.pipelines.components.utils.estimator_unable_to_handle_nans
    evalml.pipelines.components.utils.generate_component_code
    evalml.pipelines.components.utils.get_estimators
    evalml.pipelines.components.utils.handle_component_class
    evalml.pipelines.components.utils.make_balancing_dictionary


Transformers
~~~~~~~~~~~~
Transformers are components that take in data as input and output transformed data.

.. autoapisummary::

    evalml.pipelines.components.DateTimeFeaturizer
    evalml.pipelines.components.DFSTransformer
    evalml.pipelines.components.DropColumns
    evalml.pipelines.components.DropNaNRowsTransformer
    evalml.pipelines.components.DropNullColumns
    evalml.pipelines.components.DropRowsTransformer
    evalml.pipelines.components.EmailFeaturizer
    evalml.pipelines.components.Imputer
    evalml.pipelines.components.LabelEncoder
    evalml.pipelines.components.LinearDiscriminantAnalysis
    evalml.pipelines.components.LogTransformer
    evalml.pipelines.components.LSA
    evalml.pipelines.components.NaturalLanguageFeaturizer
    evalml.pipelines.components.OneHotEncoder
    evalml.pipelines.components.OrdinalEncoder
    evalml.pipelines.components.Oversampler
    evalml.pipelines.components.PCA
    evalml.pipelines.components.PerColumnImputer
    evalml.pipelines.components.PolynomialDecomposer
    evalml.pipelines.components.ReplaceNullableTypes
    evalml.pipelines.components.RFClassifierRFESelector
    evalml.pipelines.components.RFClassifierSelectFromModel
    evalml.pipelines.components.RFRegressorRFESelector
    evalml.pipelines.components.RFRegressorSelectFromModel
    evalml.pipelines.components.SelectByType
    evalml.pipelines.components.SelectColumns
    evalml.pipelines.components.SimpleImputer
    evalml.pipelines.components.StandardScaler
    evalml.pipelines.components.STLDecomposer
    evalml.pipelines.components.TargetEncoder
    evalml.pipelines.components.TargetImputer
    evalml.pipelines.components.TimeSeriesFeaturizer
    evalml.pipelines.components.TimeSeriesImputer
    evalml.pipelines.components.TimeSeriesRegularizer
    evalml.pipelines.components.Undersampler
    evalml.pipelines.components.URLFeaturizer


Estimators
~~~~~~~~~~

Classifiers
-----------

Classifiers are components that output a predicted class label.

.. autoapisummary::

    evalml.pipelines.components.BaselineClassifier
    evalml.pipelines.components.CatBoostClassifier
    evalml.pipelines.components.DecisionTreeClassifier
    evalml.pipelines.components.ElasticNetClassifier
    evalml.pipelines.components.ExtraTreesClassifier
    evalml.pipelines.components.KNeighborsClassifier
    evalml.pipelines.components.LightGBMClassifier
    evalml.pipelines.components.LogisticRegressionClassifier
    evalml.pipelines.components.RandomForestClassifier
    evalml.pipelines.components.StackedEnsembleClassifier
    evalml.pipelines.components.SVMClassifier
    evalml.pipelines.components.VowpalWabbitBinaryClassifier
    evalml.pipelines.components.VowpalWabbitMulticlassClassifier
    evalml.pipelines.components.XGBoostClassifier


Regressors
-----------

Regressors are components that output a predicted target value.

.. autoapisummary::

    evalml.pipelines.components.ARIMARegressor
    evalml.pipelines.components.BaselineRegressor
    evalml.pipelines.components.CatBoostRegressor
    evalml.pipelines.components.DecisionTreeRegressor
    evalml.pipelines.components.ElasticNetRegressor
    evalml.pipelines.components.ExponentialSmoothingRegressor
    evalml.pipelines.components.ExtraTreesRegressor
    evalml.pipelines.components.LightGBMRegressor
    evalml.pipelines.components.LinearRegressor
    evalml.pipelines.components.ProphetRegressor
    evalml.pipelines.components.RandomForestRegressor
    evalml.pipelines.components.StackedEnsembleRegressor
    evalml.pipelines.components.SVMRegressor
    evalml.pipelines.components.TimeSeriesBaselineEstimator
    evalml.pipelines.components.VowpalWabbitRegressor
    evalml.pipelines.components.XGBoostRegressor


Model Understanding
===================

Metrics
~~~~~~~~~~~~~~~
.. autoapisummary::
    :nosignatures:

    evalml.model_understanding.binary_objective_vs_threshold
    evalml.model_understanding.calculate_permutation_importance
    evalml.model_understanding.calculate_permutation_importance_one_column
    evalml.model_understanding.confusion_matrix
    evalml.model_understanding.find_confusion_matrix_per_thresholds
    evalml.model_understanding.get_linear_coefficients
    evalml.model_understanding.get_prediction_vs_actual_data
    evalml.model_understanding.get_prediction_vs_actual_over_time_data
    evalml.model_understanding.normalize_confusion_matrix
    evalml.model_understanding.partial_dependence
    evalml.model_understanding.precision_recall_curve
    evalml.model_understanding.roc_curve
    evalml.model_understanding.t_sne
    evalml.model_understanding.feature_explanations.get_influential_features
    evalml.model_understanding.feature_explanations.readable_explanation


Visualization Methods
~~~~~~~~~~~~~~~~~~~~~~~
.. autoapisummary::
    :nosignatures:

    evalml.model_understanding.graph_binary_objective_vs_threshold
    evalml.model_understanding.graph_confusion_matrix
    evalml.model_understanding.graph_partial_dependence
    evalml.model_understanding.graph_permutation_importance
    evalml.model_understanding.graph_precision_recall_curve
    evalml.model_understanding.graph_prediction_vs_actual
    evalml.model_understanding.graph_prediction_vs_actual_over_time
    evalml.model_understanding.graph_roc_curve
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

    evalml.objectives.CostBenefitMatrix
    evalml.objectives.FraudCost
    evalml.objectives.LeadScoring
    evalml.objectives.SensitivityLowAlert


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

    evalml.objectives.ExpVariance
    evalml.objectives.MAE
    evalml.objectives.MAPE
    evalml.objectives.MSE
    evalml.objectives.MeanSquaredLogError
    evalml.objectives.MedianAE
    evalml.objectives.MaxError
    evalml.objectives.R2
    evalml.objectives.RootMeanSquaredError
    evalml.objectives.RootMeanSquaredLogError


Objective Utils
~~~~~~~~~~~~~~~~~~~~~

.. autoapisummary::
    :nosignatures:

    evalml.objectives.get_all_objective_names
    evalml.objectives.get_core_objectives
    evalml.objectives.get_core_objective_names
    evalml.objectives.get_default_recommendation_objectives
    evalml.objectives.get_non_core_objectives
    evalml.objectives.get_objective
    evalml.objectives.get_optimization_objectives
    evalml.objectives.get_ranking_objectives
    evalml.objectives.normalize_objectives
    evalml.objectives.organize_objectives
    evalml.objectives.ranking_only_objectives
    evalml.objectives.recommendation_score


Problem Types
=============

.. autoapisummary::
    :nosignatures:

    evalml.problem_types.detect_problem_type
    evalml.problem_types.handle_problem_types
    evalml.problem_types.is_binary
    evalml.problem_types.is_classification
    evalml.problem_types.is_multiclass
    evalml.problem_types.is_regression
    evalml.problem_types.is_time_series
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

    evalml.data_checks.ClassImbalanceDataCheck
    evalml.data_checks.DateTimeFormatDataCheck
    evalml.data_checks.IDColumnsDataCheck
    evalml.data_checks.InvalidTargetDataCheck
    evalml.data_checks.MulticollinearityDataCheck
    evalml.data_checks.NoVarianceDataCheck
    evalml.data_checks.NullDataCheck
    evalml.data_checks.OutliersDataCheck
    evalml.data_checks.SparsityDataCheck
    evalml.data_checks.TargetDistributionDataCheck
    evalml.data_checks.TargetLeakageDataCheck
    evalml.data_checks.TimeSeriesParametersDataCheck
    evalml.data_checks.TimeSeriesSplittingDataCheck
    evalml.data_checks.UniquenessDataCheck

    evalml.data_checks.DataCheck
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

Data Check Actions
~~~~~~~~~~~~~~~~~~
.. autoapisummary::

    evalml.data_checks.DataCheckAction
    evalml.data_checks.DataCheckActionCode
    evalml.data_checks.DataCheckActionOption


Utils
=====

General Utils
~~~~~~~~~~~~~

.. autoapisummary::
    :nosignatures:

    evalml.utils.convert_to_seconds
    evalml.utils.downcast_nullable_types
    evalml.utils.drop_rows_with_nans
    evalml.utils.get_importable_subclasses
    evalml.utils.get_logger
    evalml.utils.get_time_index
    evalml.utils.import_or_raise
    evalml.utils.infer_feature_types
    evalml.utils.is_all_numeric
    evalml.utils.get_random_state
    evalml.utils.get_random_seed
    evalml.utils.pad_with_nans
    evalml.utils.safe_repr
    evalml.utils.save_plot


.. toctree::
    :hidden:

    autoapi/evalml/index
