Release Notes
-------------
**Future Releases**
    * Enhancements
        * Updated ``AutoMLSearch._check_for_high_variance`` to not emit ``RuntimeWarning`` :pr:`2024`
        * Added exception when pipeline passed to ``explain_predictions`` is a ``Stacked Ensemble`` pipeline :pr:`2033`
        * Added sensitivity at low alert rates as an objective :pr:`2001`
    * Fixes
        * Fixed bug in where Time Series Classification pipelines were not encoding targets in ``predict`` and ``predict_proba`` :pr:`2040`
    * Changes
        * Removed lists as acceptable hyperparameter ranges in ``AutoMLSearch`` :pr:`2028`
        * Renamed "details" to "metadata" for data check actions :pr:`2008`
    * Documentation Changes
        * Catch and suppress warnings in documentation :pr:`1991`
    * Testing Changes


**v0.21.0 Mar. 24, 2021**
    * Enhancements
        * Changed ``AutoMLSearch`` to default ``optimize_thresholds`` to True :pr:`1943`
        * Added multiple oversampling and undersampling sampling methods as data splitters for imbalanced classification :pr:`1775`
        * Added params to balanced classification data splitters for visibility :pr:`1966`
        * Updated ``make_pipeline`` to not add ``Imputer`` if input data does not have numeric or categorical columns :pr:`1967`
        * Updated ``ClassImbalanceDataCheck`` to better handle multiclass imbalances :pr:`1986`
        * Added recommended actions for the output of data check's ``validate`` method :pr:`1968`
        * Added error message for ``partial_dependence`` when features are mostly the same value :pr:`1994`
        * Updated ``OneHotEncoder`` to drop one redundant feature by default for features with two categories :pr:`1997`
        * Added a ``PolynomialDetrender`` component :pr:`1992`
    * Fixes
        * Updated binary classification pipelines to use objective decision function during scoring of custom objectives :pr:`1934`
    * Changes
        * Removed ``data_checks`` parameter, ``data_check_results`` and data checks logic from ``AutoMLSearch`` :pr:`1935`
        * Deleted ``random_state`` argument :pr:`1985`
        * Updated Woodwork version requirement to ``v0.0.11`` :pr:`1996`
    * Documentation Changes
    * Testing Changes
        * Removed ``build_docs`` CI job in favor of RTD GH builder :pr:`1974`
        * Added tests to confirm support for Python 3.9 :pr:`1724`
        * Changed ``build_conda_pkg`` job to use ``latest_release_changes`` branch in the feedstock. :pr:`1979`

.. warning::

    **Breaking Changes**
        * Changed ``AutoMLSearch`` to default ``optimize_thresholds`` to True :pr:`1943`
        * Removed ``data_checks`` parameter, ``data_check_results`` and data checks logic from ``AutoMLSearch``. To run the data checks which were previously run by default in ``AutoMLSearch``, please call ``DefaultDataChecks().validate(X_train, y_train)`` or take a look at our documentation for more examples. :pr:`1935`
        * Deleted ``random_state`` argument :pr:`1985`

**v0.20.0 Mar. 10, 2021**
    * Enhancements
        * Added a GitHub Action for Detecting dependency changes :pr:`1933`
        * Create a separate CV split to train stacked ensembler on for AutoMLSearch :pr:`1814`
        * Added a GitHub Action for Linux unit tests :pr:`1846`
        * Added ``ARIMARegressor`` estimator :pr:`1894`
        * Added ``DataCheckAction`` class and ``DataCheckActionCode`` enum :pr:`1896`
        * Updated ``Woodwork`` requirement to ``v0.0.10`` :pr:`1900`
        * Added ``BalancedClassificationDataCVSplit`` and ``BalancedClassificationDataTVSplit`` to AutoMLSearch :pr:`1875`
        * Update default classification data splitter to use downsampling for highly imbalanced data :pr:`1875`
        * Updated ``describe_pipeline`` to return more information, including ``id`` of pipelines used for ensemble models :pr:`1909`
        * Added utility method to create list of components from a list of ``DataCheckAction`` :pr:`1907`
        * Updated ``validate`` method to include a ``action`` key in returned dictionary for all ``DataCheck``and ``DataChecks`` :pr:`1916`
        * Aggregating the shap values for predictions that we know the provenance of, e.g. OHE, text, and date-time. :pr:`1901`
        * Improved error message when custom objective is passed as a string in ``pipeline.score`` :pr:`1941`
        * Added ``score_pipelines`` and ``train_pipelines`` methods to ``AutoMLSearch`` :pr:`1913`
        * Added support for ``pandas`` version 1.2.0 :pr:`1708`
        * Added ``score_batch`` and ``train_batch`` abstact methods to ``EngineBase`` and implementations in ``SequentialEngine`` :pr:`1913`
    * Fixes
        * Removed CI check for ``check_dependencies_updated_linux`` :pr:`1950`
        * Added metaclass for time series pipelines and fix binary classification pipeline ``predict`` not using objective if it is passed as a named argument :pr:`1874`
        * Fixed stack trace in prediction explanation functions caused by mixed string/numeric pandas column names :pr:`1871`
        * Fixed stack trace caused by passing pipelines with duplicate names to ``AutoMLSearch`` :pr:`1932`
        * Fixed ``AutoMLSearch.get_pipelines`` returning pipelines with the same attributes :pr:`1958`
    * Changes
        * Reversed GitHub Action for Linux unit tests until a fix for report generation is found :pr:`1920`
        * Updated ``add_results`` in ``AutoMLAlgorithm`` to take in entire pipeline results dictionary from ``AutoMLSearch`` :pr:`1891`
        * Updated ``ClassImbalanceDataCheck`` to look for severe class imbalance scenarios :pr:`1905`
        * Deleted the ``explain_prediction`` function :pr:`1915`
        * Removed ``HighVarianceCVDataCheck`` and convered it to an ``AutoMLSearch`` method instead :pr:`1928`
        * Removed warning in ``InvalidTargetDataCheck`` returned when numeric binary classification targets are not (0, 1) :pr:`1959`
    * Documentation Changes
        * Updated ``model_understanding.ipynb`` to demo the two-way partial dependence capability :pr:`1919`
    * Testing Changes

.. warning::

    **Breaking Changes**
        * Deleted the ``explain_prediction`` function :pr:`1915`
        * Removed ``HighVarianceCVDataCheck`` and convered it to an ``AutoMLSearch`` method instead :pr:`1928`
        * Added ``score_batch`` and ``train_batch`` abstact methods to ``EngineBase``. These need to be implemented in Engine subclasses :pr:`1913`


**v0.19.0 Feb. 23, 2021**
    * Enhancements
        * Added a GitHub Action for Python windows unit tests :pr:`1844`
        * Added a GitHub Action for checking updated release notes :pr:`1849`
        * Added a GitHub Action for Python lint checks :pr:`1837`
        * Adjusted ``explain_prediction``, ``explain_predictions`` and ``explain_predictions_best_worst`` to handle timeseries problems. :pr:`1818`
        * Updated ``InvalidTargetDataCheck`` to check for mismatched indices in target and features :pr:`1816`
        * Updated ``Woodwork`` structures returned from components to support ``Woodwork`` logical type overrides set by the user :pr:`1784`
        * Updated estimators to keep track of input feature names during ``fit()`` :pr:`1794`
        * Updated ``visualize_decision_tree`` to include feature names in output :pr:`1813`
        * Added ``is_bounded_like_percentage`` property for objectives. If true, the ``calculate_percent_difference`` method will return the absolute difference rather than relative difference :pr:`1809`
        * Added full error traceback to AutoMLSearch logger file :pr:`1840`
        * Changed ``TargetEncoder`` to preserve custom indices in the data :pr:`1836`
        * Refactored ``explain_predictions`` and ``explain_predictions_best_worst`` to only compute features once for all rows that need to be explained :pr:`1843`
        * Added custom random undersampler data splitter for classification :pr:`1857`
        * Updated ``OutliersDataCheck`` implementation to calculate the probability of having no outliers :pr:`1855`
        * Added ``Engines`` pipeline processing API :pr:`1838`
    * Fixes
        * Changed EngineBase random_state arg to random_seed and same for user guide docs :pr:`1889`
    * Changes
        * Modified ``calculate_percent_difference`` so that division by 0 is now inf rather than nan :pr:`1809`
        * Removed ``text_columns`` parameter from ``LSA`` and ``TextFeaturizer`` components :pr:`1652`
        * Added ``random_seed`` as an argument to our automl/pipeline/component API. Using ``random_state`` will raise a warning :pr:`1798`
        * Added ``DataCheckError`` message in ``InvalidTargetDataCheck`` if input target is None and removed exception raised :pr:`1866`
    * Documentation Changes
    * Testing Changes
        * Added back coverage for ``_get_feature_provenance`` in ``TextFeaturizer`` after ``text_columns`` was removed :pr:`1842`
        * Pin graphviz version for windows builds :pr:`1847`
        * Unpin graphviz version for windows builds :pr:`1851`

.. warning::

    **Breaking Changes**
        * Added a deprecation warning to ``explain_prediction``. It will be deleted in the next release. :pr:`1860`


**v0.18.2 Feb. 10, 2021**
    * Enhancements
        * Added uniqueness score data check :pr:`1785`
        * Added "dataframe" output format for prediction explanations :pr:`1781`
        * Updated LightGBM estimators to handle ``pandas.MultiIndex`` :pr:`1770`
        * Sped up permutation importance for some pipelines :pr:`1762`
        * Added sparsity data check :pr:`1797`
        * Confirmed support for threshold tuning for binary time series classification problems :pr:`1803`
    * Fixes
    * Changes
    * Documentation Changes
        * Added section on conda to the contributing guide :pr:`1771`
        * Updated release process to reflect freezing `main` before perf tests :pr:`1787`
        * Moving some prs to the right section of the release notes :pr:`1789`
        * Tweak README.md. :pr:`1800`
        * Fixed back arrow on install page docs :pr:`1795`
        * Fixed docstring for `ClassImbalanceDataCheck.validate()` :pr:`1817`
    * Testing Changes

**v0.18.1 Feb. 1, 2021**
    * Enhancements
        * Added ``graph_t_sne`` as a visualization tool for high dimensional data :pr:`1731`
        * Added the ability to see the linear coefficients of features in linear models terms :pr:`1738`
        * Added support for ``scikit-learn`` ``v0.24.0`` :pr:`1733`
        * Added support for ``scipy`` ``v1.6.0`` :pr:`1752`
        * Added SVM Classifier and Regressor to estimators :pr:`1714` :pr:`1761`
    * Fixes
        * Addressed bug with ``partial_dependence`` and categorical data with more categories than grid resolution :pr:`1748`
        * Removed ``random_state`` arg from ``get_pipelines`` in ``AutoMLSearch`` :pr:`1719`
        * Pinned pyzmq at less than 22.0.0 till we add support :pr:`1756`
        * Remove ``ProphetRegressor`` from main as windows tests were flaky :pr:`1764`
    * Changes
        * Updated components and pipelines to return ``Woodwork`` data structures :pr:`1668`
        * Updated ``clone()`` for pipelines and components to copy over random state automatically :pr:`1753`
        * Dropped support for Python version 3.6 :pr:`1751`
        * Removed deprecated ``verbose`` flag from ``AutoMLSearch`` parameters :pr:`1772`
    * Documentation Changes
        * Add Twitter and Github link to documentation toolbar :pr:`1754`
        * Added Open Graph info to documentation :pr:`1758`
    * Testing Changes

.. warning::

    **Breaking Changes**
        * Components and pipelines return ``Woodwork`` data structures instead of ``pandas`` data structures :pr:`1668`
        * Python 3.6 will not be actively supported due to discontinued support from EvalML dependencies.
        * Deprecated ``verbose`` flag is removed for ``AutoMLSearch`` :pr:`1772`


**v0.18.0 Jan. 26, 2021**
    * Enhancements
        * Added RMSLE, MSLE, and MAPE to core objectives while checking for negative target values in ``invalid_targets_data_check`` :pr:`1574`
        * Added validation checks for binary problems with regression-like datasets and multiclass problems without true multiclass targets in ``invalid_targets_data_check`` :pr:`1665`
        * Added time series support for ``make_pipeline`` :pr:`1566`
        * Added target name for output of pipeline ``predict`` method :pr:`1578`
        * Added multiclass check to ``InvalidTargetDataCheck`` for two examples per class :pr:`1596`
        * Added support for ``graphviz`` ``v0.16`` :pr:`1657`
        * Enhanced time series pipelines to accept empty features :pr:`1651`
        * Added KNN Classifier to estimators. :pr:`1650`
        * Added support for list inputs for objectives :pr:`1663`
        * Added support for ``AutoMLSearch`` to handle time series classification pipelines :pr:`1666`
        * Enhanced ``DelayedFeaturesTransformer`` to encode categorical features and targets before delaying them :pr:`1691`
        * Added 2-way dependence plots. :pr:`1690`
        * Added ability to directly iterate through components within Pipelines :pr:`1583`
    * Fixes
        * Fixed inconsistent attributes and added Exceptions to docs :pr:`1673`
        * Fixed ``TargetLeakageDataCheck`` to use Woodwork ``mutual_information`` rather than using Pandas' Pearson Correlation :pr:`1616`
        * Fixed thresholding for pipelines in ``AutoMLSearch`` to only threshold binary classification pipelines :pr:`1622` :pr:`1626`
        * Updated ``load_data`` to return Woodwork structures and update default parameter value for ``index`` to ``None`` :pr:`1610`
        * Pinned scipy at < 1.6.0 while we work on adding support :pr:`1629`
        * Fixed data check message formatting in ``AutoMLSearch`` :pr:`1633`
        * Addressed stacked ensemble component for ``scikit-learn`` v0.24 support by setting ``shuffle=True`` for default CV :pr:`1613`
        * Fixed bug where ``Imputer`` reset the index on ``X`` :pr:`1590`
        * Fixed ``AutoMLSearch`` stacktrace when a cutom objective was passed in as a primary objective or additional objective :pr:`1575`
        * Fixed custom index bug for ``MAPE`` objective :pr:`1641`
        * Fixed index bug for ``TextFeaturizer`` and ``LSA`` components :pr:`1644`
        * Limited ``load_fraud`` dataset loaded into ``automl.ipynb`` :pr:`1646`
        * ``add_to_rankings`` updates ``AutoMLSearch.best_pipeline`` when necessary :pr:`1647`
        * Fixed bug where time series baseline estimators were not receiving ``gap`` and ``max_delay`` in ``AutoMLSearch`` :pr:`1645`
        * Fixed jupyter notebooks to help the RTD buildtime :pr:`1654`
        * Added ``positive_only`` objectives to ``non_core_objectives`` :pr:`1661`
        * Fixed stacking argument ``n_jobs`` for IterativeAlgorithm :pr:`1706`
        * Updated CatBoost estimators to return self in ``.fit()`` rather than the underlying model for consistency :pr:`1701`
        * Added ability to initialize pipeline parameters in ``AutoMLSearch`` constructor :pr:`1676`
    * Changes
        * Added labeling to ``graph_confusion_matrix`` :pr:`1632`
        * Rerunning search for ``AutoMLSearch`` results in a message thrown rather than failing the search, and removed ``has_searched`` property :pr:`1647`
        * Changed tuner class to allow and ignore single parameter values as input :pr:`1686`
        * Capped LightGBM version limit to remove bug in docs :pr:`1711`
        * Removed support for `np.random.RandomState` in EvalML :pr:`1727`
    * Documentation Changes
        * Update Model Understanding in the user guide to include ``visualize_decision_tree`` :pr:`1678`
        * Updated docs to include information about ``AutoMLSearch`` callback parameters and methods :pr:`1577`
        * Updated docs to prompt users to install graphiz on Mac :pr:`1656`
        * Added ``infer_feature_types`` to the ``start.ipynb`` guide :pr:`1700`
        * Added multicollinearity data check to API reference and docs :pr:`1707`
    * Testing Changes

.. warning::

    **Breaking Changes**
        * Removed ``has_searched`` property from ``AutoMLSearch`` :pr:`1647`
        * Components and pipelines return ``Woodwork`` data structures instead of ``pandas`` data structures :pr:`1668`
        * Removed support for `np.random.RandomState` in EvalML. Rather than passing ``np.random.RandomState`` as component and pipeline random_state values, we use int random_seed :pr:`1727`


**v0.17.0 Dec. 29, 2020**
    * Enhancements
        * Added ``save_plot`` that allows for saving figures from different backends :pr:`1588`
        * Added ``LightGBM Regressor`` to regression components :pr:`1459`
        * Added ``visualize_decision_tree`` for tree visualization with ``decision_tree_data_from_estimator`` and ``decision_tree_data_from_pipeline`` to reformat tree structure output :pr:`1511`
        * Added `DFS Transformer` component into transformer components :pr:`1454`
        * Added ``MAPE`` to the standard metrics for time series problems and update objectives :pr:`1510`
        * Added ``graph_prediction_vs_actual_over_time`` and ``get_prediction_vs_actual_over_time_data`` to the model understanding module for time series problems :pr:`1483`
        * Added a ``ComponentGraph`` class that will support future pipelines as directed acyclic graphs :pr:`1415`
        * Updated data checks to accept ``Woodwork`` data structures :pr:`1481`
        * Added parameter to ``InvalidTargetDataCheck`` to show only top unique values rather than all unique values :pr:`1485`
        * Added multicollinearity data check :pr:`1515`
        * Added baseline pipeline and components for time series regression problems :pr:`1496`
        * Added more information to users about ensembling behavior in ``AutoMLSearch`` :pr:`1527`
        * Add woodwork support for more utility and graph methods :pr:`1544`
        * Changed ``DateTimeFeaturizer`` to encode features as int :pr:`1479`
        * Return trained pipelines from ``AutoMLSearch.best_pipeline`` :pr:`1547`
        * Added utility method so that users can set feature types without having to learn about Woodwork directly :pr:`1555`
        * Added Linear Discriminant Analysis transformer for dimensionality reduction :pr:`1331`
        * Added multiclass support for ``partial_dependence`` and ``graph_partial_dependence`` :pr:`1554`
        * Added ``TimeSeriesBinaryClassificationPipeline`` and ``TimeSeriesMulticlassClassificationPipeline`` classes :pr:`1528`
        * Added ``make_data_splitter`` method for easier automl data split customization :pr:`1568`
        * Integrated ``ComponentGraph`` class into Pipelines for full non-linear pipeline support :pr:`1543`
        * Update ``AutoMLSearch`` constructor to take training data instead of ``search`` and ``add_to_leaderboard`` :pr:`1597`
        * Update ``split_data`` helper args :pr:`1597`
        * Add problem type utils ``is_regression``, ``is_classification``, ``is_timeseries`` :pr:`1597`
        * Rename ``AutoMLSearch`` ``data_split`` arg to ``data_splitter`` :pr:`1569`
    * Fixes
        * Fix AutoML not passing CV folds to ``DefaultDataChecks`` for usage by ``ClassImbalanceDataCheck`` :pr:`1619`
        * Fix Windows CI jobs: install ``numba`` via conda, required for ``shap`` :pr:`1490`
        * Added custom-index support for `reset-index-get_prediction_vs_actual_over_time_data` :pr:`1494`
        * Fix ``generate_pipeline_code`` to account for boolean and None differences between Python and JSON :pr:`1524` :pr:`1531`
        * Set max value for plotly and xgboost versions while we debug CI failures with newer versions :pr:`1532`
        * Undo version pinning for plotly :pr:`1533`
        * Fix ReadTheDocs build by updating the version of ``setuptools`` :pr:`1561`
        * Set ``random_state`` of data splitter in AutoMLSearch to take int to keep consistency in the resulting splits :pr:`1579`
        * Pin sklearn version while we work on adding support :pr:`1594`
        * Pin pandas at <1.2.0 while we work on adding support :pr:`1609`
        * Pin graphviz at < 0.16 while we work on adding support :pr:`1609`
    * Changes
        * Reverting ``save_graph`` :pr:`1550` to resolve kaleido build issues :pr:`1585`
        * Update circleci badge to apply to ``main`` :pr:`1489`
        * Added script to generate github markdown for releases :pr:`1487`
        * Updated selection using pandas ``dtypes`` to selecting using Woodwork logical types :pr:`1551`
        * Updated dependencies to fix ``ImportError: cannot import name 'MaskedArray' from 'sklearn.utils.fixes'`` error and to address Woodwork and Featuretool dependencies :pr:`1540`
        * Made ``get_prediction_vs_actual_data()`` a public method :pr:`1553`
        * Updated ``Woodwork`` version requirement to v0.0.7 :pr:`1560`
        * Move data splitters from ``evalml.automl.data_splitters`` to ``evalml.preprocessing.data_splitters`` :pr:`1597`
        * Rename "# Testing" in automl log output to "# Validation" :pr:`1597`
    * Documentation Changes
        * Added partial dependence methods to API reference :pr:`1537`
        * Updated documentation for confusion matrix methods :pr:`1611`
    * Testing Changes
        * Set ``n_jobs=1`` in most unit tests to reduce memory :pr:`1505`

.. warning::

    **Breaking Changes**
        * Updated minimal dependencies: ``numpy>=1.19.1``, ``pandas>=1.1.0``, ``scikit-learn>=0.23.1``, ``scikit-optimize>=0.8.1``
        * Updated ``AutoMLSearch.best_pipeline`` to return a trained pipeline. Pass in ``train_best_pipeline=False`` to AutoMLSearch in order to return an untrained pipeline.
        * Pipeline component instances can no longer be iterated through using ``Pipeline.component_graph`` :pr:`1543`
        * Update ``AutoMLSearch`` constructor to take training data instead of ``search`` and ``add_to_leaderboard`` :pr:`1597`
        * Update ``split_data`` helper args :pr:`1597`
        * Move data splitters from ``evalml.automl.data_splitters`` to ``evalml.preprocessing.data_splitters`` :pr:`1597`
        * Rename ``AutoMLSearch`` ``data_split`` arg to ``data_splitter`` :pr:`1569`



**v0.16.1 Dec. 1, 2020**
    * Enhancements
        * Pin woodwork version to v0.0.6 to avoid breaking changes :pr:`1484`
        * Updated ``Woodwork`` to >=0.0.5 in ``core-requirements.txt`` :pr:`1473`
        * Removed ``copy_dataframe`` parameter for ``Woodwork``, updated ``Woodwork`` to >=0.0.6 in ``core-requirements.txt`` :pr:`1478`
        * Updated ``detect_problem_type`` to use ``pandas.api.is_numeric_dtype`` :pr:`1476`
    * Changes
        * Changed ``make clean`` to delete coverage reports as a convenience for developers :pr:`1464`
        * Set ``n_jobs=-1`` by default for stacked ensemble components :pr:`1472`
    * Documentation Changes
        * Updated pipeline and component documentation and demos to use ``Woodwork`` :pr:`1466`
    * Testing Changes
        * Update dependency update checker to use everything from core and optional dependencies :pr:`1480`


**v0.16.0 Nov. 24, 2020**
    * Enhancements
        * Updated pipelines and ``make_pipeline`` to accept ``Woodwork`` inputs :pr:`1393`
        * Updated components to accept ``Woodwork`` inputs :pr:`1423`
        * Added ability to freeze hyperparameters for ``AutoMLSearch`` :pr:`1284`
        * Added ``Target Encoder`` into transformer components :pr:`1401`
        * Added callback for error handling in ``AutoMLSearch`` :pr:`1403`
        * Added the index id to the ``explain_predictions_best_worst`` output to help users identify which rows in their data are included :pr:`1365`
        * The top_k features displayed in ``explain_predictions_*`` functions are now determined by the magnitude of shap values as opposed to the ``top_k`` largest and smallest shap values. :pr:`1374`
        * Added a problem type for time series regression :pr:`1386`
        * Added a ``is_defined_for_problem_type`` method to ``ObjectiveBase`` :pr:`1386`
        * Added a ``random_state`` parameter to ``make_pipeline_from_components`` function :pr:`1411`
        * Added ``DelayedFeaturesTransformer`` :pr:`1396`
        * Added a ``TimeSeriesRegressionPipeline`` class :pr:`1418`
        * Removed ``core-requirements.txt`` from the package distribution :pr:`1429`
        * Updated data check messages to include a `"code"` and `"details"` fields :pr:`1451`, :pr:`1462`
        * Added a ``TimeSeriesSplit`` data splitter for time series problems :pr:`1441`
        * Added a ``problem_configuration`` parameter to AutoMLSearch :pr:`1457`
    * Fixes
        * Fixed ``IndexError`` raised in ``AutoMLSearch`` when ``ensembling = True`` but only one pipeline to iterate over :pr:`1397`
        * Fixed stacked ensemble input bug and LightGBM warning and bug in ``AutoMLSearch`` :pr:`1388`
        * Updated enum classes to show possible enum values as attributes :pr:`1391`
        * Updated calls to ``Woodwork``'s ``to_pandas()`` to ``to_series()`` and ``to_dataframe()`` :pr:`1428`
        * Fixed bug in OHE where column names were not guaranteed to be unique :pr:`1349`
        * Fixed bug with percent improvement of ``ExpVariance`` objective on data with highly skewed target :pr:`1467`
        * Fix SimpleImputer error which occurs when all features are bool type :pr:`1215`
    * Changes
        * Changed ``OutliersDataCheck`` to return the list of columns, rather than rows, that contain outliers :pr:`1377`
        * Simplified and cleaned output for Code Generation :pr:`1371`
        * Reverted changes from :pr:`1337` :pr:`1409`
        * Updated data checks to return dictionary of warnings and errors instead of a list :pr:`1448`
        * Updated ``AutoMLSearch`` to pass ``Woodwork`` data structures to every pipeline (instead of pandas DataFrames) :pr:`1450`
        * Update ``AutoMLSearch`` to default to ``max_batches=1`` instead of ``max_iterations=5`` :pr:`1452`
        * Updated _evaluate_pipelines to consolidate side effects :pr:`1410`
    * Documentation Changes
        * Added description of CLA to contributing guide, updated description of draft PRs :pr:`1402`
        * Updated documentation to include all data checks, ``DataChecks``, and usage of data checks in AutoML :pr:`1412`
        * Updated docstrings from ``np.array`` to ``np.ndarray`` :pr:`1417`
        * Added section on stacking ensembles in AutoMLSearch documentation :pr:`1425`
    * Testing Changes
        * Removed ``category_encoders`` from test-requirements.txt :pr:`1373`
        * Tweak codecov.io settings again to avoid flakes :pr:`1413`
        * Modified ``make lint`` to check notebook versions in the docs :pr:`1431`
        * Modified ``make lint-fix`` to standardize notebook versions in the docs :pr:`1431`
        * Use new version of pull request Github Action for dependency check (:pr:`1443`)
        * Reduced number of workers for tests to 4 :pr:`1447`

.. warning::

    **Breaking Changes**
        * The ``top_k`` and ``top_k_features`` parameters in ``explain_predictions_*`` functions now return ``k`` features as opposed to ``2 * k`` features :pr:`1374`
        * Renamed ``problem_type`` to ``problem_types`` in ``RegressionObjective``, ``BinaryClassificationObjective``, and ``MulticlassClassificationObjective`` :pr:`1319`
        * Data checks now return a dictionary of warnings and errors instead of a list :pr:`1448`



**v0.15.0 Oct. 29, 2020**
    * Enhancements
        * Added stacked ensemble component classes (``StackedEnsembleClassifier``, ``StackedEnsembleRegressor``) :pr:`1134`
        * Added stacked ensemble components to ``AutoMLSearch`` :pr:`1253`
        * Added ``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` to AutoML :pr:`1255`
        * Added ``graph_prediction_vs_actual`` in ``model_understanding`` for regression problems :pr:`1252`
        * Added parameter to ``OneHotEncoder`` to enable filtering for features to encode for :pr:`1249`
        * Added percent-better-than-baseline for all objectives to automl.results :pr:`1244`
        * Added ``HighVarianceCVDataCheck`` and replaced synonymous warning in ``AutoMLSearch`` :pr:`1254`
        * Added `PCA Transformer` component for dimensionality reduction :pr:`1270`
        * Added ``generate_pipeline_code`` and ``generate_component_code`` to allow for code generation given a pipeline or component instance :pr:`1306`
        * Added ``PCA Transformer`` component for dimensionality reduction :pr:`1270`
        * Updated ``AutoMLSearch`` to support ``Woodwork`` data structures :pr:`1299`
        * Added cv_folds to ``ClassImbalanceDataCheck`` and added this check to ``DefaultDataChecks`` :pr:`1333`
        * Make ``max_batches`` argument to ``AutoMLSearch.search`` public :pr:`1320`
        * Added text support to automl search :pr:`1062`
        * Added ``_pipelines_per_batch`` as a private argument to ``AutoMLSearch`` :pr:`1355`
    * Fixes
        * Fixed ML performance issue with ordered datasets: always shuffle data in automl's default CV splits :pr:`1265`
        * Fixed broken ``evalml info`` CLI command :pr:`1293`
        * Fixed ``boosting type='rf'`` for LightGBM Classifier, as well as ``num_leaves`` error :pr:`1302`
        * Fixed bug in ``explain_predictions_best_worst`` where a custom index in the target variable would cause a ``ValueError`` :pr:`1318`
        * Added stacked ensemble estimators to to ``evalml.pipelines.__init__`` file :pr:`1326`
        * Fixed bug in OHE where calls to transform were not deterministic if ``top_n`` was less than the number of categories in a column :pr:`1324`
        * Fixed LightGBM warning messages during AutoMLSearch :pr:`1342`
        * Fix warnings thrown during AutoMLSearch in ``HighVarianceCVDataCheck`` :pr:`1346`
        * Fixed bug where TrainingValidationSplit would return invalid location indices for dataframes with a custom index :pr:`1348`
        * Fixed bug where the AutoMLSearch ``random_state`` was not being passed to the created pipelines :pr:`1321`
    * Changes
        * Allow ``add_to_rankings`` to be called before AutoMLSearch is called :pr:`1250`
        * Removed Graphviz from test-requirements to add to requirements.txt :pr:`1327`
        * Removed ``max_pipelines`` parameter from ``AutoMLSearch`` :pr:`1264`
        * Include editable installs in all install make targets :pr:`1335`
        * Made pip dependencies `featuretools` and `nlp_primitives` core dependencies :pr:`1062`
        * Removed `PartOfSpeechCount` from `TextFeaturizer` transform primitives :pr:`1062`
        * Added warning for ``partial_dependency`` when the feature includes null values :pr:`1352`
    * Documentation Changes
        * Fixed and updated code blocks in Release Notes :pr:`1243`
        * Added DecisionTree estimators to API Reference :pr:`1246`
        * Changed class inheritance display to flow vertically :pr:`1248`
        * Updated cost-benefit tutorial to use a holdout/test set :pr:`1159`
        * Added ``evalml info`` command to documentation :pr:`1293`
        * Miscellaneous doc updates :pr:`1269`
        * Removed conda pre-release testing from the release process document :pr:`1282`
        * Updates to contributing guide :pr:`1310`
        * Added Alteryx footer to docs with Twitter and Github link :pr:`1312`
        * Added documentation for evalml installation for Python 3.6 :pr:`1322`
        * Added documentation changes to make the API Docs easier to understand :pr:`1323`
        * Fixed documentation for ``feature_importance`` :pr:`1353`
        * Added tutorial for running `AutoML` with text data :pr:`1357`
        * Added documentation for woodwork integration with automl search :pr:`1361`
    * Testing Changes
        * Added tests for ``jupyter_check`` to handle IPython :pr:`1256`
        * Cleaned up ``make_pipeline`` tests to test for all estimators :pr:`1257`
        * Added a test to check conda build after merge to main :pr:`1247`
        * Removed code that was lacking codecov for ``__main__.py`` and unnecessary :pr:`1293`
        * Codecov: round coverage up instead of down :pr:`1334`
        * Add DockerHub credentials to CI testing environment :pr:`1356`
        * Add DockerHub credentials to conda testing environment :pr:`1363`

.. warning::

    **Breaking Changes**
        * Renamed ``LabelLeakageDataCheck`` to ``TargetLeakageDataCheck`` :pr:`1319`
        * ``max_pipelines`` parameter has been removed from ``AutoMLSearch``. Please use ``max_iterations`` instead. :pr:`1264`
        * ``AutoMLSearch.search()`` will now log a warning if the input is not a ``Woodwork`` data structure (``pandas``, ``numpy``) :pr:`1299`
        * Make ``max_batches`` argument to ``AutoMLSearch.search`` public :pr:`1320`
        * Removed unused argument `feature_types` from AutoMLSearch.search :pr:`1062`

**v0.14.1 Sep. 29, 2020**
    * Enhancements
        * Updated partial dependence methods to support calculating numeric columns in a dataset with non-numeric columns :pr:`1150`
        * Added ``get_feature_names`` on ``OneHotEncoder`` :pr:`1193`
        * Added ``detect_problem_type`` to ``problem_type/utils.py`` to automatically detect the problem type given targets :pr:`1194`
        * Added LightGBM to ``AutoMLSearch`` :pr:`1199`
        * Updated ``scikit-learn`` and ``scikit-optimize`` to use latest versions - 0.23.2 and 0.8.1 respectively :pr:`1141`
        * Added ``__str__`` and ``__repr__`` for pipelines and components :pr:`1218`
        * Included internal target check for both training and validation data in ``AutoMLSearch`` :pr:`1226`
        * Added ``ProblemTypes.all_problem_types`` helper to get list of supported problem types :pr:`1219`
        * Added ``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` classes :pr:`1223`
        * Added ``ProblemTypes.all_problem_types`` helper to get list of supported problem types :pr:`1219`
        * ``DataChecks`` can now be parametrized by passing a list of ``DataCheck`` classes and a parameter dictionary :pr:`1167`
        * Added first CV fold score as validation score in ``AutoMLSearch.rankings`` :pr:`1221`
        * Updated ``flake8`` configuration to enable linting on ``__init__.py`` files :pr:`1234`
        * Refined ``make_pipeline_from_components`` implementation :pr:`1204`
    * Fixes
        * Updated GitHub URL after migration to Alteryx GitHub org :pr:`1207`
        * Changed Problem Type enum to be more similar to the string name :pr:`1208`
        * Wrapped call to scikit-learn's partial dependence method in a ``try``/``finally`` block :pr:`1232`
    * Changes
        * Added ``allow_writing_files`` as a named argument to CatBoost estimators. :pr:`1202`
        * Added ``solver`` and ``multi_class`` as named arguments to ``LogisticRegressionClassifier`` :pr:`1202`
        * Replaced pipeline's ``._transform`` method to evaluate all the preprocessing steps of a pipeline with ``.compute_estimator_features`` :pr:`1231`
        * Changed default large dataset train/test splitting behavior :pr:`1205`
    * Documentation Changes
        * Included description of how to access the component instances and features for pipeline user guide :pr:`1163`
        * Updated API docs to refer to target as "target" instead of "labels" for non-classification tasks and minor docs cleanup :pr:`1160`
        * Added Class Imbalance Data Check to ``api_reference.rst`` :pr:`1190` :pr:`1200`
        * Added pipeline properties to API reference :pr:`1209`
        * Clarified what the objective parameter in AutoML is used for in AutoML API reference and AutoML user guide :pr:`1222`
        * Updated API docs to include ``skopt.space.Categorical`` option for component hyperparameter range definition :pr:`1228`
        * Added install documentation for ``libomp`` in order to use LightGBM on Mac :pr:`1233`
        * Improved description of ``max_iterations`` in documentation :pr:`1212`
        * Removed unused code from sphinx conf :pr:`1235`
    * Testing Changes

.. warning::

    **Breaking Changes**
        * ``DefaultDataChecks`` now accepts a ``problem_type`` parameter that must be specified :pr:`1167`
        * Pipeline's ``._transform`` method to evaluate all the preprocessing steps of a pipeline has been replaced with ``.compute_estimator_features`` :pr:`1231`
        * ``get_objectives`` has been renamed to ``get_core_objectives``. This function will now return a list of valid objective instances :pr:`1230`


**v0.13.2 Sep. 17, 2020**
    * Enhancements
        * Added ``output_format`` field to explain predictions functions :pr:`1107`
        * Modified ``get_objective`` and ``get_objectives`` to be able to return any objective in ``evalml.objectives`` :pr:`1132`
        * Added a ``return_instance`` boolean parameter to ``get_objective`` :pr:`1132`
        * Added ``ClassImbalanceDataCheck`` to determine whether target imbalance falls below a given threshold :pr:`1135`
        * Added label encoder to LightGBM for binary classification :pr:`1152`
        * Added labels for the row index of confusion matrix :pr:`1154`
        * Added ``AutoMLSearch`` object as another parameter in search callbacks :pr:`1156`
        * Added the corresponding probability threshold for each point displayed in ``graph_roc_curve`` :pr:`1161`
        * Added ``__eq__`` for ``ComponentBase`` and ``PipelineBase`` :pr:`1178`
        * Added support for multiclass classification for ``roc_curve`` :pr:`1164`
        * Added ``categories`` accessor to ``OneHotEncoder`` for listing the categories associated with a feature :pr:`1182`
        * Added utility function to create pipeline instances from a list of component instances :pr:`1176`
    * Fixes
        * Fixed XGBoost column names for partial dependence methods :pr:`1104`
        * Removed dead code validating column type from ``TextFeaturizer`` :pr:`1122`
        * Fixed issue where ``Imputer`` cannot fit when there is None in a categorical or boolean column :pr:`1144`
        * ``OneHotEncoder`` preserves the custom index in the input data :pr:`1146`
        * Fixed representation for ``ModelFamily`` :pr:`1165`
        * Removed duplicate ``nbsphinx`` dependency in ``dev-requirements.txt`` :pr:`1168`
        * Users can now pass in any valid kwargs to all estimators :pr:`1157`
        * Remove broken accessor ``OneHotEncoder.get_feature_names`` and unneeded base class :pr:`1179`
        * Removed LightGBM Estimator from AutoML models :pr:`1186`
    * Changes
        * Pinned ``scikit-optimize`` version to 0.7.4 :pr:`1136`
        * Removed ``tqdm`` as a dependency :pr:`1177`
        * Added lightgbm version 3.0.0 to ``latest_dependency_versions.txt`` :pr:`1185`
        * Rename ``max_pipelines`` to ``max_iterations`` :pr:`1169`
    * Documentation Changes
        * Fixed API docs for ``AutoMLSearch`` ``add_result_callback`` :pr:`1113`
        * Added a step to our release process for pushing our latest version to conda-forge :pr:`1118`
        * Added warning for missing ipywidgets dependency for using ``PipelineSearchPlots`` on Jupyterlab :pr:`1145`
        * Updated ``README.md`` example to load demo dataset :pr:`1151`
        * Swapped mapping of breast cancer targets in ``model_understanding.ipynb`` :pr:`1170`
    * Testing Changes
        * Added test confirming ``TextFeaturizer`` never outputs null values :pr:`1122`
        * Changed Python version of ``Update Dependencies`` action to 3.8.x :pr:`1137`
        * Fixed release notes check-in test for ``Update Dependencies`` actions :pr:`1172`

.. warning::

    **Breaking Changes**
        * ``get_objective`` will now return a class definition rather than an instance by default :pr:`1132`
        * Deleted ``OPTIONS`` dictionary in ``evalml.objectives.utils.py`` :pr:`1132`
        * If specifying an objective by string, the string must now match the objective's name field, case-insensitive :pr:`1132`
        * Passing "Cost Benefit Matrix", "Fraud Cost", "Lead Scoring", "Mean Squared Log Error",
            "Recall", "Recall Macro", "Recall Micro", "Recall Weighted", or "Root Mean Squared Log Error" to ``AutoMLSearch`` will now result in a ``ValueError``
            rather than an ``ObjectiveNotFoundError`` :pr:`1132`
        * Search callbacks ``start_iteration_callback`` and ``add_results_callback`` have changed to include a copy of the AutoMLSearch object as a third parameter :pr:`1156`
        * Deleted ``OneHotEncoder.get_feature_names`` method which had been broken for a while, in favor of pipelines' ``input_feature_names`` :pr:`1179`
        * Deleted empty base class ``CategoricalEncoder`` which ``OneHotEncoder`` component was inheriting from :pr:`1176`
        * Results from ``roc_curve`` will now return as a list of dictionaries with each dictionary representing a class :pr:`1164`
        * ``max_pipelines`` now raises a ``DeprecationWarning`` and will be removed in the next release. ``max_iterations`` should be used instead. :pr:`1169`


**v0.13.1 Aug. 25, 2020**
    * Enhancements
        * Added Cost-Benefit Matrix objective for binary classification :pr:`1038`
        * Split ``fill_value`` into ``categorical_fill_value`` and ``numeric_fill_value`` for Imputer :pr:`1019`
        * Added ``explain_predictions`` and ``explain_predictions_best_worst`` for explaining multiple predictions with SHAP :pr:`1016`
        * Added new LSA component for text featurization :pr:`1022`
        * Added guide on installing with conda :pr:`1041`
        * Added a “cost-benefit curve” util method to graph cost-benefit matrix scores vs. binary classification thresholds :pr:`1081`
        * Standardized error when calling transform/predict before fit for pipelines :pr:`1048`
        * Added ``percent_better_than_baseline`` to AutoML search rankings and full rankings table :pr:`1050`
        * Added one-way partial dependence and partial dependence plots :pr:`1079`
        * Added "Feature Value" column to prediction explanation reports. :pr:`1064`
        * Added LightGBM classification estimator :pr:`1082`, :pr:`1114`
        * Added ``max_batches`` parameter to ``AutoMLSearch`` :pr:`1087`
    * Fixes
        * Updated ``TextFeaturizer`` component to no longer require an internet connection to run :pr:`1022`
        * Fixed non-deterministic element of ``TextFeaturizer`` transformations :pr:`1022`
        * Added a StandardScaler to all ElasticNet pipelines :pr:`1065`
        * Updated cost-benefit matrix to normalize score :pr:`1099`
        * Fixed logic in ``calculate_percent_difference`` so that it can handle negative values :pr:`1100`
    * Changes
        * Added ``needs_fitting`` property to ``ComponentBase`` :pr:`1044`
        * Updated references to data types to use datatype lists defined in ``evalml.utils.gen_utils`` :pr:`1039`
        * Remove maximum version limit for SciPy dependency :pr:`1051`
        * Moved ``all_components`` and other component importers into runtime methods :pr:`1045`
        * Consolidated graphing utility methods under ``evalml.utils.graph_utils`` :pr:`1060`
        * Made slight tweaks to how ``TextFeaturizer`` uses ``featuretools``, and did some refactoring of that and of LSA :pr:`1090`
        * Changed ``show_all_features`` parameter into ``importance_threshold``, which allows for thresholding feature importance :pr:`1097`, :pr:`1103`
    * Documentation Changes
        * Update ``setup.py`` URL to point to the github repo :pr:`1037`
        * Added tutorial for using the cost-benefit matrix objective :pr:`1088`
        * Updated ``model_understanding.ipynb`` to include documentation for using plotly on Jupyter Lab :pr:`1108`
    * Testing Changes
        * Refactor CircleCI tests to use matrix jobs (:pr:`1043`)
        * Added a test to check that all test directories are included in evalml package :pr:`1054`


.. warning::

    **Breaking Changes**
        * ``confusion_matrix`` and ``normalize_confusion_matrix`` have been moved to ``evalml.utils`` :pr:`1038`
        * All graph utility methods previously under ``evalml.pipelines.graph_utils`` have been moved to ``evalml.utils.graph_utils`` :pr:`1060`


**v0.12.2 Aug. 6, 2020**
    * Enhancements
        * Add save/load method to components :pr:`1023`
        * Expose pickle ``protocol`` as optional arg to save/load :pr:`1023`
        * Updated estimators used in AutoML to include ExtraTrees and ElasticNet estimators :pr:`1030`
    * Fixes
    * Changes
        * Removed ``DeprecationWarning`` for ``SimpleImputer`` :pr:`1018`
    * Documentation Changes
        * Add note about version numbers to release process docs :pr:`1034`
    * Testing Changes
        * Test files are now included in the evalml package :pr:`1029`


**v0.12.0 Aug. 3, 2020**
    * Enhancements
        * Added string and categorical targets support for binary and multiclass pipelines and check for numeric targets for ``DetectLabelLeakage`` data check :pr:`932`
        * Added clear exception for regression pipelines if target datatype is string or categorical :pr:`960`
        * Added target column names and class labels in ``predict`` and ``predict_proba`` output for pipelines :pr:`951`
        * Added ``_compute_shap_values`` and ``normalize_values`` to ``pipelines/explanations`` module :pr:`958`
        * Added ``explain_prediction`` feature which explains single predictions with SHAP :pr:`974`
        * Added Imputer to allow different imputation strategies for numerical and categorical dtypes :pr:`991`
        * Added support for configuring logfile path using env var, and don't create logger if there are filesystem errors :pr:`975`
        * Updated catboost estimators' default parameters and automl hyperparameter ranges to speed up fit time :pr:`998`
    * Fixes
        * Fixed ReadtheDocs warning failure regarding embedded gif :pr:`943`
        * Removed incorrect parameter passed to pipeline classes in ``_add_baseline_pipelines`` :pr:`941`
        * Added universal error for calling ``predict``, ``predict_proba``, ``transform``, and ``feature_importances`` before fitting :pr:`969`, :pr:`994`
        * Made ``TextFeaturizer`` component and pip dependencies ``featuretools`` and ``nlp_primitives`` optional :pr:`976`
        * Updated imputation strategy in automl to no longer limit impute strategy to ``most_frequent`` for all features if there are any categorical columns :pr:`991`
        * Fixed ``UnboundLocalError`` for ``cv_pipeline`` when automl search errors :pr:`996`
        * Fixed ``Imputer`` to reset dataframe index to preserve behavior expected from  ``SimpleImputer`` :pr:`1009`
    * Changes
        * Moved ``get_estimators`` to ``evalml.pipelines.components.utils`` :pr:`934`
        * Modified Pipelines to raise ``PipelineScoreError`` when they encounter an error during scoring :pr:`936`
        * Moved ``evalml.model_families.list_model_families`` to ``evalml.pipelines.components.allowed_model_families`` :pr:`959`
        * Renamed ``DateTimeFeaturization`` to ``DateTimeFeaturizer`` :pr:`977`
        * Added check to stop search and raise an error if all pipelines in a batch return NaN scores :pr:`1015`
    * Documentation Changes
        * Updated ``README.md`` :pr:`963`
        * Reworded message when errors are returned from data checks in search :pr:`982`
        * Added section on understanding model predictions with ``explain_prediction`` to User Guide :pr:`981`
        * Added a section to the user guide and api reference about how XGBoost and CatBoost are not fully supported. :pr:`992`
        * Added custom components section in user guide :pr:`993`
        * Updated FAQ section formatting :pr:`997`
        * Updated release process documentation :pr:`1003`
    * Testing Changes
        * Moved ``predict_proba`` and ``predict`` tests regarding string / categorical targets to ``test_pipelines.py`` :pr:`972`
        * Fixed dependency update bot by updating python version to 3.7 to avoid frequent github version updates :pr:`1002`


.. warning::

    **Breaking Changes**
        * ``get_estimators`` has been moved to ``evalml.pipelines.components.utils`` (previously was under ``evalml.pipelines.utils``) :pr:`934`
        * Removed the ``raise_errors`` flag in AutoML search. All errors during pipeline evaluation will be caught and logged. :pr:`936`
        * ``evalml.model_families.list_model_families`` has been moved to ``evalml.pipelines.components.allowed_model_families`` :pr:`959`
        * ``TextFeaturizer``: the ``featuretools`` and ``nlp_primitives`` packages must be installed after installing evalml in order to use this component :pr:`976`
        * Renamed ``DateTimeFeaturization`` to ``DateTimeFeaturizer`` :pr:`977`


**v0.11.2 July 16, 2020**
    * Enhancements
        * Added ``NoVarianceDataCheck`` to ``DefaultDataChecks`` :pr:`893`
        * Added text processing and featurization component ``TextFeaturizer`` :pr:`913`, :pr:`924`
        * Added additional checks to ``InvalidTargetDataCheck`` to handle invalid target data types :pr:`929`
        * ``AutoMLSearch`` will now handle ``KeyboardInterrupt`` and prompt user for confirmation :pr:`915`
    * Fixes
        * Makes automl results a read-only property :pr:`919`
    * Changes
        * Deleted static pipelines and refactored tests involving static pipelines, removed ``all_pipelines()`` and ``get_pipelines()`` :pr:`904`
        * Moved ``list_model_families`` to ``evalml.model_family.utils`` :pr:`903`
        * Updated ``all_pipelines``, ``all_estimators``, ``all_components`` to use the same mechanism for dynamically generating their elements :pr:`898`
        * Rename ``master`` branch to ``main`` :pr:`918`
        * Add pypi release github action :pr:`923`
        * Updated ``AutoMLSearch.search`` stdout output and logging and removed tqdm progress bar :pr:`921`
        * Moved automl config checks previously in ``search()`` to init :pr:`933`
    * Documentation Changes
        * Reorganized and rewrote documentation :pr:`937`
        * Updated to use pydata sphinx theme :pr:`937`
        * Updated docs to use ``release_notes`` instead of ``changelog`` :pr:`942`
    * Testing Changes
        * Cleaned up fixture names and usages in tests :pr:`895`


.. warning::

    **Breaking Changes**
        * ``list_model_families`` has been moved to ``evalml.model_family.utils`` (previously was under ``evalml.pipelines.utils``) :pr:`903`
        * ``get_estimators`` has been moved to ``evalml.pipelines.components.utils`` (previously was under ``evalml.pipelines.utils``) :pr:`934`
        * Static pipeline definitions have been removed, but similar pipelines can still be constructed via creating an instance of ``PipelineBase`` :pr:`904`
        * ``all_pipelines()`` and ``get_pipelines()`` utility methods have been removed :pr:`904`


**v0.11.0 June 30, 2020**
    * Enhancements
        * Added multiclass support for ROC curve graphing :pr:`832`
        * Added preprocessing component to drop features whose percentage of NaN values exceeds a specified threshold :pr:`834`
        * Added data check to check for problematic target labels :pr:`814`
        * Added PerColumnImputer that allows imputation strategies per column :pr:`824`
        * Added transformer to drop specific columns :pr:`827`
        * Added support for ``categories``, ``handle_error``, and ``drop`` parameters in ``OneHotEncoder`` :pr:`830` :pr:`897`
        * Added preprocessing component to handle DateTime columns featurization :pr:`838`
        * Added ability to clone pipelines and components :pr:`842`
        * Define getter method for component ``parameters`` :pr:`847`
        * Added utility methods to calculate and graph permutation importances :pr:`860`, :pr:`880`
        * Added new utility functions necessary for generating dynamic preprocessing pipelines :pr:`852`
        * Added kwargs to all components :pr:`863`
        * Updated ``AutoSearchBase`` to use dynamically generated preprocessing pipelines :pr:`870`
        * Added SelectColumns transformer :pr:`873`
        * Added ability to evaluate additional pipelines for automl search :pr:`874`
        * Added ``default_parameters`` class property to components and pipelines :pr:`879`
        * Added better support for disabling data checks in automl search :pr:`892`
        * Added ability to save and load AutoML objects to file :pr:`888`
        * Updated ``AutoSearchBase.get_pipelines`` to return an untrained pipeline instance :pr:`876`
        * Saved learned binary classification thresholds in automl results cv data dict :pr:`876`
    * Fixes
        * Fixed bug where SimpleImputer cannot handle dropped columns :pr:`846`
        * Fixed bug where PerColumnImputer cannot handle dropped columns :pr:`855`
        * Enforce requirement that builtin components save all inputted values in their parameters dict :pr:`847`
        * Don't list base classes in ``all_components`` output :pr:`847`
        * Standardize all components to output pandas data structures, and accept either pandas or numpy :pr:`853`
        * Fixed rankings and full_rankings error when search has not been run :pr:`894`
    * Changes
        * Update ``all_pipelines`` and ``all_components`` to try initializing pipelines/components, and on failure exclude them :pr:`849`
        * Refactor ``handle_components`` to ``handle_components_class``, standardize to ``ComponentBase`` subclass instead of instance :pr:`850`
        * Refactor "blacklist"/"whitelist" to "allow"/"exclude" lists :pr:`854`
        * Replaced ``AutoClassificationSearch`` and ``AutoRegressionSearch`` with ``AutoMLSearch`` :pr:`871`
        * Renamed feature_importances and permutation_importances methods to use singular names (feature_importance and permutation_importance) :pr:`883`
        * Updated ``automl`` default data splitter to train/validation split for large datasets :pr:`877`
        * Added open source license, update some repo metadata :pr:`887`
        * Removed dead code in ``_get_preprocessing_components`` :pr:`896`
    * Documentation Changes
        * Fix some typos and update the EvalML logo :pr:`872`
    * Testing Changes
        * Update the changelog check job to expect the new branching pattern for the deps update bot :pr:`836`
        * Check that all components output pandas datastructures, and can accept either pandas or numpy :pr:`853`
        * Replaced ``AutoClassificationSearch`` and ``AutoRegressionSearch`` with ``AutoMLSearch`` :pr:`871`


.. warning::

    **Breaking Changes**
        * Pipelines' static ``component_graph`` field must contain either ``ComponentBase`` subclasses or ``str``, instead of ``ComponentBase`` subclass instances :pr:`850`
        * Rename ``handle_component`` to ``handle_component_class``. Now standardizes to ``ComponentBase`` subclasses instead of ``ComponentBase`` subclass instances :pr:`850`
        * Renamed automl's ``cv`` argument to ``data_split`` :pr:`877`
        * Pipelines' and classifiers' ``feature_importances`` is renamed ``feature_importance``, ``graph_feature_importances`` is renamed ``graph_feature_importance`` :pr:`883`
        * Passing ``data_checks=None`` to automl search will not perform any data checks as opposed to default checks. :pr:`892`
        * Pipelines to search for in AutoML are now determined automatically, rather than using the statically-defined pipeline classes. :pr:`870`
        * Updated ``AutoSearchBase.get_pipelines`` to return an untrained pipeline instance, instead of one which happened to be trained on the final cross-validation fold :pr:`876`


**v0.10.0 May 29, 2020**
    * Enhancements
        * Added baseline models for classification and regression, add functionality to calculate baseline models before searching in AutoML :pr:`746`
        * Port over highly-null guardrail as a data check and define ``DefaultDataChecks`` and ``DisableDataChecks`` classes :pr:`745`
        * Update ``Tuner`` classes to work directly with pipeline parameters dicts instead of flat parameter lists :pr:`779`
        * Add Elastic Net as a pipeline option :pr:`812`
        * Added new Pipeline option ``ExtraTrees`` :pr:`790`
        * Added precicion-recall curve metrics and plot for binary classification problems in ``evalml.pipeline.graph_utils`` :pr:`794`
        * Update the default automl algorithm to search in batches, starting with default parameters for each pipeline and iterating from there :pr:`793`
        * Added ``AutoMLAlgorithm`` class and ``IterativeAlgorithm`` impl, separated from ``AutoSearchBase`` :pr:`793`
    * Fixes
        * Update pipeline ``score`` to return ``nan`` score for any objective which throws an exception during scoring :pr:`787`
        * Fixed bug introduced in :pr:`787` where binary classification metrics requiring predicted probabilities error in scoring :pr:`798`
        * CatBoost and XGBoost classifiers and regressors can no longer have a learning rate of 0 :pr:`795`
    * Changes
        * Cleanup pipeline ``score`` code, and cleanup codecov :pr:`711`
        * Remove ``pass`` for abstract methods for codecov :pr:`730`
        * Added __str__ for AutoSearch object :pr:`675`
        * Add util methods to graph ROC and confusion matrix :pr:`720`
        * Refactor ``AutoBase`` to ``AutoSearchBase`` :pr:`758`
        * Updated AutoBase with ``data_checks`` parameter, removed previous ``detect_label_leakage`` parameter, and added functionality to run data checks before search in AutoML :pr:`765`
        * Updated our logger to use Python's logging utils :pr:`763`
        * Refactor most of ``AutoSearchBase._do_iteration`` impl into ``AutoSearchBase._evaluate`` :pr:`762`
        * Port over all guardrails to use the new DataCheck API :pr:`789`
        * Expanded ``import_or_raise`` to catch all exceptions :pr:`759`
        * Adds RMSE, MSLE, RMSLE as standard metrics :pr:`788`
        * Don't allow ``Recall`` to be used as an objective for AutoML :pr:`784`
        * Removed feature selection from pipelines :pr:`819`
        * Update default estimator parameters to make automl search faster and more accurate :pr:`793`
    * Documentation Changes
        * Add instructions to freeze ``master`` on ``release.md`` :pr:`726`
        * Update release instructions with more details :pr:`727` :pr:`733`
        * Add objective base classes to API reference :pr:`736`
        * Fix components API to match other modules :pr:`747`
    * Testing Changes
        * Delete codecov yml, use codecov.io's default :pr:`732`
        * Added unit tests for fraud cost, lead scoring, and standard metric objectives :pr:`741`
        * Update codecov client :pr:`782`
        * Updated AutoBase __str__ test to include no parameters case :pr:`783`
        * Added unit tests for ``ExtraTrees`` pipeline :pr:`790`
        * If codecov fails to upload, fail build :pr:`810`
        * Updated Python version of dependency action :pr:`816`
        * Update the dependency update bot to use a suffix when creating branches :pr:`817`

.. warning::

    **Breaking Changes**
        * The ``detect_label_leakage`` parameter for AutoML classes has been removed and replaced by a ``data_checks`` parameter :pr:`765`
        * Moved ROC and confusion matrix methods from ``evalml.pipeline.plot_utils`` to ``evalml.pipeline.graph_utils`` :pr:`720`
        * ``Tuner`` classes require a pipeline hyperparameter range dict as an init arg instead of a space definition :pr:`779`
        * ``Tuner.propose`` and ``Tuner.add`` work directly with pipeline parameters dicts instead of flat parameter lists :pr:`779`
        * ``PipelineBase.hyperparameters`` and ``custom_hyperparameters`` use pipeline parameters dict format instead of being represented as a flat list :pr:`779`
        * All guardrail functions previously under ``evalml.guardrails.utils`` will be removed and replaced by data checks :pr:`789`
        * ``Recall`` disallowed as an objective for AutoML :pr:`784`
        * ``AutoSearchBase`` parameter ``tuner`` has been renamed to ``tuner_class`` :pr:`793`
        * ``AutoSearchBase`` parameter ``possible_pipelines`` and ``possible_model_families`` have been renamed to ``allowed_pipelines`` and ``allowed_model_families`` :pr:`793`


**v0.9.0 Apr. 27, 2020**
    * Enhancements
        * Added ``Accuracy`` as an standard objective :pr:`624`
        * Added verbose parameter to load_fraud :pr:`560`
        * Added Balanced Accuracy metric for binary, multiclass :pr:`612` :pr:`661`
        * Added XGBoost regressor and XGBoost regression pipeline :pr:`666`
        * Added ``Accuracy`` metric for multiclass :pr:`672`
        * Added objective name in ``AutoBase.describe_pipeline`` :pr:`686`
        * Added ``DataCheck`` and ``DataChecks``, ``Message`` classes and relevant subclasses :pr:`739`
    * Fixes
        * Removed direct access to ``cls.component_graph`` :pr:`595`
        * Add testing files to .gitignore :pr:`625`
        * Remove circular dependencies from ``Makefile`` :pr:`637`
        * Add error case for ``normalize_confusion_matrix()`` :pr:`640`
        * Fixed ``XGBoostClassifier`` and ``XGBoostRegressor`` bug with feature names that contain [, ], or < :pr:`659`
        * Update ``make_pipeline_graph`` to not accidentally create empty file when testing if path is valid :pr:`649`
        * Fix pip installation warning about docsutils version, from boto dependency :pr:`664`
        * Removed zero division warning for F1/precision/recall metrics :pr:`671`
        * Fixed ``summary`` for pipelines without estimators :pr:`707`
    * Changes
        * Updated default objective for binary/multiclass classification to log loss :pr:`613`
        * Created classification and regression pipeline subclasses and removed objective as an attribute of pipeline classes :pr:`405`
        * Changed the output of ``score`` to return one dictionary :pr:`429`
        * Created binary and multiclass objective subclasses :pr:`504`
        * Updated objectives API :pr:`445`
        * Removed call to ``get_plot_data`` from AutoML :pr:`615`
        * Set ``raise_error`` to default to True for AutoML classes :pr:`638`
        * Remove unnecessary "u" prefixes on some unicode strings :pr:`641`
        * Changed one-hot encoder to return uint8 dtypes instead of ints :pr:`653`
        * Pipeline ``_name`` field changed to ``custom_name`` :pr:`650`
        * Removed ``graphs.py`` and moved methods into ``PipelineBase`` :pr:`657`, :pr:`665`
        * Remove s3fs as a dev dependency :pr:`664`
        * Changed requirements-parser to be a core dependency :pr:`673`
        * Replace ``supported_problem_types`` field on pipelines with ``problem_type`` attribute on base classes :pr:`678`
        * Changed AutoML to only show best results for a given pipeline template in ``rankings``, added ``full_rankings`` property to show all :pr:`682`
        * Update ``ModelFamily`` values: don't list xgboost/catboost as classifiers now that we have regression pipelines for them :pr:`677`
        * Changed AutoML's ``describe_pipeline`` to get problem type from pipeline instead :pr:`685`
        * Standardize ``import_or_raise`` error messages :pr:`683`
        * Updated argument order of objectives to align with sklearn's :pr:`698`
        * Renamed ``pipeline.feature_importance_graph`` to ``pipeline.graph_feature_importances`` :pr:`700`
        * Moved ROC and confusion matrix methods to ``evalml.pipelines.plot_utils`` :pr:`704`
        * Renamed ``MultiClassificationObjective`` to ``MulticlassClassificationObjective``, to align with pipeline naming scheme :pr:`715`
    * Documentation Changes
        * Fixed some sphinx warnings :pr:`593`
        * Fixed docstring for ``AutoClassificationSearch`` with correct command :pr:`599`
        * Limit readthedocs formats to pdf, not htmlzip and epub :pr:`594` :pr:`600`
        * Clean up objectives API documentation :pr:`605`
        * Fixed function on Exploring search results page :pr:`604`
        * Update release process doc :pr:`567`
        * ``AutoClassificationSearch`` and ``AutoRegressionSearch`` show inherited methods in API reference :pr:`651`
        * Fixed improperly formatted code in breaking changes for changelog :pr:`655`
        * Added configuration to treat Sphinx warnings as errors :pr:`660`
        * Removed separate plotting section for pipelines in API reference :pr:`657`, :pr:`665`
        * Have leads example notebook load S3 files using https, so we can delete s3fs dev dependency :pr:`664`
        * Categorized components in API reference and added descriptions for each category :pr:`663`
        * Fixed Sphinx warnings about ``BalancedAccuracy`` objective :pr:`669`
        * Updated API reference to include missing components and clean up pipeline docstrings :pr:`689`
        * Reorganize API ref, and clarify pipeline sub-titles :pr:`688`
        * Add and update preprocessing utils in API reference :pr:`687`
        * Added inheritance diagrams to API reference :pr:`695`
        * Documented which default objective AutoML optimizes for :pr:`699`
        * Create seperate install page :pr:`701`
        * Include more utils in API ref, like ``import_or_raise`` :pr:`704`
        * Add more color to pipeline documentation :pr:`705`
    * Testing Changes
        * Matched install commands of ``check_latest_dependencies`` test and it's GitHub action :pr:`578`
        * Added Github app to auto assign PR author as assignee :pr:`477`
        * Removed unneeded conda installation of xgboost in windows checkin tests :pr:`618`
        * Update graph tests to always use tmpfile dir :pr:`649`
        * Changelog checkin test workaround for release PRs: If 'future release' section is empty of PR refs, pass check :pr:`658`
        * Add changelog checkin test exception for ``dep-update`` branch :pr:`723`

.. warning::

    **Breaking Changes**

    * Pipelines will now no longer take an objective parameter during instantiation, and will no longer have an objective attribute.
    * ``fit()`` and ``predict()`` now use an optional ``objective`` parameter, which is only used in binary classification pipelines to fit for a specific objective.
    * ``score()`` will now use a required ``objectives`` parameter that is used to determine all the objectives to score on. This differs from the previous behavior, where the pipeline's objective was scored on regardless.
    * ``score()`` will now return one dictionary of all objective scores.
    * ``ROC`` and ``ConfusionMatrix`` plot methods via ``Auto(*).plot`` have been removed by :pr:`615` and are replaced by ``roc_curve`` and ``confusion_matrix`` in ``evamlm.pipelines.plot_utils`` in :pr:`704`
    * ``normalize_confusion_matrix`` has been moved to ``evalml.pipelines.plot_utils`` :pr:`704`
    * Pipelines ``_name`` field changed to ``custom_name``
    * Pipelines ``supported_problem_types`` field is removed because it is no longer necessary :pr:`678`
    * Updated argument order of objectives' ``objective_function`` to align with sklearn :pr:`698`
    * ``pipeline.feature_importance_graph`` has been renamed to ``pipeline.graph_feature_importances`` in :pr:`700`
    * Removed unsupported ``MSLE`` objective :pr:`704`


**v0.8.0 Apr. 1, 2020**
    * Enhancements
        * Add normalization option and information to confusion matrix :pr:`484`
        * Add util function to drop rows with NaN values :pr:`487`
        * Renamed ``PipelineBase.name`` as ``PipelineBase.summary`` and redefined ``PipelineBase.name`` as class property :pr:`491`
        * Added access to parameters in Pipelines with ``PipelineBase.parameters`` (used to be return of ``PipelineBase.describe``) :pr:`501`
        * Added ``fill_value`` parameter for ``SimpleImputer`` :pr:`509`
        * Added functionality to override component hyperparameters and made pipelines take hyperparemeters from components :pr:`516`
        * Allow ``numpy.random.RandomState`` for random_state parameters :pr:`556`
    * Fixes
        * Removed unused dependency ``matplotlib``, and move ``category_encoders`` to test reqs :pr:`572`
    * Changes
        * Undo version cap in XGBoost placed in :pr:`402` and allowed all released of XGBoost :pr:`407`
        * Support pandas 1.0.0 :pr:`486`
        * Made all references to the logger static :pr:`503`
        * Refactored ``model_type`` parameter for components and pipelines to ``model_family`` :pr:`507`
        * Refactored ``problem_types`` for pipelines and components into ``supported_problem_types`` :pr:`515`
        * Moved ``pipelines/utils.save_pipeline`` and ``pipelines/utils.load_pipeline`` to ``PipelineBase.save`` and ``PipelineBase.load`` :pr:`526`
        * Limit number of categories encoded by ``OneHotEncoder`` :pr:`517`
    * Documentation Changes
        * Updated API reference to remove ``PipelinePlot`` and added moved ``PipelineBase`` plotting methods :pr:`483`
        * Add code style and github issue guides :pr:`463` :pr:`512`
        * Updated API reference for to surface class variables for pipelines and components :pr:`537`
        * Fixed README documentation link :pr:`535`
        * Unhid PR references in changelog :pr:`656`
    * Testing Changes
        * Added automated dependency check PR :pr:`482`, :pr:`505`
        * Updated automated dependency check comment :pr:`497`
        * Have build_docs job use python executor, so that env vars are set properly :pr:`547`
        * Added simple test to make sure ``OneHotEncoder``'s top_n works with large number of categories :pr:`552`
        * Run windows unit tests on PRs :pr:`557`


.. warning::

    **Breaking Changes**

    * ``AutoClassificationSearch`` and ``AutoRegressionSearch``'s ``model_types`` parameter has been refactored into ``allowed_model_families``
    * ``ModelTypes`` enum has been changed to ``ModelFamily``
    * Components and Pipelines now have a ``model_family`` field instead of ``model_type``
    * ``get_pipelines`` utility function now accepts ``model_families`` as an argument instead of ``model_types``
    * ``PipelineBase.name`` no longer returns structure of pipeline and has been replaced by ``PipelineBase.summary``
    * ``PipelineBase.problem_types`` and ``Estimator.problem_types`` has been renamed to ``supported_problem_types``
    * ``pipelines/utils.save_pipeline`` and ``pipelines/utils.load_pipeline`` moved to ``PipelineBase.save`` and ``PipelineBase.load``


**v0.7.0 Mar. 9, 2020**
    * Enhancements
        * Added emacs buffers to .gitignore :pr:`350`
        * Add CatBoost (gradient-boosted trees) classification and regression components and pipelines :pr:`247`
        * Added Tuner abstract base class :pr:`351`
        * Added ``n_jobs`` as parameter for ``AutoClassificationSearch`` and ``AutoRegressionSearch`` :pr:`403`
        * Changed colors of confusion matrix to shades of blue and updated axis order to match scikit-learn's :pr:`426`
        * Added ``PipelineBase`` ``.graph`` and ``.feature_importance_graph`` methods, moved from previous location :pr:`423`
        * Added support for python 3.8 :pr:`462`
    * Fixes
        * Fixed ROC and confusion matrix plots not being calculated if user passed own additional_objectives :pr:`276`
        * Fixed ReadtheDocs ``FileNotFoundError`` exception for fraud dataset :pr:`439`
    * Changes
        * Added ``n_estimators`` as a tunable parameter for XGBoost :pr:`307`
        * Remove unused parameter ``ObjectiveBase.fit_needs_proba`` :pr:`320`
        * Remove extraneous parameter ``component_type`` from all components :pr:`361`
        * Remove unused ``rankings.csv`` file :pr:`397`
        * Downloaded demo and test datasets so unit tests can run offline :pr:`408`
        * Remove ``_needs_fitting`` attribute from Components :pr:`398`
        * Changed plot.feature_importance to show only non-zero feature importances by default, added optional parameter to show all :pr:`413`
        * Refactored ``PipelineBase`` to take in parameter dictionary and moved pipeline metadata to class attribute :pr:`421`
        * Dropped support for Python 3.5 :pr:`438`
        * Removed unused ``apply.py`` file :pr:`449`
        * Clean up ``requirements.txt`` to remove unused deps :pr:`451`
        * Support installation without all required dependencies :pr:`459`
    * Documentation Changes
        * Update release.md with instructions to release to internal license key :pr:`354`
    * Testing Changes
        * Added tests for utils (and moved current utils to gen_utils) :pr:`297`
        * Moved XGBoost install into it's own separate step on Windows using Conda :pr:`313`
        * Rewind pandas version to before 1.0.0, to diagnose test failures for that version :pr:`325`
        * Added dependency update checkin test :pr:`324`
        * Rewind XGBoost version to before 1.0.0 to diagnose test failures for that version :pr:`402`
        * Update dependency check to use a whitelist :pr:`417`
        * Update unit test jobs to not install dev deps :pr:`455`

.. warning::

    **Breaking Changes**

    * Python 3.5 will not be actively supported.

**v0.6.0 Dec. 16, 2019**
    * Enhancements
        * Added ability to create a plot of feature importances :pr:`133`
        * Add early stopping to AutoML using patience and tolerance parameters :pr:`241`
        * Added ROC and confusion matrix metrics and plot for classification problems and introduce PipelineSearchPlots class :pr:`242`
        * Enhanced AutoML results with search order :pr:`260`
        * Added utility function to show system and environment information :pr:`300`
    * Fixes
        * Lower botocore requirement :pr:`235`
        * Fixed decision_function calculation for ``FraudCost`` objective :pr:`254`
        * Fixed return value of ``Recall`` metrics :pr:`264`
        * Components return ``self`` on fit :pr:`289`
    * Changes
        * Renamed automl classes to ``AutoRegressionSearch`` and ``AutoClassificationSearch`` :pr:`287`
        * Updating demo datasets to retain column names :pr:`223`
        * Moving pipeline visualization to ``PipelinePlot`` class :pr:`228`
        * Standarizing inputs as ``pd.Dataframe`` / ``pd.Series`` :pr:`130`
        * Enforcing that pipelines must have an estimator as last component :pr:`277`
        * Added ``ipywidgets`` as a dependency in ``requirements.txt`` :pr:`278`
        * Added Random and Grid Search Tuners :pr:`240`
    * Documentation Changes
        * Adding class properties to API reference :pr:`244`
        * Fix and filter FutureWarnings from scikit-learn :pr:`249`, :pr:`257`
        * Adding Linear Regression to API reference and cleaning up some Sphinx warnings :pr:`227`
    * Testing Changes
        * Added support for testing on Windows with CircleCI :pr:`226`
        * Added support for doctests :pr:`233`

.. warning::

    **Breaking Changes**

    * The ``fit()`` method for ``AutoClassifier`` and ``AutoRegressor`` has been renamed to ``search()``.
    * ``AutoClassifier`` has been renamed to ``AutoClassificationSearch``
    * ``AutoRegressor`` has been renamed to ``AutoRegressionSearch``
    * ``AutoClassificationSearch.results`` and ``AutoRegressionSearch.results`` now is a dictionary with ``pipeline_results`` and ``search_order`` keys. ``pipeline_results`` can be used to access a dictionary that is identical to the old ``.results`` dictionary. Whereas, ``search_order`` returns a list of the search order in terms of ``pipeline_id``.
    * Pipelines now require an estimator as the last component in ``component_list``. Slicing pipelines now throws an ``NotImplementedError`` to avoid returning pipelines without an estimator.

**v0.5.2 Nov. 18, 2019**
    * Enhancements
        * Adding basic pipeline structure visualization :pr:`211`
    * Documentation Changes
        * Added notebooks to build process :pr:`212`

**v0.5.1 Nov. 15, 2019**
    * Enhancements
        * Added basic outlier detection guardrail :pr:`151`
        * Added basic ID column guardrail :pr:`135`
        * Added support for unlimited pipelines with a ``max_time`` limit :pr:`70`
        * Updated .readthedocs.yaml to successfully build :pr:`188`
    * Fixes
        * Removed MSLE from default additional objectives :pr:`203`
        * Fixed ``random_state`` passed in pipelines :pr:`204`
        * Fixed slow down in RFRegressor :pr:`206`
    * Changes
        * Pulled information for describe_pipeline from pipeline's new describe method :pr:`190`
        * Refactored pipelines :pr:`108`
        * Removed guardrails from Auto(*) :pr:`202`, :pr:`208`
    * Documentation Changes
        * Updated documentation to show ``max_time`` enhancements :pr:`189`
        * Updated release instructions for RTD :pr:`193`
        * Added notebooks to build process :pr:`212`
        * Added contributing instructions :pr:`213`
        * Added new content :pr:`222`

**v0.5.0 Oct. 29, 2019**
    * Enhancements
        * Added basic one hot encoding :pr:`73`
        * Use enums for model_type :pr:`110`
        * Support for splitting regression datasets :pr:`112`
        * Auto-infer multiclass classification :pr:`99`
        * Added support for other units in ``max_time`` :pr:`125`
        * Detect highly null columns :pr:`121`
        * Added additional regression objectives :pr:`100`
        * Show an interactive iteration vs. score plot when using fit() :pr:`134`
    * Fixes
        * Reordered ``describe_pipeline`` :pr:`94`
        * Added type check for ``model_type`` :pr:`109`
        * Fixed ``s`` units when setting string ``max_time`` :pr:`132`
        * Fix objectives not appearing in API documentation :pr:`150`
    * Changes
        * Reorganized tests :pr:`93`
        * Moved logging to its own module :pr:`119`
        * Show progress bar history :pr:`111`
        * Using ``cloudpickle`` instead of pickle to allow unloading of custom objectives :pr:`113`
        * Removed render.py :pr:`154`
    * Documentation Changes
        * Update release instructions :pr:`140`
        * Include additional_objectives parameter :pr:`124`
        * Added Changelog :pr:`136`
    * Testing Changes
        * Code coverage :pr:`90`
        * Added CircleCI tests for other Python versions :pr:`104`
        * Added doc notebooks as tests :pr:`139`
        * Test metadata for CircleCI and 2 core parallelism :pr:`137`

**v0.4.1 Sep. 16, 2019**
    * Enhancements
        * Added AutoML for classification and regressor using Autobase and Skopt :pr:`7` :pr:`9`
        * Implemented standard classification and regression metrics :pr:`7`
        * Added logistic regression, random forest, and XGBoost pipelines :pr:`7`
        * Implemented support for custom objectives :pr:`15`
        * Feature importance for pipelines :pr:`18`
        * Serialization for pipelines :pr:`19`
        * Allow fitting on objectives for optimal threshold :pr:`27`
        * Added detect label leakage :pr:`31`
        * Implemented callbacks :pr:`42`
        * Allow for multiclass classification :pr:`21`
        * Added support for additional objectives :pr:`79`
    * Fixes
        * Fixed feature selection in pipelines :pr:`13`
        * Made ``random_seed`` usage consistent :pr:`45`
    * Documentation Changes
        * Documentation Changes
        * Added docstrings :pr:`6`
        * Created notebooks for docs :pr:`6`
        * Initialized readthedocs EvalML :pr:`6`
        * Added favicon :pr:`38`
    * Testing Changes
        * Added testing for loading data :pr:`39`

**v0.2.0 Aug. 13, 2019**
    * Enhancements
        * Created fraud detection objective :pr:`4`

**v0.1.0 July. 31, 2019**
    * *First Release*
    * Enhancements
        * Added lead scoring objecitve :pr:`1`
        * Added basic classifier :pr:`1`
    * Documentation Changes
        * Initialized Sphinx for docs :pr:`1`
