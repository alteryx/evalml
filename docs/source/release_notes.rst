Release Notes
-------------
**Future Releases**
    * Enhancements
        * Add run_feature_selection to AutoMLSearch and Default Algorithm :pr:`4210`
    * Fixes
        * `IDColumnsDataCheck` now works with Unknown data type :pr:`4203`
    * Changes
    * Documentation Changes
        * Updated API reference :pr:`4213`
    * Testing Changes

.. warning::

    **Breaking Changes**
        * Removed Decision Tree and CatBoost Estimators from AutoML search :pr:`4205`


**v0.77.0 June. 07, 2023**
    * Enhancements
        * Added ``check_distribution`` function for determining if the predicted distribution matches the true one :pr:`4184`
        * Added ``get_recommendation_score_breakdown`` function for insight on the recommendation score :pr:`4188`
        * Added excluded_model_families parameter to AutoMLSearch() :pr:`4196`
        * Added option to exclude time index in ``IDColumnsDataCheck`` :pr:`4194`
    * Fixes
        * Fixed small errors in ``ARIMARegressor`` implementation :pr:`4186`
        * Fixed ``get_forecast_period`` to properly handle ``gap`` parameter :pr:`4200`
    * Changes
    * Documentation Changes
    * Testing Changes
        * Run looking glass performance tests on merge via Airflow :pr:`4198`


**v0.76.0 May. 09, 2023**
    * Enhancements
        * Added optional ``recommendation_score`` to rank pipelines during AutoMLSearch :pr:`4156`
        * Added BytesIO support to PipelinBase.load() :pr:`4179`
    * Fixes
        * Capped numpy at <=1.23.5 as a temporary measure for SHAP :pr:`4172`
        * Updated our readthedocs recipe to reenable builds :pr:`4177`


**v0.75.0 May. 01, 2023**
    * Fixes
        * Fixed bug where resetting the holdout data indices would cause time series ``predict_in_sample`` to be wrong :pr:`4161`
    * Changes
        * Changed per-pipeline timings to store as a float :pr:`4160`
        * Update Dask install commands in ``pyproject.toml`` :pr:`4164`
        * Capped `IPython` version to < 8.12.1 for readthedocs and plotly compatibility :pr:`3987`

**v0.74.0 Apr. 18, 2023**
    * Enhancements
        * Saved computed additional_objectives computed during search to AutoML object :pr:`4141`
        * Remove extra naive pipelines :pr:`4142`
    * Fixes
        * Fixed usage of codecov after uploader deprecation :pr:`4144`
        * Fixed issue where prediction intervals were becoming NaNs due to index errors :pr:`4154`
    * Changes
        * Capped size of seasonal period used for determining whether to include STLDecomposer in pipelines :pr:`4147`

**v0.73.0 Apr. 10, 2023**
    * Enhancements
        * Allowed ``InvalidTargetDataCheck`` to return a ``DROP_ROWS`` ``DataCheckActionOption`` :pr:`4116`
        * Implemented prediction intervals for non-time series native pipelines using the naïve method :pr:`4127`
    * Changes
        * Removed unnecessary logic from imputer components prior to nullable type handling :pr:`4038`, :pr:`4043`
        * Added calls to ``_handle_nullable_types`` in component fit, transform, and predict methods when needed :pr:`4046`, :pr:`4043`
        * Removed existing nullable type handling across AutoMLSearch to just use new handling :pr:`4085`, :pr:`4043`
        * Handled nullable type incompatibility in ``Decomposer`` :pr:`4105`, :pr:`4043
        * Removed nullable type incompatibility handling for ARIMA and ExponentialSmoothingRegressor :pr:`4129`
        * Changed the default value for ``null_strategy`` in ``InvalidTargetDataCheck`` to ``drop`` :pr:`4131`
        * Pinned sktime version to 0.17.0 for nullable types support :pr:`4137`
    * Testing Changes
        * Fixed installation of prophet for linux nightly tests :pr:`4114`

**v0.72.0 Mar. 27, 2023**
    * Enhancements
        * Updated `pipeline.get_prediction_intervals()` to add trend prediction interval information from STL decomposer :pr:`4093`
        * Added ``method=all`` support for ``TargetLeakageDataCheck`` :pr:`4106`
    * Fixes
        * Fixed ensemble pipelines not working with ``generate_pipeline_example`` :pr:`4102`
    * Changes
        * Pinned ipywidgets version under 8.0.5 :pr:`4097`
        * Calculated partial dependence grid values for integer data by rounding instead of truncating fractional values :pr:`4096`
    * Testing Changes
        * Updated graphviz installation in GitHub workflows to fix windows nightlies :pr:`4088`

**v0.71.0 Mar. 17, 2023***
    * Fixes
        * Fixed error in ``PipelineBase._supports_fast_permutation_importance`` with stacked ensemble pipelines :pr:`4083`

**v0.70.0 Mar. 16, 2023**
    * Changes
        * Added Oversampler nullable type incompatibility in X :pr:`4068`
        * Removed nullable handling from objective functions, ``roc_curve``, and ``correlation_matrix`` :pr:`4072`
        * Transitioned from ``prophet-prebuilt`` to ``prophet`` directly :pr:`4045`

**v0.69.0 Mar. 15, 2023**
    * Enhancements
        * Move black to regular dependency and use it for ``generate_pipeline_code`` :pr:`4005`
        * Implement ``generate_pipeline_example`` :pr:`4023`
        * Add new downcast utils for component-specific nullable type handling and begin implementation on objective and component base classes :pr:`4024`
        * Add nullable type incompatibility properties to the components that need them :pr:`4031`
        * Add ``get_evalml_requirements_file`` :pr:`4034`
        * Pipelines with DFS Transformers will run fast permutation importance if DFS features pre-exist :pr:`4037`
        * Add get_prediction_intervals() at the pipeline level :pr:`4052`
    * Fixes
        * Fixed ``generate_pipeline_example`` erroring out for pipelines with a ``DFSTransformer`` :pr:`4059`
        * Remove nullable types handling for ``OverSampler`` :pr:`4064`
    * Changes
        * Uncapped ``pmdarima`` and updated minimum version :pr:`4027`
        * Increase min catboost to 1.1.1 and xgboost to 1.7.0 to add nullable type support for those estimators :pr:`3996`
        * Unpinned ``networkx`` and updated minimum version :pr:`4035`
        * Increased ``scikit-learn`` version to 1.2.2 :pr:`4064`
        * Capped max ``holidays`` version to 0.21 :pr:`4064`
        * Stop allowing ``knn`` as a boolean impute strategy :pr:`4058`
        * Capped ``nbsphinx`` at < 0.9.0 :pr:`4071`
    * Testing Changes
        * Use ``release.yaml`` for performance tests on merge to main :pr:`4007`
        * Pin ``github-action-check-linked-issues`` at v1.4.5 :pr:`4042`
        * Updated tests to support Woodwork's object dtype inference for numeric columns :pr:`4066`
        * Updated ``TargetLeakageDataCheck`` tests to handle boolean targets properly :pr:`4066`

**v0.68.0 Feb. 15, 2023**
    * Enhancements
        * Integrated ``determine_periodicity`` into ``AutoMLSearch`` :pr:`3952`
        * Removed frequency limitations for decomposition using the ``STLDecomposer`` :pr:`3952`
    * Changes
        * Remove requirements-parser requirement :pr:`3978`
        * Updated the ``SKOptTuner`` to use a gradient boosting regressor for tuning instead of extra trees :pr:`3983`
        * Unpinned sktime from below 1.2, increased minimum to 1.2.1 :pr:`3983`
    * Testing Changes
        * Add pull request check for linked issues to CI workflow :pr:`3970`, :pr:`3980`
        * Upgraded minimum `IPython` version to 8.10.0 :pr:`3987`

**v0.67.0 Jan. 31, 2023**
    * Fixes
        * Re-added ``TimeSeriesPipeline.should_skip_featurization`` to fix bug where data would get featurized unnecessarily :pr:`3964`
        * Allow float categories to be passed into CatBoost estimators :pr:`3966`
    * Changes
        * Update pyproject.toml to correctly specify the data filepaths :pr:`3967`
    * Documentation Changes
        * Added demo for prediction intervals :pr:`3954`

**v0.66.1 Jan. 26, 2023**
    * Fixes
        * Updated ``LabelEncoder`` to store the original typing information :pr:`3960`
        * Fixed bug where all-null ``BooleanNullable`` columns would break the imputer during transform :pr:`3959`

**v0.66.0 Jan. 24, 2023**
    * Enhancements
        * Improved decomposer ``determine_periodicity`` functionality for better period guesses :pr:`3912`
        * Added ``dates_needed_for_prediction`` for time series pipelines :pr:`3906`
        * Added ``RFClassifierRFESelector``  and ``RFRegressorRFESelector`` components for feature selection using recursive feature elimination :pr:`3934`
        * Added ``dates_needed_for_prediction_range`` for time series pipelines :pr:`3941`
    * Fixes
        * Fixed ``set_period()`` not updating decomposer parameters :pr:`3932`
        * Removed second identical batch for time series problems in ``DefaultAlgorithm`` :pr:`3936`
        * Fix install command for alteryx-open-src-update-checker :pr:`3940`
        * Fixed non-prophet case of ``test_components_can_be_used_for_partial_dependence_fast_mode`` :pr:`3949`
    * Changes
        * Updated ``PolynomialDecomposer`` to work with sktime v0.15.1 :pr:`3930`
        * Add ruff and use pyproject.toml (move away from setup.cfg) :pr:`3928`
        * Pinned `category-encoders`` to 2.5.1.post0 :pr:`3933`
        * Remove requirements-parser and tomli from core requirements :pr:`3948`


**v0.65.0 Jan. 3, 2023**
    * Enhancements
        * Added the ability to retrieve prediction intervals for estimators that support time series regression :pr:`3876`
        * Added utils to handle the logic for threshold tuning objective and resplitting data :pr:`3888`
        * Integrated ``OrdinalEncoder`` into AutoMLSearch :pr:`3765`
    * Fixes
        * Fixed ARIMA not accounting for gap in prediction from end of training data :pr:`3884`
        * Fixed ``DefaultAlgorithm`` adding an extra ``OneHotEncoder`` when a categorical column is not selected :pr:`3914`
    * Changes
        * Added a threshold to ``DateTimeFormatDataCheck`` to account for too many duplicate or nan values :pr:`3883`
        * Changed treatment of ``Boolean`` columns for ``SimpleImputer`` and ``ClassImbalanceDataCheck`` to be compatible with new Woodwork inference :pr:`3892`
        * Split decomposer ``seasonal_period`` parameter into ``seasonal_smoother`` and ``period`` parameters :pr:`3896`
        * Excluded catboost from the broken link checking workflow due to 403 errors :pr:`3899`
        * Pinned scikit-learn version below 1.2.0 :pr:`3901`
        * Cast newly created one hot encoded columns as ``bool`` dtype :pr:`3913`
    * Documentation Changes
        * Hid non-essential warning messages in time series docs :pr:`3890`
    * Testing Changes


**v0.64.0 Dec. 8, 2022**
    * Enhancements
    * Fixes
        * Allowed the DFS Transformer to calculate feature values for Features with a ``dataframe_name`` that is not ``"X"`` :pr:`3873`
        * Stopped passing full list of DFS Transformer features into cloned pipeline in partial dependence fast mode :pr:`3875`
    * Changes
        * Update leaderboard names to show `ranking_score` instead of `validation_score` :pr:`3878`
        * Remove Int64Index after Pandas 1.5 Upgrade :pr:`3825`
        * Reduced the threshold for setting ``use_covariates`` to False for ARIMA models in AutoMLSearch :pr:`3868`
        * Pinned woodwork version at <=0.19.0 :pr:`3871`
        * Updated minimum Pandas version to 1.5.0 :pr:`3808`
        * Remove dsherry from automated dependency update reviews and added tamargrey :pr:`3870`
    * Documentation Changes
    * Testing Changes


**v0.63.0 Nov. 23, 2022**
    * Enhancements
        * Added fast mode to partial dependence :pr:`3753`
        * Added the ability to serialize featuretools features into time series pipelines :pr:`3836`
    * Fixes
        * Fixed ``TimeSeriesFeaturizer`` potentially selecting lags outside of feature engineering window :pr:`3773`
        * Fixed bug where ``TimeSeriesFeaturizer`` could not encode Ordinal columns with non numeric categories :pr:`3812`
        * Updated demo dataset links to point to new endpoint :pr:`3826`
        * Updated ``STLDecomposer`` to infer the time index frequency if it's not present :pr:`3829`
        * Updated ``_drop_time_index`` to move the time index from X to both ``X.index`` and ``y.index`` :pr:`3829`
        * Fixed bug where engineered features lost their origin attribute in partial dependence, causing it to fail :pr:`3830`
        * Fixed bug where partial dependence's fast mode handling for the DFS Transformer wouldn't work with multi output features :pr:`3830`
        * Allowed target to be present and ignored in partial dependence's DFS Transformer fast mode handling  :pr:`3830`
    * Changes
        * Consolidated decomposition frequency validation logic to ``Decomposer`` class :pr:`3811`
        * Removed Featuretools version upper bound and prevent Woodwork 0.20.0 from being installed :pr:`3813`
        * Updated min Featuretools version to 0.16.0, min nlp-primitives version to 2.9.0 and min Dask version to 2022.2.0 :pr:`3823`
        * Rename issue templates config.yaml to config.yml :pr:`3844`
        * Reverted change adding a ``should_skip_featurization`` flag to time series pipelines :pr:`3862`
    * Documentation Changes
        * Added information about STL Decomposition to the time series docs :pr:`3835`
        * Removed RTD failure on warnings :pr:`3864`


**v0.62.0 Nov. 01, 2022**
    * Fixes
        * Fixed bug with datetime conversion in ``get_time_index`` :pr:`3792`
        * Fixed bug where invalid anchored or offset frequencies were including the ``STLDecomposer`` in pipelines :pr:`3794`
        * Fixed bug where irregular datetime frequencies were causing errors in ``make_pipeline`` :pr:`3800`
    * Changes
        * Capped dask at < 2022.10.1 :pr:`3797`
        * Uncapped dask and excluded 2022.10.1 from viable versions :pr:`3803`
        * Removed all references to XGBoost's deprecated ``_use_label_encoder`` argument :pr:`3805`
        * Capped featuretools at < 1.17.0 :pr:`3805`
        * Capped woodwork at < 0.21.0 :pr:`3805`


**v0.61.1 Oct. 27, 2022**
    * Fixes
        * Fixed bug where ``TimeSeriesBaselinePipeline`` wouldn't preserve index name of input features :pr:`3788`
        * Fixed bug in ``TimeSeriesBaselinePipeline`` referencing a static string instead of time index var :pr:`3788`
    * Documentation Changes
        * Updated Release Notes :pr:`3788`


**v0.61.0 Oct. 25, 2022**
    * Enhancements
        * Added the STL Decomposer :pr:`3741`
        * Integrated STLDecomposer into AutoMLSearch for time series regression problems :pr:`3781`
        * Brought the PolynomialDecomposer up to parity with STLDecomposer :pr:`3768`
    * Changes
        * Cap Featuretools at < 1.15.0 :pr:`3775`
        * Remove Featuretools upper bound restriction and fix nlp-primitives import statements :pr:`3778`


**v0.60.0 Oct. 19, 2022**
    * Enhancements
        * Add forecast functions to time series regression pipeline :pr:`3742`
    * Fixes
        * Fix to allow ``IDColumnsDataCheck`` to work with ``IntegerNullable`` inputs :pr:`3740`
        * Fixed datasets name for main performance tests :pr:`3743`
    * Changes
        * Use Woodwork's ``dependence_dict`` method to calculate for ``TargetLeakageDataCheck`` :pr:`3728`
    * Documentation Changes
    * Testing Changes

.. warning::

    **Breaking Changes**
        * ``TargetLeakageDataCheck`` now uses argument ``mutual_info`` rather than ``mutual`` :pr:`3728`


**v0.59.0 Sept. 27, 2022**
    * Enhancements
        * Enhanced Decomposer with ``determine_periodicity`` function to automatically determine periodicity of seasonal target. :pr:`3729`
        * Enhanced Decomposer with ``set_seasonal_period`` function to set a ``Decomposer`` object's seasonal period automatically. :pr:`3729`
        * Added ``OrdinalEncoder`` component  :pr:`3736`
    * Fixes
        * Fixed holdout warning message showing when using default parameters :pr:`3727`
        * Fixed bug in Oversampler where categorical dtypes would fail :pr:`3732`
    * Changes
        * Automatic sorting of the ``time_index`` prior to running ``DataChecks`` has been disabled :pr:`3723`
    * Documentation Changes
    * Testing Changes
        * Update job to use new looking glass report command :pr:`3733`


**v0.58.0 Sept. 20, 2022**
    * Enhancements
        * Defined `get_trend_df()` for PolynomialDecomposer to allow decomposition of target data into trend, seasonality and residual. :pr:`3720`
        * Updated to run with Woodwork >= 0.18.0 :pr:`3700`
        * Pass time index column to time series native estimators but drop otherwise :pr:`3691`
        * Added ``errors`` attribute to ``AutoMLSearch`` for useful debugging :pr:`3702`
    * Fixes
        * Removed multiple samplers occurring in pipelines generated by ``DefaultAlgorithm`` :pr:`3696`
        * Fix search order changing when using ``DefaultAlgorithm`` :pr:`3704`
    * Changes
        * Bumped up minimum version of sktime to 0.12.0. :pr:`3720`
        * Added abstract Decomposer class as a parent to PolynomialDecomposer to support additional decomposers. :pr:`3720`
        * Pinned ``pmdarima`` < 2.0.0 :pr:`3679`
        * Added support for using ``downcast_nullable_types`` with Series as well as DataFrames :pr:`3697`
        * Added distinction between ranking and optimization objectives :pr:`3721`
    * Documentation Changes
    * Testing Changes
        * Updated pytest fixtures and brittle test files to explicitly set woodwork typing information :pr:`3697`
        * Added github workflow to run looking glass performance tests on merge to main :pr:`3690`
        * Fixed looking glass performance test script :pr:`3715`
        * Remove commit message from looking glass slack message :pr:`3719`

**v0.57.0 Sept. 6, 2022**
    * Enhancements
        *  Added ``KNNImputer`` class and created new knn parameter for Imputer :pr:`3662`
    * Fixes
        * ``IDColumnsDataCheck`` now only returns an action code to set the first column as the primary key if it contains unique values :pr:`3639`
        * ``IDColumnsDataCheck`` now can handle primary key columns containing "integer" values that are of the double type :pr:`3683`
        * Added support for BooleanNullable columns in EvalML pipelines and imputer :pr:`3678`
        * Updated StandardScaler to only apply to numeric columns :pr:`3686`
    * Changes
        * Unpinned sktime to allow for version 0.13.2 :pr:`3685`
        * Pinned ``pmdarima`` < 2.0.0 :pr:`3679`

**v0.56.1 Aug. 19, 2022**
    * Fixes
        * ``IDColumnsDataCheck`` now only returns an action code to set the first column as the primary key if it contains unique values :pr:`3639`
        * Reverted the ``make_pipeline`` changes that conditionally included the imputers :pr:`3672`

**v0.56.0 Aug. 15, 2022**
    * Enhancements
        * Add CI testing environment in Mac for install workflow :pr:`3646`
        * Updated ``make_pipeline`` to only include the Imputer in pipelines if NaNs exist in the data :pr:`3657`
        * Updated to run with Woodwork >= 0.17.2 :pr:`3626`
        * Add ``exclude_featurizers`` parameter to ``AutoMLSearch`` to specify featurizers that should be excluded from all pipelines :pr:`3631`
        * Add ``fit_transform`` method to pipelines and component graphs :pr:`3640`
        * Changed default value of data splitting for time series problem holdout set evaluation :pr:`3650`
    * Fixes
        * Reverted the Woodwork 0.17.x compatibility work due to performance regression :pr:`3664`
    * Changes
        * Disable holdout set in AutoML search by default :pr:`3659`
        * Pinned ``sktime`` at >=0.7.0,<0.13.1 due to slowdowns with time series modeling :pr:`3658`
        * Added additional testing support for Python 3.10 :pr:`3609`
    * Documentation Changes
        * Updated broken link checker to exclude stackoverflow domain :pr:`3633`
        * Add instructions to add new users to evalml-core-feedstock :pr:`3636`


**v0.55.0 July. 24, 2022**
    * Enhancements
        * Increased the amount of logical type information passed to Woodwork when calling ``ww.init()`` in transformers :pr:`3604`
        * Added ability to log how long each batch and pipeline take in ``automl.search()`` :pr:`3577`
        * Added the option to set the ``sp`` parameter for ARIMA models :pr:`3597`
        * Updated the CV split size of time series problems to match forecast horizon for improved performance :pr:`3616`
        * Added holdout set evaluation as part of AutoML search and pipeline ranking :pr:`3499`
        * Added Dockerfile.arm and .dockerignore for python version and M1 testing :pr:`3609`
        * Added ``test_gen_utils::in_container_arm64()`` fixture :pr:`3609`
    * Fixes
        * Fixed iterative graphs not appearing in documentation :pr:`3592`
        * Updated the ``load_diabetes()`` method to account for scikit-learn 1.1.1 changes to the dataset :pr:`3591`
        * Capped woodwork version at < 0.17.0 :pr:`3612`
        * Bump minimum scikit-optimize version to 0.9.0 `:pr:`3614`
        * Invalid target data checks involving regression and unsupported data types now produce a different ``DataCheckMessageCode`` :pr:`3630`
        * Updated ``test_data_checks.py::test_data_checks_raises_value_errors_on_init`` - more lenient text check :pr:`3609`
    * Changes
        * Add pre-commit hooks for linting :pr:`3608`
        * Implemented a lower threshold and window size for the ``TimeSeriesRegularizer`` and ``DatetimeFormatDataCheck`` :pr:`3627`
        * Updated ``IDColumnsDataCheck`` to return an action to set the first column as the primary key if it is identified as an ID column :pr:`3634`
    * Documentation Changes
    * Testing Changes
        * Pinned GraphViz version for Windows CI Test :pr:`3596`
        * Removed skipping of PolynomialDecomposer tests for Python 3.9 envs. :pr:`3720`
        * Removed ``pytest.mark.skip_if_39`` pytest marker :pr:`3602` :pr:`3607`
        * Updated pytest==7.1.2 :pr:`3609`
        * Added Dockerfile.arm and .dockerignore for python version and M1 testing :pr:`3609`
        * Added ``test_gen_utils::in_container_arm64()`` fixture :pr:`3609`

.. warning::

    **Breaking Changes**
        * Refactored test cases that iterate over all components to use ``pytest.mark.parametrise`` and changed the corresponding ``if...continue`` blocks to ``pytest.mark.xfail`` :pr:`3622`


**v0.54.0 June. 23, 2022**
    * Fixes
        * Updated the Imputer and SimpleImputer to work with scikit-learn 1.1.1. :pr:`3525`
        * Bumped the minimum versions of scikit-learn to 1.1.1 and imbalanced-learn to 0.9.1. :pr:`3525`
        * Added a clearer error message when ``describe`` is called on an un-instantiated ComponentGraph :pr:`3569`
        * Added a clearer error message when time series' ``predict`` is called with its X_train or y_train parameter set as None :pr:`3579`
    * Changes
        * Don't pass ``time_index`` as kwargs to sktime ARIMA implementation for compatibility with latest version :pr:`3564`
        * Remove incompatible ``nlp-primitives`` version 2.6.0 from accepted dependency versions :pr:`3572`, :pr:`3574`
        * Updated evalml authors :pr:`3581`
    * Documentation Changes
        * Fix typo in ``long_description`` field in ``setup.cfg`` :pr:`3553`
        * Update install page to remove Python 3.7 mention :pr:`3567`


**v0.53.1 June. 9, 2022**
    * Changes
        * Set the development status to ``4 - Beta`` in ``setup.cfg`` :pr:`3550`


**v0.53.0 June. 9, 2022**
    * Enhancements
        * Pass ``n_jobs`` to default algorithm :pr:`3548`
    * Fixes
        * Fixed github workflows for featuretools and woodwork to test their main branch against evalml. :pr:`3517`
        * Supress warnings in ``TargetEncoder`` raised by a coming change to default parameters :pr:`3540`
        * Fixed bug where schema was not being preserved in column renaming for XGBoost and LightGBM models :pr:`3496`
    * Changes
        * Transitioned to use pyproject.toml and setup.cfg away from setup.py :pr:`3494`, :pr:`3536`
    * Documentation Changes
        * Updated the Time Series User Guide page to include known-in-advance features and fix typos :pr:`3521`
        * Add slack and stackoverflow icon to footer :pr:`3528`
        * Add install instructions for M1 Mac :pr:`3543`
    * Testing Changes
        * Rename yml to yaml for GitHub Actions :pr:`3522`
        * Remove ``noncore_dependency`` pytest marker :pr:`3541`
        * Changed ``test_smotenc_category_features`` to use valid postal code values in response to new woodwork type validation :pr:`3544`


**v0.52.0 May. 12, 2022**
    * Changes
        * Added github workflows for featuretools and woodwork to test their main branch against evalml. :pr:`3504`
        * Added pmdarima to conda recipe. :pr:`3505`
        * Added a threshold for ``NullDataCheck`` before a warning is issued for null values :pr:`3507`
        * Changed ``NoVarianceDataCheck`` to only output warnings :pr:`3506`
        * Reverted XGBoost Classifier/Regressor patch for all boolean columns needing to be converted to int. :pr:`3503`
        * Updated ``roc_curve()`` and ``conf_matrix()`` to work with IntegerNullable and BooleanNullable types. :pr:`3465`
        * Changed ``ComponentGraph._transform_features`` to raise a ``PipelineError`` instead of a ``ValueError``. This is not a breaking change because ``PipelineError`` is a subclass of ``ValueError``. :pr:`3497`
        * Capped ``sklearn`` at version 1.1.0 :pr:`3518`
    * Documentation Changes
        * Updated to install prophet extras in Read the Docs. :pr:`3509`
    * Testing Changes
        * Moved vowpal wabbit in test recipe to ``evalml`` package from ``evalml-core`` :pr:`3502`


**v0.51.0 Apr. 28, 2022**
    * Enhancements
        * Updated ``make_pipeline_from_data_check_output`` to work with time series problems. :pr:`3454`
    * Fixes
        * Changed ``PipelineBase.graph_json()`` to return a python dictionary and renamed as ``graph_dict()``:pr:`3463`
    * Changes
        * Added ``vowpalwabbit`` to local recipe and remove ``is_using_conda`` pytest skip markers from relevant tests :pr:`3481`
    * Documentation Changes
        * Fixed broken link in contributing guide :pr:`3464`
        * Improved development instructions :pr:`3468`
        * Added the ``TimeSeriesRegularizer`` and ``TimeSeriesImputer`` to the timeseries section of the User Guide :pr:`3473`
        * Updated OSS slack link :pr:`3487`
        * Fix rendering of model understanding plotly charts in docs :pr:`3460`
    * Testing Changes
        * Updated unit tests to support woodwork 0.16.2 :pr:`3482`
        * Fix some unit tests after vowpal wabbit got added to conda recipe :pr:`3486`

.. warning::

    **Breaking Changes**
        * Renamed ``PipelineBase.graph_json()`` to ``PipelineBase.graph_dict()`` :pr:`3463`
        * Minimum supported woodwork version is now 0.16.2 :pr:`3482`

**v0.50.0 Apr. 12, 2022**
    * Enhancements
        * Added ``TimeSeriesImputer`` component :pr:`3374`
        * Replaced ``pipeline_parameters`` and ``custom_hyperparameters`` with ``search_parameters`` in ``AutoMLSearch`` :pr:`3373`, :pr:`3427`
        * Added ``TimeSeriesRegularizer`` to smooth uninferrable date ranges for time series problems :pr:`3376`
        * Enabled ensembling as a parameter for ``DefaultAlgorithm`` :pr:`3435`, :pr:`3444`
    * Fixes
        * Fix ``DefaultAlgorithm`` not handling Email and URL features :pr:`3419`
        * Added test to ensure ``LabelEncoder`` parameters preserved during ``AutoMLSearch`` :pr:`3326`
    * Changes
        * Updated ``DateTimeFormatDataCheck`` to use woodwork's ``infer_frequency`` function :pr:`3425`
        * Renamed ``graphs.py`` to ``visualizations.py`` :pr:`3439`
    * Documentation Changes
        * Updated the model understanding section of the user guide to include missing functions :pr:`3446`
        * Rearranged the user guide model understanding page for easier navigation :pr:`3457`
        * Update README text to Alteryx :pr:`3462`

.. warning::

    **Breaking Changes**
        * Renamed ``graphs.py`` to ``visualizations.py`` :pr:`3439`
        * Replaced ``pipeline_parameters`` and ``custom_hyperparameters`` with ``search_parameters`` in ``AutoMLSearch`` :pr:`3373`

**v0.49.0 Mar. 31, 2022**
    * Enhancements
        * Added ``use_covariates`` parameter to ``ARIMARegressor`` :pr:`3407`
        * ``AutoMLSearch`` will set ``use_covariates`` to ``False`` for ARIMA when dataset is large :pr:`3407`
        * Add ability to retrieve logical types to a component in the graph via ``get_component_input_logical_types`` :pr:`3428`
        * Add ability to get logical types passed to the last component via ``last_component_input_logical_types`` :pr:`3428`
    * Fixes
        * Fix conda build after PR `3407` :pr:`3429`
    * Changes
        * Moved model understanding metrics from ``graph.py`` into a separate file :pr:`3417`
        * Unpin ``click`` dependency :pr:`3420`
        * For ``IterativeAlgorithm``, put time series algorithms first :pr:`3407`
        * Use ``prophet-prebuilt`` to install prophet in extras :pr:`3407`

.. warning::

    **Breaking Changes**
        * Moved model understanding metrics from ``graph.py`` to ``metrics.py`` :pr:`3417`


**v0.48.0 Mar. 25, 2022**
    * Enhancements
        * Add support for oversampling in time series classification problems :pr:`3387`
    * Fixes
        * Fixed ``TimeSeriesFeaturizer`` to make it deterministic when creating and choosing columns :pr:`3384`
        * Fixed bug where Email/URL features with missing values would cause the imputer to error out :pr:`3388`
    * Changes
        * Update maintainers to add Frank :pr:`3382`
        * Allow woodwork version 0.14.0 to be installed :pr:`3381`
        * Moved partial dependence functions from ``graph.py`` to a separate file :pr:`3404`
        * Pin ``click`` at ``8.0.4`` due to incompatibility with ``black`` :pr:`3413`
    * Documentation Changes
        * Added automl user guide section covering search algorithms :pr:`3394`
        * Updated broken links and automated broken link detection :pr:`3398`
        * Upgraded nbconvert :pr:`3402`, :pr:`3411`
    * Testing Changes
        * Updated scheduled workflows to only run on Alteryx owned repos (:pr:`3395`)
        * Exclude documentation versions other than latest from broken link check :pr:`3401`

.. warning::

    **Breaking Changes**
        * Moved partial dependence functions from ``graph.py`` to ``partial_dependence.py`` :pr:`3404`


**v0.47.0 Mar. 16, 2022**
    * Enhancements
        * Added ``TimeSeriesFeaturizer`` into ARIMA-based pipelines :pr:`3313`
        * Added caching capability for ensemble training during ``AutoMLSearch`` :pr:`3257`
        * Added new error code for zero unique values in ``NoVarianceDataCheck`` :pr:`3372`
    * Fixes
        * Fixed ``get_pipelines`` to reset pipeline threshold for binary cases :pr:`3360`
    * Changes
        * Update maintainers :pr:`3365`
        * Revert pandas 1.3.0 compatibility patch :pr:`3378`
    * Documentation Changes
        * Fixed documentation links to point to correct pages :pr:`3358`
    * Testing Changes
        * Checkout main branch in build_conda_pkg job :pr:`3375`

**v0.46.0 Mar. 03, 2022**
    * Enhancements
        * Added ``test_size`` parameter to ``ClassImbalanceDataCheck`` :pr:`3341`
        * Make target optional for ``NoVarianceDataCheck`` :pr:`3339`
    * Changes
        * Removed ``python_version<3.9`` environment marker from sktime dependency :pr:`3332`
        * Updated ``DatetimeFormatDataCheck`` to return all messages and not return early if NaNs are detected :pr:`3354`
    * Documentation Changes
        * Added in-line tabs and copy-paste functionality to documentation, overhauled Install page :pr:`3353`

**v0.45.0 Feb. 17, 2022**
    * Enhancements
        * Added support for pandas >= 1.4.0 :pr:`3324`
        * Standardized feature importance for estimators :pr:`3305`
        * Replaced usage of private method with Woodwork's public ``get_subset_schema`` method :pr:`3325`
    * Changes
        * Added an ``is_cv`` property to the datasplitters used :pr:`3297`
        * Changed SimpleImputer to ignore Natural Language columns :pr:`3324`
        * Added drop NaN component to some time series pipelines :pr:`3310`
    * Documentation Changes
        * Update README.md with Alteryx link (:pr:`3319`)
        * Added formatting to the AutoML user guide to shorten result outputs :pr:`3328`
    * Testing Changes
        * Add auto approve dependency workflow schedule for every 30 mins :pr:`3312`

**v0.44.0 Feb. 04, 2022**
    * Enhancements
        * Updated ``DefaultAlgorithm`` to also limit estimator usage for long-running multiclass problems :pr:`3099`
        * Added ``make_pipeline_from_data_check_output()`` utility method :pr:`3277`
        * Updated ``AutoMLSearch`` to use ``DefaultAlgorithm`` as the default automl algorithm :pr:`3261`, :pr:`3304`
        * Added more specific data check errors to ``DatetimeFormatDataCheck`` :pr:`3288`
        * Added ``features`` as a parameter for ``AutoMLSearch`` and add ``DFSTransformer`` to pipelines when ``features`` are present :pr:`3309`
    * Fixes
        * Updated the binary classification pipeline's ``optimize_thresholds`` method to use Nelder-Mead :pr:`3280`
        * Fixed bug where feature importance on time series pipelines only showed 0 for time index :pr:`3285`
    * Changes
        * Removed ``DateTimeNaNDataCheck`` and ``NaturalLanguageNaNDataCheck`` in favor of ``NullDataCheck`` :pr:`3260`
        * Drop support for Python 3.7 :pr:`3291`
        * Updated minimum version of ``woodwork`` to ``v0.12.0`` :pr:`3290`
    * Documentation Changes
        * Update documentation and docstring for `validate_holdout_datasets` for time series problems :pr:`3278`
        * Fixed mistake in documentation where wrong objective was used for calculating percent-better-than-baseline :pr:`3285`


.. warning::

    **Breaking Changes**
        * Removed ``DateTimeNaNDataCheck`` and ``NaturalLanguageNaNDataCheck`` in favor of ``NullDataCheck`` :pr:`3260`
        * Dropped support for Python 3.7 :pr:`3291`


**v0.43.0 Jan. 25, 2022**
    * Enhancements
        * Updated new ``NullDataCheck`` to return a warning and suggest an action to impute columns with null values :pr:`3197`
        * Updated ``make_pipeline_from_actions`` to handle null column imputation :pr:`3237`
        * Updated data check actions API to return options instead of actions and add functionality to suggest and take action on columns with null values :pr:`3182`
    * Fixes
        * Fixed categorical data leaking into non-categorical sub-pipelines in ``DefaultAlgorithm`` :pr:`3209`
        * Fixed Python 3.9 installation for prophet by updating ``pmdarima`` version in requirements :pr:`3268`
        * Allowed DateTime columns to pass through PerColumnImputer without breaking :pr:`3267`
    * Changes
        * Updated ``DataCheck`` ``validate()`` output to return a dictionary instead of list for actions :pr:`3142`
        * Updated ``DataCheck`` ``validate()`` API to use the new ``DataCheckActionOption`` class instead of ``DataCheckAction`` :pr:`3152`
        * Uncapped numba version and removed it from requirements :pr:`3263`
        * Renamed ``HighlyNullDataCheck`` to ``NullDataCheck`` :pr:`3197`
        * Updated data check ``validate()`` output to return a list of warnings and errors instead of a dictionary :pr:`3244`
        * Capped ``pandas`` at < 1.4.0 :pr:`3274`
    * Testing Changes
        * Bumped minimum ``IPython`` version to 7.16.3 in ``test-requirements.txt`` based on dependabot feedback :pr:`3269`

.. warning::

    **Breaking Changes**
        * Renamed ``HighlyNullDataCheck`` to ``NullDataCheck`` :pr:`3197`
        * Updated data check ``validate()`` output to return a list of warnings and errors instead of a dictionary. See the Data Check or Data Check Actions pages (under User Guide) for examples. :pr:`3244`
        * Removed ``impute_all`` and ``default_impute_strategy`` parameters from the ``PerColumnImputer`` :pr:`3267`
        * Updated ``PerColumnImputer`` such that columns not specified in ``impute_strategies`` dict will not be imputed anymore :pr:`3267`


**v0.42.0 Jan. 18, 2022**
    * Enhancements
        * Required the separation of training and test data by ``gap`` + 1 units to be verified by ``time_index`` for time series problems :pr:`3208`
        * Added support for boolean features for ``ARIMARegressor`` :pr:`3187`
        * Updated dependency bot workflow to remove outdated description and add new configuration to delete branches automatically :pr:`3212`
        * Added ``n_obs`` and ``n_splits`` to ``TimeSeriesParametersDataCheck`` error details :pr:`3246`
    * Fixes
        * Fixed classification pipelines to only accept target data with the appropriate number of classes :pr:`3185`
        * Added support for time series in ``DefaultAlgorithm`` :pr:`3177`
        * Standardized names of featurization components :pr:`3192`
        * Removed empty cell in text_input.ipynb :pr:`3234`
        * Removed potential prediction explanations failure when pipelines predicted a class with probability 1 :pr:`3221`
        * Dropped NaNs before partial dependence grid generation :pr:`3235`
        * Allowed prediction explanations to be json-serializable :pr:`3262`
        * Fixed bug where ``InvalidTargetDataCheck`` would not check time series regression targets :pr:`3251`
        * Fixed bug in ``are_datasets_separated_by_gap_time_index`` :pr:`3256`
    * Changes
        * Raised lowest compatible numpy version to 1.21.0 to address security concerns :pr:`3207`
        * Changed the default objective to ``MedianAE`` from ``R2`` for time series regression :pr:`3205`
        * Removed all-nan Unknown to Double logical conversion in ``infer_feature_types`` :pr:`3196`
        * Checking the validity of holdout data for time series problems can be performed by calling ``pipelines.utils.validate_holdout_datasets`` prior to calling ``predict`` :pr:`3208`
    * Testing Changes
        * Update auto approve workflow trigger and delete branch after merge :pr:`3265`

.. warning::

    **Breaking Changes**
        * Renamed ``DateTime Featurizer Component`` to ``DateTime Featurizer`` and ``Natural Language Featurization Component`` to ``Natural Language Featurizer`` :pr:`3192`



**v0.41.0 Jan. 06, 2022**
    * Enhancements
        * Added string support for DataCheckActionCode :pr:`3167`
        * Added ``DataCheckActionOption`` class :pr:`3134`
        * Add issue templates for bugs, feature requests and documentation improvements for GitHub :pr:`3199`
    * Fixes
        * Fix bug where prediction explanations ``class_name`` was shown as float for boolean targets :pr:`3179`
        * Fixed bug in nightly linux tests :pr:`3189`
    * Changes
        * Removed usage of scikit-learn's ``LabelEncoder`` in favor of ours :pr:`3161`
        * Removed nullable types checking from ``infer_feature_types`` :pr:`3156`
        * Fixed ``mean_cv_data`` and ``validation_score`` values in AutoMLSearch.rankings to reflect cv score or ``NaN`` when appropriate :pr:`3162`
    * Testing Changes
        * Updated tests to use new pipeline API instead of defining custom pipeline classes :pr:`3172`
        * Add workflow to auto-merge dependency PRs if status checks pass :pr:`3184`

**v0.40.0 Dec. 22, 2021**
    * Enhancements
        * Added ``TimeSeriesSplittingDataCheck`` to ``DefaultDataChecks`` to verify adequate class representation in time series classification problems :pr:`3141`
        * Added the ability to accept serialized features and skip computation in ``DFSTransformer`` :pr:`3106`
        * Added support for known-in-advance features :pr:`3149`
        * Added Holt-Winters ``ExponentialSmoothingRegressor`` for time series regression problems :pr:`3157`
        * Required the separation of training and test data by ``gap`` + 1 units to be verified by ``time_index`` for time series problems :pr:`3160`
    * Fixes
        * Fixed error caused when tuning threshold for time series binary classification :pr:`3140`
    * Changes
        * ``TimeSeriesParametersDataCheck`` was added to ``DefaultDataChecks`` for time series problems :pr:`3139`
        * Renamed ``date_index`` to ``time_index`` in ``problem_configuration`` for time series problems :pr:`3137`
        * Updated ``nlp-primitives`` minimum version to 2.1.0 :pr:`3166`
        * Updated minimum version of ``woodwork`` to v0.11.0 :pr:`3171`
        * Revert `3160` until uninferrable frequency can be addressed earlier in the process :pr:`3198`
    * Documentation Changes
        * Added comments to provide clarity on doctests :pr:`3155`
    * Testing Changes
        * Parameterized tests in ``test_datasets.py`` :pr:`3145`

.. warning::

    **Breaking Changes**
        * Renamed ``date_index`` to ``time_index`` in ``problem_configuration`` for time series problems :pr:`3137`


**v0.39.0 Dec. 9, 2021**
    * Enhancements
        * Renamed ``DelayedFeatureTransformer`` to ``TimeSeriesFeaturizer`` and enhanced it to compute rolling features :pr:`3028`
        * Added ability to impute only specific columns in ``PerColumnImputer`` :pr:`3123`
        * Added ``TimeSeriesParametersDataCheck`` to verify the time series parameters are valid given the number of splits in cross validation :pr:`3111`
    * Fixes
        * Default parameters for ``RFRegressorSelectFromModel`` and ``RFClassifierSelectFromModel`` has been fixed to avoid selecting all features :pr:`3110`
    * Changes
        * Removed reliance on a datetime index for ``ARIMARegressor`` and ``ProphetRegressor`` :pr:`3104`
        * Included target leakage check when fitting ``ARIMARegressor`` to account for the lack of ``TimeSeriesFeaturizer`` in ``ARIMARegressor`` based pipelines :pr:`3104`
        * Cleaned up and refactored ``InvalidTargetDataCheck`` implementation and docstring :pr:`3122`
        * Removed indices information from the output of ``HighlyNullDataCheck``'s ``validate()`` method :pr:`3092`
        * Added ``ReplaceNullableTypes`` component to prepare for handling pandas nullable types. :pr:`3090`
        * Updated ``make_pipeline`` for handling pandas nullable types in preprocessing pipeline. :pr:`3129`
        * Removed unused ``EnsembleMissingPipelinesError`` exception definition :pr:`3131`
    * Testing Changes
        * Refactored tests to avoid using ``importorskip`` :pr:`3126`
        * Added ``skip_during_conda`` test marker to skip tests that are not supposed to run during conda build :pr:`3127`
        * Added ``skip_if_39`` test marker to skip tests that are not supposed to run during python 3.9 :pr:`3133`

.. warning::

    **Breaking Changes**
        * Renamed ``DelayedFeatureTransformer`` to ``TimeSeriesFeaturizer`` :pr:`3028`
        * ``ProphetRegressor`` now requires a datetime column in ``X`` represented by the ``date_index`` parameter :pr:`3104`
        * Renamed module ``evalml.data_checks.invalid_target_data_check`` to ``evalml.data_checks.invalid_targets_data_check`` :pr:`3122`
        * Removed unused ``EnsembleMissingPipelinesError`` exception definition :pr:`3131`


**v0.38.0 Nov. 27, 2021**
    * Enhancements
        * Added ``data_check_name`` attribute to the data check action class :pr:`3034`
        * Added ``NumWords`` and ``NumCharacters`` primitives to ``TextFeaturizer`` and renamed ``TextFeaturizer` to ``NaturalLanguageFeaturizer`` :pr:`3030`
        * Added support for ``scikit-learn > 1.0.0`` :pr:`3051`
        * Required the ``date_index`` parameter to be specified for time series problems  in ``AutoMLSearch`` :pr:`3041`
        * Allowed time series pipelines to predict on test datasets whose length is less than or equal to the ``forecast_horizon``. Also allowed the test set index to start at 0. :pr:`3071`
        * Enabled time series pipeline to predict on data with features that are not known-in-advanced :pr:`3094`
    * Fixes
        * Added in error message when fit and predict/predict_proba data types are different :pr:`3036`
        * Fixed bug where ensembling components could not get converted to JSON format :pr:`3049`
        * Fixed bug where components with tuned integer hyperparameters could not get converted to JSON format :pr:`3049`
        * Fixed bug where force plots were not displaying correct feature values :pr:`3044`
        * Included confusion matrix at the pipeline threshold for ``find_confusion_matrix_per_threshold`` :pr:`3080`
        * Fixed bug where One Hot Encoder would error out if a non-categorical feature had a missing value :pr:`3083`
        * Fixed bug where features created from categorical columns by ``Delayed Feature Transformer`` would be inferred as categorical :pr:`3083`
    * Changes
        * Delete ``predict_uses_y`` estimator attribute :pr:`3069`
        * Change ``DateTimeFeaturizer`` to use corresponding Featuretools primitives :pr:`3081`
        * Updated ``TargetDistributionDataCheck`` to return metadata details as floats rather strings :pr:`3085`
        * Removed dependency on ``psutil`` package :pr:`3093`
    * Documentation Changes
        * Updated docs to use data check action methods rather than manually cleaning data :pr:`3050`
    * Testing Changes
        * Updated integration tests to use ``make_pipeline_from_actions`` instead of private method :pr:`3047`


.. warning::

    **Breaking Changes**
        * Added ``data_check_name`` attribute to the data check action class :pr:`3034`
        * Renamed ``TextFeaturizer` to ``NaturalLanguageFeaturizer`` :pr:`3030`
        * Updated the ``Pipeline.graph_json`` function to return a dictionary of "from" and "to" edges instead of tuples :pr:`3049`
        * Delete ``predict_uses_y`` estimator attribute :pr:`3069`
        * Changed time series problems in ``AutoMLSearch`` to need a not-``None`` ``date_index`` :pr:`3041`
        * Changed the ``DelayedFeatureTransformer`` to throw a ``ValueError`` during fit if the ``date_index`` is ``None`` :pr:`3041`
        * Passing ``X=None`` to ``DelayedFeatureTransformer`` is deprecated :pr:`3041`


**v0.37.0 Nov. 9, 2021**
    * Enhancements
        * Added ``find_confusion_matrix_per_threshold`` to Model Understanding :pr:`2972`
        * Limit computationally-intensive models during ``AutoMLSearch`` for certain multiclass problems, allow for opt-in with parameter ``allow_long_running_models`` :pr:`2982`
        * Added support for stacked ensemble pipelines to prediction explanations module :pr:`2971`
        * Added integration tests for data checks and data checks actions workflow :pr:`2883`
        * Added a change in pipeline structure to handle categorical columns separately for pipelines in ``DefaultAlgorithm`` :pr:`2986`
        * Added an algorithm to ``DelayedFeatureTransformer`` to select better lags :pr:`3005`
        * Added test to ensure pickling pipelines preserves thresholds :pr:`3027`
        * Added AutoML function to access ensemble pipeline's input pipelines IDs :pr:`3011`
        * Added ability to define which class is "positive" for label encoder in binary classification case :pr:`3033`
    * Fixes
        * Fixed bug where ``Oversampler`` didn't consider boolean columns to be categorical :pr:`2980`
        * Fixed permutation importance failing when target is categorical :pr:`3017`
        * Updated estimator and pipelines' ``predict``, ``predict_proba``, ``transform``, ``inverse_transform`` methods to preserve input indices :pr:`2979`
        * Updated demo dataset link for daily min temperatures :pr:`3023`
    * Changes
        * Updated ``OutliersDataCheck`` and ``UniquenessDataCheck`` and allow for the suspension of the Nullable types error :pr:`3018`
    * Documentation Changes
        * Fixed cost benefit matrix demo formatting :pr:`2990`
        * Update ReadMe.md with new badge links and updated installation instructions for conda :pr:`2998`
        * Added more comprehensive doctests :pr:`3002`


**v0.36.0 Oct. 27, 2021**
    * Enhancements
        * Added LIME as an algorithm option for ``explain_predictions`` and ``explain_predictions_best_worst`` :pr:`2905`
        * Standardized data check messages and added default "rows" and "columns" to data check message details dictionary :pr:`2869`
        * Added ``rows_of_interest`` to pipeline utils :pr:`2908`
        * Added support for woodwork version ``0.8.2`` :pr:`2909`
        * Enhanced the ``DateTimeFeaturizer`` to handle ``NaNs`` in date features :pr:`2909`
        * Added support for woodwork logical types ``PostalCode``, ``SubRegionCode``, and ``CountryCode`` in model understanding tools :pr:`2946`
        * Added Vowpal Wabbit regressor and classifiers :pr:`2846`
        * Added `NoSplit` data splitter for future unsupervised learning searches :pr:`2958`
        * Added method to convert actions into a preprocessing pipeline :pr:`2968`
    * Fixes
        * Fixed bug where partial dependence was not respecting the ww schema :pr:`2929`
        * Fixed ``calculate_permutation_importance`` for datetimes on ``StandardScaler`` :pr:`2938`
        * Fixed ``SelectColumns`` to only select available features for feature selection in ``DefaultAlgorithm`` :pr:`2944`
        * Fixed ``DropColumns`` component not receiving parameters in ``DefaultAlgorithm`` :pr:`2945`
        * Fixed bug where trained binary thresholds were not being returned by ``get_pipeline`` or ``clone`` :pr:`2948`
        * Fixed bug where ``Oversampler`` selected ww logical categorical instead of ww semantic category :pr:`2946`
    * Changes
        * Changed ``make_pipeline`` function to place the ``DateTimeFeaturizer`` prior to the ``Imputer`` so that ``NaN`` dates can be imputed :pr:`2909`
        * Refactored ``OutliersDataCheck`` and ``HighlyNullDataCheck`` to add more descriptive metadata :pr:`2907`
        * Bumped minimum version of ``dask`` from 2021.2.0 to 2021.10.0 :pr:`2978`
    * Documentation Changes
        * Added back Future Release section to release notes :pr:`2927`
        * Updated CI to run doctest (docstring tests) and apply necessary fixes to docstrings :pr:`2933`
        * Added documentation for ``BinaryClassificationPipeline`` thresholding :pr:`2937`
    * Testing Changes
        * Fixed dependency checker to catch full names of packages :pr:`2930`
        * Refactored ``build_conda_pkg`` to work from a local recipe :pr:`2925`
        * Refactored component test for different environments :pr:`2957`

.. warning::

    **Breaking Changes**
        * Standardized data check messages and added default "rows" and "columns" to data check message details dictionary. This may change the number of messages returned from a data check. :pr:`2869`


**v0.35.0 Oct. 14, 2021**
    * Enhancements
        * Added human-readable pipeline explanations to model understanding :pr:`2861`
        * Updated to support Featuretools 1.0.0 and nlp-primitives 2.0.0 :pr:`2848`
    * Fixes
        * Fixed bug where ``long`` mode for the top level search method was not respected :pr:`2875`
        * Pinned ``cmdstan`` to ``0.28.0`` in ``cmdstan-builder`` to prevent future breaking of support for Prophet :pr:`2880`
        * Added ``Jarque-Bera`` to the ``TargetDistributionDataCheck`` :pr:`2891`
    * Changes
        * Updated pipelines to use a label encoder component instead of doing encoding on the pipeline level :pr:`2821`
        * Deleted scikit-learn ensembler :pr:`2819`
        * Refactored pipeline building logic out of ``AutoMLSearch`` and into ``IterativeAlgorithm`` :pr:`2854`
        * Refactored names for methods in ``ComponentGraph`` and ``PipelineBase`` :pr:`2902`
    * Documentation Changes
        * Updated ``install.ipynb`` to reflect flexibility for ``cmdstan`` version installation :pr:`2880`
        * Updated the conda section of our contributing guide :pr:`2899`
    * Testing Changes
        * Updated ``test_all_estimators`` to account for Prophet being allowed for Python 3.9 :pr:`2892`
        * Updated linux tests to use ``cmdstan-builder==0.0.8`` :pr:`2880`

.. warning::

    **Breaking Changes**
        * Updated pipelines to use a label encoder component instead of doing encoding on the pipeline level. This means that pipelines will no longer automatically encode non-numerical targets. Please use a label encoder if working with classification problems and non-numeric targets. :pr:`2821`
        * Deleted scikit-learn ensembler :pr:`2819`
        * ``IterativeAlgorithm`` now requires X, y, problem_type as required arguments as well as sampler_name, allowed_model_families, allowed_component_graphs, max_batches, and verbose as optional arguments :pr:`2854`
        * Changed method names of ``fit_features`` and ``compute_final_component_features`` to ``fit_and_transform_all_but_final`` and ``transform_all_but_final`` in ``ComponentGraph``, and ``compute_estimator_features`` to ``transform_all_but_final`` in pipeline classes :pr:`2902`

**v0.34.0 Sep. 30, 2021**
    * Enhancements
        * Updated to work with Woodwork 0.8.1 :pr:`2783`
        * Added validation that ``training_data`` and ``training_target`` are not ``None`` in prediction explanations :pr:`2787`
        * Added support for training-only components in pipelines and component graphs :pr:`2776`
        * Added default argument for the parameters value for ``ComponentGraph.instantiate`` :pr:`2796`
        * Added ``TIME_SERIES_REGRESSION`` to ``LightGBMRegressor's`` supported problem types :pr:`2793`
        * Provided a JSON representation of a pipeline's DAG structure :pr:`2812`
        * Added validation to holdout data passed to ``predict`` and ``predict_proba`` for time series :pr:`2804`
        * Added information about which row indices are outliers in ``OutliersDataCheck`` :pr:`2818`
        * Added verbose flag to top level ``search()`` method :pr:`2813`
        * Added support for linting jupyter notebooks and clearing the executed cells and empty cells :pr:`2829` :pr:`2837`
        * Added "DROP_ROWS" action to output of ``OutliersDataCheck.validate()`` :pr:`2820`
        * Added the ability of ``AutoMLSearch`` to accept a ``SequentialEngine`` instance as engine input :pr:`2838`
        * Added new label encoder component to EvalML :pr:`2853`
        * Added our own partial dependence implementation :pr:`2834`
    * Fixes
        * Fixed bug where ``calculate_permutation_importance`` was not calculating the right value for pipelines with target transformers :pr:`2782`
        * Fixed bug where transformed target values were not used in ``fit`` for time series pipelines :pr:`2780`
        * Fixed bug where ``score_pipelines`` method of ``AutoMLSearch`` would not work for time series problems :pr:`2786`
        * Removed ``TargetTransformer`` class :pr:`2833`
        * Added tests to verify ``ComponentGraph`` support by pipelines :pr:`2830`
        * Fixed incorrect parameter for baseline regression pipeline in ``AutoMLSearch`` :pr:`2847`
        * Fixed bug where the desired estimator family order was not respected in ``IterativeAlgorithm`` :pr:`2850`
    * Changes
        * Changed woodwork initialization to use partial schemas :pr:`2774`
        * Made ``Transformer.transform()`` an abstract method :pr:`2744`
        * Deleted ``EmptyDataChecks`` class :pr:`2794`
        * Removed data check for checking log distributions in ``make_pipeline`` :pr:`2806`
        * Changed the minimum ``woodwork`` version to 0.8.0 :pr:`2783`
        * Pinned ``woodwork`` version to 0.8.0 :pr:`2832`
        * Removed ``model_family`` attribute from ``ComponentBase`` and transformers :pr:`2828`
        * Limited ``scikit-learn`` until new features and errors can be addressed :pr:`2842`
        * Show DeprecationWarning when Sklearn Ensemblers are called :pr:`2859`
    * Testing Changes
        * Updated matched assertion message regarding monotonic indices in polynomial detrender tests :pr:`2811`
        * Added a test to make sure pip versions match conda versions :pr:`2851`

.. warning::

    **Breaking Changes**
        * Made ``Transformer.transform()`` an abstract method :pr:`2744`
        * Deleted ``EmptyDataChecks`` class :pr:`2794`
        * Removed data check for checking log distributions in ``make_pipeline`` :pr:`2806`


**v0.33.0 Sep. 15, 2021**
    * Fixes
        * Fixed bug where warnings during ``make_pipeline`` were not being raised to the user :pr:`2765`
    * Changes
        * Refactored and removed ``SamplerBase`` class :pr:`2775`
    * Documentation Changes
        * Added docstring linting packages ``pydocstyle`` and ``darglint`` to `make-lint` command :pr:`2670`


**v0.32.1 Sep. 10, 2021**
    * Enhancements
        * Added ``verbose`` flag to ``AutoMLSearch`` to run search in silent mode by default :pr:`2645`
        * Added label encoder to ``XGBoostClassifier`` to remove the warning :pr:`2701`
        * Set ``eval_metric`` to ``logloss`` for ``XGBoostClassifier`` :pr:`2741`
        * Added support for ``woodwork`` versions ``0.7.0`` and ``0.7.1`` :pr:`2743`
        * Changed ``explain_predictions`` functions to display original feature values :pr:`2759`
        * Added ``X_train`` and ``y_train`` to ``graph_prediction_vs_actual_over_time`` and ``get_prediction_vs_actual_over_time_data`` :pr:`2762`
        * Added ``forecast_horizon`` as a required parameter to time series pipelines and ``AutoMLSearch`` :pr:`2697`
        * Added ``predict_in_sample`` and ``predict_proba_in_sample`` methods to time series pipelines to predict on data where the target is known, e.g. cross-validation :pr:`2697`
    * Fixes
        * Fixed bug where ``_catch_warnings`` assumed all warnings were ``PipelineNotUsed`` :pr:`2753`
        * Fixed bug where ``Imputer.transform`` would erase ww typing information prior to handing data to the ``SimpleImputer`` :pr:`2752`
        * Fixed bug where ``Oversampler`` could not be copied :pr:`2755`
    * Changes
        * Deleted ``drop_nan_target_rows`` utility method :pr:`2737`
        * Removed default logging setup and debugging log file :pr:`2645`
        * Changed the default n_jobs value for ``XGBoostClassifier`` and ``XGBoostRegressor`` to 12 :pr:`2757`
        * Changed ``TimeSeriesBaselineEstimator`` to only work on a time series pipeline with a ``DelayedFeaturesTransformer`` :pr:`2697`
        * Added ``X_train`` and ``y_train`` as optional parameters to pipeline ``predict``, ``predict_proba``. Only used for time series pipelines :pr:`2697`
        * Added ``training_data`` and ``training_target`` as optional parameters to ``explain_predictions`` and ``explain_predictions_best_worst`` to support time series pipelines :pr:`2697`
        * Changed time series pipeline predictions to no longer output series/dataframes padded with NaNs. A prediction will be returned for every row in the `X` input :pr:`2697`
    * Documentation Changes
        * Specified installation steps for Prophet :pr:`2713`
        * Added documentation for data exploration on data check actions :pr:`2696`
        * Added a user guide entry for time series modelling :pr:`2697`
    * Testing Changes
        * Fixed flaky ``TargetDistributionDataCheck`` test for very_lognormal distribution :pr:`2748`

.. warning::

    **Breaking Changes**
        * Removed default logging setup and debugging log file :pr:`2645`
        * Added ``X_train`` and ``y_train`` to ``graph_prediction_vs_actual_over_time`` and ``get_prediction_vs_actual_over_time_data`` :pr:`2762`
        * Added ``forecast_horizon`` as a required parameter to time series pipelines and ``AutoMLSearch`` :pr:`2697`
        * Changed ``TimeSeriesBaselineEstimator`` to only work on a time series pipeline with a ``DelayedFeaturesTransformer`` :pr:`2697`
        * Added ``X_train`` and ``y_train`` as required parameters for ``predict`` and ``predict_proba`` in time series pipelines :pr:`2697`
        * Added ``training_data`` and ``training_target`` as required parameters to ``explain_predictions`` and ``explain_predictions_best_worst`` for time series pipelines :pr:`2697`

**v0.32.0 Aug. 31, 2021**
    * Enhancements
        * Allow string for ``engine`` parameter for ``AutoMLSearch``:pr:`2667`
        * Add ``ProphetRegressor`` to AutoML :pr:`2619`
        * Integrated ``DefaultAlgorithm`` into ``AutoMLSearch`` :pr:`2634`
        * Removed SVM "linear" and "precomputed" kernel hyperparameter options, and improved default parameters :pr:`2651`
        * Updated ``ComponentGraph`` initalization to raise ``ValueError`` when user attempts to use ``.y`` for a component that does not produce a tuple output :pr:`2662`
        * Updated to support Woodwork 0.6.0 :pr:`2690`
        * Updated pipeline ``graph()`` to distingush X and y edges :pr:`2654`
        * Added ``DropRowsTransformer`` component :pr:`2692`
        * Added ``DROP_ROWS`` to ``_make_component_list_from_actions`` and clean up metadata :pr:`2694`
        * Add new ensembler component :pr:`2653`
    * Fixes
        * Updated Oversampler logic to select best SMOTE based on component input instead of pipeline input :pr:`2695`
        * Added ability to explicitly close DaskEngine resources to improve runtime and reduce Dask warnings :pr:`2667`
        * Fixed partial dependence bug for ensemble pipelines :pr:`2714`
        * Updated ``TargetLeakageDataCheck`` to maintain user-selected logical types :pr:`2711`
    * Changes
        * Replaced ``SMOTEOversampler``, ``SMOTENOversampler`` and ``SMOTENCOversampler`` with consolidated ``Oversampler`` component :pr:`2695`
        * Removed ``LinearRegressor`` from the list of default ``AutoMLSearch`` estimators due to poor performance :pr:`2660`
    * Documentation Changes
        * Added user guide documentation for using ``ComponentGraph`` and added ``ComponentGraph`` to API reference :pr:`2673`
        * Updated documentation to make parallelization of AutoML clearer :pr:`2667`
    * Testing Changes
        * Removes the process-level parallelism from the ``test_cancel_job`` test :pr:`2666`
        * Installed numba 0.53 in windows CI to prevent problems installing version 0.54 :pr:`2710`

.. warning::

    **Breaking Changes**
        * Renamed the current top level ``search`` method to ``search_iterative`` and defined a new ``search`` method for the ``DefaultAlgorithm`` :pr:`2634`
        * Replaced ``SMOTEOversampler``, ``SMOTENOversampler`` and ``SMOTENCOversampler`` with consolidated ``Oversampler`` component :pr:`2695`
        * Removed ``LinearRegressor`` from the list of default ``AutoMLSearch`` estimators due to poor performance :pr:`2660`

**v0.31.0 Aug. 19, 2021**
    * Enhancements
        * Updated the high variance check in AutoMLSearch to be robust to a variety of objectives and cv scores :pr:`2622`
        * Use Woodwork's outlier detection for the ``OutliersDataCheck`` :pr:`2637`
        * Added ability to utilize instantiated components when creating a pipeline :pr:`2643`
        * Sped up the all Nan and unknown check in ``infer_feature_types`` :pr:`2661`
    * Fixes
    * Changes
        * Deleted ``_put_into_original_order`` helper function :pr:`2639`
        * Refactored time series pipeline code using a time series pipeline base class :pr:`2649`
        * Renamed ``dask_tests`` to ``parallel_tests`` :pr:`2657`
        * Removed commented out code in ``pipeline_meta.py`` :pr:`2659`
    * Documentation Changes
        * Add complete install command to README and Install section :pr:`2627`
        * Cleaned up documentation for ``MulticollinearityDataCheck`` :pr:`2664`
    * Testing Changes
        * Speed up CI by splitting Prophet tests into a separate workflow in GitHub :pr:`2644`

.. warning::

    **Breaking Changes**
        * ``TimeSeriesRegressionPipeline`` no longer inherits from ``TimeSeriesRegressionPipeline`` :pr:`2649`


**v0.30.2 Aug. 16, 2021**
    * Fixes
        * Updated changelog and version numbers to match the release.  Release 0.30.1 was release erroneously without a change to the version numbers.  0.30.2 replaces it.

**v0.30.1 Aug. 12, 2021**
    * Enhancements
        * Added ``DatetimeFormatDataCheck`` for time series problems :pr:`2603`
        * Added ``ProphetRegressor`` to estimators :pr:`2242`
        * Updated ``ComponentGraph`` to handle not calling samplers' transform during predict, and updated samplers' transform methods s.t. ``fit_transform`` is equivalent to ``fit(X, y).transform(X, y)`` :pr:`2583`
        * Updated ``ComponentGraph`` ``_validate_component_dict`` logic to be stricter about input values :pr:`2599`
        * Patched bug in ``xgboost`` estimators where predicting on a feature matrix of only booleans would throw an exception. :pr:`2602`
        * Updated ``ARIMARegressor`` to use relative forecasting to predict values :pr:`2613`
        * Added support for creating pipelines without an estimator as the final component and added ``transform(X, y)`` method to pipelines and component graphs :pr:`2625`
        * Updated to support Woodwork 0.5.1 :pr:`2610`
    * Fixes
        * Updated ``AutoMLSearch`` to drop ``ARIMARegressor`` from ``allowed_estimators`` if an incompatible frequency is detected :pr:`2632`
        * Updated ``get_best_sampler_for_data`` to consider all non-numeric datatypes as categorical for SMOTE :pr:`2590`
        * Fixed inconsistent test results from `TargetDistributionDataCheck` :pr:`2608`
        * Adopted vectorized pd.NA checking for Woodwork 0.5.1 support :pr:`2626`
        * Pinned upper version of astroid to 2.6.6 to keep ReadTheDocs working. :pr:`2638`
    * Changes
        * Renamed SMOTE samplers to SMOTE oversampler :pr:`2595`
        * Changed ``partial_dependence`` and ``graph_partial_dependence`` to raise a ``PartialDependenceError`` instead of ``ValueError``. This is not a breaking change because ``PartialDependenceError`` is a subclass of ``ValueError`` :pr:`2604`
        * Cleaned up code duplication in ``ComponentGraph`` :pr:`2612`
        * Stored predict_proba results in .x for intermediate estimators in ComponentGraph :pr:`2629`
    * Documentation Changes
        * To avoid local docs build error, only add warning disable and download headers on ReadTheDocs builds, not locally :pr:`2617`
    * Testing Changes
        * Updated partial_dependence tests to change the element-wise comparison per the Plotly 5.2.1 upgrade :pr:`2638`
        * Changed the lint CI job to only check against python 3.9 via the `-t` flag :pr:`2586`
        * Installed Prophet in linux nightlies test and fixed ``test_all_components`` :pr:`2598`
        * Refactored and fixed all ``make_pipeline`` tests to assert correct order and address new Woodwork Unknown type inference :pr:`2572`
        * Removed ``component_graphs`` as a global variable in ``test_component_graphs.py`` :pr:`2609`

.. warning::

    **Breaking Changes**
        * Renamed SMOTE samplers to SMOTE oversampler. Please use ``SMOTEOversampler``, ``SMOTENCOversampler``, ``SMOTENOversampler`` instead of ``SMOTESampler``, ``SMOTENCSampler``, and ``SMOTENSampler`` :pr:`2595`


**v0.30.0 Aug. 3, 2021**
    * Enhancements
        * Added ``LogTransformer`` and ``TargetDistributionDataCheck`` :pr:`2487`
        * Issue a warning to users when a pipeline parameter passed in isn't used in the pipeline :pr:`2564`
        * Added Gini coefficient as an objective :pr:`2544`
        * Added ``repr`` to ``ComponentGraph`` :pr:`2565`
        * Added components to extract features from ``URL`` and ``EmailAddress`` Logical Types :pr:`2550`
        * Added support for `NaN` values in ``TextFeaturizer`` :pr:`2532`
        * Added ``SelectByType`` transformer :pr:`2531`
        * Added separate thresholds for percent null rows and columns in ``HighlyNullDataCheck`` :pr:`2562`
        * Added support for `NaN` natural language values :pr:`2577`
    * Fixes
        * Raised error message for types ``URL``, ``NaturalLanguage``, and ``EmailAddress`` in ``partial_dependence`` :pr:`2573`
    * Changes
        * Updated ``PipelineBase`` implementation for creating pipelines from a list of components :pr:`2549`
        * Moved ``get_hyperparameter_ranges`` to ``PipelineBase`` class from automl/utils module :pr:`2546`
        * Renamed ``ComponentGraph``'s ``get_parents`` to ``get_inputs`` :pr:`2540`
        * Removed ``ComponentGraph.linearized_component_graph`` and ``ComponentGraph.from_list`` :pr:`2556`
        * Updated ``ComponentGraph`` to enforce requiring `.x` and `.y` inputs for each component in the graph :pr:`2563`
        * Renamed existing ensembler implementation from ``StackedEnsemblers`` to ``SklearnStackedEnsemblers`` :pr:`2578`
    * Documentation Changes
        * Added documentation for ``DaskEngine`` and ``CFEngine`` parallel engines :pr:`2560`
        * Improved detail of ``TextFeaturizer`` docstring and tutorial :pr:`2568`
    * Testing Changes
        * Added test that makes sure ``split_data`` does not shuffle for time series problems :pr:`2552`

.. warning::

    **Breaking Changes**
        * Moved ``get_hyperparameter_ranges`` to ``PipelineBase`` class from automl/utils module :pr:`2546`
        * Renamed ``ComponentGraph``'s ``get_parents`` to ``get_inputs`` :pr:`2540`
        * Removed ``ComponentGraph.linearized_component_graph`` and ``ComponentGraph.from_list`` :pr:`2556`
        * Updated ``ComponentGraph`` to enforce requiring `.x` and `.y` inputs for each component in the graph :pr:`2563`


**v0.29.0 Jul. 21, 2021**
    * Enhancements
        * Updated 1-way partial dependence support for datetime features :pr:`2454`
        * Added details on how to fix error caused by broken ww schema :pr:`2466`
        * Added ability to use built-in pickle for saving AutoMLSearch :pr:`2463`
        * Updated our components and component graphs to use latest features of ww 0.4.1, e.g. ``concat_columns`` and drop in-place. :pr:`2465`
        * Added new, concurrent.futures based engine for parallel AutoML :pr:`2506`
        * Added support for new Woodwork ``Unknown`` type in AutoMLSearch :pr:`2477`
        * Updated our components with an attribute that describes if they modify features or targets and can be used in list API for pipeline initialization :pr:`2504`
        * Updated ``ComponentGraph`` to accept X and y as inputs :pr:`2507`
        * Removed unused ``TARGET_BINARY_INVALID_VALUES`` from ``DataCheckMessageCode`` enum and fixed formatting of objective documentation :pr:`2520`
        * Added ``EvalMLAlgorithm`` :pr:`2525`
        * Added support for `NaN` values in ``TextFeaturizer`` :pr:`2532`
    * Fixes
        * Fixed ``FraudCost`` objective and reverted threshold optimization method for binary classification to ``Golden`` :pr:`2450`
        * Added custom exception message for partial dependence on features with scales that are too small :pr:`2455`
        * Ensures the typing for Ordinal and Datetime ltypes are passed through _retain_custom_types_and_initalize_woodwork :pr:`2461`
        * Updated to work with Pandas 1.3.0 :pr:`2442`
        * Updated to work with sktime 0.7.0 :pr:`2499`
    * Changes
        * Updated XGBoost dependency to ``>=1.4.2`` :pr:`2484`, :pr:`2498`
        * Added a ``DeprecationWarning`` about deprecating the list API for ``ComponentGraph`` :pr:`2488`
        * Updated ``make_pipeline`` for AutoML to create dictionaries, not lists, to initialize pipelines :pr:`2504`
        * No longer installing graphviz on windows in our CI pipelines because release 0.17 breaks windows 3.7 :pr:`2516`
    * Documentation Changes
        * Moved docstrings from ``__init__`` to class pages, added missing docstrings for missing classes, and updated missing default values :pr:`2452`
        * Build documentation with sphinx-autoapi :pr:`2458`
        * Change ``autoapi_ignore`` to only ignore files in ``evalml/tests/*`` :pr:`2530`
    * Testing Changes
        * Fixed flaky dask tests :pr:`2471`
        * Removed shellcheck action from ``build_conda_pkg`` action :pr:`2514`
        * Added a tmp_dir fixture that deletes its contents after tests run :pr:`2505`
        * Added a test that makes sure all pipelines in ``AutoMLSearch`` get the same data splits :pr:`2513`
        * Condensed warning output in test logs :pr:`2521`

.. warning::

    **Breaking Changes**
        * `NaN` values in the `Natural Language` type are no longer supported by the Imputer with the pandas upgrade. :pr:`2477`

**v0.28.0 Jul. 2, 2021**
    * Enhancements
        * Added support for showing a Individual Conditional Expectations plot when graphing Partial Dependence :pr:`2386`
        * Exposed ``thread_count`` for Catboost estimators as ``n_jobs`` parameter :pr:`2410`
        * Updated Objectives API to allow for sample weighting :pr:`2433`
    * Fixes
        * Deleted unreachable line from ``IterativeAlgorithm`` :pr:`2464`
    * Changes
        * Pinned Woodwork version between 0.4.1 and 0.4.2 :pr:`2460`
        * Updated psutils minimum version in requirements :pr:`2438`
        * Updated ``log_error_callback`` to not include filepath in logged message :pr:`2429`
    * Documentation Changes
        * Sped up docs :pr:`2430`
        * Removed mentions of ``DataTable`` and ``DataColumn`` from the docs :pr:`2445`
    * Testing Changes
        * Added slack integration for nightlies tests :pr:`2436`
        * Changed ``build_conda_pkg`` CI job to run only when dependencies are updates :pr:`2446`
        * Updated workflows to store pytest runtimes as test artifacts :pr:`2448`
        * Added ``AutoMLTestEnv`` test fixture for making it easy to mock automl tests :pr:`2406`

**v0.27.0 Jun. 22, 2021**
    * Enhancements
        * Adds force plots for prediction explanations :pr:`2157`
        * Removed self-reference from ``AutoMLSearch`` :pr:`2304`
        * Added support for nonlinear pipelines for ``generate_pipeline_code`` :pr:`2332`
        * Added ``inverse_transform`` method to pipelines :pr:`2256`
        * Add optional automatic update checker :pr:`2350`
        * Added ``search_order`` to ``AutoMLSearch``'s ``rankings`` and ``full_rankings`` tables :pr:`2345`
        * Updated threshold optimization method for binary classification :pr:`2315`
        * Updated demos to pull data from S3 instead of including demo data in package :pr:`2387`
        * Upgrade woodwork version to v0.4.1 :pr:`2379`
    * Fixes
        * Preserve user-specified woodwork types throughout pipeline fit/predict :pr:`2297`
        * Fixed ``ComponentGraph`` appending target to ``final_component_features`` if there is a component that returns both X and y :pr:`2358`
        * Fixed partial dependence graph method failing on multiclass problems when the class labels are numeric :pr:`2372`
        * Added ``thresholding_objective`` argument to ``AutoMLSearch`` for binary classification problems :pr:`2320`
        * Added change for ``k_neighbors`` parameter in SMOTE Oversamplers to automatically handle small samples :pr:`2375`
        * Changed naming for ``Logistic Regression Classifier`` file :pr:`2399`
        * Pinned pytest-timeout to fix minimum dependence checker :pr:`2425`
        * Replaced ``Elastic Net Classifier`` base class with ``Logistsic Regression`` to avoid ``NaN`` outputs :pr:`2420`
    * Changes
        * Cleaned up ``PipelineBase``'s ``component_graph`` and ``_component_graph`` attributes. Updated ``PipelineBase`` ``__repr__`` and added ``__eq__`` for ``ComponentGraph`` :pr:`2332`
        * Added and applied  ``black`` linting package to the EvalML repo in place of ``autopep8`` :pr:`2306`
        * Separated `custom_hyperparameters` from pipelines and added them as an argument to ``AutoMLSearch`` :pr:`2317`
        * Replaced `allowed_pipelines` with `allowed_component_graphs` :pr:`2364`
        * Removed private method ``_compute_features_during_fit`` from ``PipelineBase`` :pr:`2359`
        * Updated ``compute_order`` in ``ComponentGraph`` to be a read-only property :pr:`2408`
        * Unpinned PyZMQ version in requirements.txt :pr:`2389`
        * Uncapping LightGBM version in requirements.txt :pr:`2405`
        * Updated minimum version of plotly :pr:`2415`
        * Removed ``SensitivityLowAlert`` objective from core objectives :pr:`2418`
    * Documentation Changes
        * Fixed lead scoring weights in the demos documentation :pr:`2315`
        * Fixed start page code and description dataset naming discrepancy :pr:`2370`
    * Testing Changes
        * Update minimum unit tests to run on all pull requests :pr:`2314`
        * Pass token to authorize uploading of codecov reports :pr:`2344`
        * Add ``pytest-timeout``. All tests that run longer than 6 minutes will fail. :pr:`2374`
        * Separated the dask tests out into separate github action jobs to isolate dask failures. :pr:`2376`
        * Refactored dask tests :pr:`2377`
        * Added the combined dask/non-dask unit tests back and renamed the dask only unit tests. :pr:`2382`
        * Sped up unit tests and split into separate jobs :pr:`2365`
        * Change CI job names, run lint for python 3.9, run nightlies on python 3.8 at 3am EST :pr:`2395` :pr:`2398`
        * Set fail-fast to false for CI jobs that run for PRs :pr:`2402`

.. warning::

    **Breaking Changes**
        * `AutoMLSearch` will accept `allowed_component_graphs` instead of `allowed_pipelines` :pr:`2364`
        * Removed ``PipelineBase``'s ``_component_graph`` attribute. Updated ``PipelineBase`` ``__repr__`` and added ``__eq__`` for ``ComponentGraph`` :pr:`2332`
        * `pipeline_parameters` will no longer accept `skopt.space` variables since hyperparameter ranges will now be specified through `custom_hyperparameters` :pr:`2317`

**v0.25.0 Jun. 01, 2021**
    * Enhancements
        * Upgraded minimum woodwork to version 0.3.1. Previous versions will not be supported :pr:`2181`
        * Added a new callback parameter for ``explain_predictions_best_worst`` :pr:`2308`
    * Fixes
    * Changes
        * Deleted the ``return_pandas`` flag from our demo data loaders :pr:`2181`
        * Moved ``default_parameters`` to ``ComponentGraph`` from ``PipelineBase`` :pr:`2307`
    * Documentation Changes
        * Updated the release procedure documentation :pr:`2230`
    * Testing Changes
        * Ignoring ``test_saving_png_file`` while building conda package :pr:`2323`

.. warning::

    **Breaking Changes**
        * Deleted the ``return_pandas`` flag from our demo data loaders :pr:`2181`
        * Upgraded minimum woodwork to version 0.3.1. Previous versions will not be supported :pr:`2181`
        * Due to the weak-ref in woodwork, set the result of ``infer_feature_types`` to a variable before accessing woodwork :pr:`2181`

**v0.24.2 May. 24, 2021**
    * Enhancements
        * Added oversamplers to AutoMLSearch :pr:`2213` :pr:`2286`
        * Added dictionary input functionality for ``Undersampler`` component :pr:`2271`
        * Changed the default parameter values for ``Elastic Net Classifier`` and ``Elastic Net Regressor`` :pr:`2269`
        * Added dictionary input functionality for the Oversampler components :pr:`2288`
    * Fixes
        * Set default `n_jobs` to 1 for `StackedEnsembleClassifier` and `StackedEnsembleRegressor` until fix for text-based parallelism in sklearn stacking can be found :pr:`2295`
    * Changes
        * Updated ``start_iteration_callback`` to accept a pipeline instance instead of a pipeline class and no longer accept pipeline parameters as a parameter :pr:`2290`
        * Refactored ``calculate_permutation_importance`` method and add per-column permutation importance method :pr:`2302`
        * Updated logging information in ``AutoMLSearch.__init__`` to clarify pipeline generation :pr:`2263`
    * Documentation Changes
        * Minor changes to the release procedure :pr:`2230`
    * Testing Changes
        * Use codecov action to update coverage reports :pr:`2238`
        * Removed MarkupSafe dependency version pin from requirements.txt and moved instead into RTD docs build CI :pr:`2261`

.. warning::

    **Breaking Changes**
        * Updated ``start_iteration_callback`` to accept a pipeline instance instead of a pipeline class and no longer accept pipeline parameters as a parameter :pr:`2290`
        * Moved ``default_parameters`` to ``ComponentGraph`` from ``PipelineBase``. A pipeline's ``default_parameters`` is now accessible via ``pipeline.component_graph.default_parameters`` :pr:`2307`


**v0.24.1 May. 16, 2021**
    * Enhancements
        * Integrated ``ARIMARegressor`` into AutoML :pr:`2009`
        * Updated ``HighlyNullDataCheck`` to also perform a null row check :pr:`2222`
        * Set ``max_depth`` to 1 in calls to featuretools dfs :pr:`2231`
    * Fixes
        * Removed data splitter sampler calls during training :pr:`2253`
        * Set minimum required version for for pyzmq, colorama, and docutils :pr:`2254`
        * Changed BaseSampler to return None instead of y :pr:`2272`
    * Changes
        * Removed ensemble split and indices in ``AutoMLSearch`` :pr:`2260`
        * Updated pipeline ``repr()`` and ``generate_pipeline_code`` to return pipeline instances without generating custom pipeline class :pr:`2227`
    * Documentation Changes
        * Capped Sphinx version under 4.0.0 :pr:`2244`
    * Testing Changes
        * Change number of cores for pytest from 4 to 2 :pr:`2266`
        * Add minimum dependency checker to generate minimum requirement files :pr:`2267`
        * Add unit tests with minimum dependencies  :pr:`2277`


**v0.24.0 May. 04, 2021**
    * Enhancements
        * Added `date_index` as a required parameter for TimeSeries problems :pr:`2217`
        * Have the ``OneHotEncoder`` return the transformed columns as booleans rather than floats :pr:`2170`
        * Added Oversampler transformer component to EvalML :pr:`2079`
        * Added Undersampler to AutoMLSearch, as well as arguments ``_sampler_method`` and ``sampler_balanced_ratio`` :pr:`2128`
        * Updated prediction explanations functions to allow pipelines with XGBoost estimators :pr:`2162`
        * Added partial dependence for datetime columns :pr:`2180`
        * Update precision-recall curve with positive label index argument, and fix for 2d predicted probabilities :pr:`2090`
        * Add pct_null_rows to ``HighlyNullDataCheck`` :pr:`2211`
        * Added a standalone AutoML `search` method for convenience, which runs data checks and then runs automl :pr:`2152`
        * Make the first batch of AutoML have a predefined order, with linear models first and complex models last :pr:`2223` :pr:`2225`
        * Added sampling dictionary support to ``BalancedClassficationSampler`` :pr:`2235`
    * Fixes
        * Fixed partial dependence not respecting grid resolution parameter for numerical features :pr:`2180`
        * Enable prediction explanations for catboost for multiclass problems :pr:`2224`
    * Changes
        * Deleted baseline pipeline classes :pr:`2202`
        * Reverting user specified date feature PR :pr:`2155` until `pmdarima` installation fix is found :pr:`2214`
        * Updated pipeline API to accept component graph and other class attributes as instance parameters. Old pipeline API still works but will not be supported long-term. :pr:`2091`
        * Removed all old datasplitters from EvalML :pr:`2193`
        * Deleted ``make_pipeline_from_components`` :pr:`2218`
    * Documentation Changes
        * Renamed dataset to clarify that its gzipped but not a tarball :pr:`2183`
        * Updated documentation to use pipeline instances instead of pipeline subclasses :pr:`2195`
        * Updated contributing guide with a note about GitHub Actions permissions :pr:`2090`
        * Updated automl and model understanding user guides :pr:`2090`
    * Testing Changes
        * Use machineFL user token for dependency update bot, and add more reviewers :pr:`2189`


.. warning::

    **Breaking Changes**
        * All baseline pipeline classes (``BaselineBinaryPipeline``, ``BaselineMulticlassPipeline``, ``BaselineRegressionPipeline``, etc.) have been deleted :pr:`2202`
        * Updated pipeline API to accept component graph and other class attributes as instance parameters. Old pipeline API still works but will not be supported long-term. Pipelines can now be initialized by specifying the component graph as the first parameter, and then passing in optional arguments such as ``custom_name``, ``parameters``, etc. For example, ``BinaryClassificationPipeline(["Random Forest Classifier"], parameters={})``.  :pr:`2091`
        * Removed all old datasplitters from EvalML :pr:`2193`
        * Deleted utility method ``make_pipeline_from_components`` :pr:`2218`


**v0.23.0 Apr. 20, 2021**
    * Enhancements
        * Refactored ``EngineBase`` and ``SequentialEngine`` api. Adding ``DaskEngine`` :pr:`1975`.
        * Added optional ``engine`` argument to ``AutoMLSearch`` :pr:`1975`
        * Added a warning about how time series support is still in beta when a user passes in a time series problem to ``AutoMLSearch`` :pr:`2118`
        * Added ``NaturalLanguageNaNDataCheck`` data check :pr:`2122`
        * Added ValueError to ``partial_dependence`` to prevent users from computing partial dependence on columns with all NaNs :pr:`2120`
        * Added standard deviation of cv scores to rankings table :pr:`2154`
    * Fixes
        * Fixed ``BalancedClassificationDataCVSplit``, ``BalancedClassificationDataTVSplit``, and ``BalancedClassificationSampler`` to use ``minority:majority`` ratio instead of ``majority:minority`` :pr:`2077`
        * Fixed bug where two-way partial dependence plots with categorical variables were not working correctly :pr:`2117`
        * Fixed bug where ``hyperparameters`` were not displaying properly for pipelines with a list ``component_graph`` and duplicate components :pr:`2133`
        * Fixed bug where ``pipeline_parameters`` argument in ``AutoMLSearch`` was not applied to pipelines passed in as ``allowed_pipelines`` :pr:`2133`
        * Fixed bug where ``AutoMLSearch`` was not applying custom hyperparameters to pipelines with a list ``component_graph`` and duplicate components :pr:`2133`
    * Changes
        * Removed ``hyperparameter_ranges`` from Undersampler and renamed ``balanced_ratio`` to ``sampling_ratio`` for samplers :pr:`2113`
        * Renamed ``TARGET_BINARY_NOT_TWO_EXAMPLES_PER_CLASS`` data check message code to ``TARGET_MULTICLASS_NOT_TWO_EXAMPLES_PER_CLASS`` :pr:`2126`
        * Modified one-way partial dependence plots of categorical features to display data with a bar plot :pr:`2117`
        * Renamed ``score`` column for ``automl.rankings`` as ``mean_cv_score`` :pr:`2135`
        * Remove 'warning' from docs tool output :pr:`2031`
    * Documentation Changes
        * Fixed ``conf.py`` file :pr:`2112`
        * Added a sentence to the automl user guide stating that our support for time series problems is still in beta. :pr:`2118`
        * Fixed documentation demos :pr:`2139`
        * Update test badge in README to use GitHub Actions :pr:`2150`
    * Testing Changes
        * Fixed ``test_describe_pipeline`` for ``pandas`` ``v1.2.4`` :pr:`2129`
        * Added a GitHub Action for building the conda package :pr:`1870` :pr:`2148`


.. warning::

    **Breaking Changes**
        * Renamed ``balanced_ratio`` to ``sampling_ratio`` for the ``BalancedClassificationDataCVSplit``, ``BalancedClassificationDataTVSplit``, ``BalancedClassficationSampler``, and Undersampler :pr:`2113`
        * Deleted the "errors" key from automl results :pr:`1975`
        * Deleted the ``raise_and_save_error_callback`` and the ``log_and_save_error_callback`` :pr:`1975`
        * Fixed ``BalancedClassificationDataCVSplit``, ``BalancedClassificationDataTVSplit``, and ``BalancedClassificationSampler`` to use minority:majority ratio instead of majority:minority :pr:`2077`


**v0.22.0 Apr. 06, 2021**
    * Enhancements
        * Added a GitHub Action for ``linux_unit_tests``:pr:`2013`
        * Added recommended actions for ``InvalidTargetDataCheck``, updated ``_make_component_list_from_actions`` to address new action, and added ``TargetImputer`` component :pr:`1989`
        * Updated ``AutoMLSearch._check_for_high_variance`` to not emit ``RuntimeWarning`` :pr:`2024`
        * Added exception when pipeline passed to ``explain_predictions`` is a ``Stacked Ensemble`` pipeline :pr:`2033`
        * Added sensitivity at low alert rates as an objective :pr:`2001`
        * Added ``Undersampler`` transformer component :pr:`2030`
    * Fixes
        * Updated Engine's ``train_batch`` to apply undersampling :pr:`2038`
        * Fixed bug in where Time Series Classification pipelines were not encoding targets in ``predict`` and ``predict_proba`` :pr:`2040`
        * Fixed data splitting errors if target is float for classification problems :pr:`2050`
        * Pinned ``docutils`` to <0.17 to fix ReadtheDocs warning issues :pr:`2088`
    * Changes
        * Removed lists as acceptable hyperparameter ranges in ``AutoMLSearch`` :pr:`2028`
        * Renamed "details" to "metadata" for data check actions :pr:`2008`
    * Documentation Changes
        * Catch and suppress warnings in documentation :pr:`1991` :pr:`2097`
        * Change spacing in ``start.ipynb`` to provide clarity for ``AutoMLSearch`` :pr:`2078`
        * Fixed start code on README :pr:`2108`
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
        * Added a ``PolynomialDecomposer`` component :pr:`1992`
        * Added ``DateTimeNaNDataCheck`` data check :pr:`2039`
    * Fixes
        * Changed best pipeline to train on the entire dataset rather than just ensemble indices for ensemble problems :pr:`2037`
        * Updated binary classification pipelines to use objective decision function during scoring of custom objectives :pr:`1934`
    * Changes
        * Removed ``data_checks`` parameter, ``data_check_results`` and data checks logic from ``AutoMLSearch`` :pr:`1935`
        * Deleted ``random_state`` argument :pr:`1985`
        * Updated Woodwork version requirement to ``v0.0.11`` :pr:`1996`
    * Documentation Changes
    * Testing Changes
        * Removed ``build_docs`` CI job in favor of RTD GH builder :pr:`1974`
        * Added tests to confirm support for Python 3.9 :pr:`1724`
        * Added tests to support Dask AutoML/Engine :pr:`1990`
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
        * Added ability to handle index columns in ``AutoMLSearch`` and ``DataChecks`` :pr:`2138`
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
