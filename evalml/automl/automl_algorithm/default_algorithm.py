"""An automl algorithm that consists of two modes: fast and long, where fast is a subset of long."""
import logging

import numpy as np

from evalml.automl.automl_algorithm.automl_algorithm import AutoMLAlgorithm
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    EmailFeaturizer,
    OneHotEncoder,
    RFClassifierSelectFromModel,
    RFRegressorSelectFromModel,
    URLFeaturizer,
)
from evalml.pipelines.components.transformers.column_selectors import (
    SelectByType,
    SelectColumns,
)
from evalml.pipelines.components.utils import get_estimators, handle_component_class
from evalml.pipelines.utils import (
    _get_sampler,
    _make_pipeline_from_multiple_graphs,
    make_pipeline,
)
from evalml.problem_types import is_regression, is_time_series
from evalml.utils import infer_feature_types
from evalml.utils.logger import get_logger


class DefaultAlgorithm(AutoMLAlgorithm):
    """An automl algorithm that consists of two modes: fast and long, where fast is a subset of long.

    1. Naive pipelines:
        a. run baseline with default preprocessing pipeline
        b. run naive linear model with default preprocessing pipeline
        c. run basic RF pipeline with default preprocessing pipeline
    2. Naive pipelines with feature selection
        a. subsequent pipelines will use the selected features with a SelectedColumns transformer

    At this point we have a single pipeline candidate for preprocessing and feature selection

    3. Pipelines with preprocessing components:
        a. scan rest of estimators (our current batch 1).
    4. First ensembling run

    Fast mode ends here. Begin long mode.

    6. Run top 3 estimators:
        a. Generate 50 random parameter sets. Run all 150 in one batch
    7. Second ensembling run
    8. Repeat these indefinitely until stopping criterion is met:
        a. For each of the previous top 3 estimators, sample 10 parameters from the tuner. Run all 30 in one batch
        b. Run ensembling

    Args:
        X (pd.DataFrame): Training data.
        y (pd.Series): Target data.
        problem_type (ProblemType): Problem type associated with training data.
        sampler_name (BaseSampler): Sampler to use for preprocessing.
        tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
        random_seed (int): Seed for the random number generator. Defaults to 0.
        search_parameters (dict or None): Pipeline-level parameters and custom hyperparameter ranges specified for pipelines to iterate over. Hyperparameter ranges
            must be passed in as skopt.space objects. Defaults to None.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines. Defaults to -1.
        text_in_ensembling (boolean): If True and ensembling is True, then n_jobs will be set to 1 to avoid downstream sklearn stacking issues related to nltk. Defaults to False.
        top_n (int): top n number of pipelines to use for long mode.
        num_long_explore_pipelines (int): number of pipelines to explore for each top n pipeline at the start of long mode.
        num_long_pipelines_per_batch (int): number of pipelines per batch for each top n pipeline through long mode.
        allow_long_running_models (bool): Whether or not to allow longer-running models for large multiclass problems. If False and no pipelines, component graphs, or model families are provided,
            AutoMLSearch will not use Elastic Net or XGBoost when there are more than 75 multiclass targets and will not use CatBoost when there are more than 150 multiclass targets. Defaults to False.
        features (list)[FeatureBase]: List of features to run DFS on in AutoML pipelines. Defaults to None. Features will only be computed if the columns used by the feature exist in the input and if the feature has not been computed yet.
        verbose (boolean): Whether or not to display logging information regarding pipeline building. Defaults to False.
        exclude_featurizers (list[str]): A list of featurizer components to exclude from the pipelines built by DefaultAlgorithm.
            Valid options are "DatetimeFeaturizer", "EmailFeaturizer", "URLFeaturizer", "NaturalLanguageFeaturizer", "TimeSeriesFeaturizer"
    """

    def __init__(
        self,
        X,
        y,
        problem_type,
        sampler_name,
        tuner_class=None,
        random_seed=0,
        search_parameters=None,
        n_jobs=1,
        text_in_ensembling=False,
        top_n=3,
        ensembling=False,
        num_long_explore_pipelines=50,
        num_long_pipelines_per_batch=10,
        allow_long_running_models=False,
        features=None,
        verbose=False,
        exclude_featurizers=None,
    ):
        super().__init__(
            allowed_pipelines=[],
            search_parameters=search_parameters,
            tuner_class=None,
            random_seed=random_seed,
        )
        self.X = infer_feature_types(X)
        self.y = infer_feature_types(y)
        self.problem_type = problem_type
        self.sampler_name = sampler_name

        self.n_jobs = n_jobs
        self._best_pipeline_info = {}
        self.text_in_ensembling = text_in_ensembling
        self.search_parameters = search_parameters or {}
        self._top_n_pipelines = None
        self.num_long_explore_pipelines = num_long_explore_pipelines
        self.num_long_pipelines_per_batch = num_long_pipelines_per_batch
        self.top_n = top_n
        self.verbose = verbose
        self._selected_cat_cols = []
        self._split = False
        self.allow_long_running_models = allow_long_running_models
        self._X_with_cat_cols = None
        self._X_without_cat_cols = None
        self.features = features
        self.ensembling = ensembling
        self.exclude_featurizers = exclude_featurizers or []

        # TODO remove on resolution of 3186
        if is_time_series(self.problem_type) and self.ensembling:
            raise ValueError(
                "Ensembling is not available for time series problems in DefaultAlgorithm.",
            )

        if verbose:
            self.logger = get_logger(f"{__name__}.verbose")
        else:
            self.logger = logging.getLogger(__name__)
        if search_parameters and not isinstance(search_parameters, dict):
            raise ValueError(
                f"If search_parameters provided, must be of type dict. Received {type(search_parameters)}",
            )

        self._set_additional_pipeline_params()
        self._separate_hyperparameters_from_parameters()

    @property
    def default_max_batches(self):
        """Returns the number of max batches AutoMLSearch should run by default."""
        return 4 if self.ensembling else 3

    def num_pipelines_per_batch(self, batch_number):
        """Return the number of pipelines in the nth batch.

        Args:
            batch_number (int): which batch to calculate the number of pipelines for.

        Returns:
            int: number of pipelines in the given batch.
        """
        if batch_number == 0 or batch_number == 1:
            return len(self._naive_estimators())
        elif batch_number == 2:
            return len(self._non_naive_estimators())
        if self.ensembling:
            if batch_number % 2 != 0:
                return 1
            elif batch_number == 4:
                return self.num_long_explore_pipelines * self.top_n
        else:
            if batch_number == 3:
                return self.num_long_explore_pipelines * self.top_n
        return self.num_long_pipelines_per_batch * self.top_n

    def _naive_estimators(self):
        if is_regression(self.problem_type):
            naive_estimators = [
                "Elastic Net Regressor",
                "Random Forest Regressor",
            ]
        else:
            naive_estimators = [
                "Logistic Regression Classifier",
                "Random Forest Classifier",
            ]
        estimators = [
            handle_component_class(estimator) for estimator in naive_estimators
        ]
        return estimators

    def _non_naive_estimators(self):
        return [
            est
            for est in get_estimators(self.problem_type)
            if est not in self._naive_estimators()
        ]

    def _init_pipelines_with_starter_params(self, pipelines):
        next_batch = []
        for pipeline in pipelines:
            self._create_tuner(pipeline)
            starting_parameters = self._tuners[pipeline.name].get_starting_parameters(
                self._hyperparameters,
                self.random_seed,
            )
            parameters = self._transform_parameters(pipeline, starting_parameters)
            next_batch.append(
                pipeline.new(parameters=parameters, random_seed=self.random_seed),
            )
        return next_batch

    def _create_naive_pipelines(self, use_features=False):
        feature_selector = None

        if use_features:
            feature_selector = [
                (
                    RFRegressorSelectFromModel
                    if is_regression(self.problem_type)
                    else RFClassifierSelectFromModel
                ),
            ]
        else:
            feature_selector = []

        estimators = self._naive_estimators()
        pipelines = [
            make_pipeline(
                X=self.X,
                y=self.y,
                estimator=estimator,
                problem_type=self.problem_type,
                sampler_name=self.sampler_name,
                extra_components_after=feature_selector,
                parameters=self._pipeline_parameters,
                known_in_advance=self._pipeline_parameters.get("pipeline", {}).get(
                    "known_in_advance",
                    None,
                ),
                features=self.features,
                exclude_featurizers=self.exclude_featurizers,
            )
            for estimator in estimators
        ]
        pipelines = self._add_without_pipelines(pipelines, estimators, feature_selector)
        pipelines = self._init_pipelines_with_starter_params(pipelines)
        return pipelines

    def _add_without_pipelines(self, pipelines, estimators, feature_selector=[]):
        if (
            len(pipelines)
            and "STL Decomposer" in pipelines[-1].component_graph.compute_order
        ):
            without_pipelines = [
                make_pipeline(
                    X=self.X,
                    y=self.y,
                    estimator=estimator,
                    problem_type=self.problem_type,
                    sampler_name=self.sampler_name,
                    extra_components_after=feature_selector,
                    parameters=self._pipeline_parameters,
                    known_in_advance=self._pipeline_parameters.get("pipeline", {}).get(
                        "known_in_advance",
                        None,
                    ),
                    features=self.features,
                    exclude_featurizers=self.exclude_featurizers,
                    include_decomposer=False,
                )
                for estimator in estimators
            ]
            pipelines = pipelines + without_pipelines
        return pipelines

    def _find_component_names(self, original_name, pipeline):
        names = []
        for component in pipeline.component_graph.compute_order:
            split = component.split(" - ")
            split = split[1] if len(split) > 1 else split[0]
            if original_name == split:
                names.append(component)
        return names

    def _create_split_select_parameters(self):
        parameters = {
            "Categorical Pipeline - Select Columns Transformer": {
                "columns": self._selected_cat_cols,
            },
            "Numeric Pipeline - Select Columns By Type Transformer": {
                "column_types": ["Categorical", "EmailAddress", "URL"],
                "exclude": True,
            },
            "Numeric Pipeline - Select Columns Transformer": {
                "columns": self._selected_cols,
            },
        }
        return parameters

    def _create_select_parameters(self):
        parameters = {}
        if self._selected_cols:
            parameters = {
                "Select Columns Transformer": {"columns": self._selected_cols},
            }
        elif self._selected_cat_cols:
            parameters = {
                "Select Columns Transformer": {"columns": self._selected_cat_cols},
            }

        if self._split:
            parameters = self._create_split_select_parameters()
        return parameters

    def _find_component_names_from_parameters(self, old_names, pipelines):
        new_names = {}
        for component_name in old_names:
            for pipeline in pipelines:
                new_name = self._find_component_names(component_name, pipeline)
                if new_name:
                    for name in new_name:
                        if name not in new_names:
                            new_names[name] = old_names[component_name]
        return new_names

    def _rename_pipeline_search_parameters(self, pipelines):
        names_to_value_pipeline_params = self._find_component_names_from_parameters(
            self.search_parameters,
            pipelines,
        )
        self.search_parameters.update(names_to_value_pipeline_params)
        self._separate_hyperparameters_from_parameters()

    def _create_fast_final(self):
        estimators = self._non_naive_estimators()
        estimators = self._filter_estimators(
            estimators,
            self.problem_type,
            self.allow_long_running_models,
            None,
            self.y.nunique(),
            self.logger,
        )
        pipelines = self._make_pipelines_helper(estimators)

        if self._split:
            self._rename_pipeline_search_parameters(pipelines)

        next_batch = self._create_n_pipelines(
            pipelines,
            1,
            create_starting_parameters=True,
        )
        return next_batch

    def _create_n_pipelines(self, pipelines, n, create_starting_parameters=False):
        next_batch = []
        for _ in range(n):
            for pipeline in pipelines:
                if pipeline.name not in self._tuners:
                    self._create_tuner(pipeline)

                select_parameters = self._create_select_parameters()
                parameters = (
                    self._tuners[pipeline.name].get_starting_parameters(
                        self._hyperparameters,
                        self.random_seed,
                    )
                    if create_starting_parameters
                    else self._tuners[pipeline.name].propose()
                )
                parameters = self._transform_parameters(pipeline, parameters)
                parameters.update(select_parameters)
                next_batch.append(
                    pipeline.new(parameters=parameters, random_seed=self.random_seed),
                )
        return next_batch

    def _create_long_exploration(self, n):
        estimators = [
            (pipeline_dict["pipeline"].estimator, pipeline_dict["mean_cv_score"])
            for pipeline_dict in self._best_pipeline_info.values()
        ]
        estimators.sort(key=lambda x: x[1])
        estimators = estimators[:n]
        estimators = [estimator[0].__class__ for estimator in estimators]
        pipelines = self._make_pipelines_helper(estimators)
        self._top_n_pipelines = pipelines
        return self._create_n_pipelines(pipelines, self.num_long_explore_pipelines)

    def _make_pipelines_helper(self, estimators):
        pipelines = []
        if is_time_series(self.problem_type):
            pipelines = [
                make_pipeline(
                    X=self.X,
                    y=self.y,
                    estimator=estimator,
                    problem_type=self.problem_type,
                    sampler_name=self.sampler_name,
                    parameters=self._pipeline_parameters,
                    known_in_advance=self.search_parameters.get("pipeline", {}).get(
                        "known_in_advance",
                        None,
                    ),
                    features=self.features,
                    exclude_featurizers=self.exclude_featurizers,
                )
                for estimator in estimators
            ]
            pipelines = self._add_without_pipelines(pipelines, estimators)
        else:
            pipelines = [
                self._make_split_pipeline(estimator) for estimator in estimators
            ]
        return pipelines

    def next_batch(self):
        """Get the next batch of pipelines to evaluate.

        Returns:
            list(PipelineBase): a list of instances of PipelineBase subclasses, ready to be trained and evaluated.
        """
        if self.ensembling:
            if self._batch_number == 0:
                next_batch = self._create_naive_pipelines()
            elif self._batch_number == 1:
                next_batch = self._create_naive_pipelines(use_features=True)
            elif self._batch_number == 2:
                next_batch = self._create_fast_final()
            elif self.batch_number == 3:
                next_batch = self._create_ensemble(
                    self._pipeline_parameters.get("Label Encoder", {}),
                )
            elif self.batch_number == 4:
                next_batch = self._create_long_exploration(n=self.top_n)
            elif self.batch_number % 2 != 0:
                next_batch = self._create_ensemble(
                    self._pipeline_parameters.get("Label Encoder", {}),
                )
            else:
                next_batch = self._create_n_pipelines(
                    self._top_n_pipelines,
                    self.num_long_pipelines_per_batch,
                )
        else:
            if self._batch_number == 0:
                next_batch = self._create_naive_pipelines()
            elif self._batch_number == 1:
                next_batch = self._create_naive_pipelines(use_features=True)
            elif self._batch_number == 2:
                next_batch = self._create_fast_final()
            elif self.batch_number == 3:
                next_batch = self._create_long_exploration(n=self.top_n)
            else:
                next_batch = self._create_n_pipelines(
                    self._top_n_pipelines,
                    self.num_long_pipelines_per_batch,
                )

        self._pipeline_number += len(next_batch)
        self._batch_number += 1
        return next_batch

    def _get_feature_provenance_and_remove_engineered_features(
        self,
        pipeline,
        component_name,
        to_be_removed,
        to_be_added,
    ):
        component = pipeline.get_component(component_name)
        feature_provenance = component._get_feature_provenance()
        for original_col in feature_provenance:
            selected = False
            for encoded_col in feature_provenance[original_col]:
                if encoded_col in to_be_removed:
                    selected = True
                    to_be_removed.remove(encoded_col)
            if selected:
                to_be_added.append(original_col)

    def _parse_selected_categorical_features(self, pipeline):
        if list(self.X.ww.select("categorical", return_schema=True).columns):
            self._get_feature_provenance_and_remove_engineered_features(
                pipeline,
                OneHotEncoder.name,
                self._selected_cols,
                self._selected_cat_cols,
            )
        if (
            list(self.X.ww.select("URL", return_schema=True).columns)
            and "URLFeaturizer" not in self.exclude_featurizers
        ):
            self._get_feature_provenance_and_remove_engineered_features(
                pipeline,
                URLFeaturizer.name,
                self._selected_cat_cols,
                self._selected_cat_cols,
            )
        if (
            list(self.X.ww.select("EmailAddress", return_schema=True).columns)
            and "EmailFeaturizer" not in self.exclude_featurizers
        ):
            self._get_feature_provenance_and_remove_engineered_features(
                pipeline,
                EmailFeaturizer.name,
                self._selected_cat_cols,
                self._selected_cat_cols,
            )

    def add_result(
        self,
        score_to_minimize,
        pipeline,
        trained_pipeline_results,
        cached_data=None,
    ):
        """Register results from evaluating a pipeline. In batch number 2, the selected column names from the feature selector are taken to be used in a column selector. Information regarding the best pipeline is updated here as well.

        Args:
            score_to_minimize (float): The score obtained by this pipeline on the primary objective, converted so that lower values indicate better pipelines.
            pipeline (PipelineBase): The trained pipeline object which was used to compute the score.
            trained_pipeline_results (dict): Results from training a pipeline.
            cached_data (dict): A dictionary of cached data, where the keys are the model family. Expected to be of format
                {model_family: {hash1: trained_component_graph, hash2: trained_component_graph...}...}.
                Defaults to None.
        """
        cached_data = cached_data or {}
        if pipeline.model_family != ModelFamily.ENSEMBLE:
            if self.batch_number >= 3:
                super().add_result(
                    score_to_minimize,
                    pipeline,
                    trained_pipeline_results,
                )

        if (
            self.batch_number == 2
            and self._selected_cols is None
            and not is_time_series(self.problem_type)
        ):
            if is_regression(self.problem_type):
                self._selected_cols = pipeline.get_component(
                    "RF Regressor Select From Model",
                ).get_names()
            else:
                self._selected_cols = pipeline.get_component(
                    "RF Classifier Select From Model",
                ).get_names()

            self._parse_selected_categorical_features(pipeline)

        current_best_score = self._best_pipeline_info.get(
            pipeline.model_family,
            {},
        ).get("mean_cv_score", np.inf)
        if (
            score_to_minimize is not None
            and score_to_minimize < current_best_score
            and pipeline.model_family != ModelFamily.ENSEMBLE
        ):
            self._best_pipeline_info.update(
                {
                    pipeline.model_family: {
                        "mean_cv_score": score_to_minimize,
                        "pipeline": pipeline,
                        "parameters": pipeline.parameters,
                        "id": trained_pipeline_results["id"],
                        "cached_data": cached_data,
                    },
                },
            )

    def _make_split_pipeline(self, estimator, pipeline_name=None):
        if self._X_with_cat_cols is None or self._X_without_cat_cols is None:
            self._X_without_cat_cols = self.X.ww.drop(self._selected_cat_cols)
            self._X_with_cat_cols = self.X.ww[self._selected_cat_cols]

        if self._selected_cat_cols and self._selected_cols:
            self._split = True

            categorical_pipeline_parameters = {
                "Select Columns Transformer": {"columns": self._selected_cat_cols},
            }
            numeric_pipeline_parameters = {
                "Select Columns Transformer": {"columns": self._selected_cols},
                "Select Columns By Type Transformer": {
                    "column_types": ["Categorical", "EmailAddress", "URL"],
                    "exclude": True,
                },
            }

            categorical_pipeline = make_pipeline(
                self._X_with_cat_cols,
                self.y,
                estimator,
                self.problem_type,
                sampler_name=None,
                parameters=categorical_pipeline_parameters,
                extra_components_before=[SelectColumns],
                use_estimator=False,
                exclude_featurizers=self.exclude_featurizers,
            )

            numeric_pipeline = make_pipeline(
                self._X_without_cat_cols,
                self.y,
                estimator,
                self.problem_type,
                sampler_name=None,
                parameters=numeric_pipeline_parameters,
                extra_components_before=[SelectByType],
                extra_components_after=[SelectColumns],
                use_estimator=False,
                exclude_featurizers=self.exclude_featurizers,
            )
            pre_pipeline_components = (
                {"DFS Transformer": ["DFS Transformer", "X", "y"]}
                if self.features
                else {}
            )
            if self.sampler_name:
                sampler = _get_sampler(
                    self.X,
                    self.y,
                    self.problem_type,
                    estimator,
                    self.sampler_name,
                )[0]
                post_pipelines_components = {sampler.name: [sampler.name, "X", "y"]}
            else:
                post_pipelines_components = None

            input_pipelines = [numeric_pipeline, categorical_pipeline]
            sub_pipeline_names = {
                numeric_pipeline.name: "Numeric",
                categorical_pipeline.name: "Categorical",
            }
            return _make_pipeline_from_multiple_graphs(
                input_pipelines,
                estimator,
                self.problem_type,
                pipeline_name=pipeline_name,
                random_seed=self.random_seed,
                sub_pipeline_names=sub_pipeline_names,
                pre_pipeline_components=pre_pipeline_components,
                post_pipelines_components=post_pipelines_components,
            )
        elif self._selected_cat_cols and not self._selected_cols:
            categorical_pipeline_parameters = {
                "Select Columns Transformer": {"columns": self._selected_cat_cols},
            }
            categorical_pipeline = make_pipeline(
                self._X_with_cat_cols,
                self.y,
                estimator,
                self.problem_type,
                sampler_name=self.sampler_name,
                parameters=categorical_pipeline_parameters,
                extra_components_before=[SelectColumns],
                features=self.features,
                exclude_featurizers=self.exclude_featurizers,
            )
            return categorical_pipeline
        else:
            numeric_pipeline_parameters = {
                "Select Columns Transformer": {"columns": self._selected_cols},
            }
            numeric_pipeline = make_pipeline(
                self.X,
                self.y,
                estimator,
                self.problem_type,
                sampler_name=self.sampler_name,
                parameters=numeric_pipeline_parameters,
                extra_components_after=[SelectColumns],
                features=self.features,
                exclude_featurizers=self.exclude_featurizers,
            )
            return numeric_pipeline
