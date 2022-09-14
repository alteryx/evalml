"""Ensemble pipeline mix-in class."""


class EnsemblePipelineMixin:
    @property
    def _all_input_pipelines_fitted(self):
        for pipeline in self.input_pipelines:
            if not pipeline._is_fitted:
                return False
        return True

    def _fit_input_pipelines(self, X, y, force_retrain=False):
        fitted_pipelines = []
        for pipeline in self.input_pipelines:
            if pipeline._is_fitted and not force_retrain:
                fitted_pipelines.append(pipeline)
            else:
                if force_retrain:
                    new_pl = pipeline.clone()
                else:
                    new_pl = pipeline
                fitted_pipelines.append(new_pl.fit(X, y))
        self.input_pipelines = fitted_pipelines
