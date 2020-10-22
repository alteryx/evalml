from evalml.pipelines import PipelineBase


class EngineResult:
    def __init__(self):
        self.early_stop = False
        self.completed_pipelines = []
        self.pipeline_results = []
        self.unprocessed_pipelines = []

    def set_early_stop(self, current_pipeline, unprocessed_pipelines):
        self.early_stop = True
        self.unprocessed_pipelines.append(current_pipeline)
        self.unprocessed_pipelines = self.unprocessed_pipelines + unprocessed_pipelines

    def add_result(self, completed_pipelines, pipeline_results):
        if isinstance(completed_pipelines, PipelineBase) and isinstance(pipeline_results, dict):
            self.completed_pipelines.append(completed_pipelines)
            self.pipeline_results.append(pipeline_results)
        else:
            if not isinstance(completed_pipelines, PipelineBase) and (not isinstance(completed_pipelines, list) or not all(isinstance(pl, PipelineBase) for pl in completed_pipelines)):
                raise ValueError(f"`completed_pipelines` must be PipelineBase or list(PipelineBase). Recieved {type(completed_pipelines)}.")
            if not isinstance(pipeline_results, dict) and (not isinstance(pipeline_results, list) or not all(isinstance(res, dict) for res in pipeline_results)):
                raise ValueError(f"`pipeline_results` must be dict or list(dict). Recieved {(pipeline_results)}.")
            self.completed_pipelines = self.completed_pipelines + completed_pipelines
            self.pipeline_results = self.pipeline_results + pipeline_results
