import time

import tqdm


class ProgressMonitor:
    """Logs progress of automl search.

    Arguments:
        max_pipelines (int, None): If max_pipelines is set, the number of pipelines evaluated out of the total
            number of pipelines will be logged.
        logger (logging.Logger): Logger.
    """

    def __init__(self, max_pipelines, logger):
        self.logger = logger
        self.current_iteration = None
        self.max_pipelines = max_pipelines
        self.start_time = time.time()
        if max_pipelines:
            self.current_iteration = 1
            self.output_format = "({current_iteration}/{max_pipelines}) {pipeline_name} Elapsed:{time_elapsed}"
        else:
            self.output_format = "{pipeline_name} Elapsed: {time_elapsed}"

    @property
    def time_elapsed(self):
        """How much time has elapsed since the search started."""
        return tqdm.std.tqdm.format_interval(time.time() - self.start_time)

    def update(self, pipeline_name):
        """Adds the next pipeline to be evaluated to the log along with how much time has elapsed.

        Arguments:
            pipeline_name (str): Name of next pipeline to be evaluated.
        """
        if self.max_pipelines:
            self.logger.info(self.output_format.format(current_iteration=self.current_iteration,
                                                       max_pipelines=self.max_pipelines,
                                                       pipeline_name=pipeline_name,
                                                       time_elapsed=self.time_elapsed))
            self.current_iteration += 1
        else:
            self.logger.info(self.output_format.format(pipeline_name=pipeline_name,
                                                       time_elapsed=self.time_elapsed))
