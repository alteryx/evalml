class PipelineBase:
    def __init__(self):
        pass

    def fit(self, X, y, metric=None):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        self.pipeline.predict(X)

    def predict_proba(self, X):
        self.pipeline.predict_proba(X)

    def score(self, y, X):
        return .5
        # self.pipeline.predict_proba(X)
