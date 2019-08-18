from skopt import Optimizer


class SKOptTuner:
    def __init__(self, space, random_state=0):
        self.opt = Optimizer(space, "ET", acq_optimizer="sampling", random_state=random_state)

    def add(self, parameters, score):
        return self.opt.tell(list(parameters), score)

    def propose(self):
        return self.opt.ask()
