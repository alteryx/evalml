import cProfile, pstats
import time
import pandas as pd
from evalml import AutoMLSearch


def kdd():
    X = pd.read_csv("/Users/parthiv.naresh/Documents/Datasets/Regression/regress.csv")
    y = X.pop("y")
    beginning = time.time()
    aml = AutoMLSearch(X_train=X, y_train=y, problem_type="regression", max_iterations=2)
    beginning_0 = time.time()
    aml.search()
    end = time.time()
    print(f"Total: {beginning_0-beginning}")
    print(f"Total: {end-beginning_0}")
    print(aml.results)


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    kdd()
    profiler.disable()
    stats = pstats.Stats(profiler)
    sorted_stats = stats.sort_stats('cumtime')
    sorted_stats.print_stats(2000)
    stats.dump_stats('/Users/parthiv.naresh/0_16_4_stat.txt')