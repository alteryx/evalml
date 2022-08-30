import cProfile, pstats
import time
import pandas as pd
from evalml import AutoMLSearch
from evalml.utils import infer_feature_types


def kdd():
    X = pd.read_csv("/Users/parthiv.naresh/Documents/Datasets/Classification/Binary/KDDCup09_churn.csv")
    y = X.pop("CHURN")
    beginning = time.time()
    aml = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=2)
    beginning_0 = time.time()
    aml.search()
    end = time.time()
    from pprint import pprint
    pprint(aml.results)
    print(f"Total fit time: {beginning_0-beginning}")
    print(f"Total search time: {end-beginning_0}")


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    kdd()
    profiler.disable()
    stats = pstats.Stats(profiler)
    sorted_stats = stats.sort_stats('cumtime')
    sorted_stats.print_stats(800)
    stats.dump_stats('/Users/parthiv.naresh/0_16_4_stat.txt')