from .eval_algorithms import compute_semantic
from .eval_algorithms import compute_quality
import numpy as np
import pandas as pd
import time


def preprocess(line):
    return " ".join(
        [w for w in line.lower().rstrip().replace(" .", "").replace(".", "").split(" ")]
    )


class smurf_eval_captions:
    def __init__(self, gts, res, fuse=False):
        self.gts = gts
        self.res = res
        self.fuse = fuse

    # broke up evaluation into SPURTS, MIMA, SPARCS then SMURF
    def evaluate(self):
        # Set up scorers
        print("setting up scorers...")
        scorers = [
            (compute_semantic(), "SPARCS"),
            (compute_quality(use_roberta=0, distinctness=False), "MIMA"),
            (compute_quality(use_roberta=1, distinctness=True), "SPURTS"),
        ]

        # Compute scores
        metric_scores = {}
        for scorer, method in scorers:
            metric_scores[method] = []
            print("computing %s score..." % (scorer.method()))
            t = time.time()
            metric_scores[method].extend(scorer.compute_score(self.res, self.gts))
            print(
                "Mean %s score: %0.3f. Computed in %0.2f seconds."
                % (method, float(np.mean(metric_scores[method])), time.time() - t)
            )
        return metric_scores


# only getting SMURF score
def SMURF_eval(T_d, T_g, metric_scores, single):
    print("computing SMURF score...")
    t = time.time()
    estimates = pd.read_csv("smurf/standardize_estimates.txt", header=None)
    sem_ind = list(estimates[0]).index("SPARCS")
    qual_ind = list(estimates[0]).index("SPURTS")
    gram_ind = list(estimates[0]).index("MIMA")
    stand_SPARCS = (
        metric_scores["SPARCS"] - estimates.loc[sem_ind, 1]
    ) / estimates.loc[sem_ind, 2]
    stand_SPURTS = (
        metric_scores["SPURTS"] - estimates.loc[qual_ind, 1]
    ) / estimates.loc[qual_ind, 2]
    stand_MIMA = (metric_scores["MIMA"] - estimates.loc[gram_ind, 1]) / estimates.loc[
        gram_ind, 2
    ]
    detail_reward = stand_SPURTS - T_d
    detail_reward[detail_reward < 0] = 0
    gram_penalty = stand_MIMA - T_g
    gram_penalty[gram_penalty > 0] = 0
    mask = np.zeros((len(stand_SPARCS), 3))
    mask[stand_SPARCS <= T_d, :] = np.asarray([1, 0, 1])
    mask[stand_SPARCS >= T_d, :] = np.asarray([1, 1, 1])
    metric_scores["SMURF"] = [
        np.average(
            [stand_SPARCS[i], detail_reward[i], gram_penalty[i]],
            weights=mask[i, :],
        )
        for i in range(0, len(stand_SPARCS))
    ]
    print(
        "Mean SMURF score: %0.3f. Computed in %0.2f seconds."
        % (float(np.mean(metric_scores["SMURF"])), time.time() - t)
    )
    # single = whether return single score or list of scores
    if not single:
        return metric_scores
    else:
        return float(np.mean(metric_scores["SMURF"]))
