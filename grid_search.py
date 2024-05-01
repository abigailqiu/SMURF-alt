from skopt import gp_minimize
from skopt.space import Real
import json
from smurf.eval import preprocess, SMURF_eval
from scipy.stats import spearmanr, rankdata
from compute_correlation import comp_corl

ref_file = "noun/ref.json"
ref_list = json.loads(open(ref_file, "r").read())
ref_dict = {cap["image_id"]: [] for cap in ref_list}
for i, cap in enumerate(ref_list):
    ref_dict[cap["image_id"]].append(preprocess(cap["caption"]))

checked = []


def evaluate_metric(thresholds, alpha=0.9):
    D, G = thresholds
    cand = "noun/candSCORES.json"
    rcap_noun = "noun/rand_cap_nounSCORES.json"
    rcoco_noun = "noun/rand_coco_nounSCORES.json"
    # rcoco_word = "noun/rand_coco_wordSCORES.json"
    rnoun = "noun/rand_nounSCORES.json"
    rword = "noun/rand_wordSCORES.json"
    c_files = [
        cand,
        rcap_noun,
        rcoco_noun,
        # rcoco_word,
        rnoun,
        rword,
    ]
    expected = [1, 2, 3, 4, 5]  # expected rankings

    smurf_scores = []

    for cfile in c_files:
        with open(cfile, "r") as f:
            metric_scores = json.load(f)
        score = SMURF_eval(D, G, metric_scores, single=True)
        smurf_scores.append(score)
    ranks = rankdata(smurf_scores, method="ordinal")
    ranks = len(ranks) + 1 - ranks
    corr_coefficient, p_value = spearmanr(ranks, expected)
    spearmans = [corr_coefficient, p_value]
    print(corr_coefficient)

    with open("corl/cand_cocoSCORES.json", "r") as f:
        corl_scores = json.load(f)
    bigscores = SMURF_eval(D, G, corl_scores, single=False)
    human_correlation = comp_corl(bigscores)
    # print(human_correlation)
    weighted_sum = alpha * corr_coefficient + (1 - alpha) * human_correlation
    checked.append([D, G, human_correlation, corr_coefficient])
    return -weighted_sum


space = [Real(-3, 0, name="D"), Real(-3, 0, name="G")]

result = gp_minimize(evaluate_metric, space, n_calls=100, random_state=42)

best_thresholds = result.x
best_metric_value = -result.fun

print("Best Thresholds:", best_thresholds)
print("Best Metric Value:", best_metric_value)


# for i in range(20):
#     evaluate_metric([-6, (i + 20) * -0.15])

# with open("search/checked.json", "w") as f:
#     json.dump(checked, f)

# print(evaluate_metric([-1.96, -1.96]))
