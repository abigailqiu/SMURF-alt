import json
from smurf.eval import preprocess, smurf_eval_captions, SMURF_eval

cand_file = "noun/cand.json"
rwd_file = "noun/rwd2.json"
c_files = [cand_file, rwd_file]  # candidates
ref_file = "noun/ref.json"

ref_list = json.loads(open(ref_file, "r").read())
ref_dict = {cap["image_id"]: [] for cap in ref_list}
for i, cap in enumerate(ref_list):
    ref_dict[cap["image_id"]].append(preprocess(cap["caption"]))

smurf_scores = []
for cfile in c_files:
    cand_list = json.loads(open(cfile, "r").read())
    cands = [preprocess(cap["caption"]) for cap in cand_list]
    refs = [ref_dict[cap["image_id"]] for cap in cand_list]
    meta_scorer = smurf_eval_captions(refs, cands, fuse=False)
    score_list = meta_scorer.evaluate()
    scores = SMURF_eval(-1.96, -1.96, score_list, single=False)  # list of scores

    file_name = cfile.replace(".json", "SCORES.json")
    with open(file_name, "w") as f:
        json.dump(scores, f)
