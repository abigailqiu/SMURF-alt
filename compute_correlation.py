import numpy as np
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import rankdata
import pandas as pd
import pylab
from pycocotools.coco import COCO
import json
from ast import literal_eval
import re

from scipy.stats import mode


# x1 = [12, 2, 1, 12, 2]
# x2 = [1, 4, 7, 1, 0]
# tau, p_value = kendalltau(x1, x2)
# print(tau)
def human_compare(data1, data2):
    corr_set = []
    data1 = pd.DataFrame(
        [entry for i, entry in data1.iterrows() if entry[0] != entry[1][:-2]]
    )
    data2 = pd.DataFrame(
        [entry for i, entry in data2.iterrows() if entry[0] != entry[1][:-2]]
    )
    full_data_human = data1.merge(data2, on=["image_id", "cap_id"])
    num_nan = 0
    for caption_set in full_data_human.groupby("image_id"):
        experts = rankdata(np.mean(caption_set[1].iloc[:, 2:-3], axis=1))
        crowd = rankdata(caption_set[1].iloc[:, 5])
        tau, p_value = kendalltau(list(experts), list(crowd))
        if not np.isnan(tau):
            corr_set.append(tau)
        else:
            num_nan += 1
    print(np.mean(corr_set))
    print(num_nan)


def write_dict(data_judge, data_tok, ref_out_file, cand_out_file, dataset):
    if dataset == 0:
        pylab.rcParams["figure.figsize"] = (8.0, 10.0)
        dataDir = "./data/COCO"
        dataType = "val2014"
        # annFile = '{}/instances_{}.json'.format(dataDir, dataType)
        # annFile = "{}/captions_{}.json".format(dataDir, dataType)
        annFile = "annotations/captions_val2014.json"
        coco_caps = COCO(annFile)
        all_ann = json.load(open(annFile, "r"))
        id_dict = {entry["file_name"]: entry["id"] for entry in all_ann["images"]}
        ref_dict = []
        good_ids = json.load(open("adj/good_ids.json", "r"))
        for img in data_tok["Human"].keys():
            if "val" in img:
                s = img.replace("COCO_val2014_", "")
                s = s.replace(".jpg", "")
                regex = "^0+(?!$)"
                s = re.sub(regex, "", s)
                if int(s) in good_ids:
                    imgId = id_dict[img]
                    annId = coco_caps.getAnnIds(imgIds=imgId)
                    anns = coco_caps.loadAnns(annId)
                    for ann in anns:
                        if not ann["caption"].lower() == data_tok["Human"][img].lower():
                            ref_dict += [{"image_id": img, "caption": ann["caption"]}]

        json.dump(ref_dict, ref_out_file)
        ref_out_file.close()
        cand_dict = []
        frames = data_judge.loc[data_judge["participant"].isin(data_tok.keys())]
        for img in sorted(data_tok["Human"].keys()):
            if "val" in img:
                s = img.replace("COCO_val2014_", "")
                s = s.replace(".jpg", "")
                regex = "^0+(?!$)"
                s = re.sub(regex, "", s)
                if int(s) in good_ids:
                    participant_caps = [
                        {"image_id": img, "caption": data_tok[participant][img]}
                        for participant in frames["participant"]
                    ]
                    cand_dict += participant_caps
                # if len(cand_dict)>10000:
                #  break
        json.dump(cand_dict, cand_out_file)
        cand_out_file.close()
        return frames
        # cand_dict = OrderedDict()
        # for
        # for partipant in :
        #     gts[id['image_id']] = []
        # for l in ref_list:
        #     gts[l['image_id']].append({"caption": l['caption']})
        # catIds = coco.getCatIds(catNms=['person'])
        # imgIds = coco.getImgIds(catIds=catIds)
        # annIds = [coco_caps.getAnnIds(imgIds=imgId) for imgId in imgIds]
        # captions = []
        # for annIdSet in annIds:
        #    annSet = coco_caps.loadAnns(annIdSet)
        #    captions.append([entry['caption'] for entry in annSet])
        # return captions

    elif dataset == 1:
        # remove captions sourced from target image (from SPICE method)
        data_judge = pd.DataFrame(
            [entry for i, entry in data_judge.iterrows() if entry[0] != entry[1][:-2]]
        )

        no_tie_frames = pd.DataFrame(columns=["image_id", "cap_id", "score"])
        no_tie = 0
        for caption_set in data_judge.groupby("image_id"):
            experts = np.mean(caption_set[1].iloc[:, 2:], axis=1)
            # experts = pd.Series(np.median(caption_set[1].iloc[:, 2:], axis=1))
            # experts = caption_set[1].iloc[:, 2]
            if len(np.unique(experts)) == len(experts):
                new_data = pd.concat([caption_set[1].iloc[:, :2], experts], axis=1)
                new_data.columns = ["image_id", "cap_id", "score"]
                no_tie_frames = no_tie_frames.append(new_data, ignore_index=True)
                no_tie += 1
        # print(no_tie)

        # data_tok.iloc[:, 1] = ['hi ' + sent for sent in data_tok.iloc[:, 1]]
        # data_cand = data_judge.merge(data_tok, how='left', on='cap_id')
        data_cand = no_tie_frames.merge(data_tok, how="left", on="cap_id")
        ref_tok = data_tok.copy()
        ref_tok.columns = ["image_id", "caption"]
        ref_tok["image_id"] = [entry[:-2] for entry in ref_tok["image_id"]]

        ref_dict = ref_tok.to_dict("records")
        json.dump(ref_dict, ref_out_file)
        ref_out_file.close()

        cand_dict = data_cand[["image_id", "caption"]].to_dict("records")
        json.dump(cand_dict, cand_out_file)
        cand_out_file.close()
        return no_tie_frames  # data_judge#no_tie_frames

    elif dataset == 2:
        # Flickr
        ref_tok = data_tok.copy()
        ref_tok[2] = ref_tok[2].iloc[:, [0, 2]]
        ref_tok[3] = ref_tok[3].iloc[:, [0, 2]]
        ref_tok[0]["cap_id"] = [entry[:-2] for entry in ref_tok[0]["cap_id"]]
        # ref_tok[1]['cap_id'] = [entry[:-2] for entry in ref_tok[1]['cap_id']]
        ref_tok[0].columns = ref_tok[2].columns
        ref_tok[1].columns = ref_tok[2].columns
        for i in [0, 3]:
            ref_tok[i] = ref_tok[i][
                ref_tok[i]["image_id"].isin(data_judge[i].iloc[:, 0])
            ]

        # COCO
        pylab.rcParams["figure.figsize"] = (8.0, 10.0)
        dataDir = "./data/COCO"
        dataType = "val2014"
        annFile = "{}/instances_{}.json".format(dataDir, dataType)
        annFile = "{}/captions_{}.json".format(dataDir, dataType)
        coco_caps = COCO(annFile)
        all_ann = json.load(open(annFile, "r"))
        id_dict = {entry["file_name"]: entry["id"] for entry in all_ann["images"]}
        coco_frames = pd.DataFrame(columns=ref_tok[1].columns)
        for img in set(list(data_judge[4].iloc[:, 0]) + list(data_judge[5].iloc[:, 0])):
            imgId = id_dict[img]
            annId = coco_caps.getAnnIds(imgIds=imgId)
            anns = coco_caps.loadAnns(annId)
            new_data = pd.DataFrame([[img, ann["caption"]] for ann in anns])
            new_data.columns = ref_tok[1].columns
            coco_frames = coco_frames.append(new_data, ignore_index=True)

        ref_tok = pd.concat(ref_tok + [coco_frames], ignore_index=True)
        ref_dict = ref_tok.to_dict("records")
        ref_dict = [i for i in ref_dict if type(i["caption"]) is not float]
        json.dump(ref_dict, ref_out_file)
        ref_out_file.close()

        data_cand = []
        frames = []
        for df in data_judge:
            cap_names = [name for name in df.columns if "Input." in name]
            ans_names = [name for name in df.columns if "Answer." in name]
            cand_captions = pd.concat(
                [pd.DataFrame([row]) for row in df[cap_names].stack()]
            ).reset_index(drop=True)
            cand_ratings = pd.concat(
                [pd.DataFrame([row]) for row in df[ans_names].stack()]
            ).reset_index(drop=True)
            img_names = df["image_id"].repeat(len(cap_names)).reset_index(drop=True)
            data_cand.append(
                pd.concat([img_names, cand_captions], axis=1, ignore_index=True)
            )
            frames.append(
                pd.concat([img_names, cand_ratings], axis=1, ignore_index=True)
            )
        data_cand = pd.concat(data_cand, axis=0)
        data_cand.columns = ["image_id", "caption"]
        frames = pd.concat(frames, axis=0)
        frames.columns = data_cand.columns
        # drop NaN judgements and corresponding captions
        rows = [j for j, i in enumerate(frames.iloc[:, 1]) if np.isnan(i) == True]
        frames.drop(frames.index[rows])
        data_cand.drop(data_cand.index[rows])
        cand_dict = data_cand.to_dict("records")
        json.dump(cand_dict, cand_out_file)
        cand_out_file.close()
        return frames

    else:
        print("hi")


def read_result(scores, dict_entry, reference_frames, dataset):
    if dataset == 0:
        # exp1_result = json.load(result_file)
        # result_file.close()
        final_res = []
        for metric_name in dict_entry:
            res = scores[metric_name]
            # res = [np.mean(participant) for participant in np.array_split(res,len(reference_frames))]
            res = [
                np.mean(res[i :: len(reference_frames)])
                for i in range(0, len(reference_frames))
            ]
            full_res = pd.concat(
                [reference_frames, pd.DataFrame(res, columns=["metric_score"])], axis=1
            )
            tau, p_value = pearsonr(
                list(full_res["M2"]), list(full_res["metric_score"])
            )
            print(metric_name + ": " + str(tau))
            # np.argmin([pearsonr(m1_seg, m2_seg)[0] for m1_seg, m2_seg in
            # zip(np.array_split(met1, len(met1) // 4), np.array_split(met2, len(met2) // 4))])
            # met1 = exp1_result['meta_sem']
            # met2 = exp1_result['spice']
            # np.nanargmax([pearsonr(m2_seg, reference_frames['M2'])[0] - pearsonr(m1_seg, reference_frames['M2'])[0] for
            #               m1_seg, m2_seg in
            #               zip(np.array_split(met1, len(met1) // 4), np.array_split(met2, len(met2) // 4))])
            final_res.append({metric_name: p_value})
        # return sorted(final_res, key=lambda k: list(k.values())[0], reverse=True)
        return tau
    # elif dataset == 1:
    #     exp1_result = json.load(result_file)
    #     result_file.close()
    #     final_res = []
    #     for metric_name in dict_entry:
    #         res = exp1_result[metric_name]
    #         full_res = pd.concat(
    #             [reference_frames, pd.DataFrame(res, columns=["metric_score"])], axis=1
    #         )
    #         num_nan = 0
    #         corr_set = []
    #         worst = None
    #         for res_set in full_res.groupby("image_id"):
    #             experts = rankdata(res_set[1].iloc[:, -2], method="min")
    #             metric = rankdata(res_set[1].iloc[:, -1], method="min")
    #             tau, p_value = kendalltau(list(experts), list(metric))
    #             if worst is None:
    #                 worst = tau
    #                 worst_set = res_set[0]
    #             elif tau < worst:
    #                 worst = tau
    #                 worst_set = res_set[0]
    #             if not np.isnan(tau):
    #                 corr_set.append(tau)
    #             else:
    #                 corr_set.append(0)
    #                 # num_nan += 1
    #         print(metric_name + ": " + str(np.mean(corr_set)))
    #         print(num_nan)
    #         print(worst_set)
    #         final_res.append({metric_name: np.mean(corr_set)})
    #     return sorted(final_res, key=lambda k: list(k.values())[0], reverse=True)
    # elif dataset == 2:
    #     exp1_result = json.load(result_file)
    #     result_file.close()
    #     final_res = []
    #     for metric_name in dict_entry:
    #         res = exp1_result[metric_name]
    #         full_res = pd.concat(
    #             [reference_frames, pd.DataFrame(res, columns=["metric_score"])], axis=1
    #         )
    #         num_nan = 0
    #         corr_set = []
    #         worst = None
    #         for res_set in full_res.groupby("image_id"):
    #             experts = rankdata(res_set[1].iloc[:, -2], method="min")
    #             metric = rankdata(res_set[1].iloc[:, -1], method="min")
    #             tau, p_value = kendalltau(list(experts), list(metric))
    #             if worst is None:
    #                 worst = tau
    #                 worst_set = res_set[0]
    #             elif tau < worst:
    #                 worst = tau
    #                 worst_set = res_set[0]
    #             if not np.isnan(tau):
    #                 corr_set.append(tau)
    #             else:
    #                 corr_set.append(0)
    #                 num_nan += 1
    #         print(metric_name + ": " + str(np.mean(corr_set)))
    #         print(num_nan)
    #         print(worst_set)
    #         final_res.append({metric_name: np.mean(corr_set)})
    #     return sorted(final_res, key=lambda k: list(k.values())[0], reverse=True)
    # else:
    #     exp1_result = json.load(result_file[0])
    #     result_file[0].close()
    #     exp2_result = json.load(result_file[1])
    #     result_file[1].close()
    #     final_res = []
    #     for metric_name in dict_entry:
    #         res1 = exp1_result[metric_name]
    #         res2 = exp2_result[metric_name]
    #         res = np.greater_equal(res2, res1)
    #         gt = reference_frames[0] > 0
    #         accuracy = []
    #         accuracy.append(sum(np.equal(res, gt)) / len(gt))
    #         for i in range(0, 4):
    #             accuracy.append(
    #                 sum(np.equal(res, gt)[reference_frames[1] == i]) / (len(gt) // 4)
    #             )
    #         print(metric_name + ": " + str(accuracy))
    #         final_res.append({metric_name: accuracy})
    #     return sorted(final_res, key=lambda k: list(k.values())[0], reverse=True)


dataset = 0  # 0 COCO, 1 Flickr8k, 2 Composite, 3 Pascal-50s
mode = 2  # 0: human, 1: write, 2: read
dir = "corl/"
if dataset == 0:
    participants = ["Human", "Google", "Montreal", "NeuralTalk"]
    data1 = pd.read_csv(dir + "leaderboard.csv")
    data1.columns = ["participant"] + list(data1.columns[1:])
    data_tok = {}
    for participant in participants:
        caps = json.load(open(dir + participant + "_submission.json", "r"))
        if participant == "Montreal":
            participant += "/Toronto"
        data_tok.update({participant: caps})
        # data1.columns = ["img", "cap_id", "score1", "score2", "score3"]
        # data1.columns = ["img", "cap_id", "score"]
        # data_tok.columns = ["cap_id", "caption"]
elif dataset == 1:
    # data1 = pd.read_csv(dir + 'test_cand.txt', sep='\t', header=None)
    data1 = pd.read_csv(dir + "ExpertAnnotations.txt", sep="\t", header=None)
    data2 = pd.read_csv(dir + "CrowdFlowerAnnotations.txt", sep="\t", header=None)
    # data_tok = pd.read_csv(dir + 'test_ref.txt', sep='\t', header=None)
    data_tok = pd.read_csv(dir + "Flickr8k.token.txt", sep="\t", header=None)

    data1.columns = ["image_id", "cap_id", "score1", "score2", "score3"]
    # data1.columns = ["img", "cap_id", "score"]
    data2.columns = ["image_id", "cap_id", "score", "num_yes", "num_no"]
    data_tok.columns = ["cap_id", "caption"]
elif dataset == 2:
    sub_dir = "composite/"
    sets = [
        "8k_correctness.csv",
        "8k_throughness.csv",
        "30k_correctness.csv",
        "30k_throughness.csv",
        "coco_correctness.csv",
        "coco_throughness.csv",
    ]
    data1 = []
    for sing_set in sets:
        data_in = pd.read_csv(dir + sub_dir + sing_set, sep=";")
        rel_ind = [
            index
            for index, name in enumerate(data_in.columns)
            if "Input." in name or "Answer." in name
        ]
        data_in = data_in.iloc[0:-2, rel_ind]
        data_in.iloc[:, 0] = [i.split("/")[-1] for i in data_in.iloc[:, 0]]
        data_in.columns = ["image_id"] + list(data_in.columns[1:])
        data1.append(data_in)  # data_in.iloc[0:-2, -9:-2])
    data_tok = []
    ref = pd.read_csv(dir + "Flickr8k.token.txt", sep="\t", header=None)
    ref.columns = ["cap_id", "caption"]
    data_tok.append(ref)
    data_tok.append(ref)
    ref = pd.read_csv(dir + sub_dir + "flickr30k.csv", sep="|")
    ref.columns = ["image_id", "cap_id", "caption"]
    data_tok.append(ref)
    data_tok.append(ref)
else:
    files = ["pascal.csv", "pascal_candsB.json", "pascal_candsC.json"]
    ref_frame = pd.read_csv(dir + files[0])
    ref_frame["captions"] = ref_frame["captions"].apply(literal_eval)
    ref_frame = ref_frame[["image_id", "captions"]]
    candsB = json.loads(open(dir + files[1], "r").read())
    candsC = json.loads(open(dir + files[2], "r").read())
    cand_frame = [
        [entryB["image_id"], entryB["caption"], entryC["caption"]]
        for entryB, entryC in zip(candsB, candsC)
    ]
    cand_frame = pd.DataFrame(
        cand_frame, columns=["image_id", "candB_cap", "candC_cap"]
    )
    data1 = cand_frame.merge(ref_frame, how="left", on="image_id")
    entry_type = np.zeros((len(data1)))
    for i, entry in data1.iloc[:, 1:3].iterrows():
        if entry[0] in data1.iloc[i, 3] and entry[1] in data1.iloc[i, 3]:
            entry_type[i] = 0
        elif (
            entry[0] in data1.iloc[i, 3]
            and entry[1] not in [k for j in data1.iloc[:, 3] for k in j]
        ) or (
            entry[1] in data1.iloc[i, 3]
            and entry[0] not in [k for j in data1.iloc[:, 3] for k in j]
        ):
            entry_type[i] = 2
        elif entry[0] not in [k for j in data1.iloc[:, 3] for k in j] and entry[
            1
        ] not in [k for j in data1.iloc[:, 3] for k in j]:
            entry_type[i] = 3
        else:
            entry_type[i] = 1
    # import re
    # hi5=[i for i in range(0,len(data1)) if re.sub('[\W_anthe]+', '', data1.iloc[i,1].lower()) in [re.sub('[\W_anthe]+', '', sent.lower()) for line in data1.iloc[:,3] for sent in line] and re.sub('[\W_anthe]+', '', data1.iloc[i,2].lower()) in [re.sub('[\W_anthe]+', '', sent.lower()) for line in data1.iloc[:,3] for sent in line]]
    # print(len(hi5))


# made this a function of SPARCS, SPURTS, MIMA score list
def comp_corl(bigscores):
    if mode == 0:
        human_compare(data1, data2)
    elif mode == 1:
        if dataset == 0:
            ref_out_file = open(dir + "ref_coco.json", "w")
            cand_out_file = open(dir + "cand_coco.json", "w")
            # ref_out_file = open(dir + 'ref_test.json', "w")
            # cand_out_file = open(dir + 'cand_test.json', "w")
            result_frames = write_dict(
                data1, data_tok, ref_out_file, cand_out_file, dataset
            )
            result_frames.to_csv("coco_res_frames.csv")
            # result_frames.to_csv('res_test.csv')
        elif dataset == 1:
            ref_out_file = open(dir + "ref_flickr.json", "w")
            cand_out_file = open(dir + "cand_flickr.json", "w")
            # ref_out_file = open(dir + 'ref_test.json', "w")
            # cand_out_file = open(dir + 'cand_test.json', "w")
            result_frames = write_dict(
                data1, data_tok, ref_out_file, cand_out_file, dataset
            )
            result_frames.to_csv("res_frames.csv")
            # result_frames.to_csv('res_test.csv')
        elif dataset == 2:
            ref_out_file = open(dir + "ref_compos.json", "w")
            cand_out_file = open(dir + "cand_compos.json", "w")
            # ref_out_file = open(dir + 'ref_test.json', "w")
            # cand_out_file = open(dir + 'cand_test.json', "w")
            result_frames = write_dict(
                data1, data_tok, ref_out_file, cand_out_file, dataset
            )
            result_frames.to_csv("compos_res_frames.csv")
            # result_frames.to_csv('res_test.csv')
        else:
            ref_out_file = None
            cand_out_file = None
            # ref_out_file = open(dir + 'ref_test.json', "w")
            # cand_out_file = open(dir + 'cand_test.json', "w")
            result_frames = write_dict(
                data1, data_tok, ref_out_file, cand_out_file, dataset
            )
            result_frames.to_csv("pascal_res_frames.csv")
            # result_frames.to_csv('res_test.csv')
    elif mode == 2:
        if dataset == 0:
            result_frames = pd.read_csv("coco_res_frames.csv")
            # result_file = open('exp_coco_result_new_adjusted_deadline_BSnoidf.json', "r")
            # result_file = open("results/bigscore-1.645.json", "r")
            # result_file = open('exp_benchmark_nopunct.json', "r")
            # res = read_result(result_file,['CIDErD','bertscore','spice','meteor','CIDEr'],result_frames, dataset)#, 'CIDEr', 'CIDErD','bleu-1', 'bleu-2', 'rouge','spice'],result_frames, coco_dataset) # 'exp2_result.json' | 'CIDErD' 'CIDEr' 'meta_roberta' | result_frames
            res = read_result(bigscores, ["SMURF"], result_frames, dataset)
            print(res)
            return res
        elif dataset == 1:
            result_frames = pd.read_csv("res_frames.csv")
            result_file = open(
                "exp2_result.json", "r"
            )  # open('../../smurf_code/results/flickr_sub_scores.json', "r")
            res = read_result(
                result_file,
                ["bertscore", "CIDEr", "CIDErD", "spice", "meteor"],
                result_frames,
                dataset,
            )  # 'exp2_result.json' | 'CIDErD' 'CIDEr' 'meta_roberta' | result_frames
            print(res)
        elif dataset == 2:
            result_frames = pd.read_csv("compos_res_frames.csv")
            result_file = open("../../smurf_code/results/compos_scores.json", "r")
            res = read_result(
                result_file,
                [
                    "bertscore",
                    "meteor",
                    "meta_sem",
                    "meta_qual",
                    "meta_interp",
                    "bleu-1",
                    "bleu-2",
                    "rouge",
                    "CIDErD",
                    "CIDEr",
                    "spice",
                ],
                result_frames,
                dataset,
            )  # 'exp2_result.json' | 'CIDErD' 'CIDEr' 'meta_roberta' | result_frames
            print(res)
        else:
            result_frames = []
            result_frames.append(np.loadtxt(dir + "pascal_gt.txt"))
            result_frames.append(entry_type)
            result_file = []
            result_file.append(
                open("pascalB_scores.json", "r")
                # bigscores
            )
            result_file.append(
                open("../../smurf_code/results/pascalC_scores.json", "r")
                # cscores
            )

            res = read_result(
                result_file,
                [
                    "SPARCS",
                    "SPURTS",
                    "SMURF",
                    # "meta_sem",
                    # "meta_qual",
                    # "meta_interp",
                    # "CIDErD",
                    # "CIDEr",
                    # "spice",
                    # "bleu-1",
                    # "bleu-2",
                    # "rouge",
                    # "meteor",
                ],
                result_frames,
                dataset,
            )  # 'exp2_result.json' | 'CIDErD' 'CIDEr' 'meta_roberta' | result_frames
            print(res)
