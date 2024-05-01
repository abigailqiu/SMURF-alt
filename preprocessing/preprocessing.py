import spacy
import string
import json
from random import choice

nlp = spacy.load("en_core_web_sm")


def remove_punc(s):
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s


with open("preprocessing/all_nouns.json", "r") as f:
    nouns = json.load(f)


def rand_nouns(s):
    doc = nlp(s)
    noun_list = []
    for token in doc:
        if token.pos_ == "NOUN":
            noun_list.append(token.text)
    if len(noun_list) >= 1:
        s = s.replace(choice(noun_list), choice(nouns))
    return s


# with open("preprocessing/coco_adj.json", "r") as f:
#     adj = json.load(f)


# def rand_adjs(s, count):
#     doc = nlp(s)
#     adj_list = []
#     for token in doc:
#         if token.pos_ == "ADJ":
#             adj_list.append(token.text)
#     if len(adj_list) >= 2:
#         repl = sample(adj_list, 2)
#         s = s.replace(repl[0], choice(adj))
#         s = s.replace(repl[1], choice(adj))
#     elif len(adj_list) >= 1:
#         s = s.replace(choice(adj_list), choice(adj))
#     else:
#         count += 1
#     return count


with open("preprocessing/coco_words.json", "r") as f:
    words = json.load(f)


def rand_words(s):
    doc = nlp(s)
    adj_list = []
    for token in doc:
        if token.pos_ == "NOUN":
            adj_list.append(token.text)
    if len(adj_list) >= 1:
        s = s.replace(choice(adj_list), choice(words))
    return s


cand_file = "noun/cand.json"
with open(cand_file, "r") as f:
    data = json.load(f)

og = []
rnoun = []
rwd = []
for entry in data:
    image_id = entry["image_id"]
    # caption_id = entry["id"]
    cap = entry["caption"]
    cap = cap.lower()
    cap = remove_punc(cap)
    og.append({"image_id": image_id, "caption": cap})
    rnoun.append({"image_id": image_id, "caption": rand_nouns(cap)})
    rwd.append({"image_id": image_id, "caption": rand_words(cap)})

with open("noun/cand.json", "w") as f:
    json.dump(og, f)

# with open("noun/rnoun.json", "w") as f:
#     json.dump(rnoun, f)

# with open("noun/rwd.json", "w") as f:
#     json.dump(rwd, f)
