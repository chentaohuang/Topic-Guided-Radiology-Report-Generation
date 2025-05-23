import argparse
import json
from collections import defaultdict
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--pmi_dir", type=str, default="path of mimic_cxr_pmi.json", help="the output path")
parser.add_argument("--filter_pmi_dir", type=str, default="path of mimic_cxr_filter_pmi_test.json", help="the output path")
parser.add_argument("--topk_ngram", type=int, default=250, help="the output path")
parser.add_argument("--ngram_dir", type=str, default="path of mimic_cxr_ngram.txt", help="the output path")

config = parser.parse_args()

with open(config.pmi_dir, "r", encoding="utf-8") as f:
    pmi_data = json.load(f)

group_pmi = defaultdict(list)
old_group_pmi = defaultdict(list)
max_mention_pmi = defaultdict(float)
max_gram_pmi = defaultdict(float)

for mention in pmi_data:
    pmi = pmi_data[mention]
    mention, ngram = mention.split("@")
    max_gram_pmi[ngram] = max(max_gram_pmi[ngram], pmi)

for mention in pmi_data:
    pmi = pmi_data[mention]
    mention, ngram = mention.split("@")
    if pmi >= 0.5 * max_gram_pmi[ngram]:
        group_pmi[mention].append((ngram, pmi))
    max_mention_pmi[mention] = max(max_mention_pmi[mention], pmi)

ngram_stat = pd.read_csv(config.ngram_dir)

ngram_stat = {a: b for a, b in zip(ngram_stat["ngram"], ngram_stat["count"])}

ratio = 0.5
unique_ngram = set()
for mention in group_pmi:
    group_pmi[mention] = sorted(
        [
            ngram
            for ngram in group_pmi[mention]
            if ngram[1] >= max(ratio * max_mention_pmi[mention],0.5)
        ],
        key=lambda x: (
            -x[1],  # PMI
            -len(x[0].split()),  # N Gram Length
            -len(x[0]),  # Sequence Length
        )[: config.topk_ngram],
    )

    #old_group_pmi[mention] = group_pmi[mention]
    group_pmi[mention] = [x[0] for x in group_pmi[mention]]
    print(len(group_pmi[mention]))
    #     if "No Finding" not in mention:
    unique_ngram.update(group_pmi[mention])



with open(config.filter_pmi_dir, "w", encoding="utf-8") as f:
    json.dump(group_pmi, f, ensure_ascii=False, indent=4)

print("Unique N-Gram Count %d" % len(unique_ngram))
