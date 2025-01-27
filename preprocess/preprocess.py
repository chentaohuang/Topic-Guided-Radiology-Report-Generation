
import os
import json
import torch.nn as nn
import torch
import math

from transformers import (
    AutoTokenizer,
    AutoModel,
)

import numpy as np
import  csv
import pandas as pd

def NumIn(s):
    for char in s:
        if char.isdigit():
            return True
    if "xx" in s:
        return True
    return False


def get_graph():
    f_label = pd.read_csv('path of mimic_label_full.csv')
    with open("path of MIMIC-CXR_graphs.json", "r", encoding="utf-8") as f:
        f_read = json.load(f)

    with open("path of mimic_addr2id.json", "r", encoding="utf-8") as f:
        addr2id = json.load(f)

    trp_set = set()

    topic_KG = {}
    topic_stat = {}
    topics = ['Atelectasis_True', 'Cardiomegaly_True', 'Consolidation_True', 'Edema_True',
              'Enlarged Cardiomediastinum_True', 'Fracture_True', 'Lung Lesion_True', 'Lung Opacity_True',
              'No Finding_True', 'Pleural Effusion_True', 'Pleural Other_True', 'Pneumonia_True', 'Pneumothorax_True',
              'Support Devices_True',
              'Atelectasis_False', 'Cardiomegaly_False', 'Consolidation_False', 'Edema_False',
              'Enlarged Cardiomediastinum_False', 'Fracture_False', 'Lung Lesion_False', 'Lung Opacity_False',
              'No Finding_False', 'Pleural Effusion_False', 'Pleural Other_False', 'Pneumonia_False',
              'Pneumothorax_False', 'Support Devices_False']
    for topic in topics:
        topic_stat[topic] = 0
        triplets = {'old_triplets': {}, 'triplets': {}, 'triplets_pmi': {}}
        topic_KG[topic] = triplets

    # print(len(f_read))
    stat = {}

    for addr in f_read:
        if addr not in addr2id:
            continue;

        ids = addr2id[addr]
        for id in ids:
            label = f_label[f_label['id'] == id]

            id = addr
            label = label.iloc[0].tolist()[2:]


            for l, t in zip(label, topics):
                if l == 0.0:
                    continue;
                else:
                    for entity in f_read[id]["entities"]:

                        if len(f_read[id]["entities"][entity]["relations"]) == 0:
                            continue;
                        else:
                            head = f_read[id]["entities"][entity]["tokens"]
                            idx = f_read[id]["entities"][entity]["relations"][0][1]
                            rel = f_read[id]["entities"][entity]["relations"][0][0]
                            tail = f_read[id]["entities"][idx]["tokens"]
                            trp = head.lower() + '@' + rel + '@' + tail.lower()
                            if NumIn(trp):
                                continue;

                            # trp_set.add(trp)

                            # if trp not in topic_KG[t]['triplets']:
                            #     print('====================')
                            #     topic_KG[t]['triplets'][trp] = 1
                            # else:
                            #     topic_KG[t]['triplets'][trp] += 1

                            if trp not in topic_KG[t]['old_triplets']:
                                topic_KG[t]['old_triplets'][trp] = 1
                                if trp not in stat:
                                    stat[trp] = 1
                                else:
                                    stat[trp] += 1
                                topic_stat[t] += 1
                            else:
                                topic_KG[t]['old_triplets'][trp] += 1
                                stat[trp] += 1
                                topic_stat[t] += 1

    total_count = 0
    for topic in topic_stat:
        total_count += topic_stat[topic]

    for topic in topics:
        topic_KG[topic]["triplets"] = {}
        for trp in topic_KG[topic]['old_triplets']:
            if topic_KG[topic]['old_triplets'][trp] > 10:
                trp_set.add(trp)
                p_x = stat[trp] / total_count
                p_xy = topic_KG[topic]['old_triplets'][trp] / topic_stat[topic]
                pmi_xy = math.log(p_xy / p_x, 2)
                topic_KG[topic]['triplets_pmi'][trp] = pmi_xy

    for topic in topic_KG:
        del topic_KG[topic]["old_triplets"]
        ordered = sorted(topic_KG[topic]['triplets_pmi'].items(), key=lambda a: a[1], reverse=True)

        if len(ordered) > 80:
            ordered = ordered[:80]
        topic_KG[topic]['triplets'] = dict(ordered)
        topic_KG[topic]['entity'] = set()

        for triplet in topic_KG[topic]['triplets']:
            head, rel, tail = triplet.split("@")
            topic_KG[topic]['entity'].add(head)
            topic_KG[topic]['entity'].add(tail)
        topic_KG[topic]['entity'] = list(topic_KG[topic]['entity'])

        n = len(topic_KG[topic]['entity'])
        print('===', n)
        topic_KG[topic]['entity2id'] = {}
        i = 0
        for ent in topic_KG[topic]['entity']:
            topic_KG[topic]['entity2id'][ent] = i
            i += 1

        n = len(topic_KG[topic]['entity'])
        if n == 0:
            matrix = 0
        else:
            matrix = np.zeros((n, n), dtype=int)
            for triplet in topic_KG[topic]['triplets']:
                head, rel, tail = triplet.split("@")
                x = topic_KG[topic]['entity2id'][head]
                y = topic_KG[topic]['entity2id'][tail]
                matrix[x, y] = 1
                matrix[y, x] = 1
                matrix[x, x] = 1
                matrix[y, y] = 1
            topic_KG[topic]['matrix'] = matrix.tolist()

    with open('path of mimic_cxr_KG.json', "w",
              encoding="utf-8") as f:
        json.dump(topic_KG, f, ensure_ascii=False, indent=4)

def get_node_embed():
    
    with open('path of mimic_cxr_KG.json', "r", encoding="utf-8") as f:
        f_read = json.load(f)
    
    text_tokenizer = AutoTokenizer.from_pretrained(
        "path of \clinicalbert",
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        "path of \clinicalbert",
        local_files_only=True,
    )

    
    model.eval()
    with torch.no_grad():
        for topic in f_read:
            emb = torch.empty((0, 768))
            for ent in f_read[topic]['entity2id']:
                ids = text_tokenizer.encode(ent)
                outputs = model(input_ids=torch.LongTensor([ids])).last_hidden_state
                emb = torch.cat([emb, torch.mean(outputs[0][1:], dim=0).unsqueeze(dim=0)], dim=0)
            f_read[topic]['embedding'] = np.array(emb).tolist()
    
    with open('path of mimic_cxr_KG.json', "w", encoding="utf-8") as f:
        json.dump(f_read, f, ensure_ascii=False, indent=4)


def get_gram_embed():
    with open('path of mimic_cxr_filter_pmi_test.json', "r", encoding="utf-8") as f:
        f_read = json.load(f)

    text_tokenizer = AutoTokenizer.from_pretrained(
        "path of \clinicalbert",
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        "path of \clinicalbert",
        local_files_only=True,
    )

    dct = {}
    model.eval()
    with torch.no_grad():
        for topic in f_read:
            emb = torch.empty((0, 768))
            for ent in f_read[topic]:
                ids = text_tokenizer.encode(ent)
                outputs = model(input_ids=torch.LongTensor([ids])).last_hidden_state
                emb = torch.cat([emb, torch.mean(outputs[0][1:], dim=0).unsqueeze(dim=0)], dim=0)
            print(emb.size())
            dct[topic] = np.array(emb).tolist()

    with open("path of mimic_cxr_gram_embed.json", "w", encoding="utf-8") as f:
        json.dump(dct, f, ensure_ascii=False, indent=4)

def get_report_embed():
    with open("path of annotation.json", "r", encoding="utf-8") as f:
        ff = json.load(f)

    text_tokenizer = AutoTokenizer.from_pretrained(
            "path of \clinicalbert",
            local_files_only = True,
        )
    model = AutoModel.from_pretrained(
        "path of \clinicalbert",
        local_files_only=True,
    )

    dt = {}
    model.eval()
    with torch.no_grad():
        for i in range(len(ff['train'])):
            id = ff['train'][i]['id']
            rep =  ff['train'][i]['report']
            ids = text_tokenizer.encode(rep)
            outputs = model(input_ids=torch.LongTensor([ids])).last_hidden_state
            emb =  torch.mean(outputs[0][1:], dim=0) #outputs[0][0]
            emb = np.array(emb).tolist()
            dt[id] = emb
    with open('path of mimic_cxr_ReportEmb.json', "w", encoding="utf-8") as f:
        json.dump(dt, f, ensure_ascii=False, indent=4)

def kmeans(x):
    import numpy as np
    from scipy.spatial.distance import cdist

    class K_Means(object):
        def __init__(self, n_clusters=50, max_iter=300, centroids=[]):
            self.n_clusters = n_clusters
            self.max_iter = max_iter
            self.centroids = np.array(centroids, dtype=np.float)

        def fit(self, data):
            if (self.centroids.shape == (0,)):
                self.centroids = data[np.random.randint(0, data.shape[0], self.n_clusters),
                                 :]
            for i in range(self.max_iter):
                distances = cdist(data, self.centroids)
                c_index = np.argmin(distances, axis=1)
                for i in range(self.n_clusters):
                    if i in c_index:
                        self.centroids[i] = np.mean(data[c_index == i], axis=0)
        def predict(self, samples):
            distances = cdist(samples, self.centroids)
            c_index = np.argmin(distances, axis=1)
            return c_index

    kmeans = K_Means(max_iter=10000)
    kmeans.fit(x)

    return kmeans.centroids

def get_topic_reportemb():
    with open("path of mimic_cxr_ReportEmb.json", "r", encoding="utf-8") as f:
        ff = json.load(f)

    with open("path of annotation.json", "r", encoding="utf-8") as f:
        f_an = json.load(f)

    f_label = pd.read_csv('path of mimic_label_full.csv')

    topics = ['Atelectasis_True', 'Cardiomegaly_True', 'Consolidation_True', 'Edema_True',
              'Enlarged Cardiomediastinum_True', 'Fracture_True', 'Lung Lesion_True', 'Lung Opacity_True',
              'No Finding_True', 'Pleural Effusion_True', 'Pleural Other_True', 'Pneumonia_True', 'Pneumothorax_True',
              'Support Devices_True',
              'Atelectasis_False', 'Cardiomegaly_False', 'Consolidation_False', 'Edema_False',
              'Enlarged Cardiomediastinum_False', 'Fracture_False', 'Lung Lesion_False', 'Lung Opacity_False',
              'No Finding_False', 'Pleural Effusion_False', 'Pleural Other_False', 'Pneumonia_False',
              'Pneumothorax_False', 'Support Devices_False']
    dt = {}
    for topic in topics:
        dt[topic] = []
    for id in ff:
        for i in range(len(f_an['train'])):
            if f_an['train'][i]['id'] == id:
                stuid = f_an['train'][i]['study_id']
                subid = f_an['train'][i]['subject_id']
                break;
        #label = f_label[f_label['id']==id]
        label = f_label[(f_label['subject_id'] == int(subid)) & (f_label['study_id'] == int(stuid))]
        label = label.iloc[0].tolist()[3:]
        #label = label.iloc[0].tolist()[2:]
        for l, t in zip(label, topics):
            if l==0.0:
                continue
            else:
                dt[t].append(ff[id])
    
    dict ={}
    for i in dt:
        if len(dt[i])>=50:
            means = kmeans(np.array(dt[i]))
        else:
            means = np.array(dt[i])
        print(means.shape)
        dict[i] = means.tolist()
    
    with open("path of mimic_cxr_topics_embed.json", "w", encoding="utf-8") as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)

def get_topic():
    name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
            'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices']

    f1 = pd.read_csv('path of mimic_label.csv')
    f = f1[name]

    f1 = f1.replace(-1.0,1)
    f_a = f.replace(1.0, -1.0).replace(0.0, 1.0).replace(-1.0, 0.0)

    f_aa = f_a[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                'Pneumothorax', 'Support Devices']]

    result = pd.concat([f1[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                'Pneumothorax', 'Support Devices']], f_aa], axis=1)

    result = result.fillna(0.0)

    result.to_csv('path of mimic_label_full.csv')


def main():
    #construct topic
    get_topic()

    #get topic graph
    get_graph()
    get_node_embed()
    
    #get ngram embedding
    #first run  pmi_ngram.py -> pmi_mention_ngram->filter_ngram.py to get mimic_cxr_filter_pmi_test.json
    #then run:
    get_gram_embed()
    
    # get topic textual centroids
    get_report_embed()
    get_topic_reportemb()



if __name__ == '__main__':
   main()



