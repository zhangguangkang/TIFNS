import pickle
import numpy as np
import torch
def get_infomation_by_path(path):
    with open(path, 'rb') as handle:
        result = pickle.load(handle)
    return result

def get_entity_eigenvector(vectors):
    # 求和
    # result = torch.sum(vectors, dim=0)
    # 平均
    result = torch.mean(vectors, dim=0)
    return result

def load_entity_num(train_data, aux_data):
    train_entity_num = set()
    for triplet in train_data:
        train_entity_num.add(triplet[0])
        train_entity_num.add(triplet[2])

    ookb_entity_num = set()
    for triplet in aux_data:
        if int(triplet[0]) >= len(train_entity_num):
            ookb_entity_num.add(triplet[0])
        if int(triplet[2]) >= len(train_entity_num):
            ookb_entity_num.add(triplet[2])
    return len(train_entity_num), len(ookb_entity_num)