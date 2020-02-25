import glob
import pandas as pd
import pickle
import time

import cv2
import imagehash
import numpy as np
import networkx as nx
from tqdm.notebook import tqdm
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
from networkx.algorithms.dag import dag_longest_path

def get_frame_alignment(query_features, refer_features, top_K=5, min_sim=0.80, max_step=10):
    """
      用于计算两组特征(已经做过l2-norm)之间的帧匹配结果
      Args:
        query_features: shape: [N, D]
        refer_features: shape: [M, D]
        top_K: 取前K个refer_frame
        min_sim: 要求query_frame与refer_frame的最小相似度
        max_step: 有边相连的结点间的最大步长
      Returns:
        path_query: shape: [1, L]
        path_refer: shape: [1, L]
    """
    node_pair2id = {}
    node_id2pair = {}
    node_id2pair[0] = (-1, -1) # source
    node_pair2id[(-1, -1)] = 0
    node_num = 1

    DG = nx.DiGraph()
    DG.add_node(0)

    idxs, unsorted_dists, sorted_dists = compute_dists(query_features, refer_features)

    # add nodes
    for qf_idx in range(query_features.shape[0]):
        for k in range(top_K):
            rf_idx = idxs[qf_idx][k]
            sim = 1 - sorted_dists[qf_idx][k]
            if sim < min_sim:
                break
            node_id2pair[node_num] = (qf_idx, rf_idx)
            node_pair2id[(qf_idx, rf_idx)] = node_num
            DG.add_node(node_num)
            node_num += 1
    
    node_id2pair[node_num] = (query_features.shape[0], refer_features.shape[0]) # sink
    node_pair2id[(query_features.shape[0], refer_features.shape[0])] = node_num
    DG.add_node(node_num)
    node_num += 1

    # link nodes

    for i in range(0, node_num - 1):
        for j in range(i + 1, node_num - 1):
            
            pair_i = node_id2pair[i]
            pair_j = node_id2pair[j]

            if(pair_j[0] > pair_i[0] and pair_j[1] > pair_i[1] and
               pair_j[0] - pair_i[0] <= max_step and pair_j[1] - pair_i[1] <= max_step):
               qf_idx = pair_j[0]
               rf_idx = pair_j[1]
               DG.add_edge(i, j, weight=1 - unsorted_dists[qf_idx][rf_idx])

    for i in range(0, node_num - 1):
        j = node_num - 1

        pair_i = node_id2pair[i]
        pair_j = node_id2pair[j]

        if(pair_j[0] > pair_i[0] and pair_j[1] > pair_i[1] and
            pair_j[0] - pair_i[0] <= max_step and pair_j[1] - pair_i[1] <= max_step):
            qf_idx = pair_j[0]
            rf_idx = pair_j[1]
            DG.add_edge(i, j, weight=0)

    longest_path = dag_longest_path(DG)
    if 0 in longest_path:
        longest_path.remove(0) # remove source node
    if node_num - 1 in longest_path:
        longest_path.remove(node_num - 1) # remove sink node
    path_query = [node_id2pair[node_id][0] for node_id in longest_path]
    path_refer = [node_id2pair[node_id][1] for node_id in longest_path]

    score = 0.0
    for (qf_idx, rf_idx) in zip(path_query, path_refer):
        score += 1 - unsorted_dists[qf_idx][rf_idx]

    return path_query, path_refer, score