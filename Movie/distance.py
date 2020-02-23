
def compute_similarities(query_features, refer_features):
    """
      用于计算两组特征(已经做过l2-norm)之间的相似度
      Args:
        query_features: shape: [N, D]
        refer_features: shape: [M, D]
      Returns:
        sorted_sims: shape: [N, M]
        unsorted_sims: shape: [N, M]
    """
    sorted_sims = []
    unsorted_sims = []
    # 计算待查询视频和所有视频的距离
    dist = np.nan_to_num(cdist(query_features, refer_features, metric='cosine'))
    for i, v in enumerate(query_features):
        # 归一化，将距离转化成相似度
        # sim = np.round(1 - dist[i] / dist[i].max(), decimals=6)
        sim = 1 - dist[i]
        # 按照相似度的从大到小排列，输出index
        unsorted_sims += [sim]
        sorted_sims += [[(s, sim[s]) for s in sim.argsort()[::-1] if not np.isnan(sim[s])]]
    return sorted_sims, unsorted_sims

def compute_dists(query_features, refer_features):
    """
      用于计算两组特征(已经做过l2-norm)之间的余弦距离
      Args:
        query_features: shape: [N, D]
        refer_features: shape: [M, D]
      Returns:
        idxs: shape [N, M]
        unsorted_dists: shape: [N, M]
        sorted_dists: shape: [N, M]
    """
    sims = np.dot(query_features, refer_features.T)
    unsorted_dists = 1 - sims # sort 不好改降序
    # unsorted_dist = np.nan_to_num(cdist(query_features, refer_features, metric='cosine'))
    idxs = np.argsort(unsorted_dists)
    rows = np.dot(np.arange(idxs.shape[0]).reshape((idxs.shape[0], 1)), np.ones((1, idxs.shape[1]))).astype(int)
    sorted_dists = unsorted_dists[rows, idxs]
    # sorted_dists = np.sort(unsorted_dists)
    return idxs, unsorted_dists, sorted_dists