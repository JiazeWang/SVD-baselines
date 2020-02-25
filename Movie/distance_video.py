import h5py
import time
import numpy as np
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

refer_video_num = []
copy_video_num = []
with open("useful_train_test_groundtruth", 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.rstrip()
    line = line.split(" ")
    refer_video_num.append(line[1])
    copy_video_num.append(line[0])
test_video = []
with open("test_groundtruth", 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.rstrip()
    line = line.split(' ')
    test_video.append(line[0])
"""
with open("video_list.txt", 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.rstrip()
    copy_video_num.append(line)
"""
"""
#copy_video_h5 = h5py.File("videos-features.h5")
#copy_video_sum = copy_video_h5["000001.mp4"]
#for i in range(1, len(copy_video)):
#    append_video = copy_video_h5[copy_video[i]]
#    copy_video_sum = np.vstack((copy_video_sum, append_video))
#for in i range(0,len(refer_video))
refer_video_h5 = h5py.File("refer-video-features.h5")
refer_video_sum = refer_video_h5["tt0006864.mp4"]
for i in range(1, len(refer_video)):
    if i%100 == 0:
        print(i)
    append_video = refer_video_h5[refer_video[i]]
    refer_video_sum = np.vstack((refer_video_sum, append_video))
print(refer_video_sum.shape)
np.save("refer_video_sum.npy", refer_video_sum)
"""
refer_video = np.load("refer_video_225.npy")
#time_start = time.time()
copy_video_h5 = h5py.File("/mnt/SSD/jzwang/dataset/features/copy-video-feature-1fps.h5")
refer_video_h5 = h5py.File("/mnt/SSD/jzwang/dataset/features_1s/refer-video-features.h5")
generate = []
new = []
for i in range(0,len(test_video)):
    #copy_video = np.load("copy_video_sum.npy")[i].reshape(1,-1)
    copy_video = copy_video_h5[test_video[i]].value
    t1 = time.time()
    idxs, unsorted_dists, sorted_dists = compute_dists(copy_video, refer_video)
    t2 = time.time()
    time_per = (t2-t1)
    if i%100 ==0:
        print(i)
#np.save("idx1.npy",idxs)
#time_end = time.time()
#print("totally cost:",time_end - time_start)
    result = (refer_video_num[int(idxs[0][0])])
    new.append(result+' '+str(time_per))
with open("test_video_retrieve.txt", 'w') as f:
    f.write("\n".join(new))
