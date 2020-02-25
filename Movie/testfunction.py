import h5py
import numpy as np
def shotagg(feature, txt):
    with open(txt ,'r') as f:
        lines = f.readlines()
    total_lines = int(lines[-1].rstrip().split(' ')[1])
    print(total_lines)
    f_dim = feature.shape[0]
    num = 0
    for line in lines:
        line = line.rstrip()
        line = line.split(' ')
        line[0] = int(line[0])
        line[1] = int(line[1])
        if (line[1]-line[0])*f_dim/total_lines < 2:
            continue
        else:
            start = round(line[0]*f_dim/total_lines)
            end = round(line[1]*f_dim/total_lines)
            print(start,end)
            vfeature = feature[start:end]
            vfeature = vfeature.mean(axis=0, keepdims=True)
            #norm
            if num = 0:
                vtotal = vfeature
            else:
                vtotal = np.vstack((vtotal, vfeature))
            num = num + 1

    return vtotal

video = "000001.mp4"
txt = "/mnt/SSD/jzwang/dataset/shot_movie/000001.txt"
video_h = h5py.File("/mnt/SSD/jzwang/dataset/features/copy-frame-feature-1fps.h5","r")
framefeatures = np.array(video_h[video][()]).squeeze()
print(framefeatures.shape)
newfeature = shotagg(framefeatures,txt)
print(newfeature.shape)
