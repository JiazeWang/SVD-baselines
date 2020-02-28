# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: videofeatures_extraction.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-8-25
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import h5py
import time
parenddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parenddir)

import numpy as np
import multiprocessing as mp

from utils.util import *
from utils.args import opt
from utils.logger import logger
refer_video_num = [] 
copy_video_num = []
with open("/mnt/SSD/jzwang/dataset/list/useful_train_test_groundtruth", 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.rstrip()
    line = line.split(" ")
    refer_video_num.append(line[1])
    copy_video_num.append(line[0])


class VideoFeatureExtractor(object):
    def __init__(self):
        self.procs = []
        self.num_procs = opt['num_procs']

        vfeatures = mp.Manager()
        self.vfeatures = vfeatures.dict()

        self.input = mp.Queue()

        for idx in range(self.num_procs):
            p = mp.Process(target=self.worker, args=(idx, ))
            p.start()
            self.procs.append(p)

    @staticmethod
    def __normalize__(X):
        X -= X.mean(axis=1, keepdims=True)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-15
        return X

    def normalization(self, params):
        copy_featurepath = os.path.join(opt['featurepath'], 'copy-scene.h5')
        refer_featurepath = os.path.join(opt['featurepath'], 'refer-scene.h5')
        copy_fp = h5py.File(copy_featurepath, mode='r')
        refer_fp = h5py.File(refer_featurepath, mode='r')
        index, copy_video = params[0], params[1]
        copy_features = np.array(copy_fp[copy_video][()])
        vfeature, vfeature_avg = self.shotcompare(copy_features, refer_fp)
        self.vfeatures[copy_video[0:-4] + 'max'] = vfeature
        self.vfeatures[copy_video[0:-4] + 'avg'] = vfeature_avg
        if index % 1 == 0:
            logger.info('index: {:6d}, video: {}'.format(index, copy_video))
        copy_fp.close()
        refer_fp.close()

    def worker(self, idx):
        while True:
            params = self.input.get()
            if params is None:
                self.input.put(None)
                break
            try:
                self.normalization(params)
            except Exception as e:
                logger.info('Exception: {}. video: {}'.format(e, params[1]))

    def start(self, videolists):
        for idx, video in enumerate(videolists):
            self.input.put([idx, video])
        self.input.put(None)

    def stop(self):
        for idx, proc in enumerate(self.procs):
            proc.join()
            logger.info('process: {} is done.'.format(idx))

    def save_features(self):
        vfeatures = dict(self.vfeatures)
        vfeaturepath = os.path.join(opt['featurepath'], 'search-max-scene.h5')
        fp = h5py.File(vfeaturepath, mode='w')
        for video in vfeatures:
            feature = vfeatures[video]
            fp.create_dataset(name=video, data=feature)
        fp.close()
        logger.info('saving feature done')
    
    def shotcompare(self, feature, refer_fp):     
        copy_video = feature
        #t1 = time.time()
        sims = np.zeros((len(refer_video_num)))
        simsavg = np.zeros((len(refer_video_num)))
        for j in range(0, len(refer_video_num)):
            refer_video = refer_fp[refer_video_num[j]].value
            sim = np.dot(copy_video, refer_video.T)
            
            sim_total=np.zeros(sim.shape[0])
            for k in range(0,(sim.shape[0])):
                sim_v = np.max(sim[k])
                sim_total[k] = (sim_v)
            #print(sim_total.shape)
            avgsim = np.sum(sim_total)/len(sim_total)
            
            maxsim = np.max(sim)
            sims[j] = maxsim
            simsavg[j] = avgsim
        return sims, simsavg
    

def main():
    vfe = VideoFeatureExtractor()
    videos = get_video_id()
    logger.info('#video: {}'.format(len(videos)))
    vfe.start(videos)
    vfe.stop()
    vfe.save_features()
    logger.info('all done')


if __name__ == "__main__":
    main()

'''bash
python videoprocess/videofeatures_extraction.py --dataname svd-example
'''
