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

parenddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parenddir)

import numpy as np
import multiprocessing as mp

from utils.util import *
from utils.args import opt
from utils.logger import logger
from sklearn.cluster import AgglomerativeClustering

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
        featurepath = os.path.join(opt['featurepath'], 'copy-shot2.h5')
        fp = h5py.File(featurepath, mode='r')
        index, video = params[0], params[1]
        framefeatures = np.array(fp[video][()]).squeeze()
        if framefeatures.ndim == 1:
            framefeatures = np.array([framefeatures, framefeatures])
        vfeature = self.__normalize__(framefeatures)
        vfeature = self.__normalize__(vfeature)
        #vfeature = vfeature.mean(axis=0, keepdims=True)
        vfeature = self.shotagg(vfeature)
        vfeature = self.__normalize__(vfeature)
        self.vfeatures[video] = vfeature
        if index % 100 == 0:
            logger.info('index: {:6d}, video: {}'.format(index, video))
        fp.close()

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
        vfeaturepath = os.path.join(opt['featurepath'], 'copy-scene.h5')
        fp = h5py.File(vfeaturepath, mode='w')
        for video in vfeatures:
            feature = vfeatures[video]
            fp.create_dataset(name=video, data=feature)
        fp.close()
        logger.info('saving feature done')
    
    def shotagg(self, feature):
        if feature.shape[0]>5:
            featurenew = np.zeros((5, 4096))
            clustering = AgglomerativeClustering(5).fit(feature)
            itemindex0 = np.argwhere(clustering.labels_ == 0)
            for i in range(len(itemindex0)):
                featurenew[0] = featurenew[0] + feature[int(itemindex0[i])]
            featurenew[0] = featurenew[0] / len(itemindex0)

            itemindex1 = np.argwhere(clustering.labels_ == 1)
            for i in range(len(itemindex0)):
                featurenew[1] = featurenew[1] + feature[int(itemindex0[i])]
            featurenew[1] = featurenew[1] / len(itemindex1)

            itemindex2 = np.argwhere(clustering.labels_ == 2)
            for i in range(len(itemindex0)):
                featurenew[2] = featurenew[2] + feature[int(itemindex0[i])]
            featurenew[2] = featurenew[2] / len(itemindex2)

            itemindex3 = np.argwhere(clustering.labels_ == 3)
            for i in range(len(itemindex0)):
                featurenew[3] = featurenew[3] + feature[int(itemindex0[i])]
            featurenew[3] = featurenew[3] / len(itemindex3)

            itemindex4 = np.argwhere(clustering.labels_ == 4)
            for i in range(len(itemindex0)):
                featurenew[4] = featurenew[4] + feature[int(itemindex0[i])]
            featurenew[4] = featurenew[4] / len(itemindex4)
        else:
            featurenew = feature
        
        return featurenew
    

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
