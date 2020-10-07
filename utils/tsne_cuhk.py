#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:10:06 2020

@author: hu
"""

import os
import shutil
import sys, getopt
parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curpath)
from shell import MODELS, ANCHORS
sys.path.remove(curpath)

sys.path.append(parpath)
from tools.config import Config
from tools.DataAugmentation import DataAugmentation as DA
sys.path.remove(parpath)
import numpy as np
import cv2
import pickle
import scipy.io

class EVALUATION():
    def __init__(self, dataset, detector = 'yolov3', experiment_name = 'default', overwrite = False):
        if dataset is None:return
        self.detector = detector
        self.dataset = dataset
        self.overwrite = overwrite
        self.experiment_name = experiment_name
        self.checkdir()
        self.config = Config(detector)
        self.DA = DA('validation', self.config)
        self.load_gallery()
    
    def checkdir(self,):
        dirpath = os.path.join(parpath, 'experiment_results', self.experiment_name)
        if os.path.exists(dirpath):
            print('experiment [%s] existed'%self.experiment_name)
            if self.overwrite:
                print('cleaning experiment [%s] [overwrite == True]'%self.experiment_name)
                shutil.rmtree(dirpath, ignore_errors = True)
                if os.path.exists(dirpath):
                    print('it seems the experiment directory can not be deleted. please check the status of the directory %s'%dirpath)
                os.mkdir(dirpath)
                assert os.path.exists(dirpath)
            else:
                print('the results of experiment [%s] will be reused [overwrite == False]'%self.experiment_name)
        else:
            os.mkdir(dirpath)
            assert os.path.exists(dirpath)
        
    def load_gallery(self,):
        pool_path = os.path.join(self.dataset, 'dataset/annotation/pool.mat')
        Person_path = os.path.join(self.dataset, 'dataset/annotation/Person.mat')
        if not os.path.exists(pool_path):
            raise ValueError('cannot found %s'%pool_path)
        pool = scipy.io.loadmat(pool_path)['pool'].squeeze()
        Person = scipy.io.loadmat(Person_path)['Person'].squeeze()
        imnames = [imname[0] for imname in pool]
        
        images = {}
        for person in Person:
            pid = int(person[0][0][1:])
            for item in person[2][0]:
                if item[0][0] in images:
                    images[item[0][0]][0].append(item[1][0].tolist())
                    images[item[0][0]][1].append(pid)
                else:
                    images[item[0][0]] = [[item[1][0].tolist()], [pid]]
            
        self.gallery_dict = {}
        for key in images.keys():
            if key in imnames:
                self.gallery_dict[key] = images[key]
    
    def load_model(self,):
        self.reid_model = MODELS(self.config).load_model()
    
    def TSNE(self,):
        fTSNE_features = os.path.join(parpath, 'experiment_results', self.experiment_name, 'cuhk_%s_TSNE_features.pkl'%self.config.M)
        fTSNE_id = os.path.join(parpath, 'experiment_results', self.experiment_name, 'cuhk_%s_TSNE_id.pkl'%self.config.M)
        fTSNE_imgs = os.path.join(parpath, 'experiment_results', self.experiment_name, 'cuhk_%s_TSNE_imgs.pkl'%self.config.M)
        if os.path.exists(fTSNE_features):
            f = open(fTSNE_features, 'rb')
            TSNE_features = pickle.load(f)
            f = open(fTSNE_id, 'rb')
            TSNE_id = pickle.load(f)
            f.close()
        else:
            if not hasattr(self, 'reid_model'):
                self.load_model()
            
            if self.config.M == 'mrcnn' and not hasattr(self, 'anchors'):
                self.anchors = ANCHORS(self.config)
            
            TSNE_features = []
            TSNE_id = []
            TSNE_imgs = []
            gallery = []
            
            for imname in self.gallery_dict.keys():
                boxes, ids = self.gallery_dict[imname]
                img = cv2.imread(os.path.join(self.dataset, 'dataset/Image/SSM', imname))
                input_img, input_box, _, meta = self.DA(img, boxes)
                if self.config.M == 'mrcnn':
                    feats, _, _, _, det_features, det, _ = self.reid_model.predict([np.stack([input_img]), np.stack([input_box]), np.stack(meta), np.stack([self.anchors.get_anchors(input_img.shape)])])
                else:
                    feats, _, _, _, det_features, det, _ = self.reid_model.predict([np.stack([input_img]), np.stack([input_box])])
                for i, feat in enumerate(feats[0]):
                    TSNE_features.append(feat)
                    TSNE_id.append(ids[i])
                    x, y, w, h = boxes[i]
                    TSNE_imgs.append(img[int(y):int(y+h), int(x):int(x+w), :])
                gallery.append(imname)
                print("\r%d|%d"%(len(gallery), len(self.gallery_dict)), end = '')
            print('')
            TSNE_features = np.array(TSNE_features)
            TSNE_id = np.array(TSNE_id)

            f = open(fTSNE_features, 'wb')
            pickle.dump(TSNE_features, f)
            f.close()
            f = open(fTSNE_id, 'wb')
            pickle.dump(TSNE_id, f)
            f.close()
            f = open(fTSNE_imgs, 'wb')
            pickle.dump(TSNE_imgs, f)
            f.close()
        
def main(argv):
    M = 'yolov3'
    path = "/home/ronghua/Projects/data/dataset-v2/"
    gpu = '0'
    overwrite = False
    experiment_name = 'cuhk_default'
    try:
        opts, args = getopt.getopt(argv[1:], 'hm:p:g:w:e:', ['m=', 'path=', 'gpu=', 'ow=', 'en'])
    except getopt.GetoptError:
        print(argv[0] + ' -m <M> -p <path> -g <gpu>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(argv[0] + ' -m <M> -p <path> -g <gpu>')
        elif opt in ['-m', '--M']:
            M = arg
        elif opt in ['-p', '--path']:
            path = arg
        elif opt in ['-g', '--gpu']:
            gpu = arg
        elif opt in ['-w', '--ow']:
            if arg == 'False':overwrite = False
        elif opt in ['-e', '--en']:
            experiment_name = arg
    
    if not os.path.exists(path):
        raise ValueError('you should specify the dataset path by [python evaluation -p /path/to/dataset #or --path /path/to/dataset #structure example: /home/dataset-v2]')
    
    print('CUHK: [%s] gpu: [%s] model: [%s] experiment name: [%s]'%(path, gpu, M, experiment_name))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu  
    
    evaluation = EVALUATION(dataset = path, detector = M, experiment_name = experiment_name, overwrite = overwrite)
    evaluation.TSNE()
    
if __name__ == "__main__":
   main(sys.argv)
