#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:12:53 2019

@author: hu
"""
import os
import sys, getopt
import numpy as np
import cv2
from scipy.io import loadmat

parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curpath)
from shell import MODELS, ANCHORS
sys.path.remove(curpath)

sys.path.append(parpath)
from tools.load_weights import load_weights_by_name
from tools.config import Config
from tools.DataAugmentation import DataAugmentation as DA
sys.path.remove(parpath)

import time

class ELAPSEDTIME():
    def __init__(self, dataset, reid, detector = 'yolov3'):
        self.detector = detector
        self.reid = reid
        self.dataset = dataset
        self.config = Config(detector)
        self.DA = DA('validation', self.config)
    
    def load_model(self,):
        if self.reid:
            self.model = MODELS(self.config).load_model()
        else:
            self.model = MODELS(self.config, model_type = 'detection').load_model(model_name = '')
        
    def timer(self,):
        if self.config.M == 'mrcnn' and not hasattr(self, 'anchors'):
            self.anchors = ANCHORS(self.config)
        pool = loadmat(os.path.join(self.dataset, 'dataset/annotation/pool.mat'))['pool'].squeeze()
        gallery = [imname[0] for imname in pool]
        if not hasattr(self, 'model'):
            self.load_model()
        t_step1, t_step2 = 0, 0
        overall_timer_start = time.time()
        for i, inmame in enumerate(gallery):
            step0_timer = time.time()
            img = cv2.imread(os.path.join(self.dataset, 'dataset/Image/SSM/', inmame))
            input_img, input_box, input_ids, meta = self.DA(img, [])
            step1_timer = time.time()
            t_step1 += (step1_timer - step0_timer)
            if self.config.M == 'mrcnn':
                self.model.predict([np.stack([input_img]), np.stack([input_box]), np.stack(meta), np.stack([self.anchors.get_anchors(input_img.shape)])])[0]
            else:
                self.model.predict([np.stack([input_img]), np.stack([input_box])])[0]
            step2_timer = time.time()
            t_step2 += (step2_timer - step1_timer)
            print("\r%d|%d|%.3f|%.3f|%.3f"%(i+1, len(gallery), step2_timer - overall_timer_start, t_step1, t_step2), end = '')
        t_all = time.time() - overall_timer_start
        return [t_all, t_step1, t_step2]
        
        
def main(argv):
    M = 'yolov3'
    CUHK_SYSU = "/home/ronghua/Projects/data/dataset-v2/"
    reid = 0
    gpu = '0'
    try:
        opts, args = getopt.getopt(argv[1:], 'hm:p:g:r:w:e:', ['m=', 'path=', 'gpu=', 'reid=', 'ow=', 'en'])
    except getopt.GetoptError:
        print(argv[0] + ' -m <M> -p <path> -g <gpu> -r <reid>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(argv[0] + ' -m <M> -p <path> -g <gpu> -r<reid>')
        elif opt in ['-m', '--M']:
            M = arg
        elif opt in ['-p', '--path']:
            CUHK_SYSU = arg
        elif opt in ['-r', '--reid']:
            reid = int(arg)
        elif opt in ['-g', '--gpu']:
            gpu = arg
    
    if not os.path.exists(CUHK_SYSU):
        raise ValueError('you should specify the CUHK_SYSU dataset by [python evaluation -p /path/to/dataset]')
    
    print('CUHK_SYSU: [%s], gpu: [%s], model: [%s] %s reid module'%(CUHK_SYSU, gpu, M, 'with' if reid else 'without'))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    timer = ELAPSEDTIME(dataset = CUHK_SYSU, detector = M, reid = reid)
    print(timer.timer())
    
if __name__ == "__main__":
   main(sys.argv)
    
    
    
