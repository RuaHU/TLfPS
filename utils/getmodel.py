#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:12:53 2019

@author: hu
"""
import os
import sys, getopt

parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curpath)
from shell import MODELS
sys.path.remove(curpath)

sys.path.append(parpath)
from tools.load_weights import load_weights_by_name
from tools.config import Config
sys.path.remove(parpath)

def main(argv):
    M = 'yolov3'
    gpu = '0'
    try:
        opts, args = getopt.getopt(argv[1:], 'hm:g:', ['m=', 'gpu='])
    except getopt.GetoptError:
        print(argv[0] + ' -m <M> -g <gpu>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(argv[0] + ' -m <M> -g <gpu>')
        elif opt in ['-m', '--M']:
            M = arg
        elif opt in ['-g', '--gpu']:
            gpu = arg
    
    print('model: [%s] gpu: [%s]'%(M, gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu  
    
    config = Config(M)
    model = MODELS(config, model_type = 'detection').load_model()
    print('saving model to pretrained_weights/%s.h5'%config.M)
    model.save_weights(os.path.join(parpath, 'pretrained_weights/%s.h5'%config.M))
    
if __name__ == "__main__":
    main(sys.argv)
    
