#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:40:26 2019

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
    #default:
    M = 'yolov3'
    gpu = '0'
    path = None
    try:
        opts, args = getopt.getopt(argv[1:], 'hm:p:g:', ['m=', 'path=', 'gpu='])
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
    
    if path == None:
        raise ValueError('you should specify the model path via [python convertor -p /path/to/model/weights]')
    print('model: [%s], gpu: [%s], weights: [%s]'%(M, gpu, path))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    config = Config(M)
    config.mgn = True
    model = MODELS(config = config)
    reid_model = model.reid_model
    load_weights_by_name(reid_model, path)
    
    load_weights_by_name(reid_model, os.path.join(parpath, 'pretrained_weights/%s.h5'%M))
    saved_path = os.path.join(parpath, 'saved_weights/%s_reid.h5'%M)
    print('weights saving to %s'%saved_path)
    reid_model.save_weights(saved_path)
          
if __name__ == "__main__":
   main(sys.argv)
