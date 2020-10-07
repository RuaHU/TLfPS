#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:07:33 2019

@author: hu
"""
import sys
import os, getopt
from trainshell import MOT

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def main(argv):
    #default:
    M = 'yolov3'
    gpu = '0'
    CUHK_SYSU = "/home/ronghua/Projects/data/dataset-v2/"
    PRW = '/home/ronghua/Projects/data/PRW-v16.04.20/'
    name_idx = '1'

    try:
        opts, args = getopt.getopt(argv[1:], 'hm:p:c:g:n:', ['M=', 'prw=', 'cuhk=', 'gpu=', 'name='])
    except getopt.GetoptError:
        print(argv[0] + ' -m <M> -p <prw> -c <cuhk> -g <gpu>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(argv[0] + ' -m <M> -p <path> -g <gpu>')
        elif opt in ['-m', '--M']:
            M = arg
        elif opt in ['-p', '--prw']:
            PRW = arg
        elif opt in ['-c', '--cuhk']:
            CUHK_SYSU = arg
        elif opt in ['-g', '--gpu']:
            gpu = arg
        elif opt in ['-n', '--name']:
            name_idx = arg
    
    dataset = False
    if os.path.exists(PRW):
        dataset = True
    else:
        PRW = None
    if os.path.exists(CUHK_SYSU):
        dataset = True
    else:
        CUHK_SYSU = None
    if not dataset:
        raise ValueError('you should specify at least one dataset')
        
    print('model: [%s], gpu: [%s], CUHK_SYSU: [%s], PRW: [%s]'%(M, gpu, CUHK_SYSU, PRW))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    set_session(sess)

    mot = MOT(CUHK_SYSU = CUHK_SYSU, PRW=PRW, M = M)
    mot.train(idx = int(name_idx))
    
if __name__ == "__main__":
   main(sys.argv)
