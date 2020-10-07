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

parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))

sys.path.append(parpath)
from shell import MODELS, ANCHORS
from tools.load_weights import load_weights_by_name
from tools.config import Config
from tools.DataAugmentation import DataAugmentation as DA
sys.path.remove(parpath)
    
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

def draw(image, boxes, scores):
    img = image.copy()
    scores = scores.ravel()
    for i, box in enumerate(boxes):
        x, y, w, h = box
        score = scores[i]
        img = cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        img = cv2.putText(img, '%.3f'%score, (int(x), int(y - 12)), cv2.FONT_HERSHEY_TRIPLEX , 0.5, (255, 255, 255), 1)
    return img
    
def main(argv):
    M = 'yolov3'
    gpu = '0'
    image_path = None
    try:
        opts, args = getopt.getopt(argv[1:], 'hm:g:p:', ['m=', 'gpu=', 'path='])
    except getopt.GetoptError:
        print(argv[0] + ' -m <M> -g <gpu>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(argv[0] + ' -m <M> -g <gpu>')
        elif opt in ['-m', '--M']:
            M = arg
        elif opt in ['-p', '--path']:
            image_path = arg
        elif opt in ['-g', '--gpu']:
            gpu = arg
    
    print('model: [%s] gpu: [%s]'%(M, gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu  
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)


    if not os.path.exists(image_path):
        raise ValueError('you must specify a image for testing %s'%('' if image_path is None else 'bad image path: [%s]'%image_path))
    
    config = Config(M)
    model_path = os.path.join(parpath, 'pretrained_weights/%s.h5'%config.M)
    model = MODELS(config, model_type = 'detection').load_model(model_name = '')
    
    #test
    da = DA('validation', config)
    img = cv2.imread(image_path)
    input_img, input_box, input_ids, meta = da(img, [])
    cv2.imwrite('input_img.jpg', (input_img*255).astype(np.uint8))
    if config.M == 'mrcnn':
        anchors = ANCHORS(config)
        detection, detection_score = model.predict([np.stack([input_img]), np.stack([input_box]), np.stack(meta), np.stack([anchors.get_anchors(input_img.shape)])])
    else:
        detection, detection_score = model.predict([np.stack([input_img]), np.stack([input_box])])
    detection = da.unmold(detection[0], meta)
    image = draw(img, detection, detection_score)
    print(detection_score)
    print(detection)
    cv2.imwrite('%s_test.jpg'%config.M, image)
    print('detection results saved as: %s_test.jpg'%config.M)
    
if __name__ == "__main__":
   main(sys.argv)
    
    
    
