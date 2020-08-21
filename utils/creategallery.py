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
import pickle
from scipy.io import loadmat
parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curpath)
from mrcnnshell import mrcnn, ANCHORS
from yoloshell import yolo
sys.path.remove(curpath)

sys.path.append(parpath)
from tools.load_weights import load_weights_by_name
from tools.config import Config
from tools.DataAugmentation import DataAugmentation as DA
sys.path.remove(parpath)
        
def unmold(bboxes, meta, mode = 'uxywh'):
    if mode == 'uxywh':
        ih, iw = meta[0][4:6]
        window = meta[0][7:11]
        scale = meta[0][11]
        bboxes[:, [0, 2]] *= ih
        bboxes[:, [1, 3]] *= iw
        bboxes[:, [1, 3]] -= window[1]
        bboxes[:, [0, 2]] -= window[0]
        bboxes /= scale
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes[:, [1, 0, 3, 2]]
    elif mode == 'nxywh':
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes[:, [1, 0, 3, 2]]
    else:
        'default mode: normalized yxyx (nyxyx)'
    return bboxes



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
    CUHK_SYSU = None
    PRW = None
    try:
        opts, args = getopt.getopt(argv[1:], 'hm:g:p:c:', ['m=', 'gpu=', 'prw=', 'cuhk='])
    except getopt.GetoptError:
        print(argv[0] + ' -m <M> -g <gpu>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(argv[0] + ' -m <M> -g <gpu>')
        elif opt in ['-m', '--M']:
            M = arg
        elif opt in ['-p', '--path']:
            PRW = arg
        elif opt in ['-c', '--cuhk']:
            CUHK_SYSU = arg
        elif opt in ['-g', '--gpu']:
            gpu = arg
    
    print('model: [%s] gpu: [%s], CUHK_SYSU: [%s], PRW: [%s]'%(M, gpu, CUHK_SYSU, PRW))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    dataset = False
    if os.path.exists(CUHK_SYSU):
        dataset = True
    if os.path.exists(PRW):
        dataset = True
    
    if not dataset:
        raise ValueError('you should specify at least one dataset [CUHK_SYSU or PRW]')
    
    config = Config(M)
    model_path = os.path.join(parpath, 'pretrained_weights/%s.h5'%config.M)
    
    if config.M == 'yolov3' or config.M == 'yolov4':
        model = yolo(config, True)
    else:
        model = mrcnn(config, True)
        anchors = ANCHORS(config)
        
    load_weights_by_name(model, model_path)
    da = DA('validation', config)
    
    
    if os.path.exists(CUHK_SYSU):
        filepath = os.path.join(parpath, 'experiment_results', 'cuhk_%s_gallery.pkl'%config.M)
        if os.path.exists(filepath):
            print('cuhk_%s_gallery.pkl exists.'%config.M)
        else:
            print('creating cuhk-sysu gallery for %s'%config.M)
            gallery = []
            pool_path = os.path.join(CUHK_SYSU, 'dataset/annotation/pool.mat')
            if not os.path.exists(pool_path):
                raise ValueError('cannot found %s'%pool_path)
            pool = loadmat(pool_path)['pool'].squeeze()
            imnames = [imname[0] for imname in pool]
            for imname in imnames:
                img = cv2.imread(os.path.join(CUHK_SYSU, 'dataset/Image/SSM/', imname))
                input_img, input_box, input_ids, meta = da(img, [])
                if config.M == 'mrcnn':
                    detection, scores = model.predict([np.stack([input_img]), np.stack([input_box]), np.stack(meta), np.stack([anchors.get_anchors(input_img.shape)])])
                else:
                    detection, scores = model.predict([np.stack([input_img]), np.stack([input_box])])
                detection = unmold(detection[0], meta)
                detection[:, 2:] += detection[:, :2]
                features = np.zeros([len(detection), 0])
                gallery.append([imname, features, detection])
                print("\r%d|%d"%(len(gallery), len(imnames)), end = '')
            print('')
            f = open(filepath, 'wb')
            pickle.dump(gallery, f)
            f.close()
            
    if os.path.exists(PRW):
        filepath = os.path.join(parpath, 'experiment_results', 'prw_%s_gallery.pkl'%config.M)
        if os.path.exists(filepath):
            print('prw_%s_gallery.pkl exists.'%config.M)
        else:
            print('creating prw gallery for %s'%config.M)
            gallery = []
            frame_test_path = os.path.join(PRW, 'frame_test.mat')
            if not os.path.exists(frame_test_path):
                raise ValueError('cannot found %s'%frame_test_path)
            frame_indices = loadmat(frame_test_path)['img_index_test'].squeeze()
            imnames = [imname[0]+'.jpg' for imname in frame_indices]
            for imname in imnames:
                img = cv2.imread(os.path.join(PRW, 'frames', imname))
                input_img, input_box, input_ids, meta = da(img, [])
                if config.M == 'mrcnn':
                    detection, scores = model.predict([np.stack([input_img]), np.stack([input_box]), np.stack(meta), np.stack([anchors.get_anchors(input_img.shape)])])
                else:
                    detection, scores = model.predict([np.stack([input_img]), np.stack([input_box])])
                detection = unmold(detection[0], meta)
                detection[:, 2:] += detection[:, :2]
                features = np.zeros([len(detection), 0])
                gallery.append([imname, features, detection])
                print("\r%d|%d"%(len(gallery), len(imnames)), end = '')
            print('')
            f = open(filepath, 'wb')
            pickle.dump(gallery, f)
            f.close()
        
if __name__ == "__main__":
   main(sys.argv)
    
    
    
