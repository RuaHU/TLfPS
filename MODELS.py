#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:12:53 2019

@author: hu
"""
import os
import sys
parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))
import tensorflow as tf
sys.path.append(curpath)
from BaseNet import *
import keras.models as KM
from YOLO import YOLOv3, YOLOv4
from MaskRCNN import MaskRCNN
from dla_34 import DLASeg
sys.path.remove(curpath)


class MODELS():
    def __init__(self, cfg, mode = 'training'):
        self.config = cfg
        self.mode = mode
        
    def model(self,):
        input_img = KL.Input(shape = [None, None, 3], name = "input_1")
        input_bbox = KL.Input(shape = [None, 4], name = "bbox1")
        input_id = KL.Input(shape = [None, 1], name = 'id')
        input_loc = KL.Input(shape = [None, 1], name = 'loc')
        if self.config.M == 'mrcnn':
            mrcnn = MaskRCNN(self.config)
            reid_feature_map = mrcnn.reid(input_img, training = False)[1]
        elif self.config.M in ['yolov3', 'yolov3-tiny']:
            yolov3 = YOLOv3(self.config, self.mode)
            reid_feature_map = yolov3.reid(input_img, training = False)[1]
        elif self.config.M in ['yolov4', 'yolov4-tinye']:
            yolov4 = YOLOv4(self.config)
            reid_feature_map = yolov4.reid(input_img, training = False)[1]
        elif self.config.M == 'dla_34':
            reid_feature_map = DLASeg(config=self.config).reid(input_img, training = False)[1]
            
        reid_map = ATLnet(reid_feature_map, layer = self.config.layer, SEnet = self.config.SEnet)
        pooled = feature_pooling(self.config, name = "AlignedROIPooling")([input_bbox] + reid_map)
        pooled = gatherValidItems(pooled, input_id)
        fl = sMGN(pooled, _eval = False, l2_norm = self.config.l2_norm)
        fl = KL.Lambda(lambda x : tf.squeeze(x, [1, 2]))(fl)
        loss = LookUpTable(self.config, name = 'lut')([fl, input_id, input_loc])
        return KM.Model([input_img, input_bbox, input_id, input_loc], [loss], name = 'person_search_model')
        
