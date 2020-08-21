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
from yolov3 import *
from maskrcnn import *
sys.path.remove(curpath)


class MODELS():
    def __init__(self, cfg):
        self.config = cfg
        
    def model(self,):
        input_img = KL.Input(shape = [*self.config.IMAGE_SIZE, 3], name = "input_1")
        input_bbox = KL.Input(shape = [None, 4], name = "bbox1")
        input_id = KL.Input(shape = [None, 1], name = 'id')
        input_loc = KL.Input(shape = [None, 1], name = 'loc')
        if self.config.M == 'mrcnn':
            feature_maps = resnet_graph(input_img, training = False)
            reid_feature_map = mrcnn_proposal_map(feature_maps)[-2:]
        elif self.config.M == 'yolov3':
            feature_maps = darknet_body_v3(input_img, training = False)
            reid_feature_map = yolo_proposal_map_v3(feature_maps, training = False)[-2:]
        elif self.config.M == 'yolov4':
            feature_maps = darknet_body_v4(input_img, training = False)
            reid_feature_map = yolo_proposal_map_v4(feature_maps, training = False)[-2:]
        reid_map = ATLnet(reid_feature_map, layer = self.config.layer, SEnet = self.config.SEnet)
        pooled = feature_pooling(self.config, name = "AlignedROIPooling")([input_bbox] + reid_map)
        pooled = gatherValidItems(pooled, input_id)
        fl = sMGN(pooled, _eval = False, l2_norm = self.config.l2_norm)
        fl = KL.Lambda(lambda x : tf.squeeze(x, [1, 2]))(fl)
        loss = LookUpTable(self.config, name = 'lut')([fl, input_id, input_loc])
        return KM.Model([input_img, input_bbox, input_id, input_loc], [loss], name = 'person_search_model')
        
