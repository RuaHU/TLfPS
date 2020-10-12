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

sys.path.append(parpath)
from YOLO import YOLOv3, YOLOv4
from MaskRCNN import MaskRCNN, ANCHORS
from dla_34 import DLASeg
from tools.load_weights import load_weights_by_name
sys.path.remove(parpath)

class MODELS():
    def __init__(self, config = None, model_type = 'reid'):
        self.config = config
        if self.config.M in ['yolov3', 'yolov3-tiny']:
            yolov3 = YOLOv3(self.config)
            self.model = yolov3.model(model_type)
        elif self.config.M in ['yolov4', 'yolov4-tiny']:
            yolov4 = YOLOv4(self.config)
            self.model = yolov4.model(model_type)
        elif self.config.M in ['mrcnn']:
            mrcnn = MaskRCNN(self.config)
            self.model = mrcnn.model(model_type)
        elif self.config.M in ['dla_34']:
            self.model = DLASeg(self.config).model(model_type)
        else:
            raise ValueError('unsupported model type...')
    
    def load_model(self, model_name = 'reid'):
        if model_name == 'reid':
            model_path = os.path.join(parpath, 'saved_weights/%s_%s.h5'%(self.config.M, model_name))
        else:
            model_path = os.path.join(parpath, 'pretrained_weights/%s.h5'%self.config.M)
        print('loading weights for %s_%s from %s'%(self.config.M, model_name, model_path))
        load_weights_by_name(self.model, model_path)
        return self.model
    
