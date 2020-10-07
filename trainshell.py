# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 11:02:40 2019
MOT
@author: hu
"""
import os
import sys
import time
import _thread
import logging
import numpy as np

parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))

sys.path.append(curpath)
from tools.load_weights import load_weights_by_name
from tools.config import Config
from tools.DataAugmentation import DataAugmentation as DA
from tools.generator import GENERATORS
from utils.evaluation import EVALUATION as EVAL_CUHK
from utils.prw_evaluation import EVALUATION as EVAL_PRW
from MODELS import MODELS
sys.path.remove(curpath)

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import keras.models as KM
import keras.layers as KL
import tensorflow as tf

class MOT():
    def __init__(self, CUHK_SYSU, PRW = None, M = 'mrcnn'):
        self.config = Config(M = M)
        self.CUHK_SYSU = CUHK_SYSU
        self.PRW = PRW
        self.generators = GENERATORS(cfg = self.config, CUHK_SYSU = self.CUHK_SYSU, PRW=self.PRW)
        self.gen = self.generators.CUHK_SYSU_generator(data_type = 'training')
        self.config.TABLE_SIZE = len(self.generators.id_dict)
        self.config.TRAIN_LIST = self.generators.bTrain
        self.models = MODELS(self.config)
        self.TRAIN_BATCH = []
        self.MS_TRAIN_BATCH = {}
        self.createLogger()

    def loss_1(self, idx = 0):
        def loss_train(y_true, y_pred):
            return y_pred[0, 0 + idx]
        return loss_train

    def acc_1(self, idx = 1):
        def acc_train(y_true, y_pred):
            return y_pred[0, 0 + idx]
        return acc_train

    def createLogger(self,):
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('log.txt', mode = 'w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s") 
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def thread_data_generator(self,):
        _thread.start_new_thread(self.thread_batch_generator, ('thread_1', 0.01, 6))
        while True:
            cnt = 0
            while len(self.TRAIN_BATCH) == 0:
                cnt += 1
                if cnt > 100:
                    self.logger.warn('thread __main stacked')
                    cnt = 0
                time.sleep(0.01)
            yield  self.TRAIN_BATCH.pop()
    
    def thread_da(self, thread_name, delay):
        self.logger.info("start new thread: %s"%thread_name)
        da = DA('training', self.config)
        while 1:
            cnt = 0
            while len(self.thread_task[thread_name]) == 0:
               cnt += 1
               if cnt > 200:
                   self.logger.warn("thread %s stacked"%thread_name)
                   cnt = 0
               time.sleep(delay)
            img1, box1 = self.thread_task[thread_name].pop()

            image_1, gt_bbox1, gt_id, gt_loc = da(img1, box1)
            self.thread_task_results[thread_name].append([image_1, gt_bbox1, gt_id, gt_loc])

    def thread_batch_generator(self, thread_name, delay, workers = 6):
        self.logger.info("entering thread: %s with delay: %f"%(thread_name, delay))
        self.thread_task = {"task%d"%x : [] for x in range(workers)}
        self.thread_task_results = {"task%d"%x : [] for x in range(workers)}
        [_thread.start_new_thread(self.thread_da, ("task%d"%x, 0.01)) for x in range(workers)]
        batch_counter = 0
        images_1, bboxes_1, ids1, loc1 = [], [], [], []
        outputs1 = []
        while 1:
            #assign tasks
            for key in self.thread_task:
                while len(self.thread_task[key]) < 5:
                    img1, boxes1 = next(self.gen)
                    self.thread_task[key].append([img1, boxes1])
            #fetch results
            for key in self.thread_task_results:
                while len(self.thread_task_results[key]) > 0:
                    image, gt_bbox, gt_id, gt_loc = self.thread_task_results[key].pop()
                    if image.shape[0] not in self.MS_TRAIN_BATCH:
                        self.MS_TRAIN_BATCH[image.shape[0]]={'batch_counter':0, 'images':[], 'bboxes':[], 'ids':[], 'loc':[], 'outputs':[]}
                    self.MS_TRAIN_BATCH[image.shape[0]]['batch_counter'] += 1
                    self.MS_TRAIN_BATCH[image.shape[0]]['images'].append(image)
                    self.MS_TRAIN_BATCH[image.shape[0]]['bboxes'].append(gt_bbox)
                    self.MS_TRAIN_BATCH[image.shape[0]]['ids'].append(gt_id)
                    self.MS_TRAIN_BATCH[image.shape[0]]['loc'].append(gt_loc)
                    self.MS_TRAIN_BATCH[image.shape[0]]['outputs'].append(np.array([0, 0]))

                    batch_counter = max([self.MS_TRAIN_BATCH[key1]['batch_counter'] for key1 in self.MS_TRAIN_BATCH])

                    if batch_counter >= self.config.BATCH_SIZE:
                        for key1 in self.MS_TRAIN_BATCH:
                            if self.MS_TRAIN_BATCH[key1]['batch_counter'] >= self.config.BATCH_SIZE:
                                BATCH = self.MS_TRAIN_BATCH[key1]
                                break
                        inputs = [np.stack(BATCH['images']), np.stack(BATCH['bboxes']), np.stack(BATCH['ids']), np.stack(BATCH['loc'])]
                        outputs = [np.stack(BATCH['outputs']) for i in range(1)]
                        BATCH['batch_counter'] = 0
                        BATCH['images'] = []
                        BATCH['bboxes'] = []
                        BATCH['ids'] = []
                        BATCH['loc'] = []
                        BATCH['outputs'] = []
                        cnt = 0
                        while len(self.TRAIN_BATCH) > 5:
                            cnt+=1
                            if cnt > 100:
                                self.logger.warn('thread %s stacked'%thread_name)
                                cnt = 0
                            time.sleep(0.01)
                        self.TRAIN_BATCH.append([inputs, outputs])  

    
    
    def getCtrl(self,):
        assert os.path.exists(os.path.join(curpath, 'ctrl'))
        ctrl = open(os.path.join(curpath, 'ctrl'))
        return [item[:-1] if item[-1] == '\n' else item for item in ctrl]

    def train(self, idx = 1):
        print(self.getCtrl())
        self.model = self.models.model()
        load_weights_by_name(self.model, os.path.join(curpath, 'pretrained_weights/%s.h5'%self.config.M))
    
        print('creating evaluation model...')
        inputs = self.model.inputs[:3]
        output = KL.Lambda(lambda x : tf.expand_dims(tf.nn.l2_normalize(tf.concat(x, axis = -1), dim = -1), axis = 0))(self.model.get_layer('l2_norm').output)
        eval_model = KM.Model(inputs = inputs, outputs = output)
        
        eval_prw = EVAL_PRW(dataset = self.PRW, detector = self.config.M)
        eval_cuhk = EVAL_CUHK(dataset = self.CUHK_SYSU, detector = self.config.M)
        
        self.generators.setImg(['CUHK_SYSU', 'PRW'])
        steps_per_epoch = self.generators.getLen(data_type = 'training') // self.config.BATCH_SIZE

        train_dataset_batch = self.thread_data_generator()

        checkpoint = ModelCheckpoint(os.path.join(curpath, 'saved_weights/%s_model-%d-{epoch:03d}.h5'%(self.config.M, idx)),
                                     monitor = 'loss',
                                     verbose = 1,
                                     save_best_only = False,
                                     save_weights_only = True,
                                     mode = 'min')

        tensorboard = TensorBoard(log_dir = os.path.join(curpath, 'logs'))

        callback_list = [tensorboard, checkpoint]

        sgd = optimizers.SGD(lr = 1e-2, decay = 0., momentum = 0.9, nesterov = True, clipnorm = 5.)

        self.model.compile(loss = self.loss_1(), optimizer = sgd, metrics = [self.loss_1(0), self.acc_1(1)])
        
        self.model.fit_generator(train_dataset_batch, 
                              steps_per_epoch = steps_per_epoch, 
                              epochs = 20,
                              callbacks = callback_list,
                              verbose = 1,
                              )

        #train on CUHK_SYSU and PRW
        self.generators.setImg(['CUHK_SYSU', 'PRW'])
        steps_per_epoch = self.generators.getLen(data_type = 'training') // self.config.BATCH_SIZE

        sgd = optimizers.SGD(lr = 1e-3, decay = 0., momentum = 0.9, nesterov = True, clipnorm = 5.)

        self.model.compile(loss = self.loss_1(), optimizer = sgd, metrics = [self.loss_1(0), self.acc_1(1)])
        
        acc = 0
        epoch = 1
        while True:
            ctrl = self.getCtrl()
            print(ctrl)
            if 'eval' in ctrl:
                evaluation = [eval_cuhk, eval_prw]['default_prw' in ctrl]
                aps, accs = evaluation.simple_evaluation(eval_model)
                new_acc = accs[0]
                savepath = os.path.join(curpath, 'saved_weights/%s_model-%d-%d-%.3f.h5'%(self.config.M, idx, epoch, new_acc))
                if new_acc > acc:
                    print('%d: acc improved from %f to %f saving model to %s'%(epoch, acc, new_acc, savepath))
                    self.model.save_weights(savepath)
                    acc = new_acc
                else:
                    print('%d: new acc %.3f did not improve from %.3f'%(epoch, new_acc, acc))
                    if 'force' in ctrl:
                        print('%d: force saving model to %s'%(epoch, savepath))
                        self.model.save_weights(savepath)
            if 'always_save' in ctrl:
                savepath = os.path.join(curpath, 'saved_weights/%s_model-%d-%d-alwayssave.h5'%(self.config.M, idx, epoch))
                self.model.save_weights(savepath)
            if 'eval_prw' in ctrl:
                aps, accs = eval_prw.simple_evaluation(eval_model)
                print('prw acc:', accs[0])
            if 'eval_cuhk' in ctrl:
                aps, accs = eval_cuhk.simple_evaluation(eval_model)
                print('cuhk acc:', cuhk[0])

            callback_list = [tensorboard]
            self.model.fit_generator(train_dataset_batch, 
                              steps_per_epoch = steps_per_epoch, 
                              epochs = 1, 
                              callbacks = callback_list,
                              verbose = 1,
                              )
            epoch += 1
            #
            if epoch > 20:break
    
    
