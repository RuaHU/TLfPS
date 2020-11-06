#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:10:06 2020

@author: hu
"""

import os
import shutil
import sys, getopt
parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curpath)
from shell import MODELS, ANCHORS
sys.path.remove(curpath)

sys.path.append(parpath)
from tools.config import Config
from tools.DataAugmentation import DataAugmentation as DA
sys.path.remove(parpath)
import numpy as np
import cv2
import pickle
import scipy.io
from sklearn.metrics import average_precision_score
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


class EVALUATION():
    def __init__(self, dataset, detector = 'yolov3', experiment_name = 'default', overwrite = False):
        if dataset is None:return
        self.detector = detector
        self.dataset = dataset
        self.overwrite = overwrite
        self.experiment_name = experiment_name
        self.checkdir()
        self.config = Config(detector)
        self.DA = DA('validation', self.config)
        self.load_gallery()
        self.load_query()
        
        
    def get_sims(self, gfeats, qfeat, _eval = True):
        '''
        gfeats: gallery features
        qfeat: query feature
        '''
        if _eval:return gfeats.dot(qfeat.ravel()).ravel()
        gfeats_norm = np.linalg.norm(gfeats, keepdims = True, axis = -1)
        qfeat_norm = np.linalg.norm(qfeat, keepdims = True)
        gfeats_nl = gfeats / gfeats_norm
        qfeat_nl = qfeat / qfeat_norm
        sim = gfeats_nl.dot(qfeat_nl.ravel()).ravel()
        return sim
    
    def checkdir(self,):
        dirpath = os.path.join(parpath, 'experiment_results', self.experiment_name)
        if os.path.exists(dirpath):
            print('experiment [%s] existed'%self.experiment_name)
            if self.overwrite:
                print('cleaning experiment [%s] [overwrite == True]'%self.experiment_name)
                shutil.rmtree(dirpath, ignore_errors = True)
                if os.path.exists(dirpath):
                    print('it seems the experiment directory can not be deleted. please check the status of the directory %s'%dirpath)
                os.mkdir(dirpath)
                assert os.path.exists(dirpath)
            else:
                print('the results of experiment [%s] will be reused [overwrite == False]'%self.experiment_name)
        else:
            os.mkdir(dirpath)
            assert os.path.exists(dirpath)
            
    def load_model(self,):
        self.reid_model = MODELS(config = self.config).load_model()
        
    def load_gallery(self):
        self.gallery_dict = {}
        frame_test = scipy.io.loadmat(os.path.join(self.dataset, 'frame_test.mat'))
        frame_indices = frame_test['img_index_test']
        for index, im_name in enumerate(frame_indices[:, 0]):
            mat = scipy.io.loadmat(os.path.join(self.dataset, 'annotations', im_name[0] + '.jpg.mat'))
            boxes = mat[list(mat.keys())[-1]]
            ids = boxes[:, 0]
            boxes = boxes[:, 1:5]
            self.gallery_dict[im_name[0]+'.jpg'] = [im_name[0]+'.jpg', boxes, ids]

    def load_query(self):
        file = open(os.path.join(self.dataset, 'query_info.txt'))
        self.query_list = []
        for line in file:
            items = line.split()
            ids, roi = int(items[0]), [float(items[1]), float(items[2]), float(items[3]), float(items[4])]
            self.query_list.append([items[-1]+'.jpg', roi, ids])
    
    def query_feature_extractor(self,):
        filepath = os.path.join(parpath, 'experiment_results', self.experiment_name, 'prw_%s_query_features.pkl'%self.config.M)
        if os.path.exists(filepath):
            return
        
        if not hasattr(self, 'reid_model'):
            self.load_model()
            
        if self.config.M == 'mrcnn' and not hasattr(self, 'anchors'):
            self.anchors = ANCHORS(self.config)
        
        query_features = []
        for item in self.query_list:
            img_name, roi, _ = item
            img = cv2.imread(os.path.join(self.dataset, 'frames', img_name))
            input_img, input_box, _, meta = self.DA(img, [roi])
            if self.config.M == 'mrcnn':
                feature = self.reid_model.predict([np.stack([input_img]), np.stack([input_box]), np.stack(meta), np.stack([self.anchors.get_anchors(input_img.shape)])])[0]
            else:
                feature = self.reid_model.predict([np.stack([input_img]), np.stack([input_box])])[0]
            query_features.append([img_name, feature[0], np.array([roi])])
            print("\r%d|%d"%(len(query_features), len(self.query_list)), end = '')
        print('')
        self.query_features = query_features
        f = open(filepath, 'wb')
        pickle.dump(query_features, f)
        f.close()
        return
    
    def gallery_feature_extractor(self,):
        filepath = os.path.join(parpath, 'experiment_results', self.experiment_name, 'prw_%s_gallery_features.pkl'%self.config.M)
        if os.path.exists(filepath):
            return
        
        if not hasattr(self, 'reid_model'):
            self.load_model()
        
        if self.config.M == 'mrcnn' and not hasattr(self, 'anchors'):
            self.anchors = ANCHORS(self.config)

        gallery = []
        for imname in self.gallery_dict.keys():
            img = cv2.imread(os.path.join(self.dataset, 'frames', imname))
            input_img, input_box, _, meta = self.DA(img, [])
            if self.config.M == 'mrcnn':
                feats, _, _, _, det_features, det, _ = self.reid_model.predict([np.stack([input_img]), np.stack([input_box]), np.stack(meta), np.stack([self.anchors.get_anchors(input_img.shape)])])
            elif self.config.M == 'dla_34':
                feats, det_features, det, _ = self.reid_model.predict([np.stack([input_img]), np.stack([input_box])])
            else:
                feats, _, _, _, det_features, det, _ = self.reid_model.predict([np.stack([input_img]), np.stack([input_box])])
            det = self.DA.unmold(det[0], meta)
            det[:, 2:] += det[:, :2]
            gallery.append([imname, det_features[0], det])
            print("\r%d|%d"%(len(gallery), len(self.gallery_dict)), end = '')
        print('')
        f = open(filepath, 'wb')
        pickle.dump(gallery, f)
        f.close()
        return

    def simple_evaluation(self, model, gallery_size = 50):
        #extract query feature vectors
        qfeatures = []
        for item in self.query_list:
            img_name, roi, _ = item
            img = cv2.imread(os.path.join(self.dataset, 'frames', img_name))
            input_img, input_box, input_ids, meta = self.DA(img, [roi])
            feature = model.predict([np.stack([input_img]), np.stack([input_box]), np.stack([input_ids])])
            qfeatures.append([img_name, feature[0], np.array([roi])])
            print("\r%d|%d"%(len(qfeatures), len(self.query_list)), end = '')
        print('')
        #extract gallery feature vectors
        filepath = os.path.join(parpath, 'experiment_results/prw_%s_gallery.pkl'%self.config.M)
        assert os.path.exists(filepath)
        f = open(filepath, 'rb')
        oim_gallery = pickle.load(f, encoding='latin1')
        gallery = []
        for item in oim_gallery:
            imname, oim_features, oim_boxes = item
            if oim_features is None:
                oim_features = np.zeros([0, 256], dtype = np.float32)
                oim_boxes = np.zeros([0, 5], dtype = np.float32)
            img = cv2.imread(os.path.join(self.dataset, 'frames', imname))
            #xyxy 2 xywh
            toim_boxes = oim_boxes.copy()
            toim_boxes[:, 2:4] -= toim_boxes[:, :2]
            input_img, input_box, input_ids, meta = self.DA(img, toim_boxes[:, :4])
            feats = model.predict([np.stack([input_img]), np.stack([input_box]), np.stack([input_ids])])
            gallery.append([imname, feats[0, :, 0, 0, :], oim_boxes])
            print("\r%d|%d"%(len(gallery), len(oim_gallery)), end = '')
        print('')

        name_to_det_feat = {}
        for img_name, features, boxes in gallery:
            name_to_det_feat[img_name] = (boxes, features)
        return self.evaluation(qfeatures, name_to_det_feat, _eval = True)

    def evaluation(self, qfeatures, name_to_det_feat, _eval):
        aps, accs, topk = [], [], [1, 5, 10]
        log = open('log.txt', 'w')
        sysout = sys.stdout
        all_recall_rate = []
        #tape = {}
        for i, query in enumerate(self.query_list):
            sys.stdout = log
            qimg_name, qroi, qid = query
            y_true, y_score = [], []
            count_gt, count_tp = 0, 0
            qfeat = qfeatures[i][1].ravel()
            gallery_items = [self.gallery_dict[key] for key in self.gallery_dict if qid in self.gallery_dict[key][-1] and key != qimg_name ]
            gallery_gts = {}
            for item in gallery_items:gallery_gts[item[0]] = item[1][item[2]==qid]
            gallery_imgs = [key for key in self.gallery_dict if key != qimg_name]
            imgs, y_boxes, y_gname = [], [], []
            for gallery_imname in gallery_imgs:
                count_gt += (gallery_imname in gallery_gts)
                if gallery_imname not in name_to_det_feat:continue
                gboxes, gfeatures = name_to_det_feat[gallery_imname]
                sim = self.get_sims(gfeatures, qfeat, _eval)
                label = np.zeros(len(sim), dtype=np.int32)
                if gallery_imname in gallery_gts:
        
                    gt = gallery_gts[gallery_imname].ravel()
                    w, h = gt[2], gt[3]
                    gt[2], gt[3] = gt[0] + gt[2], gt[1] + gt[3]
                    iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    gboxes = gboxes[inds]
                    for j, roi in enumerate(gboxes[:, :4]):
                        if self._compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                
                y_true.extend(list(label))
                y_score.extend(list(sim))
                y_boxes.extend(list(gboxes))
                y_gname.extend([gallery_imname for _ in gboxes])

            y_score = np.array(y_score)
            y_true = np.array(y_true)
            y_boxes = np.array(y_boxes)
            y_gname = np.array(y_gname)
            assert count_tp <= count_gt
            recall_rate = count_tp * 1.0 / count_gt
            all_recall_rate.append(recall_rate)
            ap = 0 if count_tp == 0 else \
                average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            
            inds = np.argsort(y_score)[::-1]
            y_score = y_score[inds]
            y_true = y_true[inds]
            y_boxes = y_boxes[inds]
            y_gname = y_gname[inds]
            acc = [min(1, sum(y_true[:k])) for k in topk]
            accs.append(acc)
            #tape[qimg_name] = [qid, qroi, ap, acc, recall_rate, y_score, y_true, y_boxes, y_gname]
            sys.stdout = sysout
            print("\r%d:\t%d|%d|%.2f|%.2f"%(-1, len(aps), len(qfeatures), np.mean(aps), np.mean(accs, axis = 0)[0]), end = '')
        print('')
        print('search ranking:')
        print('aRR:%.4f'%np.mean(all_recall_rate))
        print('  mAP = {:.2%}'.format(np.mean(aps)))
        accs = np.mean(accs, axis=0)
        for i, k in enumerate(topk):
            print('  top-{:2d} = {:.2%}'.format(k, accs[i]))
        
        #record_aps = []
        #new_tape = {}
        #for key in tape.keys():
        #    record_aps.append(tape[key][2])
        #record_aps.sort()
        #th = record_aps[50]
        #for key in tape.keys():
        #    if tape[key][2] > th:continue
        #    new_tape[key] = tape[key]
        
        #filepath = os.path.join(parpath, 'experiment_results', self.experiment_name, 'prw_%s_tape.pkl'%self.config.M)
        #pickle.dump(new_tape, filepath)
        
        return aps, accs

    def private_detector_evaluation(self):
        print('the results of this experiment using end-to-end detector [%s] + feature extractor [%s%s]'%(self.config.M, self.config.M, '_mgn' if self.config.mgn else ''))
        topk = [1, 5, 10]
        respath = os.path.join(parpath, 'experiment_results', self.experiment_name, 'prw_%s_res.pkl'%self.config.M)
        if os.path.exists(respath):
            f = open(respath, 'rb')
            res = pickle.load(f)
            f.close()
            aps, accs = res
            print('  mAP = {:.2%}'.format(np.mean(aps)))
            for i, k in enumerate(topk):
                print('  top-{:2d} = {:.2%}'.format(k, accs[i]))
            return
        
        qfilepath = os.path.join(parpath, 'experiment_results', self.experiment_name, 'prw_%s_query_features.pkl'%self.config.M)
        assert os.path.exists(qfilepath)
        f = open(qfilepath, 'rb')
        qfeatures = pickle.load(f)
        f.close()
        gfilepath = os.path.join(parpath, 'experiment_results', self.experiment_name, 'prw_%s_gallery_features.pkl'%self.config.M)
        assert os.path.exists(gfilepath)
        f = open(gfilepath, 'rb')
        gfeatures = pickle.load(f , encoding='latin1')
        f.close()
        name_to_det_feat = {}
        for img_name, features, boxes in gfeatures:
            name_to_det_feat[img_name] = (boxes, features)
        
        res = self.evaluation(qfeatures, name_to_det_feat, _eval = True)
        
        f = open(respath, 'wb')
        pickle.dump(res, f)
        f.close()
        
    def _compute_iou(self, box1, box2):
        a, b = box1.copy(), box2.copy()
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter * 1.0 / union
    

def main(argv):
    M = 'yolov3'
    path = "/home/ronghua/Projects/data/PRW-v16.04.20/"
    gpu = '0'
    overwrite = False
    experiment_name = 'prw_default'
    try:
        opts, args = getopt.getopt(argv[1:], 'hm:p:g:w:e:', ['m=', 'path=', 'gpu=', 'ow=', 'en'])
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
        elif opt in ['-w', '--ow']:
            if arg == 'False':overwrite = False
        elif opt in ['-e', '--en']:
            experiment_name = arg
    
    if not os.path.exists(path):
        raise ValueError('you should specify the dataset path by [python evaluation -p /path/to/dataset #or --path /path/to/dataset #structure example: /home/dataset-v2]')
    
    print('PRW: [%s] gpu: [%s] model: [%s] experiment name: [%s]'%(path, gpu, M, experiment_name))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu  
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    evaluation = EVALUATION(dataset = path, detector = M, experiment_name = experiment_name, overwrite = overwrite)
    evaluation.gallery_feature_extractor()
    evaluation.query_feature_extractor()
    evaluation.private_detector_evaluation()
    
if __name__ == "__main__":
   main(sys.argv)
