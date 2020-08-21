#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:55:51 2020

@author: hu
"""

import os
import cv2
import numpy as np
import random
import scipy.io
import _thread
import time
import copy

class GENERATORS():
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.images = []
        self.bTrain = []
        self.id_dict = {}
        self.id_ptr = 0
        if 'PRW' in kwargs:
            if kwargs['PRW'] is not None:
                self.PRW = {'dataset' : kwargs['PRW']}
                self.PRW_parser()
        if 'CUHK_SYSU' in kwargs:
            if kwargs['CUHK_SYSU'] is not None:
                self.CUHK_SYSU = {'dataset' : kwargs['CUHK_SYSU']}
                self.CUHK_SYSU_parser()
                self.CUHK_SYSU_get_images()
        if 'Market1501' in kwargs:
            if kwargs['Market1501'] is not None:
                self.Market1501 = {'dataset' : kwargs['Market1501']}
                self.Market1501_parser()


    def setImg(self, keys):
        self.images = []
        for key in keys:
            if hasattr(self, key+'_images'):
                self.images += eval('self.%s_images'%key)
                print('get %d images from %s, the training dataset expanded to %d'%(len(eval('self.%s_images'%key)), key, len(self.images)))


    def getLen(self, data_type = 'all'):
        return len(self.images)
    
    def id_map(self, id_):
        if id_ in self.id_dict:return self.id_dict[id_]
        self.bTrain.append(1)
        self.id_dict[id_] = self.id_ptr
        self.id_ptr+=1
        return self.id_dict[id_]
        
    def Market1501_parser(self,):
        bounding_box_train = os.path.join(self.Market1501['dataset'], 'bounding_box_train')
        instances = []
        imgs = os.listdir(bounding_box_train)
        loc_dict = {}
        for img in imgs:
            if img[-4:] != '.jpg':continue
            ist_id = int(img[:-4].split('_')[0])
            cam_id = int(img[:-4].split('_')[1][1])
            ser_id = int(img[:-4].split('_')[1][3])
            unique_id = 'Market1501' + str(ist_id)
            if unique_id in loc_dict:
                loc_dict[unique_id] += 1
            else:
                loc_dict[unique_id] = 0
            instances.append([unique_id, cam_id, ser_id, loc_dict[unique_id], os.path.join(bounding_box_train, img)])
        ids = np.array(list(list(zip(*instances))[0]))
        ids = list(set(ids))
        
        self.Market1501.update({'instances' : instances,
                                'ids' : ids,
                                })    

    def _compute_iou(self, box1, box2):
        a, b = box1.copy(), box2.copy()
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter * 1.0 / union
    
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
    
    
    
    def CUHK_SYSU_parser(self,):
        Images = scipy.io.loadmat(os.path.join(self.CUHK_SYSU['dataset'], 'dataset/annotation/Images.mat'))
        Imgs = Images['Img'][0]
        boxes_dict = {}
        all_imgs_name = []
        for i, img in enumerate(Imgs):
            imname = img[0][0]
            all_imgs_name.append(imname)
            boxes = np.array([im[0][0].tolist() for im in img[2][0]])
            boxes[:, 2:] += boxes[:, :2]
            boxes = boxes.tolist()
            boxes_dict[imname] = boxes
        del Images
        
        pool = scipy.io.loadmat(os.path.join(self.CUHK_SYSU['dataset'], 'dataset/annotation/pool.mat'))
        test = pool['pool'].squeeze()
        test_imgs_name = [str(a[0]) for a in test]
        test_id = [int(im[1:-4]) for im in test_imgs_name]
        del pool
        
        train_imgs_name = [imname for imname in all_imgs_name if imname not in test_imgs_name]
        
        PersonSet = scipy.io.loadmat(os.path.join(self.CUHK_SYSU['dataset'], 'dataset/annotation/test/train_test/Train.mat'))
        Person = PersonSet['Train'][:, 0]
        nPerson = Person.shape[0]
        train_person_id = []
        for i in range(nPerson):
            person = Person[i][0, 0]
            person_id = int(person[0][0][1:])
            train_person_id.append(person_id)
            
        PersonSet = scipy.io.loadmat(os.path.join(self.CUHK_SYSU['dataset'], 'dataset/annotation/Person.mat'))
        Person = PersonSet['Person']
        nPerson = Person.shape[1]
        track_list = []
        track_id = []
        all_person_instance = []
        bTrain_list = []
        pidentities = 0
        track_length = []
        train_id = []
        for i in range(nPerson):
            person = Person[0, i]
            pidentities += person[1][0, 0]
            person_id = int(person[0][0][1:])
            person_na = person[1][0][0]
            person_instance = []
            if person_id in train_person_id:
                bTrain = 1
            else:
                bTrain = 0
                continue
            track_length.append(person_na)
            indices = list(range(person_na))
            if person_na > self.cfg.max_volume:
                random.Random(i).shuffle(indices)
            for j in range(person_na):
                #[imageid, personid, x, y, w, h, ishard]
                image_id = int(person[2][0, j][0][0][1:-4])
                if bTrain:
                    train_id.append(image_id)
                elif image_id not in test_id:
                    continue
                box = person[2][0, j][1][0]
                ishard = person[2][0, j][2][0][0]
                person_instance.append([image_id, person_id, indices[j], *box.tolist(), ishard, bTrain])
            if len(person_instance) == 0:continue
            track_list.append(person_instance)
            track_id.append(person_id)
            bTrain_list.append(bTrain)
            all_person_instance+=person_instance
            
        gt = []
        gt_id = np.array(list(list(zip(*all_person_instance))[0]))
        gt_id_set = set(gt_id)
        gt_id_list = []
        train_id = list(set(train_id))
        for i in gt_id_set:
            imname = 's' + str(i) + '.jpg'
            if imname in train_imgs_name:
                bTrain = 1
            else:
                bTrain = 0
            #[xyxy]
            boxes = boxes_dict[imname]
            idx = np.where(gt_id == i)
            items = [all_person_instance[j] for j in idx[0].tolist()]
            new_items = []
            for q_box in boxes:
                x1, y1, x2, y2 = q_box
                if (y2 - y1 < 60) or (x2 - x1) < 30:continue
                match = False
                for item in items:
                    #xywh
                    box = np.array(item[3:7])
                    box[2:] += box[:2]
                    if self._compute_iou(q_box, box) > 0.5:
                        match = True
                        break
                if not match:
                    box = np.array(q_box)
                    box[2:] -= box[:2]
                    new_items.append([i, -3 + bTrain, -1, *box.tolist(), 0, bTrain])
            items += new_items
                
                
              
            gt.append(items)
            gt_id_list.append(i)
            
            
    
        gt_id_array = np.array(gt_id_list)
        self.CUHK_SYSU.update({'gt' : gt,
                                        'gt_id_array' : gt_id_array,
                                        'track_list' : track_list,
                                        'track_id' : track_id,
                                        'bTrain' : bTrain_list,
                                        'train_img_id' : train_id,
                                        'test_img_id' : test_id,
                })
    
    def CUHK_SYSU_worker(self, images_instance, images):
        while True:
            while len(images) == 0:
                time.sleep(0.01)
            while len(images_instance) > 20:
                time.sleep(0.01)
            image = images.pop()
            boxes, frame_path = image
            img1 = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            images_instance.append([img1, boxes])
    
    def CUHK_SYSU_get_images(self, image_type = 'all'):
        gt = self.CUHK_SYSU['gt']
        gt_id_array = self.CUHK_SYSU['gt_id_array']
        frame_ids = []
        self.CUHK_SYSU_images = []
        for track in self.CUHK_SYSU['track_list']:
            lt = len(track)
            for i in range(lt):
                frameID = track[i][0]
                if frameID in frame_ids:continue
                frame_ids.append(frameID)
                idx1 = np.where(gt_id_array == frameID)[0]
                boxes = [x[3 : 7] + [self.id_map(x[1]) if x[1] >= 0 else x[1]] + x[2:3] + x[-1:] for x in gt[idx1[0]][:self.cfg.MAX_TRACKING_VOLUME]]
                img_info = [boxes, os.path.join(self.CUHK_SYSU['dataset'], 'dataset/Image/SSM/s%d.jpg'%(track[i][0]))]
                self.CUHK_SYSU_images.append(img_info)
                self.images.append(img_info)
                
    def PRW_parser(self,):
        self.PRW_images = []
        frame_train = scipy.io.loadmat(os.path.join(self.PRW['dataset'], 'frame_train.mat'))
        frame_indices = frame_train['img_index_train']
        for index, im_name in enumerate(frame_indices[:, 0]):
            mat = scipy.io.loadmat(os.path.join(self.PRW['dataset'], 'annotations', im_name[0] + '.jpg.mat'))
            boxes_ids = mat[list(mat.keys())[-1]].tolist()
            boxes = [x[1:] + [self.id_map('prw'+str(int(x[0]))) if x[0] >= 0 else x[0]] + [0, 1] for x in boxes_ids]
            img_info = [boxes, os.path.join(self.PRW['dataset'], 'frames', im_name[0]+'.jpg')]
            self.PRW_images.append(img_info)
            self.images.append(img_info)
            
        
    def CUHK_SYSU_generator(self, data_type = 'all', workers = 6):
        
        instances = []
        thread_instances = [[] for i in range(workers)]
        thread_pairs = [[] for i in range(workers)]
        [_thread.start_new_thread(self.CUHK_SYSU_worker, (thread_instances[i], thread_pairs[i])) for i in range(workers)]
        images = self.images
        
        while True:
            tmp_images = copy.deepcopy(images)
            #generate data.Random(len(images))
            random.shuffle(tmp_images)
            while len(tmp_images) > 0:
                for thread_pair in thread_pairs:
                    if len(tmp_images) == 0:break
                    while len(thread_pair) < 10:
                        thread_pair.append(tmp_images.pop())
                        if len(tmp_images) == 0:break
                for instance in thread_instances:
                    while len(instance) > 0:
                        instances.append(instance.pop())
                while len(instances) > 0:
                    yield instances.pop()
