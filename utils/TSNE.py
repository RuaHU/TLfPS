#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:10:06 2020

@author: hu
"""

import os
import shutil
import sys, getopt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#from cuml.manifold import TSNE
#from cuml.dask.decomposition import PCA
import seaborn as sns
import numpy as np
parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parpath)
from tools.config import Config
from tools.DataAugmentation import DataAugmentation as DA
sys.path.remove(parpath)
import pickle
import matplotlib.pyplot as plt
import cv2

class EVALUATION():
    def __init__(self, dataset, detector = 'yolov3', experiment_name = 'default', overwrite = False):
        self.detector = detector
        self.dataset = dataset
        self.overwrite = overwrite
        self.experiment_name = experiment_name
        self.checkdir()
        self.config = Config(detector)
        self.DA = DA('validation', self.config)
        
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
    
    def TSNE(self, draw_image = True):
        fTSNE_features = os.path.join(parpath, 'experiment_results', self.experiment_name, '%s_%s_TSNE_features.pkl'%(self.dataset, self.config.M))
        fTSNE_id = os.path.join(parpath, 'experiment_results', self.experiment_name, '%s_%s_TSNE_id.pkl'%(self.dataset, self.config.M))
        fTSNE_imgs = os.path.join(parpath, 'experiment_results', self.experiment_name, '%s_%s_TSNE_imgs.pkl'%(self.dataset, self.config.M))
        f = open(fTSNE_features, 'rb')
        TSNE_features = pickle.load(f)
        f = open(fTSNE_id, 'rb')
        TSNE_id = pickle.load(f)
        f = open(fTSNE_imgs, 'rb')
        TSNE_imgs = pickle.load(f)
        f.close()
        TSNE_features = TSNE_features[TSNE_id > 0, :]
        TSNE_imgs = np.array(TSNE_imgs)
        TSNE_imgs = TSNE_imgs[TSNE_id > 0]
        TSNE_id = TSNE_id[TSNE_id > 0]
        print('TSNE_features shape:', TSNE_features.shape)
        print('TSNE_id shape:', TSNE_id.shape)
        print('id number:', len(set(TSNE_id.tolist())))
        os.path.join(parpath, 'experiment_results', self.experiment_name, '%s_%s_Embeddding.pkl'%(self.dataset, self.config.M))
        fEmbedding = os.path.join(parpath, 'experiment_results', self.experiment_name, '%s_%s_Embeddding.pkl'%(self.dataset, self.config.M))
        if self.dataset == 'cuhk':
            #sklearn can outputs correct results, while cuml not, the two solutions different a lot
            tsne = TSNE(n_iter=5000, verbose = True)
        else:
            #cuml works fine for PRW, sklearn not
            tsne = TSNE(n_iter=8000, verbose = True)
        '''
        if os.path.exists(fEmbedding):
            f = open(fEmbedding, 'rb')
            Embedding = pickle.load(f)
            print('Embedding shape:', Embedding.shape)
        else:
            Embedding = tsne.fit_transform(TSNE_features)
        '''
        Embedding = tsne.fit_transform(TSNE_features)
        f = open(fEmbedding, 'wb')
        pickle.dump(Embedding, f)
        f.close()
        
        collect = {}
        for i, index in enumerate(TSNE_id):
            if index in collect:
                collect[index].append([Embedding[i, :], TSNE_imgs[i]])
            else:
                collect[index] = [[Embedding[i, :], TSNE_imgs[i]]]
        
        palette = sns.color_palette("bright", len(set(collect)))
        for i in range(len(palette)):
            palette[i] = tuple(np.random.randint(0, 256, 3).astype(np.float32)/255)
            
        '''Image version'''
        if draw_image:
            print('max:', np.abs(Embedding).max())
            Embedding*=10
            image_radius = int(((np.abs(Embedding).max()//1000)+1)*1000)
            sparse_factor = 5
            l = image_radius*sparse_factor
            canvas = np.ones([image_radius*sparse_factor*2, image_radius*sparse_factor*2, 3], dtype = 'uint8') * 255
            for i, key in enumerate(collect.keys()):
                color = (np.array(palette[i])*255).astype(np.uint8)
                identity_set = collect[key]
                for identity in identity_set:
                    pos, img = identity
                    x, y = pos.astype(np.int32) * sparse_factor
                    ih, iw = img.shape[:2]
                    if img.size == 0:continue
                    img = cv2.resize(img, (64, 128), interpolation = cv2.INTER_LINEAR)
                    if (y-64+l) < 0 or (y+64+l) > image_radius*sparse_factor*2 or (x-32+l) < 0 or (x+32+l) > image_radius*sparse_factor*2:
                        print(pos, x, y)
                        continue
                    canvas[(y-64-4+l):(y+64+4+l), (x-32-4+l):(x+32+4+l)] = color
                    canvas[(y-64+l):(y+64+l), (x-32+l):(x+32+l), :] = img[::-1, :, :]
            canvas = canvas[::-1, :, :]
            half_canvas = canvas[::2, ::2, :]
            cv2.imwrite('%s.jpg'%self.dataset, half_canvas)
            cv2.imwrite('%s.bmp'%self.dataset, half_canvas)
            
        length = []
        for i in collect.keys():
            length.append(len(collect[i]))
        
        length.sort()
        
        valid_index = []
        for i in collect.keys():
            if len(collect[i]) >= 0:
                valid_index.append(i)

        new_embedding = np.array([Embedding[i] for i, j in enumerate(TSNE_id) if j in valid_index])
        new_ids = np.array([TSNE_id[i] for i, j in enumerate(TSNE_id) if j in valid_index])
        
        sns.scatterplot(new_embedding[:,0], new_embedding[:,1], hue=new_ids, palette=palette, s=10, legend=False)
        plt.show()
    
def main(argv):
    M = 'yolov3'
    path = 'prw'
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
    
    assert path in ['prw', 'cuhk']
    print('dataset: [%s] gpu: [%s] model: [%s] experiment name: [%s]'%(path, gpu, M, experiment_name))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu  
    
    evaluation = EVALUATION(dataset = path, detector = M, experiment_name = experiment_name, overwrite = overwrite)
    evaluation.TSNE()
    
if __name__ == "__main__":
   main(sys.argv)
