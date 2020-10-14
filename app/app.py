#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 08:15:02 2020

@author: hu
"""
import os
import sys
import wx
import cv2
import glob
import pickle
import numpy as np

parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parpath)
from utils.shell import MODELS, ANCHORS
from tools.config import Config
from tools.DataAugmentation import DataAugmentation as DA
sys.path.remove(parpath)

from wx.lib.pubsub import pub
import wx.lib.statbmp as statbmp
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

class SelectCtrl(statbmp.GenStaticBitmap):
    def __init__(self, parent, ID, bitmap, name = 'mybitmap'):
        statbmp.GenStaticBitmap.__init__(self, parent, ID, bitmap, name = name)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.bLBtnD = False
        self.Bind(wx.EVT_MOTION, self.onMouseMove)
        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftBtnDown)
        self.Bind(wx.EVT_LEFT_UP, self.onLeftBtnUP)
        self.Bind(wx.EVT_RIGHT_UP, self.onRightBtnUP)
        self.timer = wx.Timer(self)
        self.interval = 100
        self.Bind(wx.EVT_TIMER, self.onTimer, self.timer)
    
    def getQueryRect(self,):
        if hasattr(self, 'queryRect'):
            return self.queryRect
        else:
            return None
    
    def getSelectRect(self,):
        if hasattr(self, 'corner1') and hasattr(self, 'corner2'):
            x1, y1 = self.corner1
            x2, y2 = self.corner2
            self.selectRect = [min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)]
        
    def onLeftBtnDown(self, event):
        self.corner1 = self.corner2 = event.GetPosition()
        self.getSelectRect()
        self.bLBtnD = True
        self.timer.Start(self.interval)
    
    def onTimer(self, event):
        self.Refresh()
        
    def setQueryRect(self,):
        if hasattr(self, 'selectRect'):
            self.queryRect = self.selectRect
            del self.selectRect
     
    def onRightBtnUP(self, event):
        self.setQueryRect()
        self.Refresh()
        if hasattr(self, 'queryRect'):
            pub.sendMessage('query')
        
    
    def onLeftBtnUP(self, event):
        self.bLBtnD = False
        self.timer.Stop()
        self.Refresh()
        
    def onMouseMove(self, event):
        if self.bLBtnD:
            self.corner2 = event.GetPosition()
            self.getSelectRect()
    
    def AcceptsFocus(self):
        return True
    
    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        if self._bitmap:
            dc.DrawBitmap(self._bitmap, 0, 0, True)
            dc.SetBrush(wx.Brush("red", style = wx.BRUSHSTYLE_TRANSPARENT))
            #draw query rect
            if hasattr(self, 'queryRect'):
                dc.SetPen(wx.Pen("blue", width = 5,style = wx.PENSTYLE_SOLID))
                dc.DrawRectangle(*list(self.queryRect))
                dc.SetPen(wx.Pen("yellow", width = 1,style = wx.PENSTYLE_SOLID))
                dc.DrawRectangle(*list(self.queryRect))
            #draw detection
            if hasattr(self, 'detRects'):
                dc.SetPen(wx.Pen("white", width = 1,style = wx.PENSTYLE_SOLID))
                for rect in self.detRects:
                    dc.DrawRectangle(*list(rect))
            if hasattr(self, 'selectRect'):
                dc.SetPen(wx.Pen('yellow', width = 1, style = wx.PENSTYLE_SOLID))
                dc.DrawRectangle(*list(self.selectRect))


class myPanel(wx.Panel):
    def __init__(self, parent, image):
        '''
        image:opencv BGR image
        '''
        wx.Panel.__init__(self, parent)
        #BGR2RGB
        self.setImage(image)
        self.setlayout()
    
    def fitAll(self,):
        ih, iw, _ = self.image.shape
        self.fit_image = cv2.resize(self.image, (int(iw * self.ratio), int(ih * self.ratio)), interpolation = cv2.INTER_LINEAR)
        
    def getQuery(self,):
        self.query = list(np.array(self.selectCtrl.getQueryRect()) / self.ratio)
        return self.query
    
    def setlayout(self):
        self.mySizer = wx.BoxSizer(wx.VERTICAL)
        self.selectCtrl = SelectCtrl(self, wx.ID_ANY, wx.BitmapFromBuffer(*self.fit_image.shape[:2][::-1], self.fit_image))
        self.mySizer.Add(self.selectCtrl, 0, wx.ALL|wx.CENTER, 5)
        self.SetSizer(self.mySizer)
    
    def setImage(self, image):
        self.mSize = min(wx.DisplaySize()) - 256
        self.image = image[:, :, ::-1].astype(np.ubyte)
        self.ratio = min(self.mSize / max(self.image.shape), 1)
        self.fitAll()
        if hasattr(self, 'mySizer'):
            if self.mySizer.GetChildren():
                self.mySizer.Hide(0)
                self.mySizer.Remove(0)
                del self.selectCtrl
                self.selectCtrl = SelectCtrl(self, wx.ID_ANY, wx.BitmapFromBuffer(*self.fit_image.shape[:2][::-1], self.fit_image))
                self.mySizer.Add(self.selectCtrl, 0, wx.ALL|wx.CENTER, 5)
                self.Fit()
    
class MyFrame(wx.Frame):
    def __init__(self, detector, det = None, name = 'SelectFrame'):
        wx.Frame.__init__(self, None, title=name)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        self.config = Config(detector)
        self.DA = DA('validation', self.config)
        self.fmts = ['*.jpg', '*.png', '*.bmp','*.jpeg']
        self.menuBar()
        self.Center()
        self.query = {}
        self.gallery = {}
        pub.subscribe(self.queryReid, 'query')
        
    def create_anchors(self,):
        '''
        only mask rcnn needs it
        '''
        if self.config.M == 'mrcnn' and not hasattr(self, 'anchors'):
            self.anchors = ANCHORS(self.config)
    def menuBar(self,):
        _gallery = wx.Menu()
        _gallery_Img = _gallery.Append(wx.ID_ANY, 'Gallery Image', 'Select a single image')
        _gallery_Dir = _gallery.Append(wx.ID_ANY, 'Gallery Dir', 'Select a Directory')
        _gallery_DirR = _gallery.Append(wx.ID_ANY, '&Gallery_Dir(R)...\tCtrl-G', 'Select a Directory Recursively')
        _gallery_clear = _gallery.Append(wx.ID_ANY, 'Clear Gallery', 'Clear gallery')
        _file = wx.Menu()
        _file_query = _file.Append(wx.ID_ANY, '&Query...\tCtrl-Q', 'Open query image')
        _file_gallery = _file.AppendSubMenu(_gallery, '&Gallery')
        _file_search = _file.Append(wx.ID_ANY, '&Search...\tCtrl-S', 'Search')
        _file.AppendSeparator()
        _file_save_gallery = _file.Append(wx.ID_ANY, 'Save Gallery', 'Save Gallery')
        _file_load_gallery = _file.Append(wx.ID_ANY, 'Load Gallery', 'load gallery from file')
        _file.AppendSeparator()
        _file_exit = _file.Append(2, '&Exit...\tCtrl-E', 'Exit program')
        _load = wx.Menu()
        _load_model = _load.Append(wx.ID_ANY, '&Load...\tCtrl-L', 'Load model')
        
        menuBar = wx.MenuBar()
        menuBar.Append(_file, "&File")
        menuBar.Append(_load, '&Load')
        
        # Give the menu bar to the frame
        self.SetMenuBar(menuBar)

        self.Bind(wx.EVT_MENU, self.onQuery, _file_query)
        
        self.Bind(wx.EVT_MENU, self.onGalleryImg, _gallery_Img)
        self.Bind(wx.EVT_MENU, self.onGalleryDir, _gallery_Dir)
        self.Bind(wx.EVT_MENU, self.onGalleryDirR, _gallery_DirR)
        self.Bind(wx.EVT_MENU, self.onClear, _gallery_clear)
        self.Bind(wx.EVT_MENU, self.onSearch, _file_search)
        
        self.Bind(wx.EVT_MENU, self.onSaveGallery, _file_save_gallery)
        self.Bind(wx.EVT_MENU, self.onLoadGallery, _file_load_gallery)
        
        self.Bind(wx.EVT_MENU, self.onExit, _file_exit)
        
        self.Bind(wx.EVT_MENU, self.onLoad, _load_model)
    
    def onLoadGallery(self, event):
        f = open('gallery.pkl', 'rb')
        gallery = pickle.load(f, encoding='latin1')
        f.close()
        for pimage in gallery:
            if pimage in self.gallery:continue
            self.gallery[pimage] = gallery[pimage]
            
    def onSaveGallery(self, event):
        f = open('gallery.pkl', 'wb')
        pickle.dump(self.gallery, f)
        f.close()
    
    def onExit(self, event):
        self.Close(True)
    def queryReid(self,):
        if not hasattr(self, 'panel'):return
        roi = self.panel.getQuery()
        if roi is None:return False
        if 'path' in self.query:
            img = cv2.imread(self.query['path'])
            input_img, input_box, _, meta = self.DA(img, [roi])
            feat = self.feature_extraction(input_img, input_box, meta)[0]
            self.query['feature'] = feat[0]
            self.query['roi'] = 'roi'
    
    def onSearch(self, event, top_n = 20, fnLen = 30):
        if 'feature' not in self.query:
            print('no query')
        query = self.query['feature']
        infolen = 0
        self.toplist = []
        for i, pimage in enumerate(self.gallery):
            features = self.gallery[pimage]['features']
            detection = self.gallery[pimage]['detection']
            if len(detection) == 0:continue
            sims = self.get_sims(features, query, _eval = True)
            maxids = np.argmax(sims)
            for _ in range(infolen):print("\x1b[1A\x1b[2K", end = '')
            self.toplist.append([pimage, sims[maxids], detection[maxids]])
            self.toplist.sort(key = lambda x : x[1], reverse = True)
            print("%d|%d>>%s"%(i+1, len(self.gallery), pimage))
            for item in self.toplist[:top_n]:
                print('\x1b[0;31;47m' + '%.6f\t'%item[1] + '\x1b[0m', '...%30s\t'%item[0][-fnLen:], item[2].astype(np.int32))
            infolen = len(self.toplist[:top_n]) + 1
            
    def onQuery(self, event):
        fDlg = wx.FileDialog(self, 'select query image.', 
                             wildcard = 'image (*.jpg;*.png;*.bmp;*.jpeg)|*.jpg;*.png;*.bmp;*.jpeg',
                             style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
                             )
        if fDlg.ShowModal() == wx.ID_CANCEL:
            return
        pimage = fDlg.GetPath()
        self.query['path'] = pimage
        print('get query image: %s'%pimage)
        image = cv2.imread(pimage)
        if not hasattr(self, 'panel'):
            self.panel = myPanel(self, image)
            self.sizer.Add(self.panel, 1, wx.EXPAND)
        else:
            self.panel.setImage(image)
        self.sizer.Fit(self)
        self.Fit()
        self.Center()
    
    def onClear(self,):
        self.gallery = {}
    
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
    
    def feature_extraction(self, input_img, input_box, meta):
        if not hasattr(self, 'model'):
            self.load()
        
        self.create_anchors()
        
        if self.config.M == 'mrcnn':
            feats, _, _, _, features, detection, _ = self.model.predict([np.stack([input_img]), np.stack([input_box]), np.stack([self.anchors.get_anchors(input_img.shape)])])
        elif self.config.M == 'dla_34':
            feats, det_features, det, _ = self.reid_model.predict([np.stack([input_img]), np.stack([input_box])])
        else:
            feats, _, _, _, features, detection, _ = self.model.predict([np.stack([input_img]), np.stack([input_box])])
            detection = self.DA.unmold(detection[0], meta)
        return feats[0], features[0], detection
    
    def load(self,):
        if hasattr(self, 'model'):
            del self.model
        self.model = MODELS(config = self.config).load_model()
    
    def onLoad(self, event):
        self.load()
    
    def extract(self, pimage):
        if pimage in self.gallery:
            #print('image existed:', pimage)
            return
        if not hasattr(self, 'model'):
            self.load()
        print('extracting>>%s'%pimage)
        img = cv2.imread(pimage)
        input_img, input_box, _, meta = self.DA(img, [])
        _, features, detection = self.feature_extraction(input_img, input_box, meta)
        self.gallery[pimage] = {'features':features, 'detection':detection}
    
    def onGalleryImg(self, event):
        fDlg = wx.FileDialog(self, 'select query image.', 
                             wildcard = 'image (*.jpg;*.png;*.bmp;*.jpeg)|*.jpg;*.png;*.bmp;*.jpeg',
                             style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
                             )
        if fDlg.ShowModal() == wx.ID_CANCEL:
            return
        pimage = fDlg.GetPath()
        self.extract(pimage)
    
    def parseGalleryPath(self, gallery_path, mode = ''):
        paths = []
        for fmt in self.fmts:paths+=glob.glob(os.path.join(gallery_path, fmt))
        if mode == 'r':
            for item in os.listdir(gallery_path):
                path = os.path.join(gallery_path, item)
                if os.path.isdir(path):
                    paths+=self.parseGalleryPath(path, mode = mode)
        return list(set(paths))
    
    def onGalleryDir(self, event):
        pDlg = wx.DirDialog(self, "select a directory",
                           style=wx.DD_DEFAULT_STYLE)
        
        if pDlg.ShowModal() == wx.ID_CANCEL:
            return
        gallery_path = pDlg.GetPath()
        gallery_images = list(set(self.parseGalleryPath(gallery_path)))
        for pimage in gallery_images:
            self.extract(pimage)
            
    
    def onGalleryDirR(self, event):
        pDlg = wx.DirDialog(self, "select a directory",
                           style=wx.DD_DEFAULT_STYLE)
        
        if pDlg.ShowModal() == wx.ID_CANCEL:
            return
        gallery_path = pDlg.GetPath()
        gallery_images = list(set(self.parseGalleryPath(gallery_path, mode = 'r')))
        for pimage in gallery_images:
            self.extract(pimage)
    
import sys, getopt
def main(argv):
    M = 'yolov3'
    gpu = '0'
    try:
        opts, args = getopt.getopt(argv[1:], 'hm:g:', ['m=', 'gpu='])
    except getopt.GetoptError:
        print(argv[0] + ' -m <M> -g <gpu>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(argv[0] + ' -m <M> -g <gpu>')
        elif opt in ['-m', '--M']:
            M = arg
        elif opt in ['-g', '--gpu']:
            gpu = arg
    
    
    print('gpu=[%s] model: [%s]'%(gpu, M))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    app = wx.App()
    myFrame = MyFrame(M)
    myFrame.Show()
    app.MainLoop()
    del app

if __name__ == "__main__":
    main(sys.argv)
    

    
    
    
