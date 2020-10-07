#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:04:53 2020

@author: hu
"""
import tensorflow as tf
import keras.layers as KL
import keras.models as KM
class DCNv2(KL.Layer):
    def __init__(self, filters, 
                 kernel_size,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (1, 1, 1, 1)
        self.dilation = (1, 1)
        self.deformable_groups = 1
        self.use_bias = use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer
        super(DCNv2, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name = 'kernel',
            shape = self.kernel_size + (int(input_shape[-1]), self.filters),
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            trainable = True,
            dtype = 'float32',
            )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name = 'bias',
                shape = (self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype='float32',
                )
        
        #[kh, kw, ic, 3 * groups * kh, kw]--->3 * groups * kh * kw = oc [output channels]
        self.offset_kernel = self.add_weight(
            name = 'offset_kernel',
            shape = self.kernel_size + (input_shape[-1], 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]), 
            initializer = 'zeros',
            trainable = True,
            dtype = 'float32')
        
        self.offset_bias = self.add_weight(
            name = 'offset_bias',
            shape = (3 * self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups,),
            initializer='zeros',
            trainable = True,
            dtype = 'float32',
            )
        self.ks = self.kernel_size[0] * self.kernel_size[1]
        self.ph, self.pw = (self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        self.phw = tf.constant([self.ph, self.pw], dtype = 'int32')
        self.patch_yx = tf.stack(tf.meshgrid(tf.range(-self.phw[1], self.phw[1] + 1), tf.range(-self.phw[0], self.phw[0] + 1))[::-1], axis = -1)
        self.patch_yx = tf.reshape(self.patch_yx, [-1, 2])
        super(DCNv2, self).build(input_shape)
        
        
    def call(self, x):
        #x: [B, H, W, C]
        #offset: [B, H, W, ic] convx [kh, kw, ic, 3 * groups * kh * kw] ---> [B, H, W, 3 * groups * kh * kw]
        offset = tf.nn.conv2d(x, self.offset_kernel, strides = self.stride, padding = 'SAME')
        offset += self.offset_bias
        bs, ih, iw, ic = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        #[B, H, W, 18], [B, H, W, 9]
        oyox, mask = offset[..., :2*self.ks], offset[..., 2*self.ks:]
        mask = tf.nn.sigmoid(mask)
        #[H, W, 2]
        grid_yx = tf.stack(tf.meshgrid(tf.range(iw), tf.range(ih))[::-1], axis = -1)
        #[1, H, W, 9, 2]
        grid_yx = tf.reshape(grid_yx, [1, ih, iw, 1, 2]) + self.phw + self.patch_yx
        #[B, H, W, 9, 2]
        grid_yx = tf.cast(grid_yx, 'float32') + tf.reshape(oyox, [bs, ih, iw, -1, 2])
        grid_iy0ix0 = tf.floor(grid_yx)
        up_limit = tf.cast(tf.stack([ih+1, iw+1]), 'float32')
        grid_iy1ix1 = tf.clip_by_value(grid_iy0ix0 + 1, 0, up_limit)
        #[B, H, W, 9, 1] * 2
        grid_iy1, grid_ix1 = tf.split(grid_iy1ix1, 2, axis = -1)
        grid_iy0ix0 = tf.clip_by_value(grid_iy0ix0, 0, up_limit)
        grid_iy0, grid_ix0 = tf.split(grid_iy0ix0, 2, axis = -1)
        grid_yx = tf.clip_by_value(grid_yx, 0, up_limit)
        #[B, H, W, 9, 4, 1]
        batch_index = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1, 1, 1, 1]), [1, ih, iw, self.ks, 4, 1])
        #[B, H, W, 9, 4, 2]
        grid = tf.reshape(tf.concat([grid_iy1ix1, grid_iy1, grid_ix0, grid_iy0, grid_ix1, grid_iy0ix0], axis = -1), [bs, ih, iw, self.ks, 4, 2])
        #[B, H, W, 9, 4, 3]
        grid = tf.concat([batch_index, tf.cast(grid, 'int32')], axis = -1)
        #[B, H, W, 9, 2, 2]
        delta = tf.reshape(tf.concat([grid_yx - grid_iy0ix0, grid_iy1ix1 - grid_yx], axis = -1), [bs, ih, iw, self.ks, 2, 2])
        #[B, H, W, 9, 2, 1] * [B, H, W, 9, 1, 2] = [B, H, W, 9, 2, 2]
        w = tf.expand_dims(delta[..., 0], axis = -1) * tf.expand_dims(delta[..., 1], axis = -2)
        #[B, H+2, W+2, C]
        x = tf.pad(x, [[0, 0], [int(self.ph), int(self.ph)], [int(self.pw), int(self.pw)], [0, 0]])
        #[B, H, W, 9, 4, C]
        map_sample = tf.gather_nd(x, grid)
        #([B, H, W, 9, 4, 1] * [B, H, W, 9, 4, C]).SUM(-2) * [B, H, W, 9, 1] = [B, H, W, 9, C]
        map_bilinear = tf.reduce_sum(tf.reshape(w, [bs, ih, iw, self.ks, 4, 1]) * map_sample, axis = -2) * tf.expand_dims(mask, axis = -1)
        #[B, H, W, 9*C]
        map_all = tf.reshape(map_bilinear, [bs, ih, iw, -1])
        #[B, H, W, OC]
        output = tf.nn.conv2d(map_all, tf.reshape(self.kernel, [1, 1, -1, self.filters]), strides = self.stride, padding = 'SAME')
        if self.use_bias:
            output += self.bias
        return output
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    

class BasicBlock():
    def __init__(self, planes, name, stride = 1, dilation = 1):
        self.channels = planes
        self.stride = stride
        self.dilation = (dilation, dilation)
        self.name = name
    def __call__(self, x, residual=None, training = None):
        if residual is None:
            residual = x
        padding = 'same'
        if self.stride == 2:
            x = KL.ZeroPadding2D(((1,0),(1,0)))(x)
            padding = 'valid'
        x = KL.Conv2D(self.channels, 3, name = self.name + '.conv1', strides = self.stride, padding = padding, bias = False)(x)
        x = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.bn1')(x, training = training)
        x = KL.Activation('relu')(x)
        x = KL.Conv2D(self.channels, 3, name = self.name + '.conv2', strides = 1, padding = 'same', bias = False)(x)
        x = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.bn2')(x, training = training)
        x = KL.Add()([x, residual])
        x = KL.Activation('relu')(x)
        return x
    
class Root():
    def __init__(self, out_channels, residual, name):
        self.channels = out_channels
        self.residual = residual
        self.name = name
    def __call__(self, x, training = None):
        children = x
        x = KL.Concatenate(axis = -1)(x)
        x = KL.Conv2D(self.channels, 1, name = self.name + '.conv', strides = 1, bias = False, padding='same')(x)
        x = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.bn')(x, training = training)
        if self.residual:
            x = KL.Add()([x, children[0]])
        x = KL.Activation('relu')(x)
        return x
        

class Tree():
    def __init__(self, levels, out_channels, name, stride = 1, level_root = False, root_dim = 0, root_kernel_size = 1, dilation = 1, root_residual = False):
        if root_dim == 0:
            self.root_dim = 2 * out_channels
        else:
            self.root_dim = root_dim
        self.level_root = level_root
        self.levels = levels
        self.root_kernel_size = root_kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.root_residual = root_residual
        self.name = name
        
    def tree1(self, x, residual, children, level, out_channels, stride = 1, dilation = 1, root_kernel_size=1, root_residual=False, training = None):
        if level == 1:
            x = BasicBlock(out_channels, self.name + '.tree1', stride, dilation = dilation)(x, residual, training = training)
        else:
            x = Tree(level-1, out_channels, self.name + '.tree1', stride, root_dim=0, root_kernel_size=root_kernel_size, dilation = dilation, root_residual = root_residual)(x, residual, children, training = training)
        return x
    
    def tree2(self, x, residual, children, level, out_channels, dilation = 1, root_kernel_size=1, root_residual=False, training = None):
        if level == 1:
            x = BasicBlock(out_channels, self.name + '.tree2', 1, dilation = dilation)(x, residual, training = training)
        else:
            x = Tree(level-1, out_channels, self.name + '.tree2', 1, root_dim=self.root_dim+out_channels, root_kernel_size=root_kernel_size, dilation = dilation, root_residual = root_residual)(x, residual, children, training = training)
        return x
            
    def downsample(self, x):
        if self.stride > 1:
            x = KL.MaxPooling2D(self.stride, self.stride)(x)
        return x
    def project(self, x, training = None):
        if self.in_channels != self.out_channels:
            x = KL.Conv2D(self.out_channels, 1, name = self.name + '.project.0', strides = 1, bias = False)(x)
            x = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.project.1')(x, training = training)
        return x
    
    def root(self, x, training = None):
        return Root(self.out_channels, self.root_residual, self.name + '.root')(x, training = training)
    
    def __call__(self, x, residual = None, children=None, training = None):
        self.in_channels = x.get_shape()[-1]
        if self.level_root: self.root_dim += self.in_channels
        children = [] if children is None else children
        bottom = self.downsample(x)
        residual = self.project(bottom, training = training)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual, None, self.levels, self.out_channels, self.stride, self.dilation, self.root_kernel_size, self.root_residual, training = training)
        if self.levels == 1:
            x2 = self.tree2(x1, None, None, self.levels, self.out_channels, self.dilation, self.root_kernel_size, self.root_residual, training = training)
            x = self.root([x2, x1, *children], training = training)
        else:
            children.append(x1)
            x = self.tree2(x1, None, children, self.levels, self.out_channels, self.dilation, self.root_kernel_size, self.root_residual, training = training)
        return x
    

def make_conv_level(x, num_filters, level, name, stride = 1, dilation = 1, training = None):
    for i in range(level):
        padding = 'same'
        if stride == 2:
            x = KL.ZeroPadding2D(((1,0),(1,0)))(x)
            padding = 'valid'
        x = KL.Conv2D(num_filters, 3, name = name + '.0', strides = stride if i == 0 else 1, \
                      padding = padding, \
                      bias = False)(x)
        x = KL.BatchNormalization(epsilon=1e-5, name = name + '.1')(x, training = training)
        x = KL.Activation('relu')(x)
    return x
        

class Base():
    def __init__(self, levels, channels, num_classes = 1000, residual_root = False):
        self.channels = channels
        self.residual_root = residual_root
        self.levels = levels
        self.num_classes = num_classes
        self.name = 'base'
        
    def __call__(self, x, training = None):
        y = []
        #base layer
        x = KL.Conv2D(self.channels[0], 7, name = self.name + '.base_layer.0', padding = 'same', bias = False)(x)
        x = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.base_layer.1')(x, training = training)
        x = KL.Activation('relu')(x)
        #level0
        x = make_conv_level(x, self.channels[0], self.levels[0], name = self.name + '.level0', training = training)
        y.append(x)
        #level1
        x = make_conv_level(x, self.channels[1], self.levels[1], name = self.name + '.level1', stride = 2, training = training)
        y.append(x)
        #level2
        for i in range(2, 6):
            x = Tree(self.levels[i], self.channels[i], name = self.name + '.level'+str(i), stride = 2, level_root = i > 2, root_residual = self.residual_root)(x, training = training)
            y.append(x)
        return y

class IDAUp():
    def __init__(self, o, channels, up_f, name):
        self.channels = channels
        self.o = o
        self.up_f = up_f
        self.name = name
    
    def DepthwiseConv2DTranspose(self, x, kernel_size, name = None, pad = 1, strides = 1, use_bias = False):
        #keras and tensorflow grouped transpose convolutinoal unsupported
        up_x = KL.Lambda(lambda x : tf.reshape(tf.transpose(tf.reshape(tf.concat([x, tf.tile(tf.zeros_like(x), [1, 1, 1, strides*strides-1])], axis = -1), [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], strides, strides, x.shape[3]]), [0, 1, 3, 2, 4, 5]), [tf.shape(x)[0], tf.shape(x)[1]*strides, tf.shape(x)[2]*strides, x.shape[3]]))(x)
        up_x = KL.ZeroPadding2D(((pad, pad - (strides - 1)), (pad, pad - (strides - 1))))(up_x)
        return  KL.DepthwiseConv2D((kernel_size, kernel_size), name = name, use_bias = use_bias)(up_x)
        
    def __call__(self, x, startp, endp, training = None):
        for i in range(startp + 1, endp):
            x[i] = DCNv2(self.o, 3, name = self.name + '.proj_%d.conv'%(i-startp))(x[i])
            x[i] = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.proj_%d.actf.0'%(i-startp))(x[i], training = training)
            x[i] = KL.Activation('relu')(x[i])
            x[i] = self.DepthwiseConv2DTranspose(x[i], kernel_size = self.up_f[i-startp]*2, \
                                                 name = self.name + '.up_%d'%(i-startp), \
                                                 pad = self.up_f[i-startp] * 2 - 1 - self.up_f[i-startp]//2, \
                                                 strides = self.up_f[i-startp])
            
            x[i] = KL.Add()([x[i], x[i-1]])
            x[i] = DCNv2(self.o, 3, name = self.name + '.node_%d.conv'%(i-startp))(x[i])
            x[i] = KL.BatchNormalization(epsilon=1e-5, name = self.name + '.node_%d.actf.0'%(i-startp))(x[i], training = training)
            x[i] = KL.Activation('relu')(x[i])
            
import numpy as np    
class DLAUp():
    def __init__(self, startp, channels, scales):
        self.startp = startp
        self.in_channels = channels
        self.channels = list(channels)
        self.scales = np.array(scales, dtype = int)
        self.name = 'dla_up'
        
    def __call__(self, x, training = None):
        out = [x[-1]]
        for i in range(len(x) - self.startp - 1):
            j = -i - 2
            IDAUp(self.channels[j], self.in_channels[j:], self.scales[j:]//self.scales[j], name = self.name + '.ida_%d'%i)(x, len(x)-i-2, len(x), training = training)
            out.insert(0, x[-1])
            self.scales[j+1:] = self.scales[j]
            self.in_channels[j+1:] = [self.channels[j] for _ in self.channels[j+1:]]
        return out
            
class DLASeg():
    def __init__(self, heads, down_ratio, final_kernel, last_level, head_conv, out_channel=0):
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.heads = heads
        self.head_conv = head_conv
        self.out_channel = out_channel
        self.final_kernel = final_kernel
        self.down_ratio = down_ratio
        self.name = 'dlaseg'
        
    def detection(self, hm, wh, ids, reg, num_classes = 1, K = 128):
        hm = tf.nn.sigmoid(hm)
        hmax = tf.nn.max_pool2d(hm, 3, 1, 'SAME')
        km = tf.where(tf.equal(hmax, hm), hmax, tf.zeros_like(hmax))
        bs, h, w, c = km.shape
        bs = tf.shape(km)[0]   
        
        scores, indices = tf.nn.top_k(tf.reshape(km, [bs, -1]), K)
        
        classes = indices // (h * w)
        y, x = (indices % (h * w)) // w, (indices % (h * w)) % w
        batch_index = tf.reshape(tf.range(bs), [bs, 1]) * tf.ones_like(classes) 
        index = tf.stack([batch_index, y, x, classes], axis = -1)
        kwh = tf.gather_nd(tf.reshape(wh, [bs, h, w, num_classes, -1]), tf.reshape(index, [-1, 4]))
        kid = tf.gather_nd(tf.reshape(ids, [bs, h, w, num_classes, -1]), tf.reshape(index, [-1, 4]))
        krg = tf.gather_nd(tf.reshape(reg, [bs, h, w, num_classes, -1]), tf.reshape(index, [-1, 4]))
        
        kid = tf.nn.l2_normalize(kid, axis = -1)
        x, y = tf.reshape(tf.cast(x, 'float32'), [-1, 1]) + krg[:, 0:1], tf.reshape(tf.cast(y, 'float32'), [-1, 1]) + krg[:, 1:2]
        
        bboxes = tf.concat([x - kwh[:, 0:1]/2, \
                            y - kwh[:, 1:2]/2, \
                            x + kwh[:, 0:1]/2, \
                            y + kwh[:, 1:2]/2, \
                            tf.reshape(scores, [-1, 1]), \
                            tf.reshape(tf.cast(classes, 'float32'), [-1, 1]), \
                            tf.reshape(kid, [-1, tf.shape(kid)[-1]])], axis = -1)
        bboxes = tf.reshape(bboxes, [bs, -1, 4 + 2 + tf.shape(kid)[-1]])
        return bboxes
        
        
        
    def __call__(self,x, training = None):
        _base = Base([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512])
        x = _base(x, training = training)
        channels = _base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        x = DLAUp(self.first_level, channels[self.first_level:], scales)(x, training = training)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i])
            
        reid_feature_map = y[-1]
        if self.out_channel == 0:
            out_channel = channels[self.first_level]
        IDAUp(out_channel, channels[self.first_level:self.last_level], [2 ** i for i in range(self.last_level - self.first_level)], self.name + '.idaup')(y, 0, len(y), training = training)
        outputs = []
        for head in self.heads:
            classes = self.heads[head]
            if self.head_conv > 0:
                x = KL.Conv2D(self.head_conv, 3, name = self.name +'.' + head + '.conv1', padding = 'same', activation='relu')(y[-1])
                x = KL.Conv2D(classes, self.final_kernel, name = self.name + '.' + head + '.conv2', padding = 'same')(x)
            else:
                x = KL.Conv2D(classes, self.final_kernel, name = self.name + '.' + head + '.conv', padding = 'same')(y[-1])
            outputs.append(x)
        #detection = KL.Lambda(lambda x : self.detection(*x))(outputs)
        output = [KL.Lambda(lambda x : tf.stop_gradient(x))(reid_feature_map)]
        return output
        
def DLA_MODEL():
    inputs = KL.Input(shape = [608, 1088, 3])
    outputs = DLASeg(heads = {'hm': 1, 'wh': 2, 'id': 512, 'reg': 2},\
                     down_ratio = 4,\
                     final_kernel = 1,\
                     last_level = 5,\
                     head_conv=256
                     )(inputs)
    return KM.Model(inputs, outputs)

from tools.load_weights import load_weights_by_name
if __name__ == '__main__':
    model = DLA_MODEL()
    load_weights_by_name(model, 'dla_34.h5')
    
    def letterbox(img, height=608, width=1088,
                  color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh
    
    import cv2
    import time
    while 1:
        start = time.time()
        img0 = cv2.imread('test.jpg')
        img, ratio, dw, dh = letterbox(img0, height=608, width=1088)
        
        img = img[:, :, ::-1].astype(np.float32)/255.
        dets = model.predict([[img]])
        
        dets = dets[0]
        
        min_confidence = 0.42
        
        remain_indices = dets[:, 4] > min_confidence
        dets = dets[remain_indices]
        
        boxes, scores, features = dets[:, :4], dets[:, 4], dets[:, 6:]
        
        scale_x, scale_y = img0.shape[0] / img.shape[0], img0.shape[1] / img.shape[1]               
        
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * 4 - dw)/ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * 4 - dh)/ratio
        print(time.time() - start)
