#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:59:26 2019

@author: hu
"""
import keras.layers as KL
import tensorflow as tf
from functools import wraps

reg = [None]

def resnet_graph(inputs, architecture = "resnet101", stage5=True, training=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    reg[0] = None
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(inputs)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = KL.BatchNormalization(name='bn_conv1')(x, training=training)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D(3, strides=2, padding="same")(x)
    # Stage 2
    x = __conv_block(x, [64, 64, 256], set_shortcut = True, stage=2, block='a', training=training)
    x = __conv_block(x, [64, 64, 256], stage=2, block='b', training=training)
    C2 = x = __conv_block(x, [64, 64, 256], stage=2, block='c', training=training)
    # Stage 3
    x = __conv_block(x, [128, 128, 512], set_shortcut = True, strides = (2, 2), stage=3, block='a', training=training)
    x = __conv_block(x, [128, 128, 512], stage=3, block='b', training=training)
    x = __conv_block(x, [128, 128, 512], stage=3, block='c', training=training)
    C3 = x = __conv_block(x, [128, 128, 512], stage=3, block='d', training=training)
    # Stage 4
    x = __conv_block(x, [256, 256, 1024], set_shortcut = True, strides = (2, 2), stage=4, block='a', training=training)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = __conv_block(x, [256, 256, 1024], stage=4, block=chr(98 + i), training=training)
    C4 = x
    # Stage 5
    if stage5:
        x = __conv_block(x, [512, 512, 2048], set_shortcut = True, strides = (2, 2), stage=5, block='a', training=training)
        x = __conv_block(x, [512, 512, 2048], stage=5, block='b', training=training)
        C5 = x = __conv_block(x, [512, 512, 2048], stage=5, block='c', training=training)
    else:
        C5 = None

    return [C2, C3, C4, C5]

@wraps(KL.Conv2D)
def __DarknetConv2D1(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': reg[0]}
    darknet_conv_kwargs.update(kwargs)
    return KL.Conv2D(*args, **darknet_conv_kwargs)


def mrcnn_proposal_map(inputs):
    reg[0] = None
    C2, C3, C4, C5 = inputs
    P5 = __DarknetConv2D1(256, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        __DarknetConv2D1(256, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        __DarknetConv2D1(256, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        __DarknetConv2D1(256, (1, 1), name='fpn_c2p2')(C2)])
    
    #reid feature map
    C3_enhance = KL.Concatenate()([P3, C3])
    C3_enhance = KL.Lambda(lambda x : tf.stop_gradient(x))(C3_enhance)
    C3 = KL.Lambda(lambda x : tf.stop_gradient(x))(C3)
    
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = __DarknetConv2D1(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = __DarknetConv2D1(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = __DarknetConv2D1(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = __DarknetConv2D1(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    
    return [P2, P3, P4, P5, P6, C3, C3_enhance]


def __conv_block(inputs, filters, stage, block, strides=(1, 1), set_shortcut = False, use_bias=True, training=None):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = __DarknetConv2D1(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(inputs)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x, training=training)
    x = KL.Activation('relu')(x)

    x = __DarknetConv2D1(nb_filter2, 3, padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x, training=training)
    x = KL.Activation('relu')(x)

    x = __DarknetConv2D1(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x, training=training)

    if set_shortcut:
        shortcut = __DarknetConv2D1(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias)(inputs)
        shortcut = KL.BatchNormalization(name=bn_name_base + '1')(shortcut, training=training)
    else:
        shortcut = inputs
    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x
