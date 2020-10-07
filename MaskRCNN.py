#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:59:26 2019

@author: hu
"""
import keras.layers as KL
import tensorflow as tf
import keras.engine as KE
import keras.models as KM
import numpy as np
from BaseNet import *
class MaskRCNN:
    def __init__(self, config):
        self.cfg = config

    def resnet_graph(self, inputs, architecture = "resnet101", stage5=True, training=True):
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
        x = mconv_block(x, [64, 64, 256], set_shortcut = True, stage=2, block='a', training=training)
        x = mconv_block(x, [64, 64, 256], stage=2, block='b', training=training)
        C2 = x = mconv_block(x, [64, 64, 256], stage=2, block='c', training=training)
        # Stage 3
        x = mconv_block(x, [128, 128, 512], set_shortcut = True, strides = (2, 2), stage=3, block='a', training=training)
        x = mconv_block(x, [128, 128, 512], stage=3, block='b', training=training)
        x = mconv_block(x, [128, 128, 512], stage=3, block='c', training=training)
        C3 = x = mconv_block(x, [128, 128, 512], stage=3, block='d', training=training)
        # Stage 4
        x = mconv_block(x, [256, 256, 1024], set_shortcut = True, strides = (2, 2), stage=4, block='a', training=training)
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            x = mconv_block(x, [256, 256, 1024], stage=4, block=chr(98 + i), training=training)
        C4 = x
        # Stage 5
        if stage5:
            x = mconv_block(x, [512, 512, 2048], set_shortcut = True, strides = (2, 2), stage=5, block='a', training=training)
            x = mconv_block(x, [512, 512, 2048], stage=5, block='b', training=training)
            C5 = x = mconv_block(x, [512, 512, 2048], stage=5, block='c', training=training)
        else:
            C5 = None
    
        return [C2, C3, C4, C5]
    
    def reid(self, inputs, training = False):
        feature_maps = self.resnet_graph(inputs, training = training)
        feature_maps = self.proposal_map(feature_maps, training = training)
        return feature_maps[:-2], feature_maps[-2:]
    
    def model(self, model_type):
        input_image = KL.Input(shape = [None, None, 3], name ='input_image')
        input_bbox = KL.Input(shape = [None, 4], name = 'input_bbox')
        input_image_meta = KL.Input(shape=[self.cfg.IMAGE_META_SIZE], name="input_image_meta")
        input_anchors = KL.Input(shape=[None, 4], name="input_anchors")
        
        detection_map, reid_map = self.reid(input_image)
        rpn = self.build_rpn_model(self.cfg.RPN_ANCHOR_STRIDE, len(self.cfg.RPN_ANCHOR_RATIOS), self.cfg.TOP_DOWN_PYRAMID_SIZE)
        outputs = list(zip(*[rpn(p) for p in detection_map]))
        rpn_class_logits, rpn_class, rpn_bbox =  [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, ["rpn_class_logits", "rpn_class", "rpn_bbox"])]
        rpn_rois = ProposalLayer(proposal_count=self.cfg.POST_NMS_ROIS_INFERENCE, nms_threshold = self.cfg.RPN_NMS_THRESHOLD, name="ROI", config=self.cfg)([rpn_class, rpn_bbox, input_anchors])
        rois = KL.Lambda(lambda x : tf.concat(x, axis = 1))([input_bbox, rpn_rois])
        x = PyramidROIAlign([self.cfg.POOL_SIZE, self.cfg.POOL_SIZE], name="roi_align_classifier")([rois, input_image_meta] + detection_map[:-1])
        x = self.mrcnn_feature(x)
        scores, bboxes = self.mrcnn_box_reg([rois, x])
        detection_scores, regression_scores, detection, regression = KL.Lambda(lambda x : [x[0][:, tf.shape(input_bbox)[1]:, :], \
                                                                                           x[0][:, :tf.shape(input_bbox)[1], :], \
                                                                                           x[1][:, tf.shape(input_bbox)[1]:, :], \
                                                                                           x[1][:, :tf.shape(input_bbox)[1], :]
                                                                                           ])([scores, bboxes])
            
        detection_scores, detection = KL.Lambda(lambda x : self.nms_selection(x[0], x[1]))([detection_scores, detection])
        if model_type == 'detection':
            return KM.Model([input_image, input_bbox, input_image_meta, input_anchors], [detection, detection_scores], name = 'mrcnn')

        bboxes = KL.Lambda(lambda x : tf.concat(x, axis = 1))([input_bbox, regression, detection])
        
        reid_map = ATLnet(reid_map, layer = self.cfg.layer, SEnet = self.cfg.SEnet)
        pooled = feature_pooling(self.cfg, name = 'alignedROIPooling')([bboxes] + reid_map)
        pooled = KL.Lambda(lambda x : tf.squeeze(x, axis = 0))(pooled)
        vectors = sMGN(pooled, _eval = True, return_all = self.cfg.mgn, return_mgn = True, l2_norm = self.cfg.l2_norm)
        vectors = KL.Lambda(lambda x : tf.expand_dims(x[:, 0, 0, :], axis = 0))(vectors)
        prediction_vector, regression_vector, detection_vector = KL.Lambda(lambda x : [x[:, :tf.shape(input_bbox)[1], :], \
                                                                                       x[:, tf.shape(input_bbox)[1]:(2*tf.shape(input_bbox)[1]), :], \
                                                                                       x[:, (2*tf.shape(input_bbox)[1]):, :], \
                                                                                    ])(vectors)


        return KM.Model([input_image, input_bbox, input_image_meta, input_anchors], [prediction_vector, regression_vector, regression, regression_scores, detection_vector, detection, detection_scores])

    
    def proposal_map(self, inputs, training = None):
        reg[0] = None
        C2, C3, C4, C5 = inputs
        P5 = DarknetConv2D1(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            DarknetConv2D1(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            DarknetConv2D1(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            DarknetConv2D1(256, (1, 1), name='fpn_c2p2')(C2)])
        
        #reid feature map
        C3_enhance = KL.Concatenate()([P3, C3])
        C3_enhance = KL.Lambda(lambda x : tf.stop_gradient(x))(C3_enhance)
        C3 = KL.Lambda(lambda x : tf.stop_gradient(x))(C3)
        
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = DarknetConv2D1(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = DarknetConv2D1(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = DarknetConv2D1(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = DarknetConv2D1(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
        
        return [P2, P3, P4, P5, P6, C3, C3_enhance]
    
    def rpn_graph(self, feature_map, anchors_per_location, anchor_stride):
        shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                           strides=anchor_stride,
                           name='rpn_conv_shared')(feature_map)
        x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                      activation='linear', name='rpn_class_raw')(shared)
        rpn_class_logits = KL.Lambda(
            lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
        rpn_probs = KL.Activation(
            "softmax", name="rpn_class_xxx")(rpn_class_logits)
        x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                      activation='linear', name='rpn_bbox_pred')(shared)
        rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
        return [rpn_class_logits, rpn_probs, rpn_bbox]
    
    def build_rpn_model(self, anchor_stride, anchors_per_location, depth):
        input_feature_map = KL.Input(shape=[None, None, depth],
                                     name="input_rpn_feature_map")
        outputs = self.rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
        return KM.Model([input_feature_map], outputs, name="rpn_model")
    
    def box_reg(self, bboxes, mrcnn_box):
        height, width = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]
        cy, cx = (bboxes[:, 2] + bboxes[:, 0]) * 0.5, (bboxes[:, 3] + bboxes[:, 1]) * 0.5
        delta = mrcnn_box[:, 1, :] * self.cfg.BBOX_STD_DEV
        cy, cx = cy + delta[:, 0] * height, cx + delta[:, 1] * width
        height, width = height * tf.exp(delta[:, 2]), width * tf.exp(delta[:, 3])
        y1 = cy - 0.5 * height
        x1 = cx - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width
        return tf.expand_dims(tf.stack([y1, x1, y2, x2], axis = 1), axis = 0)

    def mrcnn_feature(self, inputs):
        x = KL.TimeDistributed(KL.Conv2D(1024, self.cfg.POOL_SIZE, padding="valid"), name="mrcnn_class_conv1")(inputs)
        x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=self.cfg.TRAIN_BN)
        x = KL.Activation('relu')(x)
        x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1)), name="mrcnn_class_conv2")(x)
        x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=self.cfg.TRAIN_BN)
        x = KL.Activation('relu')(x)
        x = KL.Lambda(lambda x : tf.squeeze(x, [2, 3]))(x)
        return x
    
    def mrcnn_box_reg(self, inputs):
        input_bbox, mrcnn_feature = inputs
        mrcnn_vector = KL.Lambda(lambda x : x[0, :, :])(mrcnn_feature)
        mrcnn_class_logits = KL.Dense(self.cfg.NUM_CLASSES, name='mrcnn_class_logits')(mrcnn_vector)
        mrcnn_probs = KL.Activation("softmax")(mrcnn_class_logits)
        mrcnn_probs = KL.Lambda(lambda x : tf.expand_dims(x, axis = 0))(mrcnn_probs)
        mrcnn_bbox = KL.Dense(self.cfg.NUM_CLASSES * 4, activation='linear', name='mrcnn_bbox_fc')(mrcnn_vector)
        mrcnn_bbox = KL.Reshape((self.cfg.NUM_CLASSES, 4), name="mrcnn_bbox")(mrcnn_bbox)
        reg_box = KL.Lambda(lambda x : self.box_reg(x[0][0, :, :], x[1]))([input_bbox, mrcnn_bbox])
        return [mrcnn_probs, reg_box]
    
    def nms_selection(self, probs, boxes):
        probs = tf.squeeze(probs, axis = 0)
        boxes = tf.squeeze(boxes, axis = 0)
        class_ids = tf.cast(tf.argmax(probs[:, :], axis = -1), 'int32')
        keep = tf.where(tf.logical_and(class_ids > 0, class_ids < 2))[:, 0]
        #only 'person' considered
        if self.cfg.DETECTION_MIN_CONFIDENCE:
            conf_keep = tf.where(probs[:, 1] >= self.cfg.DETECTION_MIN_CONFIDENCE)[:, 0]
            keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
            keep = tf.sparse_tensor_to_dense(keep)[0]
        #filter out low confidence boxes
        probs = tf.gather(probs[:, :], keep)
        boxes = tf.gather(boxes[:, :], keep)
    
        nms_keep = tf.image.non_max_suppression(boxes, probs[:, 1], max_output_size = self.cfg.DETECTION_MAX_INSTANCES, iou_threshold = self.cfg.DETECTION_NMS_THRESHOLD)
    
        nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
        nms_probs = tf.expand_dims(tf.gather(probs, nms_keep), axis = 0)
        nms_boxes = tf.expand_dims(tf.gather(boxes, nms_keep), axis = 0)
        return [nms_probs[..., 1], nms_boxes]
    

def batch_slice(inputs, graph_fn, batch_size, names=None):

    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)) : output_slice = [output_slice]
        outputs.append(output_slice)

    outputs = list(zip(*outputs))

    if names is None : names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]

    if len(result) == 1 : result = result[0]

    return result
    
def apply_box_deltas_graph(boxes, deltas):
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }

def log2_graph(x):
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        boxes = inputs[0]
        image_meta = inputs[1]
        feature_maps = inputs[2:]
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))

        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)
            box_indices = tf.cast(ix[:, 0], tf.int32)
            box_to_level.append(ix)
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))
        pooled = tf.concat(pooled, axis=0)
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)

        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)

    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

class ProposalLayer(KE.Layer):

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        anchors = inputs[2]
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])
        boxes =batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
#proposals: [1, 1000, 4]
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)
    
import math
class ANCHORS():
    def __init__(self, config):
        self.config = config
        
    def generate_anchors(self, scales, ratios, shape, feature_stride, anchor_stride):
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
        box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return boxes
    
    def generate_pyramid_anchors(self, scales, ratios, feature_shapes, feature_strides, anchor_stride):
        anchors = []
        for i in range(len(scales)): anchors.append(self.generate_anchors(scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride))
        return np.concatenate(anchors, axis=0)
    
    def compute_backbone_shapes(self, image_shape):
        return np.array( [[int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))] for stride in self.config.BACKBONE_STRIDES])
    
    def norm_boxes(self, boxes, shape):
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.divide((boxes - shift), scale).astype(np.float32)
    
    def get_anchors(self, image_shape):
        backbone_shapes = self.compute_backbone_shapes(image_shape)
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            a = self.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            self.anchors = a
            self._anchor_cache[tuple(image_shape)] = self.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]
