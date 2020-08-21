# TLfPS
Transfer Learning for Person Search
## Installation
- tensorflow 1.14.0
- keras
- scikit-learn [conda install -c anaconda scikit-learn]
- ~~wxPython [conda install -c anaconda wxpython]~~
- yolov3_reid.h5
- ~~[mrcnn.h5](https://github.com/matterport/Mask_RCNN/releases)~~
- ~~[yolov3.h5](https://github.com/qqwweee/keras-yolo3)~~
- ~~[yolov4.h5](https://github.com/Ma-Dan/keras-yolo4)~~
- ~~yolov4_reid.h5~~
- ~~mrcnn_reid.h5~~

## Simple application
Download yolov3_reid.h5, save to ./saved_weights
```python
#model: [yolov3], gpu: [0]
#ctrl+L: load model; ctrl+R: select query image & draw roi; ctrl+G:select gallery images; ctrl+S:search
python app/app.py -m yolov3 -g 0
```

## Evaluation
- Download yolov3_reid.h5, save to ./saved_weights
- Download CUHK_SYSU dataset
```python
#model: [yolov3], gpu: [0], dataset: [path/to/CUHK_SYSU], experiment name: [default]
python utils/evaluation.py -m yolov3 -g 0 -p path/to/CUHK_SYSU -e default
```

## Training
- Download [yolov3.h5](https://github.com/qqwweee/keras-yolo3), save to ./pretrained_weights
- Download CUHK_SYSU dataset
```python
#model: [yolov3], gpu: [0], dataset: [path/to/CUHK_SYSU]
python utils/creategallery.py -m yolov3 -g 0 -c path/to/CUHK_SYSU
#model: [yolov3], gpu: [0], dataset: [path/to/CUHK_SYSU], name [1]
python train.py -m yolov3 -g 0 -p path/to/CUHK_SYSU -n 1
#model: [yolov3], gpu: [0], weights path: [path/to/yolov3_model-1-x-0.xxx.h5]
python utils/convertor.py -m yolov3 -g 0 -p path/to/yolov3_model-1-x-0.xxx.h5
'''
Error: [None Gradients], please modify the functon keras->optimizer.py->get_gradients by this:
def get_gradients(self, loss, params):
    grads = K.gradients(loss, params)
    grads = [g if g is not None else tf.constant(0, dtype = 'float32') for g in grads]
'''

```
