# TLfPS
Transfer Learning for Person Search
## Installation
- tensorflow 1.14.0
- keras
- scikit-learn [conda install -c anaconda scikit-learn]
- ~~cuml[conda install -c rapidsai cuml]~~
- ~~wxPython [conda install -c anaconda wxpython]~~
- [yolov3_reid.h5](https://drive.google.com/file/d/1Dne2_ZCOAA4nn8PySBjzPFUaZbHpYsd5/view?usp=sharing)
- ~~[mrcnn.h5](https://github.com/matterport/Mask_RCNN/releases)~~
- ~~[yolov3.h5](https://github.com/qqwweee/keras-yolo3)~~
- ~~[yolov4.h5](https://github.com/Ma-Dan/keras-yolo4)~~
- ~~yolov4_reid.h5~~
- ~~mrcnn_reid.h5~~

## Simple application
Download [yolov3_reid.h5](https://drive.google.com/file/d/1Dne2_ZCOAA4nn8PySBjzPFUaZbHpYsd5/view?usp=sharing), save to ./saved_weights
```python
#model: [yolov3], gpu: [0]
#ctrl+L: load model; ctrl+Q: select query image & draw roi; ctrl+G:select gallery images; ctrl+S:search
python app/app.py -m yolov3 -g 0
```

## Evaluation
- Download [yolov3_reid.h5](https://drive.google.com/file/d/1Dne2_ZCOAA4nn8PySBjzPFUaZbHpYsd5/view?usp=sharing), save to ./saved_weights
- Download [CUHK_SYSU](https://drive.google.com/file/d/1D7VL43kIV9uJrdSCYl53j89RE2K-IoQA/view?usp=sharing) dataset
- Download [PRW](https://drive.google.com/file/d/116_mIdjgB-WJXGe8RYJDWxlFnc_4sqS8/view?usp=sharing) dataset
```python
#model: [yolov3], gpu: [0], dataset: [path/to/CUHK_SYSU], experiment name: [cuhk_default]; evaluate on CUHK-SYSU dataset
python utils/evaluation.py -m yolov3 -g 0 -p path/to/CUHK_SYSU -e cuhk_default
#model: [yolov3], gpu: [0], dataset: [path/to/PRW], experiment name: [default]; evaluate on PRW dataset
python utils/prw_evaluation.py -m yolov3 -g 0 -p path/to/PRW -e prw_default
#model: [yolov3], gpu: [0], dataset: [path/to/CUHK_SYSU], experiment name: [cuhk_default]; TSNE visualization for CUHK-SYSU dataset
python utils/tsne_cuhk.py -m yolov3 -g 0 -p path/to/CUHK_SYSU -e cuhk_default
python utils/TSNE.py -m yolov3 -g 0 -p cuhk -e cuhk_default
#model: [yolov3], gpu: [0], dataset: [path/to/PRW], experiment name: [prw_default]; TSNE visualization for PRW dataset
python utils/tsne_prw.py -m yolov3 -g 0 -p path/to/PRW -e cuhk_default
python utils/TSNE.py -m yolov3 -g 0 -p prw -e prw_default
#model: [yolov3], gpu: [0], dataset: [path/to/CUHK_SYSU], reid: [0]; test the running time without reid module
python utils/elapsedtime.py -m yolov3 -g 0 -p [path/to/CUHK_SYSU] -r 0
#model: [yolov3], gpu: [0], dataset: [path/to/CUHK_SYSU], reid: [1]; test the running time with reid module
python utils/elapsedtime.py -m yolov3 -g 0 -p [path/to/CUHK_SYSU] -r 1
```

## Training
- Download [yolov3_reid.h5](https://drive.google.com/file/d/1Dne2_ZCOAA4nn8PySBjzPFUaZbHpYsd5/view?usp=sharing), save to ./saved_weights
- Download [CUHK_SYSU](https://drive.google.com/file/d/1D7VL43kIV9uJrdSCYl53j89RE2K-IoQA/view?usp=sharing) dataset
- Download [PRW](https://drive.google.com/file/d/116_mIdjgB-WJXGe8RYJDWxlFnc_4sqS8/view?usp=sharing) dataset
```python
#model: [yolov3], gpu: [0]; convert yolov3_reid.h5 to yolov3.h5
python utils/getmodel.py -m yolov3 -g 0
#model: [yolov3], gpu: [0]; test yolov3 detection network
python utils/testmodel.py -m yolov3 -g 0 -p path/to/image
#model: [yolov3], gpu: [0], dataset: [path/to/CUHK_SYSU], [path/to/PRW]; create detection, help to evaluate the running time evaluation
python utils/creategallery.py -m yolov3 -g 0 -c path/to/CUHK_SYSU -p path/to/PRW
#model: [yolov3], gpu: [0], dataset: [path/to/CUHK_SYSU], [path/to/PRW], model name: [1]; start training
python train.py -m yolov3 -g 0 -c path/to/CUHK_SYSU -p path/to/PRW -n 1
#model: [yolov3], gpu: [0], weights path: [path/to/yolov3_model-1-x-0.xxx.h5]; convert a trained model yolov3_xxx.h5 to yolov3_reid.h5
python utils/convertor.py -m yolov3 -g 0 -p path/to/yolov3_model-1-x-0.xxx.h5
'''
if you meet the [none gradient] problem, please modify the functon keras->optimizer.py->get_gradients by this:
def get_gradients(self, loss, params):
    grads = K.gradients(loss, params)
    grads = [g if g is not None else tf.constant(0, dtype = 'float32') for g in grads]
'''
```

## Train YOLOv3 with you own dataset
- 
