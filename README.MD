# Vehicle Detection

## 1. Introduction

This project aims to implement a Vehicle detection model.


The model which we have trained over VOC dataset is being referred as vehicle detection model here.

## 2. Performance

### 2.1 mAP

VGG16 train on `trainval` and test on `test` split. 


|              Implementation              |     mAP     |
| :--------------------------------------: | :---------: |
| [origin paper](https://arxiv.org/abs/1506.01497) |    0.699    |
| vehicle detection model  |   0.7053    |

### 2.2 Speed

|              Implementation              |   GPU    | Inference | Trainining |
| :--------------------------------------: | :------: | :-------: | :--------: |
| [origin paper](https://arxiv.org/abs/1506.01497) |   K40    |   5 fps   |     NA     |
|                 vehicle detection                 | TITAN Xp | 14-15 fps |   6 fps    |

[1]: make sure you install cupy correctly and only one program run on the GPU. The training speed is sensitive to your gpu status. see [troubleshooting](troubleshooting) for more info. Morever it's slow in the start of the program. 

It could be faster by removing visualization, logging, averaging loss etc.
## 3. Install dependencies

requires python3 and PyTorch 0.3

- install PyTorch >=0.3 with GPU (code are GPU-only), refer to [official website](http://pytorch.org)

- install cupy, you can install via `pip install` but it's better to read the [docs](https://docs-cupy.chainer.org/en/latest/install.html#install-cupy-with-cudnn-and-nccl) and make sure the environ is correctly set

- install other dependencies:  `pip install -r requirements.txt `

- Optional, but strongly recommended: build cython code `nms_gpu_post`: 

  ```Bash
  cd model/utils/nms/
  python3 build.py build_ext --inplace
  ```

- start vidom for visualization

```Bash
nohup python3 -m visdom.server &
```


## 4. Demo

Download pretrained model from [Google Drive](https://drive.google.com/file/d/1a6pjWxv1ZDwJPHGP5XsKi4F4voooYBDT/view?usp=sharing)

See
[vehicle-detection-in-image.ipynb](https://github.com/mahavird/vehicle_detection/blob/master/vehicle-detection-in-image.ipynb)
[vehicle-detection-in-video.ipynb](https://github.com/mahavird/vehicle_detection/blob/master/vehicle-detection-in-video.ipynb) for more detail.


## 5. Train

### 5.1 Prepare data

#### Pascal VOC2007

1. Download the training, validation, test data and VOCdevkit

   ```Bash
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   ```

2. Extract all of these tars into one directory named `VOCdevkit`

   ```Bash
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

3. It should have this basic structure

   ```Bash
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

4. modify `voc_data_dir` cfg item in `utils/config.py`, or pass it to program using argument like `--voc-data-dir=/path/to/VOCdevkit/VOC2007/` .


### 5.3 begin training

```Bash
mkdir checkpoints/ # folder for snapshots
```

```bash
python3 train.py train --env='fasterrcnn-caffe' --plot-every=100 --caffe-pretrain    
```

you may refer to `utils/config.py` for more argument.

Some Key arguments:

- `--caffe-pretrain=False`: use pretrain model from caffe or torchvision (Default: torchvison)
- `--plot-every=n`: visualize prediction, loss etc every `n` batches.
- `--env`: visdom env for visualization
- `--voc_data_dir`: where the VOC data stored
- `--use-drop`: use dropout in RoI head, default False
- `--use-Adam`: use Adam instead of SGD, default SGD. (You need set a very low `lr` for Adam)
- `--load-path`: pretrained model path, default `None`, if it's specified, it would be loaded.

you may open browser, visit `http://<ip>:8097` and see the visualization of training procedure as below:
