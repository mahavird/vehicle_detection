
# coding: utf-8

# In[1]:


import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import cv2

# In[10]:


img = read_image('misc/demo.jpg')
imgFile = cv2.imread('misc/demo.jpg')
#cv2.imshow('Image', imgFile)
img = t.from_numpy(img)[None]

print (img.shape)
print (imgFile.shape)


# In[3]:


faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()


# You'll need to download pretrained model from [google dirve](https://drive.google.com/open?id=1cQ27LIn-Rig4-Uayzy_gH5-cW-NRGVzY) 
# # 1. model converted from chainer

# In[9]:


# in this machine the cupy isn't install correctly... 
# so it's a little slow
trainer.load('/home/ml/Downloads/chainer_best_model_converted_to_pytorch_0.7053.pth')
opt.caffe_pretrain=True # this model was trained from caffe-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))


print ("Hero")
