import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import cv2
import numpy
from numpy  import array

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

LABEL_NAMES = (
    'fly',
    'bike',
    'bird',
    'boat',
    'pin',
    'bus',
    'c',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'p',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
)


trainer.load('/home/ml/Downloads/chainer_best_model_converted_to_pytorch_0.7053.pth')
opt.caffe_pretrain=True # this model was trained from caffe-pretrained model


cap = cv2.VideoCapture(0)



print ("capture starts")


while (cap.isOpened()):
    print ("cap is opened")
    ret,image_np = cap.read()
    
    cv2.imshow('image',image_np)
    
    image_np = image_np.transpose((2, 0, 1))
    
    
    #cv2.imshow('image',image_np)
    img = t.from_numpy(image_np)[None]
    
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
    
    print (_labels)
    
    X = numpy.array(_labels)
    
    i=0
    for i in range (X.shape[1]):
        #print (X[0,i])
        print (LABEL_NAMES[X[0,i]])
    
    #print (_labels)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break