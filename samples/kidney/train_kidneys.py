#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:49:35 2018

@author: ch194093
"""
import sys
sys.path.insert(0, '/home/ch194093/Desktop/kidneydcemri/deepLearningSegmentation')
from selectTrainAndTestSubjects import selectTrainAndTestSubjects
import os

import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from config import Config
import utils
import model as modellib
import visualize
from model import log
import kidney
config = kidney.KidneysConfig()
Kid_DIR = "path to Kidney dataset"  # TODO: enter value here

trainMode=1;
testMode=0;

ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
TestSetNum=2;init_with = "coco" #"imagenet","coco","scratch",
maskBranchNet='def';
config.head=maskBranchNet;
fileNumModel='Mask_RCNN_testSet'+str(TestSetNum)+'_pretrained_'+init_with+'_maskNet_'+config.head;
    
address = "/fileserver/abd/marzieh/deepLearningModels/MaskRCNN/"+fileNumModel+"/"
MODEL_DIR = os.path.join(address)
#MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


subjectNamesNormalTrain,subjectNamesNormalTest,_,_=selectTrainAndTestSubjects(TestSetNum);
if trainMode:


    dataset_train = kidney.KidneysDataset()
    dataset_train.load_Kidneys(subjectNamesNormalTrain)
    dataset_train.prepare()


#subjectNamesNormalTest=['Niedzielski_Bianca',\
#                    'Alreem_Alhamed','Do_Lee'];

dataset_val = kidney.KidneysDataset()
dataset_val.load_Kidneys(subjectNamesNormalTest)
dataset_val.prepare()


"""
for image_id in range(len(subjectNamesNormalTrain)):
    image = dataset_train.load_image(image_id,subjectNamesNormalTrain)
    mask, class_ids = dataset_train.load_mask(image_id,subjectNamesNormalTrain)
    visualize.display_top_masks(image[:,:,0], mask, class_ids, dataset_train.class_names)
#plt.figure();plt.close('all')
    
# Load random image and mask.
image_id = random.choice(dataset_train.image_ids)
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset_train.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image[:,:,0:3], bbox, mask, class_ids, dataset_train.class_names)   

image_id=3;
image = dataset_val.load_image(image_id)
mask, class_ids = dataset_val.load_mask(image_id)
bbox = utils.extract_bboxes(mask)
visualize.display_instances(image[:,:,0:3], bbox, mask, class_ids, dataset_val.class_names)  
"""


 

if trainMode:
# Which weights to start with?
 # imagenet, coco, or last
# Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)   

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
        
#    model.load_weights(model.find_last()[1], by_name=True)   #marzieh    

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

#model.train(dataset_train, dataset_val, 
#            learning_rate=config.LEARNING_RATE, 
#            epochs=1, 
#            layers='heads')    

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.

#    model.load_weights(COCO_MODEL_PATH, by_name=True,
#                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
#                                "mrcnn_bbox", "mrcnn_mask"])
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=400, 
                layers="heads");
#$#################################################

if testMode:
    from networks import calculatedPerfMeasures as calcPerf
    class InferenceConfig(kidney.KidneysConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    inference_config = InferenceConfig()
    inference_config.head=maskBranchNet;
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
    
    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#    model_path = model.find_last()[1]
    
    #  best weights for test1 ---> train using coco model
    txt_file = open(address+'selectedEpoc.txt','r')
    selectedEpoch=str(int(txt_file.read()))[1:]
    model_path=[x[0] for x in os.walk(address)][1]+'/mask_rcnn_kidneys_'+selectedEpoch+'.h5';
#    model_path ='/home/ch194093/Desktop/kidneydcemri/deepLearningSegmentation\
#    /Mask_RCNN/logs/kidneys20180228T1214/mask_rcnn_kidneys_0161.h5'
    
    
#    model_path ='/home/ch194093/Desktop/kidneydcemri/deepLearningSegmentation\
#    /Mask_RCNN/logs/kidneys20180216T1826/mask_rcnn_kidneys_0399.h5'    
    
    
    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    
    
    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
#    for i in dataset_val.image_ids:
    for i in [5]:    
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config, 
                                   i, use_mini_mask=False)
        
        log("original_image", original_image)
        log("image_meta", image_meta)
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
        
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                    dataset_val.class_names, figsize=(8, 8))
        ########################################################################
        results = model.detect([original_image], verbose=1)
        
        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset_val.class_names, r['scores'])
        
        rightK=calcPerf(results[0]['masks'][:,:,1],gt_mask[:,:,1]);
    ###############################################################################
    
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id,
                             r["rois"], r["class_ids"], r["scores"])
        APs.append(AP)
        
    print("mAP: ", np.mean(APs))

 
#######################################################################    
"""
import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import utils
import visualize
from visualize import display_images
import model as modellib
from model import log

#%matplotlib inline 

ROOT_DIR = os.getcwd()

COCO_DIR="/home/ch194093/Desktop/kidneydcemri/deepLearningSegmentation/Mask_RCNN/coco-master/"
    
import coco
config = coco.CocoConfig()    
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR,"train",year="2017")    

dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))



# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

# Load random image and mask.
image_id = random.choice(dataset.image_ids)
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
"""