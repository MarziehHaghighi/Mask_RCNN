"""
Mask R-CNN
Configurations and data loading code for MS COCO.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco
    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True
    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5
    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last
    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
# import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from pycocotools import mask as maskUtils
from ast import literal_eval
import CellClass
import skimage.io
import zipfile
# import urllib.request
import shutil
from skimage.util import img_as_ubyte
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



############################################################
#  Configurations
############################################################


class cellConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Cell"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

#     config.IMAGE_SHAPE
    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
#     NUM_CLASSES = 1 + 80  # COCO has 80 classes    # Marz
    # Train on 1 GPU and 8-->2 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8-->2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 19  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
#     numOchannels=4;
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8,16, 32, 64,128)  # anchor side in pixels
#    RPN_ANCHOR_SCALES = (32, 64,128,256)  # anchor side in pixels
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    
    TRAIN_ROIS_PER_IMAGE = 128

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
#    head='def'; #'def','unet'


    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
#     MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    MEAN_PIXEL =np.ones((IMAGE_CHANNEL_COUNT,), dtype=int)*128;
#     IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + NUM_CLASSES-1
############################################################
#  Dataset
############################################################

class cellsDataset(utils.Dataset):
    def load_cells(self, dataset_dir,df_Info, subset,class_ids=None,
                  class_map=None, return_cell=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        df_Info: dataframe containg all data and mask addresses and annotations
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """


#         coco = CellClass("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))


    
        if subset == "minival" or subset == "valminusminival":
            subset = "val"

#         if subset == "train":
        df_Info_t = df_Info[df_Info['subset_label']=="train"];
        cell_train = CellClass.CELL(df_Info_t)
        
        df_Info = df_Info[df_Info['subset_label']==subset];
#         print(df_Info.shape) 
        cell = CellClass.CELL(df_Info)

        
#         image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
#             if subset == "train" or subset == "val":
            class_ids = sorted(cell_train.getCatIds())
#             else:
#                 class_ids = sorted(cell.getCatIds())
        
#         print(class_ids)
            

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                
                image_ids.extend(list(cell.getImgIds(catIds=[id])))
#                 print(image_ids,id)
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(cell.imgs.keys())
            
#         if subset == "test":   
#             image_ids = list(cell.imgs.keys())
#         print(image_ids)
        # Add classes
        for i in class_ids:
#             if subset == "train" or subset == "val":
              self.add_class("cell", i, cell_train.loadCats(i)[0]["name"])
#             else:
#                 self.add_class("cell", i, cell.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
#             print(cell.imgs[i]['im_paths'],len(cell.imgs[i]['im_paths']))
            self.add_image(
                "cell", image_id=i,
                path=cell.imgs[i]['im_paths'],
#                 path=os.path.join(image_dir, cell.imgs[i]['file_name']),
                width=cell.imgs[i]["width"],
                height=cell.imgs[i]["height"],
                annotations=cell.loadAnns(cell.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))
            annotations=cell.loadAnns(cell.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None))
#             print(i)
#             print(cell.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None))
#             print(annotations[0]['mask'])
#             print(cell.imgs[i]['im_paths'])
        
        if return_cell:
            return cell
        
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,nChannels] Numpy array.
        """
        # Load images

        listOfPaths=literal_eval(self.image_info[image_id]['path'])
#         print(len(listOfPaths))
        listOfPaths=[listOfPaths[0],listOfPaths[2],listOfPaths[1]]
#         listOfPaths=[listOfPaths[2]]
        imagesList=[]
        for imPath in listOfPaths:
#             print(imPath)
            im_uint16=skimage.io.imread(imPath) # images are 'uint16'
        # if you want to convert to unit8
            im_uint8=img_as_ubyte(im_uint16)
            imagesList.append(im_uint8)
        # If grayscale. Convert to RGB for consistency.
#         if image.ndim != 3:
#             image = skimage.color.gray2rgb(image)
        image=np.stack(imagesList, axis=2)

        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cell":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
#         objectOrders=[]
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        
        
        maskAddress=annotations[0]['mask'];
#         print(maskAddress)
        mask2DImage=skimage.io.imread(maskAddress)
#         print(np.unique(mask2DImage))
#         one_hot_masks = np.zeros((im.shape[0], im.shape[1], n_classes))
#         i=0
#         print(maskAddress)
        for annotation in annotations:
#             print(annotation['mask'])
#             print(annotation['ObjectNumber'])
            class_id = self.map_source_class_id(
                "cell.{}".format(annotation['category_id']))
            if class_id:
                class_ids.append(class_id)
                objectNum=annotation['ObjectNumber']
#                 OneHot2D_instance_masks=mask2DImage==objectNum
                bboxx=literal_eval(annotation['bbox'])
#                 print(annotation['P-W-S'],annotation['ObjectNumber'])
#                 print(bboxx[0]+int(bboxx[2]/2),bboxx[1]+int(bboxx[3]/2))
#                 print(bboxx[1]+int(bboxx[3]/2),bboxx[0]+int(bboxx[2]/2))
                objectPixValue=mask2DImage[bboxx[1]+int(bboxx[3]/2),bboxx[0]+int(bboxx[2]/2)]
#                 objectPixValue=mask2DImage[bboxx[0]+int(bboxx[2]/2),bboxx[1]+int(bboxx[3]/2)]
                OneHot2D_instance_masks=mask2DImage==objectPixValue
                instance_masks.append(OneHot2D_instance_masks)
#                 i=i+1
        
        
        
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.


    
#     def annToMask(self, ann, height, width):
#         """
#         read mask .png file and
#         :return: binary mask (numpy 2D array)
#         """
        
# #         p,w,s=ann['P-W-S'].split('-')[0],ann['P-W-S'].split('-')[1],ann['P-W-S'].split('-')[2]
#         maskAddress=ann['mask']
#         maskFileName=masksSaveFolder+imDf.Metadata_Well[0]+'_s'+str(imDf.Metadata_Site[0])+'_cell_mask.png'
#         rle = self.annToRLE(ann, height, width)
#         m = maskUtils.decode(rle)
#         return m

#     def annToMask(self, ann, height, width):
#         """
#         Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
#         :return: binary mask (numpy 2D array)
#         """
#         rle = self.annToRLE(ann, height, width)
#         m = maskUtils.decode(rle)
#         return m

############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
