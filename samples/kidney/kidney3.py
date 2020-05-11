"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import random
import numpy as np
import cv2
import pickle
from config import Config
import utils


class KidneysConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "kidneys"

    # Train on 1 GPU and 8-->2 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8-->2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64,128,256)  # anchor side in pixels
#    RPN_ANCHOR_SCALES = (32, 64,128,256)  # anchor side in pixels
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class KidneysDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    
    def load_Kidneys(self, subjectNamesNormalTrain):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        self.subjectNamesNormalTrain=subjectNamesNormalTrain;
        # Add classes
#        self.add_class("kidneys", 0, "BG")
        self.add_class("kidneys", 1, "right")
        self.add_class("kidneys", 2, "left")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        self.numOfSlices=32;
        idCount=0;
        for i in range(len(subjectNamesNormalTrain)):            
            subName=self.subjectNamesNormalTrain[i];
            data4D = pickle.load(open( "/home/ch194093/Desktop/kidneydcemri/deepLearningSegmentation/normalData1Slice3D_Pcs.p","rb" ))
            sliceNum=i# % self.numOfSlices;
#            print(subName);
#            maxSlice=data4D[subName+'M'].shape[2];
#            if maxSlice<self.numOfSlices and sliceNum>=maxSlice:
#                m0=data4D[subName+'M'][:,:,20].astype('uint8');
#            else:
            m0=data4D[subName+'M'].astype('uint8');            
                        
            annotations=[];
#            if 1 not in set(m0.flatten()) and 2 not in set(m0.flatten()) :
#                annotations.append('BG');
#            else:
            if 1 in set(m0.flatten()):
                annotations.append('right');
            
            if 2 in set(m0.flatten()):
                annotations.append('left');
                
#                print(i,set(m0.flatten()),annotations)
#            kidneys = self.load_image(i)
#                idCount=+1;
                self.add_image("kidneys", image_id=i, path=None,                
                               subName=subName,sliceNum=sliceNum,width=256,
                               height=256,annotations=annotations);
                           
    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
#        print(self.image_info["height"])
#        subName=self.subjectNamesNormalTrain[int(image_id/self.numOfSlices)];
        subName=self.image_info[image_id]['subName'];
#        sliceNum=self.image_info[image_id]['sliceNum'];
        data4D = pickle.load( open( "/home/ch194093/Desktop/kidneydcemri/deepLearningSegmentation/normalData1Slice3D_Pcs.p","rb" ))
#        data4D = pickle.load( open( "/common/abd/marzieh/preprocessedData/singleSubjectsV4/"+subName+".p","rb" ));
#        print(image_id)
#        sliceNum=image_id % self.numOfSlices;
#        maxSlice=data4D[subName+'M'].shape[2];
#        if maxSlice<self.numOfSlices and sliceNum>=maxSlice:
#            image0=data4D[subName+'D'][:,:,20,:];
#        else:
        image0=data4D[subName+'D'];
        image000=(image0+abs(min([0,image0.min()])));image000=(image000/image000.max())*255;
#        image000=image0*255;
        image00=image000.astype('uint8');
#        image=np.lib.pad(image0, ((16, 16), (2, 3)), 'minimum')
        image = cv2.copyMakeBorder(image00,16,16,16,16,cv2.BORDER_REPLICATE)
        
#        info = self.image_info[image_id]

        return image[:,:,0:3]

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "kidneys":
            return info["kidneys"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        subName=self.image_info[image_id]['subName'];
        sliceNum=self.image_info[image_id]['sliceNum'];
        
#        subName=self.subjectNamesNormalTrain[int(image_id/self.numOfSlices)];
        data4D = pickle.load( open( "/home/ch194093/Desktop/kidneydcemri/deepLearningSegmentation/normalData1Slice3D_Pcs.p","rb" ))
#        data4D = pickle.load( open( "/common/abd/marzieh/preprocessedData/singleSubjectsV4/"+subName+".p","rb" ));
#        sliceNum=image_id % self.numOfSlices;
#        maxSlice=data4D[subName+'M'].shape[2];
#        if maxSlice<self.numOfSlices and sliceNum>=maxSlice:
#            m0=data4D[subName+'M'][:,:,20].astype('uint8');
#        else:
#            m0=data4D[subName+'M'][:,:,sliceNum].astype('uint8');
        
        m0=data4D[subName+'M'].astype('uint8');   
#        m0=data4D[subName+'M'][:,:,sliceNum].astype('uint8');
        m0 = cv2.copyMakeBorder(m0,16,16,16,16,cv2.BORDER_REPLICATE)
        
#        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
#        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        annotations=[];classIds4Anns={};
        classIds4Anns['BG']=0;classIds4Anns['right']=1;classIds4Anns['left']=2;
        annotations=self.image_info[image_id]['annotations'];  
        
#        print(annotations)
#        if 1 not in set(m0.flatten()) and 2 not in set(m0.flatten()) :
#            annotations.append('BG');
#            classIds4Anns['BG']=0;
#        else:
#            if 1 in set(m0.flatten()):
#                annotations.append('right');
#                classIds4Anns['right']=1;
#            
#            if 2 in set(m0.flatten()):
#                annotations.append('left');
#                classIds4Anns['left']=2;
#        annotations=['right','left'];classIds4Anns={};
#        classIds4Anns[annotations[0]]=1;classIds4Anns[annotations[1]]=2;
        
#        print(annotations)
        for a in range(len(annotations)):
#            class_id = self.map_source_class_id(classIds4Anns[annotations[a]])
            class_id = classIds4Anns[annotations[a]];
            m=np.copy(m0);
            m[m!=classIds4Anns[annotations[a]]]=0;
            if annotations[a]=='BG':
                m[m==0]=1;
            instance_masks.append(m);
            class_ids.append(class_id)

        # Pack instance masks into an array
        if len(annotations)>1:
            mask = np.stack(instance_masks, axis=2);
        else:
            mask=instance_masks;
            
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            mask = np.ones([256,256], dtype=bool)
            return mask, class_ids  
            
            
#        class_ids = np.array(class_ids, dtype=np.int32)
#        return mask, class_ids

