#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:10:38 2018

@author: ch194093
"""

import sys
sys.path.insert(0, '/home/ch194093/Desktop/kidneydcemri/deepLearningSegmentation')
from selectTrainAndTestSubjects import selectTrainAndTestSubjects
import os

import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from config import Config
import utils
from decimal import Decimal
import model as modellib
from model import log
import DCRF_postprocess_2D
import kidney
#config = kidney.KidneysConfig()
from networks import calculatedPerfMeasures as calcPerf
import pandas as pd
from skimage import morphology

def maskRCNNInferenceMRI(preTrainednet,maskBranchNet,testSetNdx):
#    preTrainednet='imagenet';testSetNdx=1;maskBranchNet='def';
    crfEnabled=1;
    morfFilteringEnabled=0;
    TestSetNum=testSetNdx;init_with = preTrainednet 
    fileName='Mask_RCNN_testSet'+str(TestSetNum)+'_pretrained_'+init_with+'_maskNet_'+maskBranchNet;
        
    address = "/fileserver/abd/marzieh/deepLearningModels/MaskRCNN/"+fileName+"/"
    MODEL_DIR = os.path.join(address)
    #MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    class InferenceConfig(kidney.KidneysConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    inference_config = InferenceConfig()
    inference_config.head=maskBranchNet;
    model = modellib.MaskRCNN(mode="inference",config=inference_config,model_dir=MODEL_DIR)  
    subjectNamesNormalTrain,subjectNamesNormalTest,trainKidCond,testKidCond=selectTrainAndTestSubjects(TestSetNum);

    txt_file = open(address+'selectedEpoc.txt','r')
    selectedEpoch=str(int(txt_file.read()))[1:]
    model_path=[x[0] for x in os.walk(address)][1]+'/mask_rcnn_kidneys_'+selectedEpoch+'.h5';
    model.load_weights(model_path, by_name=True)

    
    scores=[]
    columns = ['Name','kidney Condition','F1-Score', 'Prec','Rec','VEE','testSet','Model'];
    index=np.arange(len(subjectNamesNormalTest));
    df= pd.DataFrame(index=index, columns=columns)
    df= df.fillna(0);dfNd=0;
    for s in subjectNamesNormalTest:
#        s=subjectNamesNormalTest[0];
        dataset_val = kidney.KidneysDataset()
        isValidationData=1;
        dataset_val.load_Kidneys([s],isValidationData);
        dataset_val.prepare()
        sliceIds=dataset_val.image_ids;
        rightMask3Dorig=np.zeros((256,256,len(sliceIds)));
        rightMask3Dpred=np.zeros((256,256,len(sliceIds)));
        leftMask3Dorig=np.zeros((256,256,len(sliceIds)));
        leftMask3Dpred=np.zeros((256,256,len(sliceIds)));   
        print(s);
        for sl in range(len(sliceIds)):
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config, sliceIds[sl], use_mini_mask=False)
#            molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
            # Run object detection
            results = model.detect([image], verbose=0)
#            print(sl,results[0]['class_ids'],gt_class_id)
            if 1 in results[0]['class_ids'] and 2 in results[0]['class_ids']:                
                rightNdx=np.where(results[0]['class_ids']==1)[0][0];
                leftNdx=np.where(results[0]['class_ids']==2)[0][0];
                rightMask3Dpred[:,:,sl] = results[0]['masks'][:,:,rightNdx]
                leftMask3Dpred[:,:,sl]=results[0]['masks'][:,:,leftNdx]
            elif 1 in results[0]['class_ids'] and 2  not in results[0]['class_ids']:
                rightMask3Dpred[:,:,sl] = results[0]['masks'][:,:,0] 
            elif 1 not in results[0]['class_ids'] and 2 in results[0]['class_ids']:
                leftMask3Dpred[:,:,sl]=results[0]['masks'][:,:,0]
                
                
            if 1 in gt_class_id and 2 in gt_class_id:                
                rightMask3Dorig[:,:,sl] = gt_mask[:,:,0]       
                leftMask3Dorig[:,:,sl] = gt_mask[:,:,1]   
            elif 1 in gt_class_id and 2  not in gt_class_id:
                rightMask3Dorig[:,:,sl] = gt_mask[:,:,0]       
            elif 1 not in gt_class_id and 2 in gt_class_id:
                leftMask3Dorig[:,:,sl] = gt_mask[:,:,0]                 
                
                    
        if crfEnabled:
            rightMask3DpredCRF=np.zeros((256,256,len(sliceIds)));
            leftMask3DpredCRF=np.zeros((256,256,len(sliceIds)));                  
            for ndx in range(leftMask3Dpred.shape[1]):
                rightMask3DpredCRF[ndx,:,:]=DCRF_postprocess_2D.DCRF_postprocess_2D(rightMask3Dpred[ndx,:,:],rightMask3Dpred[ndx,:,:]);
                leftMask3DpredCRF[ndx,:,:]=DCRF_postprocess_2D.DCRF_postprocess_2D(leftMask3Dpred[ndx,:,:],leftMask3Dpred[ndx,:,:]); 
            avgPerfOverKidneys=np.mean([calcPerf(rightMask3DpredCRF,rightMask3Dorig), calcPerf(leftMask3DpredCRF,leftMask3Dorig)],axis=0)
        
#        import matplotlib.pyplot as plt
#        plt.figure();ndx=100;
#        plt.subplot(131);plt.imshow(rightMask3Dorig[ndx,:,:])     
#        plt.subplot(132);plt.imshow(rightMask3Dpred[ndx,:,:])        
#        plt.subplot(133);plt.imshow(rightMask3DpredCRF[ndx,:,:])  
        
        elif morfFilteringEnabled:
            #            rightMask3Dpred2=morphology.remove_small_objects(rightMask3Dpred.astype(int), 10000)
#            leftMask3Dpred2=morphology.remove_small_objects(leftMask3Dpred.astype(int), 10000)                                            
            rightMask3DpredCRF=np.zeros((256,256,len(sliceIds)));
            leftMask3DpredCRF=np.zeros((256,256,len(sliceIds)));                  
            selem = morphology.disk(3);
            for ndx in range(leftMask3Dpred.shape[1]):
#                rightMask3DpredCRF[ndx,:,:]=morphology.binary_closing(rightMask3Dpred[ndx,:,:],selem)
#                leftMask3DpredCRF[ndx,:,:]=morphology.binary_closing(leftMask3Dpred[ndx,:,:],selem)
                rightMask3DpredCRF=morphology.remove_small_objects(rightMask3Dpred.astype(int), 1000)
                leftMask3DpredCRF=morphology.remove_small_objects(leftMask3Dpred.astype(int), 1000)                   
                
                
            avgPerfOverKidneys=np.mean([calcPerf(rightMask3DpredCRF,rightMask3Dorig), calcPerf(leftMask3DpredCRF,leftMask3Dorig)],axis=0)

        else:
            avgPerfOverKidneys=np.mean([calcPerf(rightMask3Dpred,rightMask3Dorig), calcPerf(leftMask3Dpred,leftMask3Dorig)],axis=0)
#        isValidationData=0;array([  0.91786584,   0.93257748,   0.90517073, 848.5,3.97734375])
#        iisValidationData=1;       0.86431,       0.932577,     0.807285,   3929.5,   18.4195]
#iisValidationData=1;+CRF  0.916862,0.893233,0.942325,-1215,-5.69531

        scores.append(avgPerfOverKidneys);
        df.ix[dfNd]=pd.Series({'Name':s,'kidney Condition':testKidCond[dfNd],'F1-Score':avgPerfOverKidneys[0]*100,'Prec':avgPerfOverKidneys[1]*100,
           'Rec':avgPerfOverKidneys[2]*100,'VEE':avgPerfOverKidneys[4],'testSet':TestSetNum,'Model':preTrainednet+'-'+maskBranchNet});
        dfNd+=1;
#        print(dfNd)
    averageOverSubjectPrefsNormal=np.mean(scores[0:4],axis=0);
    averageOverSubjectPrefsAbnormal=np.mean(scores[4:],axis=0);
    f1Normal=np.round(averageOverSubjectPrefsNormal[0]*100,2);
    precNormal=np.round(averageOverSubjectPrefsNormal[1]*100,2);
    recNormal=np.round(averageOverSubjectPrefsNormal[2]*100,2);
    veeNormal=np.round(abs(averageOverSubjectPrefsNormal[4]),2);
    
    f1Abnormal=np.round(averageOverSubjectPrefsAbnormal[0]*100,2);
    precAbnormal=np.round(averageOverSubjectPrefsAbnormal[1]*100,2);
    recAbnormal=np.round(averageOverSubjectPrefsAbnormal[2]*100,2);
    veeAbnormal=np.round(abs(averageOverSubjectPrefsAbnormal[4]),2);
    return [f1Normal,precNormal,recNormal,veeNormal,f1Abnormal,precAbnormal,recAbnormal,veeAbnormal],df