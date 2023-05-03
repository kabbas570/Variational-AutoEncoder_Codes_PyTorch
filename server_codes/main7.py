import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np
import torch.nn.functional as F
from helper_utils import set_deterministic, set_all_seeds

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
#from typing import List, Union, Tuple
import torch
import albumentations as A
import cv2
from torch.utils.data import SubsetRandomSampler
import torchio as tio
from sklearn.model_selection import KFold
           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256
   

def Generate_Meta_(vendors_,scanners_,diseases_): 
    temp = np.zeros([2,17,8,8])
    if vendors_=='GE MEDICAL SYSTEMS': 
        temp[0,:,0:4,0:4] = 0.1
        temp[0,:,4:9,4:9] = 0.2
        temp[0,:,0:4,4:9] = 0.3
        temp[0,:,4:9,0:4] = 0.4
    if vendors_=='SIEMENS':
        temp[0,:,0:4,0:4] = 0.7
        temp[0,:,4:9,4:9] = 0.3
        temp[0,:,0:4,4:9] = 0.1
        temp[0,:,4:9,0:4] = 0.5
    if vendors_=='Philips Medical Systems':
        temp[0,:,0:4,0:4] = 0.8
        temp[0,:,4:9,4:9] = 0.6
        temp[0,:,0:4,4:9] = 0.9
        temp[0,:,4:9,0:4] = 0.1
    if scanners_=='Symphony':
        temp[1,:,0:4,0:4] = 0
        temp[1,:,4:9,0:4] = 0
        temp[1,:,0:4,4:9] = 1.6
        temp[1,:,5:9,5:9] = 0.9
    if scanners_=='SIGNA EXCITE':
        temp[1,:,0:3,0:8] = -0.1
        temp[1,:,3:8,0:2] = .3
        temp[1,:,6:8,6:8] = 1.9
    if scanners_=='Signa Explorer':
        temp[1,:,1:8,1:5] = 1.1
        temp[1,:,0:8,5:8] = .8
    if scanners_=='SymphonyTim':
        temp[1,:,0:3,:] = 1.6
        temp[1,:,5:8,:] = 1.1
    if scanners_=='Avanto Fit':
        temp[1,:,:,0:3] = -0.8
        temp[1,:,5:8] = 0.9
    if scanners_=='Avanto':
        temp[1,:,0:2,:] = -0.9
        temp[1,:,:,0:2] = 1.8
        temp[1,:,:,6:8] = -0.9
        temp[1,:,6:8,2:6] = 1.2
    if scanners_=='Achieva':
        temp[1,:,0:4,0:4] = 0.8
        temp[1,:,4:9,4:9] = 0.6
    if scanners_=='Signa HDxt':
        temp[1,:,0:4,4:9] = 0.2
        temp[1,:,4:9,0:4] = 0.4
    if scanners_=='TrioTim':
        temp[1,:,0:4,0:4] = 1.2
        temp[1,:,4:9,0:4] = 1.8
    
    return temp
           
def same_depth(img):
    temp = np.zeros([img.shape[0],17,DIM_,DIM_])
    temp[:,0:img.shape[1],:,:] = img
    return temp  
    
def resample_image_SA(itk_image):
    # get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_spacing=(1.25, 1.25, original_spacing[2])
    
    # calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / original_spacing[2])))
    ]
    # instantiate resample filter with properties and execute it
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(itk_image)
    

def resample_image_LA(itk_image):

    # get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_spacing=(1.25, 1.25, original_spacing[2])
    
    # calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    # instantiate resample filter with properties and execute it
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(itk_image)
    
def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_):
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_


def Normalization_1(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 

     
def generate_label_3(gt,org_dim3):
        temp_ = np.zeros([4,org_dim3,DIM_,DIM_])
        temp_[0,:,:,:][np.where(gt==1)]=1
        temp_[1,:,:,:][np.where(gt==2)]=1
        temp_[2,:,:,:][np.where(gt==3)]=1
        temp_[3,:,:,:][np.where(gt==0)]=1
        return temp_


def generate_label_4(gt,org_dim3):
        temp_ = np.zeros([1,org_dim3,DIM_,DIM_])
        temp_[0,:,:,:][np.where(gt==1)]=1
        temp_[0,:,:,:][np.where(gt==2)]=1
        temp_[0,:,:,:][np.where(gt==3)]=1
        return temp_

transforms_all = tio.OneOf({
        tio.RandomBiasField(): .3,  ## axis [0,1] or [1,2]
        #tio.RandomGhosting(axes=([1,2])): 0.3,
        #tio.RandomFlip(axes=([1,2])): .3,  ## axis [0,1] or [1,2]
        tio.RandomFlip(axes=([0,1])): .3,  ## axis [0,1] or [1,2]
        #tio.RandomAffine(degrees=(30,0,0)): 0.3, ## for 2D rotation 
        #tio.RandomMotion(degrees =(30) ):0.3 ,
        tio.RandomBlur(): 0.3,
        tio.RandomGamma(): 0.3,   
        #tio.RandomNoise(mean=0.1,std=0.1):0.20,
})

def Normalization_LA_ES(img):
        img = (img-114.8071)/191.2891
        return img 
def Normalization_LA_ED(img):
        img = (img-114.7321)/189.8573
        return img 
        
def Normalization_SA_ES(img):
        img = (img-62.5983)/147.4826
        return img 
def Normalization_SA_ED(img):
        img = (img-62.9529)/147.6579
        return img 
    
class Dataset_io(Dataset): 
    def __init__(self, df, images_folder,transformations=None):
        self.df = df
        self.images_folder = images_folder
        self.vendors = df['VENDOR']
        self.scanners = df['SCANNER']
        self.diseases=df['DISEASE']
        self.fields=df['FIELD']        
        self.images_name = df['SUBJECT_CODE'] 
        self.transformations = transformations
    def __len__(self):
        return self.vendors.shape[0]
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        ## sa_es_img ####
        img_SA_path = img_path+'_SA_ES.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_SA_ES(img_SA)
        # img_SA = Normalization_1(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ES = same_depth(img_SA)
        
        ## sa_es_gt ####
        img_SA_gt_path = img_path+'_SA_ES_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt) 
        img_SA_gt = np.expand_dims(img_SA_gt, axis=0)
        temp_SA_ES = same_depth(img_SA_gt)
       ### Augmentation for img_SA ####
        
        d = {}
        d['Image'] = tio.Image(tensor = img_SA_ES, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = temp_SA_ES, type=tio.LABEL)
        sample = tio.Subject(d)


        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_SA_ES = transformed_tensor['Image'].data
            temp_SA_ES = transformed_tensor['Mask'].data
        
        temp_SA_gt = temp_SA_ES[0,:]  ## expand dim is removed here
        temp_SA_ES = generate_label_3(temp_SA_gt,17) 
        
        all_three_ES_SA = generate_label_4(temp_SA_gt,17)
        

        #####    LA Images #####
        ## la_es_img ####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ES.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        img_LA_ES = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA_ES = Normalization_LA_ES(img_LA_ES)
        img_LA_ES = np.expand_dims(img_LA_ES, axis=0)
        ## la_es_gt ####
        
        img_LA_gt_path = img_path+'_LA_ES_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt) 
        img_LA_gt = np.expand_dims(img_LA_gt, axis=0)
                
        ### Augmentation for img_LA #### 
        
        d = {}
        d['Image'] = tio.Image(tensor = img_LA_ES, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = img_LA_gt, type=tio.LABEL)
        sample = tio.Subject(d)
        
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_LA_ES = transformed_tensor['Image'].data
            img_LA_gt = transformed_tensor['Mask'].data
        

        img_LA_ES = img_LA_ES[0,:]
        img_LA_gt = img_LA_gt[0,:]
        
        temp_LA_ES = generate_label_3(img_LA_gt,1)
        temp_LA_ES = temp_LA_ES[:,0,:,:]
        
        all_three_ES = generate_label_4(img_LA_gt,1)
        all_three_ES = all_three_ES[:,0,:,:]
        
        ## ED images ##
        ## sa_ED_img ####
        
        img_SA_path = img_path+'_SA_ED.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_SA_ED(img_SA)
        # img_SA = Normalization_1(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ED = same_depth(img_SA)
        

        ## sa_ed_gt ####
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt) 
        img_SA_gt = np.expand_dims(img_SA_gt, axis=0)
        temp_SA_ED = same_depth(img_SA_gt)
        
        ### Augmentation for img_SA ####
        d = {}
        d['Image'] = tio.Image(tensor = img_SA_ED, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = temp_SA_ED, type=tio.LABEL)
        sample = tio.Subject(d)


        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_SA_ED = transformed_tensor['Image'].data
            temp_SA_ED = transformed_tensor['Mask'].data
        
        temp_SA_gt = temp_SA_ED[0,:]  ## expand im is removed here
        temp_SA_ED = generate_label_3(temp_SA_gt,17) 
        
        all_three_ED_SA = generate_label_4(temp_SA_gt,17)
        
        #####    LA Images #####
        ## la_ed_img ####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ED.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        img_LA_ED = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA_ED = Normalization_LA_ED(img_LA_ED)
        img_LA_ED = np.expand_dims(img_LA_ED, axis=0)

        img_LA_gt_path = img_path+'_LA_ED_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt) 
        img_LA_gt = np.expand_dims(img_LA_gt, axis=0)
        
        ### Augmentation for img_LA #### 
        
        d = {}
        d['Image'] = tio.Image(tensor = img_LA_ED, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = img_LA_gt, type=tio.LABEL)
        sample = tio.Subject(d)
        
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_LA_ED = transformed_tensor['Image'].data
            img_LA_gt = transformed_tensor['Mask'].data
        
        img_LA_ED = img_LA_ED[0,:]
        img_LA_gt = img_LA_gt[0,:]
        
        temp_LA_ED = generate_label_3(img_LA_gt,1)
        temp_LA_ED = temp_LA_ED[:,0,:,:]
        
        all_three_ED = generate_label_4(img_LA_gt,1)
        all_three_ED = all_three_ED[:,0,:,:]
        
        ## meta data ##
        vendors_ = self.vendors[index]
        scanners_ = self.scanners[index]
        diseases_ = self.diseases[index]
        M = Generate_Meta_(vendors_,scanners_,diseases_)
        return img_LA_ES,temp_LA_ES[:,:,:],img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED[:,:,:],img_SA_ED,temp_SA_ED,self.images_name[index],M,all_three_ES,all_three_ED,all_three_ES_SA,all_three_ED_SA

def Data_Loader_io_transforms(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

class Dataset_V(Dataset): 
    def __init__(self, df, images_folder,transformations=None):
        self.df = df
        self.images_folder = images_folder
        self.vendors = df['VENDOR']
        self.scanners = df['SCANNER']
        self.diseases=df['DISEASE']
        self.fields=df['FIELD']        
        self.images_name = df['SUBJECT_CODE'] 
        self.transformations = transformations
    def __len__(self):
        return self.vendors.shape[0]
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        ## sa_es_img ####
        img_SA_path = img_path+'_SA_ES.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_SA_ES(img_SA)
        # img_SA = Normalization_1(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ES = same_depth(img_SA)
        
        ## sa_es_gt ####
        img_SA_gt_path = img_path+'_SA_ES_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt) 
        img_SA_gt = np.expand_dims(img_SA_gt, axis=0)
        temp_SA_ES = same_depth(img_SA_gt)
       ### Augmentation for img_SA ####
        
        d = {}
        d['Image'] = tio.Image(tensor = img_SA_ES, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = temp_SA_ES, type=tio.LABEL)
        sample = tio.Subject(d)


        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_SA_ES = transformed_tensor['Image'].data
            temp_SA_ES = transformed_tensor['Mask'].data
        
        temp_SA_gt = temp_SA_ES[0,:]  ## expand dim is removed here
        temp_SA_ES = generate_label_3(temp_SA_gt,17) 
        
        all_three_ES_SA = generate_label_4(temp_SA_gt,17)
        

        #####    LA Images #####
        ## la_es_img ####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ES.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        img_LA_ES = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA_ES = Normalization_LA_ES(img_LA_ES)
        img_LA_ES = np.expand_dims(img_LA_ES, axis=0)
        ## la_es_gt ####
        
        img_LA_gt_path = img_path+'_LA_ES_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt) 
        img_LA_gt = np.expand_dims(img_LA_gt, axis=0)
                
        ### Augmentation for img_LA #### 
        
        d = {}
        d['Image'] = tio.Image(tensor = img_LA_ES, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = img_LA_gt, type=tio.LABEL)
        sample = tio.Subject(d)
        
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_LA_ES = transformed_tensor['Image'].data
            img_LA_gt = transformed_tensor['Mask'].data
        

        img_LA_ES = img_LA_ES[0,:]
        img_LA_gt = img_LA_gt[0,:]
        
        temp_LA_ES = generate_label_3(img_LA_gt,1)
        temp_LA_ES = temp_LA_ES[:,0,:,:]
        
        all_three_ES = generate_label_4(img_LA_gt,1)
        all_three_ES = all_three_ES[:,0,:,:]
        
        ## ED images ##
        ## sa_ED_img ####
        
        img_SA_path = img_path+'_SA_ED.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_SA_ED(img_SA)
        # img_SA = Normalization_1(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ED = same_depth(img_SA)
        

        ## sa_ed_gt ####
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt) 
        img_SA_gt = np.expand_dims(img_SA_gt, axis=0)
        temp_SA_ED = same_depth(img_SA_gt)
        
        ### Augmentation for img_SA ####
        d = {}
        d['Image'] = tio.Image(tensor = img_SA_ED, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = temp_SA_ED, type=tio.LABEL)
        sample = tio.Subject(d)


        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_SA_ED = transformed_tensor['Image'].data
            temp_SA_ED = transformed_tensor['Mask'].data
        
        temp_SA_gt = temp_SA_ED[0,:]  ## expand im is removed here
        temp_SA_ED = generate_label_3(temp_SA_gt,17) 
        
        all_three_ED_SA = generate_label_4(temp_SA_gt,17)
        
        #####    LA Images #####
        ## la_ed_img ####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ED.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        img_LA_ED = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA_ED = Normalization_LA_ED(img_LA_ED)
        img_LA_ED = np.expand_dims(img_LA_ED, axis=0)

        img_LA_gt_path = img_path+'_LA_ED_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt) 
        img_LA_gt = np.expand_dims(img_LA_gt, axis=0)
        
        ### Augmentation for img_LA #### 
        
        d = {}
        d['Image'] = tio.Image(tensor = img_LA_ED, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = img_LA_gt, type=tio.LABEL)
        sample = tio.Subject(d)
        
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_LA_ED = transformed_tensor['Image'].data
            img_LA_gt = transformed_tensor['Mask'].data
        
        img_LA_ED = img_LA_ED[0,:]
        img_LA_gt = img_LA_gt[0,:]
        
        temp_LA_ED = generate_label_3(img_LA_gt,1)
        temp_LA_ED = temp_LA_ED[:,0,:,:]
        
        all_three_ED = generate_label_4(img_LA_gt,1)
        all_three_ED = all_three_ED[:,0,:,:]
        
        ## meta data ##
        vendors_ = self.vendors[index]
        scanners_ = self.scanners[index]
        diseases_ = self.diseases[index]
        M = Generate_Meta_(vendors_,scanners_,diseases_)
        return img_LA_ES,temp_LA_ES[:,:,:],img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED[:,:,:],img_SA_ED,temp_SA_ED,self.images_name[index],M,all_three_ES,all_three_ED,all_three_ES_SA,all_three_ED_SA

def Data_Loader_V(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_V(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

# val_imgs = r'C:\My_Data\M2M Data\data\data_2/val'
# val_csv_path = r'C:\My_Data\M2M Data\data\val.csv'
# df_val = pd.read_csv(val_csv_path)
# train_loader = Data_Loader_io_transforms(df_val,val_imgs,batch_size = 1)
# #train_loader = Data_Loader_V(df_val,val_imgs,batch_size = 1)
# a = iter(train_loader)
# a1 =next(a)
# gt2=a1[10][0,0,:,:]
# plt.figure()
# plt.imshow(gt2)


# Hyperparameters
RANDOM_SEED = 123
BATCH_SIZE = 3

set_deterministic
set_all_seeds(RANDOM_SEED)

   #### Specify all the paths here #####
   
train_imgs='/data/scratch/acw676/MnM/data_2/train/'
val_imgs='/data/scratch/acw676/MnM/data_2/val/'

train_csv_path='/data/scratch/acw676/MnM/train.csv'
val_csv_path='/data/scratch/acw676/MnM/val.csv'

df_train = pd.read_csv(train_csv_path)
df_val = pd.read_csv(val_csv_path)

train_loader = Data_Loader_io_transforms(df_train,train_imgs,BATCH_SIZE)
val_loader = Data_Loader_V(df_val,val_imgs,BATCH_SIZE)
   
print(len(train_loader)) ### this shoud be = Total_images/ batch size
print(len(val_loader))   ### same here
#print(len(test_loader))   ### same here

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('Device:', DEVICE)



##########################
### Dataset
##########################

    
### uper part is fine ###

Max_Epochs = 30
LEARNING_RATE = 5e-3

        #### Import All libraies used for training  #####
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#import numpy as np
from tqdm import tqdm
import torch.optim as optim

avg_train_losses1 = []   
avg_valid_losses1 = [] 
avg_valid_DS1 = []  


# #class IoULoss(nn.Module):
# #    def __init__(self, weight=None, size_average=True):
# #        super(IoULoss, self).__init__()
# #
# #    def forward(self, inputs, targets, smooth=1):
# #        # inputs = torch.sigmoid(inputs) 
# #        inputs = inputs.view(-1)
# #        targets = targets.view(-1)
# #        intersection = (inputs * targets).sum()
# #        total = (inputs + targets).sum()
# #        union = total - intersection   
# #        IoU = (intersection + smooth)/(union + smooth)          
# #        return 1 - IoU

# #loss_fn1 =IoULoss()
# loss_fn1 = F.mse_loss

# #def loss_func(z_mean, z_log_var, decoded,features):
# #    reconstruction_term_weight = 1
# ##    kl_div = -0.5 * torch.sum(1 + z_log_var 
# ##                              - z_mean**2 
# ##                              - torch.exp(z_log_var), 
# ##                              axis=1) # sum over latent dimension
# #    kl_div = -0.5 * 100 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
# #
# #    # kl_div = kl_div.mean() # average over batch dimension
# #    recons = loss_fn1(decoded,features)
# #    loss = reconstruction_term_weight*recons + kl_div
# #    
# #    return loss

# beta = 2
# def loss_func(mu, log_var,recon_x,x):
#     bce = loss_fn1(recon_x, x)
#     kld = -0.5 * beta * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#     return bce + kld

    
from losses import DiceLoss

loss_fn_dice = DiceLoss()
loss_fn_mse = F.mse_loss

def train_fn(loader_train1,loader_valid1,model1,model2,model3, optimizer1, scaler):  ### Loader_1--> ED and Loader2-->ES
    train_losses1 = [] # loss of each batch
    valid_losses1 = []  # loss of each batch

    loop = tqdm(loader_train1)
    model1.eval()
    model2.eval() 
    model3.train() 
    
    for batch_idx, (img_LA_ES,temp_LA_ES,img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED,img_SA_ED,temp_SA_ED,label,M,all_three_ES,all_three_ED,three_ES_SA,three_ED_SA) in enumerate(loop):
       
       img_LA_ES = img_LA_ES.to(device=DEVICE,dtype=torch.float)  
       temp_LA_ES = temp_LA_ES.to(device=DEVICE,dtype=torch.float)
       img_SA_ES = img_SA_ES.to(device=DEVICE,dtype=torch.float)  
       temp_SA_ES = temp_SA_ES.to(device=DEVICE,dtype=torch.float)
       
       img_LA_ED = img_LA_ED.to(device=DEVICE,dtype=torch.float)  
       temp_LA_ED = temp_LA_ED.to(device=DEVICE,dtype=torch.float)
       img_SA_ED = img_SA_ED.to(device=DEVICE,dtype=torch.float)  
       temp_SA_ED = temp_SA_ED.to(device=DEVICE,dtype=torch.float)

       M = M.to(device=DEVICE,dtype=torch.float)
       all_three_ES = all_three_ES.to(device=DEVICE,dtype=torch.float)  
       all_three_ED = all_three_ED.to(device=DEVICE,dtype=torch.float)
       
       three_ES_SA = three_ES_SA.to(device=DEVICE,dtype=torch.float)
       three_ED_SA = three_ED_SA.to(device=DEVICE,dtype=torch.float)

       with torch.cuda.amp.autocast():
           
             features_2d = model1.encoder(temp_LA_ES)
             # x_L_2D = model1.L_Space(features_2d)
             
             features_3d = model2.encoder(temp_SA_ES)
             # x_L_3D = model2.L_Space(features_3d)
             
             encoded_2d,encoded_3d = model3(features_2d,features_3d)
             
             x_L_2 = model1.L_Space(encoded_2d)
             out_2D = model1.decoder(x_L_2)
             
             x_L_3 = model2.L_Space(encoded_3d)
             out_3D = model2.decoder(x_L_3)
             
             #print(x_L_2D.shape)
             #print(encoded_2d.shape)
             
             loss1 = loss_fn_mse(encoded_2d, features_2d)
             loss2 = loss_fn_mse(encoded_3d, features_3d)
             loss3 = loss_fn_dice(out_2D, temp_LA_ES)
             loss4 = loss_fn_dice(out_3D, temp_SA_ES)
             loss_1 = (loss1+loss2+loss3+loss4)/4
             
             ### FOR ED ###
             features_2d = model1.encoder(temp_LA_ED)
             # x_L_2D = model1.L_Space(features_2d)
             
             features_3d = model2.encoder(temp_SA_ED)
             # x_L_3D = model2.L_Space(features_3d)
             
             encoded_2d,encoded_3d = model3(features_2d,features_3d)
             
             x_L_2 = model1.L_Space(encoded_2d)
             out_2D = model1.decoder(x_L_2)
             
             x_L_3 = model2.L_Space(encoded_3d)
             out_3D = model2.decoder(x_L_3)
             
             loss1 = loss_fn_mse(encoded_2d, features_2d)
             loss2 = loss_fn_mse(encoded_3d, features_3d)
             loss3 = loss_fn_dice(out_2D, temp_LA_ED)
             loss4 = loss_fn_dice(out_3D, temp_SA_ED)
             loss_2 = (loss1+loss2+loss3+loss4)/4


            # backward
       loss = (loss_1+loss_2)/2
       optimizer1.zero_grad()
       scaler.scale(loss).backward()
       scaler.step(optimizer1)
       scaler.update()
        # update tqdm loop
       loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
       train_losses1.append(float(loss))
        
    loop_v = tqdm(loader_valid1)
    model1.eval() 
    model2.eval() 
    model3.eval() 
    for batch_idx, (img_LA_ES,temp_LA_ES,img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED,img_SA_ED,temp_SA_ED,label,M,all_three_ES,all_three_ED,three_ES_SA,three_ED_SA) in enumerate(loop_v):
       
       img_LA_ES = img_LA_ES.to(device=DEVICE,dtype=torch.float)  
       temp_LA_ES = temp_LA_ES.to(device=DEVICE,dtype=torch.float)
       img_SA_ES = img_SA_ES.to(device=DEVICE,dtype=torch.float)  
       temp_SA_ES = temp_SA_ES.to(device=DEVICE,dtype=torch.float)
       
       img_LA_ED = img_LA_ED.to(device=DEVICE,dtype=torch.float)  
       temp_LA_ED = temp_LA_ED.to(device=DEVICE,dtype=torch.float)
       img_SA_ED = img_SA_ED.to(device=DEVICE,dtype=torch.float)  
       temp_SA_ED = temp_SA_ED.to(device=DEVICE,dtype=torch.float)

       M = M.to(device=DEVICE,dtype=torch.float)
       all_three_ES = all_three_ES.to(device=DEVICE,dtype=torch.float)  
       all_three_ED = all_three_ED.to(device=DEVICE,dtype=torch.float)
       
       three_ES_SA = three_ES_SA.to(device=DEVICE,dtype=torch.float)
       three_ED_SA = three_ED_SA.to(device=DEVICE,dtype=torch.float)

       with torch.cuda.amp.autocast():
           
             features_2d = model1.encoder(temp_LA_ES)
             #x_L_2D = model1.L_Space(features_2d)
             
             features_3d = model2.encoder(temp_SA_ES)
             #x_L_3D = model2.L_Space(features_3d)
             
             encoded_2d,encoded_3d = model3(features_2d,features_3d)
             
             x_L_2 = model1.L_Space(encoded_2d)
             out_2D = model1.decoder(x_L_2)
             
             x_L_3 = model2.L_Space(encoded_3d)
             out_3D = model2.decoder(x_L_3)
             
             loss1 = loss_fn_mse(encoded_2d, features_2d)
             loss2 = loss_fn_mse(encoded_3d, features_3d)
             loss3 = loss_fn_dice(out_2D, temp_LA_ES)
             loss4 = loss_fn_dice(out_3D, temp_SA_ES)
             loss_1 = (loss1+loss2+loss3+loss4)/4
             
             ### FOR ED ###
             features_2d = model1.encoder(temp_LA_ED)
             #x_L_2D = model1.L_Space(features_2d)
             
             features_3d = model2.encoder(temp_SA_ED)
             #x_L_3D = model2.L_Space(features_3d)
             
             encoded_2d,encoded_3d = model3(features_2d,features_3d)
             
             x_L_2 = model1.L_Space(encoded_2d)
             out_2D = model1.decoder(x_L_2)
             
             x_L_3 = model2.L_Space(encoded_3d)
             out_3D = model2.decoder(x_L_3)
             
             loss1 = loss_fn_mse(encoded_2d, features_2d)
             loss2 = loss_fn_mse(encoded_3d, features_3d)
             loss3 = loss_fn_dice(out_2D, temp_LA_ED)
             loss4 = loss_fn_dice(out_3D, temp_SA_ED)
             loss_2 = (loss1+loss2+loss3+loss4)/4
             
            # backward
       loss = (loss_1+loss_2)/2
       loop_v.set_postfix(loss = loss.item())
       valid_losses1.append(float(loss))

    train_loss_per_epoch1 = np.average(train_losses1)
    valid_loss_per_epoch1 = np.average(valid_losses1)
    ## all epochs
    avg_train_losses1.append(train_loss_per_epoch1)
    avg_valid_losses1.append(valid_loss_per_epoch1)
    
    return train_loss_per_epoch1, valid_loss_per_epoch1


def check_Acc(loader, model1,model2,model3, device=DEVICE):
    loop = tqdm(loader)
    
    Dice_score_LA_RV_ES = 0
    Dice_score_LA_MYO_ES = 0
    Dice_score_LA_LV_ES = 0
    
    Dice_score_LA_RV_ED = 0
    Dice_score_LA_MYO_ED = 0
    Dice_score_LA_LV_ED = 0
    
    Dice_score_SA_RV_ES = 0
    Dice_score_SA_MYO_ES = 0
    Dice_score_SA_LV_ES = 0
    
    Dice_score_SA_RV_ED = 0
    Dice_score_SA_MYO_ED = 0
    Dice_score_SA_LV_ED = 0
    

    model1.eval() 
    model2.eval() 
    model3.eval() 
    
    for batch_idx, (img_LA_ES,temp_LA_ES,img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED,img_SA_ED,temp_SA_ED,label,M,all_three_ES,all_three_ED,three_ES_SA,three_ED_SA) in enumerate(loop):
       
       img_LA_ES = img_LA_ES.to(device=DEVICE,dtype=torch.float)  
       temp_LA_ES = temp_LA_ES.to(device=DEVICE,dtype=torch.float)
       img_SA_ES = img_SA_ES.to(device=DEVICE,dtype=torch.float)  
       temp_SA_ES = temp_SA_ES.to(device=DEVICE,dtype=torch.float)
       
       img_LA_ED = img_LA_ED.to(device=DEVICE,dtype=torch.float)  
       temp_LA_ED = temp_LA_ED.to(device=DEVICE,dtype=torch.float)
       img_SA_ED = img_SA_ED.to(device=DEVICE,dtype=torch.float)  
       temp_SA_ED = temp_SA_ED.to(device=DEVICE,dtype=torch.float)

       M = M.to(device=DEVICE,dtype=torch.float)
       all_three_ES = all_three_ES.to(device=DEVICE,dtype=torch.float)  
       all_three_ED = all_three_ED.to(device=DEVICE,dtype=torch.float)
       
       three_ES_SA = three_ES_SA.to(device=DEVICE,dtype=torch.float)
       three_ED_SA = three_ED_SA.to(device=DEVICE,dtype=torch.float)
       
       with torch.cuda.amp.autocast():
            
            features_2d = model1.encoder(temp_LA_ES)            
            features_3d = model2.encoder(temp_SA_ES)
            encoded_2d,encoded_3d = model3(features_2d,features_3d)
            
            x_L_2 = model1.L_Space(encoded_2d)
            out_2D_ES = model1.decoder(x_L_2)
            
            x_L_3 = model2.L_Space(encoded_3d)
            out_3D_ES = model2.decoder(x_L_3)
            
            out_LA_ES = (out_2D_ES > 0.5)*1
            out_LA_ES_LV = out_LA_ES[:,0:1,:]
            out_LA_ES_MYO = out_LA_ES[:,1:2,:]
            out_LA_ES_RV = out_LA_ES[:,2:3,:]
            
            out_SA_ES = (out_3D_ES > 0.5)*1
            out_SA_ES_LV = out_SA_ES[:,0:1,:]
            out_SA_ES_MYO = out_SA_ES[:,1:2,:]
            out_SA_ES_RV = out_SA_ES[:,2:3,:]
            

            features_2d = model1.encoder(temp_LA_ED)            
            features_3d = model2.encoder(temp_SA_ED)
            encoded_2d,encoded_3d = model3(features_2d,features_3d)
            
            x_L_2 = model1.L_Space(encoded_2d)
            out_2D_ED = model1.decoder(x_L_2)
            
            x_L_3 = model2.L_Space(encoded_3d)
            out_3D_ED = model2.decoder(x_L_3)
            
            out_LA_ED = (out_2D_ED > 0.5)*1
            out_LA_ED_LV = out_LA_ED[:,0:1,:]
            out_LA_ED_MYO = out_LA_ED[:,1:2,:]
            out_LA_ED_RV = out_LA_ED[:,2:3,:]
            
            out_SA_ED = (out_3D_ED > 0.5)*1
            out_SA_ED_LV = out_SA_ED[:,0:1,:]
            out_SA_ED_MYO = out_SA_ED[:,1:2,:]
            out_SA_ED_RV = out_SA_ED[:,2:3,:]
            
            
            ## Dice Score for ES-LA ###

            single_LA_LV_ES = (2 * (out_LA_ES_LV * temp_LA_ES[:,0:1,:]).sum()) / (
               (out_LA_ES_LV + temp_LA_ES[:,0:1,:]).sum() + 1e-8)
           
            Dice_score_LA_LV_ES +=single_LA_LV_ES
           
            single_LA_MYO_ES = (2 * (out_LA_ES_MYO*temp_LA_ES[:,1:2,:]).sum()) / (
   (out_LA_ES_MYO + temp_LA_ES[:,1:2,:]).sum() + 1e-8)
            Dice_score_LA_MYO_ES += single_LA_MYO_ES

            single_LA_RV_ES = (2 * (out_LA_ES_RV* temp_LA_ES[:,2:3,:]).sum()) / (
       (out_LA_ES_RV + temp_LA_ES[:,2:3,:]).sum() + 1e-8)
            Dice_score_LA_RV_ES += single_LA_RV_ES
            
            ## Dice Score for ED-LA ###

            single_LA_LV_ED = (2 * (out_LA_ED_LV * temp_LA_ED[:,0:1,:]).sum()) / (
                (out_LA_ED_LV + temp_LA_ED[:,0:1,:]).sum() + 1e-8)
            
            Dice_score_LA_LV_ED +=single_LA_LV_ED
            
            single_LA_MYO_ED = (2 * (out_LA_ED_MYO*temp_LA_ED[:,1:2,:]).sum()) / (
    (out_LA_ED_MYO + temp_LA_ED[:,1:2,:]).sum() + 1e-8)
            Dice_score_LA_MYO_ED += single_LA_MYO_ED

            single_LA_RV_ED = (2 * (out_LA_ED_RV* temp_LA_ED[:,2:3,:]).sum()) / (
        (out_LA_ED_RV + temp_LA_ED[:,2:3,:]).sum() + 1e-8)
            Dice_score_LA_RV_ED += single_LA_RV_ED
            
            
            ## Dice Score for ES-SA ###

            single_SA_LV_ES = (2 * (out_SA_ES_LV * temp_SA_ES[:,0:1,:]).sum()) / (
                        (out_SA_ES_LV + temp_SA_ES[:,0:1,:]).sum() + 1e-8)
                    
            Dice_score_SA_LV_ES +=single_SA_LV_ES
                    
            single_SA_MYO_ES = (2 * (out_SA_ES_MYO*temp_SA_ES[:,1:2,:]).sum()) / (
            (out_SA_ES_MYO + temp_SA_ES[:,1:2,:]).sum() + 1e-8)
            Dice_score_SA_MYO_ES += single_SA_MYO_ES

            single_SA_RV_ES = (2 * (out_SA_ES_RV* temp_SA_ES[:,2:3,:]).sum()) / (
                (out_SA_ES_RV + temp_SA_ES[:,2:3,:]).sum() + 1e-8)
            Dice_score_SA_RV_ES += single_SA_RV_ES
                     
                     ## Dice Score for ED-SA ###

            single_SA_LV_ED = (2 * (out_SA_ED_LV * temp_SA_ED[:,0:1,:]).sum()) / (
                         (out_SA_ED_LV + temp_SA_ED[:,0:1,:]).sum() + 1e-8)
                     
            Dice_score_SA_LV_ED +=single_SA_LV_ED
                     
            single_SA_MYO_ED = (2 * (out_SA_ED_MYO*temp_SA_ED[:,1:2,:]).sum()) / (
             (out_SA_ED_MYO + temp_SA_ED[:,1:2,:]).sum() + 1e-8)
            Dice_score_SA_MYO_ED += single_SA_MYO_ED

            single_SA_RV_ED = (2 * (out_SA_ED_RV* temp_SA_ED[:,2:3,:]).sum()) / (
                 (out_SA_ED_RV + temp_SA_ED[:,2:3,:]).sum() + 1e-8)
            Dice_score_SA_RV_ED += single_SA_RV_ED
            
            
    Dice_RV_LA = (Dice_score_LA_RV_ES + Dice_score_LA_RV_ED)/2
    Dice_MYO_LA = (Dice_score_LA_MYO_ES + Dice_score_LA_MYO_ED)/2
    Dice_LV_LA = (Dice_score_LA_LV_ES + Dice_score_LA_LV_ED)/2
    
    print(f"Dice_RV_LA  : {Dice_RV_LA/len(loader)}")
    print(f"Dice_MYO_LA  : {Dice_MYO_LA/len(loader)}")
    print(f"Dice_LV_LA  : {Dice_LV_LA/len(loader)}")

    Overall_Dicescore_LA =( Dice_RV_LA + Dice_MYO_LA + Dice_LV_LA ) /3
    
    Dice_RV_SA = (Dice_score_SA_RV_ES + Dice_score_SA_RV_ED)/2
    Dice_MYO_SA = (Dice_score_SA_MYO_ES + Dice_score_SA_MYO_ED)/2
    Dice_LV_SA = (Dice_score_SA_LV_ES + Dice_score_SA_LV_ED)/2
    
    print(f"Dice_RV_SA  : {Dice_RV_SA/len(loader)}")
    print(f"Dice_MYO_SA  : {Dice_MYO_SA/len(loader)}")
    print(f"Dice_LV_SA  : {Dice_LV_SA/len(loader)}")

    Overall_Dicescore_SA = ( Dice_RV_SA + Dice_MYO_SA + Dice_LV_SA ) /3
    
    Overall_Both  = (Overall_Dicescore_LA+Overall_Dicescore_SA)/2

    return Overall_Both/len(loader)
    
    
### 6 - This is Focal Tversky Loss loss function ### 
     
epoch_len = len(str(Max_Epochs))

path_to_save_Learning_Curve='/data/scratch/acw676/VAE_weights/'+'/Recons_Model'
path_to_save_check_points='//data/scratch/acw676/VAE_weights/'+'/Recons_Model'  ##these are weights
### 3 - this function will save the check-points 
def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
 
from models import Autoencoder_2D2_1, Autoencoder_3D3_1,Recons_Model
model_1 = Autoencoder_2D2_1()
model_2 = Autoencoder_3D3_1()
model_3 = Recons_Model()

def main():
    
    model1 = model_1.to(device=DEVICE,dtype=torch.float)
    optimizer = optim.Adam(model1.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
    weights_paths= "/data/scratch/acw676/VAE_weights/Autoencoder_2D2_1.pth.tar"
    checkpoint = torch.load(weights_paths,map_location=DEVICE)
    model1.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    model2 = model_2.to(device=DEVICE,dtype=torch.float)
    optimizer = optim.Adam(model2.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
    weights_paths= "/data/scratch/acw676/VAE_weights/Autoencoder_3D3_1.pth.tar"
    checkpoint = torch.load(weights_paths,map_location=DEVICE)
    model2.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    model3 = model_3.to(device=DEVICE,dtype=torch.float)
    optimizer1 = optim.Adam(model3.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(Max_Epochs):
        train_loss,valid_loss=train_fn(train_loader,val_loader, model1,model2,model3, optimizer1,scaler)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        dice_score = check_Acc(val_loader, model1,model2,model3, device=DEVICE)
        # avg_valid_DS1.append(dice_score)
        avg_valid_DS1.append(dice_score.detach().cpu().numpy())
        checkpoint = {
                "state_dict": model3.state_dict(),
                "optimizer":optimizer1.state_dict(),
            }
        save_checkpoint(checkpoint)

if __name__ == "__main__":
    main()

### This part of the code will generate the learning curve ......

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses1)+1),avg_train_losses1, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses1)+1),avg_valid_losses1,label='Validation Loss')
plt.plot(range(1,len(avg_valid_DS1)+1),avg_valid_DS1,label='Validation DS')
# find position of lowest validation loss
minposs = avg_valid_losses1.index(min(avg_valid_losses1))+1 
plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')

font1 = {'size':20}
plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scalev
plt.xlim(0, len(avg_train_losses1)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')