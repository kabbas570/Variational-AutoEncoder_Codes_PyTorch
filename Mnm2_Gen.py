from typing import List, Union, Tuple

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

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
#from typing import List, Union, Tuple
import torch
import cv2
from torch.utils.data import SubsetRandomSampler
import torchio as tio
from sklearn.model_selection import KFold
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256

   
def same_depth(img):
    temp = np.zeros([img.shape[0],17,DIM_,DIM_])
    temp[:,0:img.shape[1],:,:] = img
    return temp  
   
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
   
def generate_label_3(gt):
        temp_ = np.zeros([4,gt.shape[1],DIM_,DIM_])
        temp_[0:1,:,:,:][np.where(gt==1)]=1
        temp_[1:2,:,:,:][np.where(gt==2)]=1
        temp_[2:3,:,:,:][np.where(gt==3)]=1
        temp_[3:4,:,:,:][np.where(gt==0)]=1
        return temp_

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

def resample_image(image: sitk.Image ,
                       out_size: Union[None, Tuple[int]] = None, is_label: bool = False,
                       pad_value: float = 0) -> sitk.Image:
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        
        out_spacing = (1.25, 1.25,original_spacing[2])

        if original_size[-1] == 1:
            out_spacing = list(out_spacing)
            out_spacing[-1] = original_spacing[-1]
            out_spacing = tuple(out_spacing)
    
        if out_size is None:
            #out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
            out_size = (np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)

        else:
            out_size = np.array(out_size)
            
    
        original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
        original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
        out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    
        original_center = np.matmul(original_direction, original_center)
        out_center = np.matmul(original_direction, out_center)
        out_origin = np.array(image.GetOrigin()) + (original_center - out_center)
    
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size.tolist())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(out_origin.tolist())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(pad_value)
    
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
    
        return resample.Execute(image)
    
def SA_to_LA(SA_img,LA_img):    ## new LA image 
    size = (LA_img.GetSize()) 
    new_img = sitk.Image(LA_img)
    
    for x in range(0,size[0]):
        for y in range(0,size[1]):
            for z in range(0,size[2]):
                new_img[x,y,z] = 0
                point = LA_img.TransformIndexToPhysicalPoint([x,y,z])  ##  determine the physical location of a pixel:
                index_LA = SA_img.TransformPhysicalPointToIndex(point)
                if index_LA[0] < 0 or index_LA[0]>= SA_img.GetSize()[0]:
                    continue
                if index_LA[1] < 0 or index_LA[1]>= SA_img.GetSize()[1]:
                    continue
                if index_LA[2] < 0 or index_LA[2]>= SA_img.GetSize()[2]:
                    continue
                new_img[x,y,z] = SA_img[index_LA[0],index_LA[1],index_LA[2]]
    
    return new_img
    
class Dataset_io(Dataset): 
    def __init__(self, df, images_folder,transformations=None):
        self.df = df
        self.images_folder = images_folder
        self.vendors = df['VENDOR']
        self.images_name = df['SUBJECT_CODE'] 
        self.transformations = transformations
    def __len__(self):
        return self.vendors.shape[0]
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        ## SA_ES_img ####
        img_SA_path = img_path+'_SA_ES.nii.gz'
        img = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img = resample_image(img,is_label=False)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2] 
        img = Normalization_SA_ES(img)
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img)
        img = np.expand_dims(img, axis=0)
        img_SA_ES = same_depth(img)
        
        ## SA_ES_gt ####
        img_SA_gt_path = img_path+'_SA_ES_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=True)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img) 
        img = np.expand_dims(img, axis=0)
        img = generate_label_3(img) 
        temp_SA_ES = same_depth(img)
        
        ## SA_ED_img ####
        img_SA_path = img_path+'_SA_ED.nii.gz'
        img = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img = resample_image(img,is_label=False)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2] 
        img = Normalization_SA_ED(img)
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img)
        img = np.expand_dims(img, axis=0)
        img_SA_ED = same_depth(img)
        
        ## SA_ED_gt ####
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=True)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img) 
        img = np.expand_dims(img, axis=0)
        img = generate_label_3(img) 
        temp_SA_ED = same_depth(img)
        
        #### LA PART ### LA PART ###
        
        ## LA_ES_img ####
        img_SA_path = img_path+'_LA_ES.nii.gz'
        img = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img = resample_image(img,is_label=False)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2] 
        img = Normalization_LA_ES(img)
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img)
        img_LA_ES = np.expand_dims(img, axis=0)
        
        ## LA_ES_gt ####
        img_SA_gt_path = img_path+'_LA_ES_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=True)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img) 
        img = np.expand_dims(img, axis=0)
        temp_LA_ES = generate_label_3(img) 
        
        ## LA_ED_img ####
        img_SA_path = img_path+'_LA_ED.nii.gz'
        img = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img = resample_image(img,is_label=False)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2]
        img = Normalization_LA_ED(img)
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img)
        img_LA_ED = np.expand_dims(img, axis=0)
        
        ## LA_ED_gt ####
        img_SA_gt_path = img_path+'_LA_ED_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=True)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img) 
        img = np.expand_dims(img, axis=0)
        temp_LA_ED = generate_label_3(img) 
        
        print(self.images_name[index])
        
        return img_LA_ES,temp_LA_ES,img_LA_ED,temp_LA_ED,img_SA_ES,temp_SA_ES,img_SA_ED,temp_SA_ED
        

def Data_Loader_io_transforms(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


val_imgs = r"C:\My_Data\M2M Data\data\data_2\val"
val_csv_path = r"C:\My_Data\M2M Data\data\val.csv"
df_val = pd.read_csv(val_csv_path)
train_loader = Data_Loader_io_transforms(df_val,val_imgs,batch_size = 1)
a = iter(train_loader)
a1 = next(a)


for i in range(1):
    plt.figure()
    plt.imshow(a1[4][0,0,1,:])
    
for i in range(1):
    plt.figure()
    plt.imshow(a1[6][0,0,1,:])

# for i in range(7,8):
#     plt.figure()
#     plt.imshow(a1[3][0,3,i,:])

    
img = a1[6][0,0,10,:].numpy()

path = r'C:\My_Data\M2M Data\save2\img'

for t in range(17):
    img = a1[7][0,3,t,:].numpy().astype('uint8')
    #plt.imsave(os.path.join(path,str(t)+".png"),img,cmap='gray')
    cv2.imwrite(os.path.join(path , str(t)+".png"),img*255)
