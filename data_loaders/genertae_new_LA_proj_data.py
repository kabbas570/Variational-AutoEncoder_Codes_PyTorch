import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
import SimpleITK as sitk
import cv2
from typing import List, Union, Tuple

           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256

def Normalization_LA_ES(img):
        img = (img-114.8071)/191.2891
        return img 
def Normalization_LA_ED(img):
        img = (img-114.7321)/189.8573
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
   
   
class Dataset_V(Dataset): 
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
       
        #### LA PART ### LA PART ###
        
        ## LA_ES_img ####
        img_SA_gt_path = img_path+'_LA_ES.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=False)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_LA_ES(img)
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2] 
        LA_ES_img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img)         
        
        ## LA_ED_img ####
        img_SA_gt_path = img_path+'_LA_ED.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=False)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_LA_ED(img)
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2] 
        LA_ED_img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img) 
        
        ## LA_ES_gt ####
        img_SA_gt_path = img_path+'_LA_ES_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=True)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2] 
        temp_LA_ES = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img) 

        ## LA_ED_gt ####
        img_SA_gt_path = img_path+'_LA_ED_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=True)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2]
        temp_LA_ED = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img) 

        
        # ### sa_to_la_ES mapping ####
        img_path_SA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_SA_path = img_path_SA +'_SA_ES_gt.nii.gz'
        img_SA_1 = sitk.ReadImage(img_SA_path)
        
        img_path_LA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path = img_path_LA +'_LA_ES_gt.nii.gz'
        img_LA_1 = sitk.ReadImage(img_LA_path)
        
        new_SA_img = SA_to_LA(img_SA_1,img_LA_1)
        new_SA_img = resample_image(new_SA_img,is_label=True)
        new_SA_img = sitk.GetArrayFromImage(new_SA_img)
        
        org_dim3 = new_SA_img.shape[0]
        org_dim1 = new_SA_img.shape[1]
        org_dim2 = new_SA_img.shape[2] 
        proj_2d_ES = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,new_SA_img)
        
        # ### sa_to_la_ED mapping ####
        img_path_SA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_SA_path = img_path_SA +'_SA_ED_gt.nii.gz'
        img_SA_1 = sitk.ReadImage(img_SA_path)
        
        img_path_LA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path = img_path_LA +'_LA_ED_gt.nii.gz'
        img_LA_1 = sitk.ReadImage(img_LA_path)
        
        new_SA_img = SA_to_LA(img_SA_1,img_LA_1)
        new_SA_img = resample_image(new_SA_img,is_label=True)
        new_SA_img = sitk.GetArrayFromImage(new_SA_img)
        
        org_dim3 = new_SA_img.shape[0]
        org_dim1 = new_SA_img.shape[1]
        org_dim2 = new_SA_img.shape[2] 
        proj_2d_ED = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,new_SA_img)


        return LA_ES_img,temp_LA_ES,proj_2d_ES,LA_ED_img,temp_LA_ED,proj_2d_ED,self.images_name[index]
        
def Data_Loader_V(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_V(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader



val_imgs = r"C:\My_Data\M2M Data\data\data_2\train"
val_csv_path = r"C:\My_Data\M2M Data\data\train.csv"
df_val = pd.read_csv(val_csv_path)
train_loader = Data_Loader_V(df_val,val_imgs,batch_size = 1)
a = iter(train_loader)


# a1 = next(a)

# for i in range(1):
#     plt.figure()
#     plt.imshow(a1[0][0,i,:])

# for i in range(1):
#     plt.figure()
#     plt.imshow(a1[1][0,i,:])
    
# for i in range(1):
#     plt.figure()
#     plt.imshow(a1[2][0,i,:])
  

import nibabel as nib  

img_path = r'C:\My_Data\M2M Data\data\new_data\LA_Data\train\img'
gt_path = r'C:\My_Data\M2M Data\data\new_data\LA_Data\train\gt'
proj_path = r'C:\My_Data\M2M Data\data\new_data\LA_Data\train\proj'


for i in range(160):
    
    a1 = next(a)
    name = a1[6][0].numpy()
    
    img = a1[0][0,:].numpy() 
    gt = a1[1][0,:].numpy() 
    proj = a1[2][0,:].numpy() 
    
    to_format_img = nib.Nifti1Image(img, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(img_path,str(name)+'_ES'+'.nii.gz'))
    
    to_format_img = nib.Nifti1Image(gt, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(gt_path,str(name)+'_ES'+'.nii.gz'))
    
    to_format_img = nib.Nifti1Image(proj, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(proj_path,str(name)+'_ES'+'.nii.gz'))
    
    
    img = a1[3][0,:].numpy() 
    gt = a1[4][0,:].numpy() 
    proj = a1[5][0,:].numpy() 
    
    to_format_img = nib.Nifti1Image(img, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(img_path,str(name)+'_ED'+'.nii.gz'))
    
    to_format_img = nib.Nifti1Image(gt, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(gt_path,str(name)+'_ED'+'.nii.gz'))
    
    to_format_img = nib.Nifti1Image(proj, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(proj_path,str(name)+'_ED'+'.nii.gz'))
    
    
       
# es = sitk.ReadImage(r"C:\My_Data\M2M Data\data\new_data\val\img\197_ES.nii.gz")
# es = sitk.GetArrayFromImage(es)
 
# for i in range(3):
#     plt.figure()
#     plt.imshow(es[:,:,i])
