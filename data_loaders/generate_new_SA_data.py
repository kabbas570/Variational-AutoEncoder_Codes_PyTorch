import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),
            
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1,2,2),(1,2,2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        
        return self.maxpool_conv(x)
class Down_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((2,2,2),(2,2,2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels,out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
                
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_2(nn.Module):
    def __init__(self, in_channels,out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(2,2,2), stride=(2,2,2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
                
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class Up_last(nn.Module):
    def __init__(self, in_channels, in_channels1,out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2))
            self.conv = DoubleConv(in_channels1, out_channels)

    def forward(self, x1, x2):
                
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Tanh()
            )

    def forward(self, x):
        return self.conv(x)

class UNet_512(nn.Module):
    def __init__(self, n_channels = 1, bilinear=False):
        super(UNet_512, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down_2(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down_2(128,256)
        self.down4 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        self.up0 = Up(512, 256 // factor, bilinear)
        self.up1 = Up_2(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up_2(64, 32 // factor, bilinear)
        self.outc = OutConv(32,3)
                
        self.Drop_Out= nn.Dropout(p=0.20)
        

    def forward(self,x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = self.Drop_Out(x5)
        
        z1 = self.up0(x5, x4)
        z2 = self.up1(z1, x3)
        z3 = self.up2(z2, x2)
        z4 = self.up3(z3, x1)
        logits1 = self.outc(z4)
        return logits1
            
    
# Input_Image_Channels = 1
# def model() -> UNet_512:
#     model = UNet_512()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels,16, 256,256)])

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

   
def same_depth(img):
    temp = np.zeros([img.shape[0],16,DIM_,DIM_])
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
                
        out_spacing = (1.25, 1.25,10)

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

import torchio as tio

transforms_all = tio.OneOf({
        tio.RandomAffine(degrees=(30,0,0)): 0.3, ## for 2D rotation 

})
    
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
        img_SA_ES = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img)
        
        ## SA_ES_gt ####
        img_SA_gt_path = img_path+'_SA_ES_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=True)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        temp_SA_ES = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img) 
        
        ## SA_ED_img ####
        img_SA_path = img_path+'_SA_ED.nii.gz'
        img = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img = resample_image(img,is_label=False)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        org_dim3 = img.shape[0]
        org_dim1 = img.shape[1]
        org_dim2 = img.shape[2] 
        img = Normalization_SA_ED(img)
        img_SA_ED = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img)
        
        ## SA_ED_gt ####
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        img = resample_image(img,is_label=True)
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        temp_SA_ED = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img)         

        return img_SA_ES,temp_SA_ES,img_SA_ED,temp_SA_ED,self.images_name[index]
        
def Data_Loader_train(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


val_imgs = r"C:\My_Data\M2M Data\data\data_2\train"
val_csv_path = r"C:\My_Data\M2M Data\data\train.csv"
df_val = pd.read_csv(val_csv_path)
train_loader = Data_Loader_train(df_val,val_imgs,batch_size = 1)
a = iter(train_loader)


img_path = r'C:\My_Data\M2M Data\data\new_data\SA_Data\imgs'
gt_path = r'C:\My_Data\M2M Data\data\new_data\SA_Data\gts'
import nibabel as nib  

for i in range(1):
    a1 = next(a)
    
    name = a1[4][0].numpy()
    
    ### saving ESESES
    img_SA = a1[0][0,:].numpy()
    gt_SA = a1[1][0,:].numpy()
    
    new_img_sa = []
    new_gt_sa = []
    
    for k in range(img_SA.shape[0]):
        gt_slice = gt_SA[k,:]
        img_slice = img_SA[k,:]
        if np.sum(gt_slice)!=0:
            new_gt_sa.append(gt_slice)
            new_img_sa.append(img_slice)
        
    new_img_sa = np.array(new_img_sa)
    new_gt_sa = np.array(new_gt_sa)
    
    to_format_img = nib.Nifti1Image(new_img_sa, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(img_path,str(name)+'_ES'+'.nii.gz'))
    
    to_format_img = nib.Nifti1Image(new_gt_sa, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(gt_path,str(name)+'_ES'+'.nii.gz'))
    
    ## saving EDEDED  
    
    img_SA = a1[2][0,:].numpy()
    gt_SA = a1[3][0,:].numpy()
    
    new_img_sa = []
    new_gt_sa = []
    
    for k in range(img_SA.shape[0]):
        gt_slice = gt_SA[k,:]
        img_slice = img_SA[k,:]
        if np.sum(gt_slice)!=0:
            new_gt_sa.append(gt_slice)
            new_img_sa.append(img_slice)
        
    new_img_sa = np.array(new_img_sa)
    new_gt_sa = np.array(new_gt_sa)
    
    to_format_img = nib.Nifti1Image(new_img_sa, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(img_path,str(name)+'_ED'+'.nii.gz'))
    
    to_format_img = nib.Nifti1Image(new_gt_sa, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(gt_path,str(name)+'_ED'+'.nii.gz'))
    
            
        
    
img = sitk.ReadImage(r"C:\My_Data\M2M Data\data\new_data\SA_Data\gts\142_ES.nii.gz")
img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
for i in range(7):
         plt.imsave(r'C:\My_Data\M2M Data\data\new_data\SA_Data/' + str(i)+ 'ES'+ '.png', img[:,:,i])
         
         

for i in range(16):
         gt = a1[0][0,:].numpy()
         plt.imsave(r'C:\My_Data\M2M Data\data\new_data\SA_Data\check/' + str(i)+'img'+ 'ES'+ '.png', gt[i,:])
         
